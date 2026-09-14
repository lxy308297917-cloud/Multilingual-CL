import datasets
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, default_data_collator)

from torch.utils.data import DataLoader
# from utils.replay_buffer import ReplayBuffer
from transformers import TrainerCallback
from utils.ewc import EWC, EWCConfig
from utils.lwf import LwF, LwFConfig
import os
import gc
from utils.gpm import GPMManager, GPMConfig,GPMCallback


from typing import Optional, List
try:
    from peft import (
        LoraConfig,
        AdaLoraConfig,
        TaskType,
        get_peft_model,
    )
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

from utils import (CustomArgumentParser,
                   freeze_random_parameters,
                   create_calibration_dataloader,
                   create_gmt_trainer,
                   # LoTA
                   lota_calibrate_mask, lota_prepare_sparse_training, lota_parameter_summary,
                   # S2FT
                   s2ft_enable,)

from plnd import apply_plnd_mask_from_env
# LA-SSU: Language-aware Soft SSU
# 这里直接从 model_utils 导入，避免必须修改 utils/__init__.py。
from utils.model_utils import (
    collect_lassu_column_scores,
    apply_lassu_from_score_files,
    apply_lassu_english_anchor_from_score_files,
)
from utils.s2_utils import convert_s2_modules_to_linear
from utils.assistant_loss import per_sequence_causal_lm_loss

class ExactSaveStepsCallback(TrainerCallback):
    """Only allow step checkpoints explicitly registered by the controller."""

    def __init__(self, steps):
        self.steps = {int(step) for step in steps}

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % int(args.save_steps) == 0:
            control.should_save = int(state.global_step) in self.steps
        return control


def apply_trainable_layer_range_from_env(model):
    """
    Optional layer-range training.

    用环境变量控制：
        export TRAINABLE_LAYER_RANGE="23-27"

    效果：
        只允许 model.layers.23 ~ model.layers.27 更新；
        其它层、embedding、norm、lm_head 全部冻结。

    设计目的：
        和 LA-SSU current-only top20 同时使用：
        - LA-SSU 控制每层哪些 neuron 更新
        - 这里控制哪些 layer 可以更新
    """
    import re

    layer_range = os.environ.get("TRAINABLE_LAYER_RANGE", "").strip()

    if not layer_range:
        print("[LayerRange] TRAINABLE_LAYER_RANGE not set, skip layer-range freezing.")
        return

    ranges = []
    for part in layer_range.split(","):
        part = part.strip()
        if "-" not in part:
            raise ValueError(
                f"Invalid TRAINABLE_LAYER_RANGE={layer_range}. "
                "Expected format like '23-27' or '1-5,23-27'."
            )
        s, e = part.split("-", 1)
        s, e = int(s), int(e)
        if s > e:
            raise ValueError(f"Invalid range: {part}")
        ranges.append((s, e))

    def in_trainable_ranges(layer_id):
        return any(s <= layer_id <= e for s, e in ranges)

    print("=" * 80)
    print("[LayerRange] Enable layer-range update")
    print(f"[LayerRange] TRAINABLE_LAYER_RANGE = {layer_range}")
    print(f"[LayerRange] Trainable ranges = {ranges}")
    print("[LayerRange] Freeze all parameters first...")
    print("=" * 80)

    # 1) 先全部冻结
    for _, param in model.named_parameters():
        param.requires_grad = False

    # 2) 只打开指定 transformer layer
    trainable_names = []
    frozen_names = []

    for name, param in model.named_parameters():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)

        if match is None:
            frozen_names.append(name)
            continue

        layer_id = int(match.group(1))

        if in_trainable_ranges(layer_id):
            param.requires_grad = True
            trainable_names.append(name)
        else:
            frozen_names.append(name)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("=" * 80)
    print("[LayerRange] Summary")
    print(f"[LayerRange] Trainable ranges = {ranges}")
    print(f"[LayerRange] trainable tensors : {len(trainable_names)}")
    print(f"[LayerRange] frozen tensors    : {len(frozen_names)}")
    print(f"[LayerRange] trainable params  : {trainable_params:,}")
    print(f"[LayerRange] total params      : {total_params:,}")
    print(f"[LayerRange] trainable ratio   : {trainable_params / total_params:.4%}")

    if trainable_params == 0:
        raise RuntimeError(
            "[LayerRange] No trainable parameters found. "
            f"Please check TRAINABLE_LAYER_RANGE={layer_range} "
            "and model parameter names."
        )
    print("=" * 80)

    print("[LayerRange] First 20 trainable tensors:")
    for n in trainable_names[:20]:
        print(f"  {n}")

    print("=" * 80)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_layer_spec(spec: str, num_layers: int):
    """
    Parse layer specs such as:
      "23-27"
      "1-5,23-27"
      "all"
    """
    spec = spec.strip().lower()
    if spec in {"", "all"}:
        return list(range(num_layers))

    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if start > end:
                raise ValueError(f"Invalid layer range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))

    bad = sorted(i for i in selected if i < 0 or i >= num_layers)
    if bad:
        raise ValueError(
            f"Layer ids out of range: {bad}; valid range is 0-{num_layers - 1}"
        )
    return sorted(selected)


def _extract_top_indices_from_score_entry(entry, expected_size: int, top_ratio: float):
    """
    Read one module entry from a LA-SSU score file.

    Supported formats:
      - tensor of scores, length == expected_size
      - tensor/list of selected indices
      - dict containing common keys such as:
        top_indices, topk_indices, selected_indices, indices,
        mask, top_mask, scores, score, column_scores, importance

    The function is intentionally defensive because historical score files may
    use slightly different field names.
    """
    index_keys = (
        "top_indices",
        "topk_indices",
        "selected_indices",
        "indices",
        "top_idx",
        "topk_idx",
    )
    mask_keys = (
        "mask",
        "top_mask",
        "selected_mask",
        "important_mask",
    )
    score_keys = (
        "scores",
        "score",
        "column_scores",
        "importance",
        "importance_scores",
        "wanda_scores",
    )

    k = max(1, int(expected_size * top_ratio))

    def as_tensor(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, (list, tuple)):
            return torch.as_tensor(value)
        return None

    def indices_from_value(value, key_hint=""):
        tensor = as_tensor(value)
        if tensor is None:
            return None

        tensor = tensor.flatten()

        if tensor.numel() == 0:
            return None

        # Explicit index fields.
        if "indice" in key_hint or key_hint.endswith("idx"):
            idx = tensor.long()
            if idx.min().item() < 0 or idx.max().item() >= expected_size:
                raise ValueError(
                    f"Index field {key_hint} contains values outside "
                    f"[0, {expected_size - 1}]"
                )
            return torch.unique(idx, sorted=True)

        # Explicit mask fields.
        if "mask" in key_hint:
            if tensor.numel() != expected_size:
                raise ValueError(
                    f"Mask field {key_hint} has {tensor.numel()} elements; "
                    f"expected {expected_size}"
                )
            return torch.nonzero(tensor != 0, as_tuple=False).flatten().long()

        # Integer vector shorter than the number of neurons: likely indices.
        if (
            tensor.numel() < expected_size
            and tensor.dtype
            in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
        ):
            idx = tensor.long()
            if idx.min().item() >= 0 and idx.max().item() < expected_size:
                return torch.unique(idx, sorted=True)

        # Length == expected_size: mask or score vector.
        if tensor.numel() == expected_size:
            if tensor.dtype == torch.bool:
                return torch.nonzero(tensor, as_tuple=False).flatten().long()

            # A 0/1 vector is treated as a mask.
            unique = torch.unique(tensor)
            if unique.numel() <= 2 and all(float(v) in (0.0, 1.0) for v in unique):
                return torch.nonzero(tensor != 0, as_tuple=False).flatten().long()

            # Otherwise it is treated as an importance score vector.
            return torch.topk(tensor.float(), k=k, largest=True).indices.sort().values

        return None

    if isinstance(entry, dict):
        # Prefer raw score vectors: the requested top_ratio then determines
        # the exact number of selected neurons even when the file also stores
        # a historical top_indices/top_mask field.
        for key in score_keys:
            if key in entry:
                idx = indices_from_value(entry[key], key)
                if idx is not None:
                    return idx

        for key in index_keys:
            if key in entry:
                idx = indices_from_value(entry[key], key)
                if idx is not None:
                    return idx

        for key in mask_keys:
            if key in entry:
                idx = indices_from_value(entry[key], key)
                if idx is not None:
                    return idx

        # Last-resort scan over tensor-like values.
        for key, value in entry.items():
            idx = indices_from_value(value, str(key).lower())
            if idx is not None:
                return idx

        raise KeyError(
            "Could not find indices/mask/scores in score entry. "
            f"Available keys: {list(entry.keys())}"
        )

    idx = indices_from_value(entry)
    if idx is None:
        raise TypeError(f"Unsupported score entry type: {type(entry)}")
    return idx


def apply_coupled_ffn_topk_from_env(model):
    """
    Coupled FFN-neuron Top-K training.

    Selection:
      use down_proj Wanda/LA-SSU score to select intermediate neuron id j.

    Update the same j across all three SwiGLU matrices:
      gate_proj.weight[j, :]
      up_proj.weight[j, :]
      down_proj.weight[:, j]

    Freeze:
      attention, norms, embeddings, lm_head, all non-selected FFN entries.

    Environment variables:
      COUPLED_FFN_ENABLE=1
      COUPLED_FFN_SCORE_PATH=/path/to/lang_scores.pt
      COUPLED_FFN_LAYER_RANGE=23-27   # or all
      COUPLED_FFN_TOP_RATIO=0.20
    """
    if not _env_flag("COUPLED_FFN_ENABLE"):
        return False

    score_path = os.environ.get("COUPLED_FFN_SCORE_PATH", "").strip()
    layer_spec = os.environ.get("COUPLED_FFN_LAYER_RANGE", "all").strip()
    top_ratio = float(os.environ.get("COUPLED_FFN_TOP_RATIO", "0.20"))

    if not score_path:
        raise ValueError(
            "COUPLED_FFN_ENABLE=1 requires COUPLED_FFN_SCORE_PATH"
        )
    if not os.path.isfile(score_path):
        raise FileNotFoundError(f"Score file not found: {score_path}")
    if not (0.0 < top_ratio <= 1.0):
        raise ValueError(f"Invalid COUPLED_FFN_TOP_RATIO={top_ratio}")

    # QwenForCausalLM -> model.model.layers
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise AttributeError(
            "Could not find model.model.layers; this helper currently expects "
            "a Qwen/LLaMA-like causal LM."
        )

    selected_layers = _parse_layer_spec(layer_spec, len(layers))
    score_obj = torch.load(score_path, map_location="cpu")

    print("=" * 88)
    print("[CoupledFFN] Enable coupled FFN Top-K training")
    print(f"[CoupledFFN] score_path  = {score_path}")
    print(f"[CoupledFFN] layers      = {selected_layers}")
    print(f"[CoupledFFN] top_ratio   = {top_ratio:.2%}")
    print("[CoupledFFN] selection   = down_proj score")
    print("[CoupledFFN] update      = gate rows + up rows + down columns")
    print("[CoupledFFN] frozen      = attention/norm/embed/lm_head/other neurons")
    print("=" * 88)

    # Freeze the whole model first.
    for param in model.parameters():
        param.requires_grad = False

    # Gradient checkpointing can require the embedding output to carry grad,
    # while the embedding weights themselves remain frozen.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    total_effective = 0
    handles = []

    for layer_id in selected_layers:
        mlp = layers[layer_id].mlp
        gate_w = mlp.gate_proj.weight
        up_w = mlp.up_proj.weight
        down_w = mlp.down_proj.weight

        intermediate_size = down_w.shape[1]

        if gate_w.shape[0] != intermediate_size:
            raise RuntimeError(
                f"Layer {layer_id}: gate_proj rows={gate_w.shape[0]} "
                f"!= down_proj columns={intermediate_size}"
            )
        if up_w.shape[0] != intermediate_size:
            raise RuntimeError(
                f"Layer {layer_id}: up_proj rows={up_w.shape[0]} "
                f"!= down_proj columns={intermediate_size}"
            )

        score_key = f"layers.{layer_id}.mlp.down_proj.weight"
        if score_key not in score_obj:
            # Defensive fallback for score files that include a model prefix.
            candidates = [
                key
                for key in score_obj.keys()
                if key.endswith(score_key)
            ]
            if len(candidates) != 1:
                raise KeyError(
                    f"Cannot uniquely find score entry for layer {layer_id}. "
                    f"Expected key {score_key}; candidates={candidates}"
                )
            score_key = candidates[0]

        top_idx = _extract_top_indices_from_score_entry(
            score_obj[score_key],
            expected_size=intermediate_size,
            top_ratio=top_ratio,
        )

        if top_idx.numel() == 0:
            raise RuntimeError(f"Layer {layer_id}: selected zero neurons")

        top_idx = top_idx.to(device=gate_w.device)

        # The optimizer sees these three tensors, but only the masked entries
        # receive non-zero gradients. Use weight_decay=0.0 for exact freezing.
        gate_w.requires_grad = True
        up_w.requires_grad = True
        down_w.requires_grad = True

        row_mask = torch.zeros(
            (intermediate_size, 1),
            dtype=gate_w.dtype,
            device=gate_w.device,
        )
        row_mask[top_idx, 0] = 1

        col_mask = torch.zeros(
            (1, intermediate_size),
            dtype=down_w.dtype,
            device=down_w.device,
        )
        col_mask[0, top_idx] = 1

        handles.append(
            gate_w.register_hook(
                lambda grad, mask=row_mask: grad * mask.to(
                    device=grad.device, dtype=grad.dtype
                )
            )
        )
        handles.append(
            up_w.register_hook(
                lambda grad, mask=row_mask: grad * mask.to(
                    device=grad.device, dtype=grad.dtype
                )
            )
        )
        handles.append(
            down_w.register_hook(
                lambda grad, mask=col_mask: grad * mask.to(
                    device=grad.device, dtype=grad.dtype
                )
            )
        )

        hidden_size = down_w.shape[0]
        effective = top_idx.numel() * hidden_size * 3
        total_effective += effective

        print(
            f"[CoupledFFN] layer={layer_id:02d} "
            f"selected={top_idx.numel()}/{intermediate_size} "
            f"effective_params={effective:,}"
        )

    trainable_tensor_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    # Keep hook handles alive for the whole training process.
    model._coupled_ffn_hook_handles = handles

    print("=" * 88)
    print(f"[CoupledFFN] selected layers               = {len(selected_layers)}")
    print(f"[CoupledFFN] optimizer-visible parameters = {trainable_tensor_params:,}")
    print(f"[CoupledFFN] effective masked parameters  = {total_effective:,}")
    print(f"[CoupledFFN] total model parameters       = {total_params:,}")
    print(
        f"[CoupledFFN] effective update ratio       = "
        f"{total_effective / total_params:.4%}"
    )
    print("=" * 88)

    return True


def main(args, training_args):
    #####
    # Load the dataset
    #####
    train_dataset = datasets.load_from_disk(args.dataset_path)
    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    from utils.data_utils import maybe_concat_replay_datasets


    raw_train_dataset = train_dataset   
    if args.cl_method == "replay":
        if not args.replay_dataset_path:
            raise ValueError("--cl_method replay requires --replay_dataset_path")

        replay_paths = [p.strip() for p in args.replay_dataset_path.split(",") if p.strip()]
        replay_datasets = [datasets.load_from_disk(p) for p in replay_paths]

        train_dataset = maybe_concat_replay_datasets(
            train_dataset=train_dataset,
            replay_datasets=replay_datasets,
            replay_ratio=args.replay_ratio,
            seed=args.replay_seed,
        )
        print(f"[Replay] Mixed train_dataset size = {len(train_dataset)}")


    if args.val_dataset_path is not None:
        val_dataset = datasets.load_from_disk(args.val_dataset_path)
    else:
        val_dataset = None

    # replay_buffer = None
    # if args.cl_method == "er":
    #         print("=== Experience Replay (ER) Enabled ===")
    #         replay_buffer = ReplayBuffer(
    #             capacity=args.replay_buffer_size
    #         )




    #####
    # Load the tokenizer
    #####
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #####
    # Set up the data collator
    #####
    class CLTrainer(Trainer):
        def __init__(
            self,
            *args,
            ewc: EWC = None,
            lwf: LwF = None,
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.ewc = ewc
            self.lwf = lwf

        def compute_loss(self, model, inputs, return_outputs=False,**kwargs,):
            # ===== 1. 标准 LM loss / assistant-only loss =====
            if assistant_only:
                labels = inputs["labels"]
                model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                outputs = model(**model_inputs)
                loss = per_sequence_causal_lm_loss(outputs.logits, labels)
            else:
                outputs = model(**inputs)
                loss = outputs.loss

            # ===== 2. EWC penalty =====
            if self.ewc is not None:
                loss = loss + self.ewc.penalty(model)

            # ===== 3. LwF distillation loss =====
            if self.lwf is not None:
                lwf_loss = self.lwf.kd_loss(
                    student_logits=outputs.logits,
                    inputs=inputs,
                    attention_mask=inputs.get("attention_mask", None),
                )
                loss = loss + lwf_loss

            return (loss, outputs) if return_outputs else loss



    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    assistant_only = os.environ.get("ASSISTANT_ONLY_LOSS", "0") == "1"
    base_collator = default_data_collator if assistant_only else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print(f"[LossMode] assistant_only_per_sequence={assistant_only}")

    data_collator = base_collator


    #####
    # Load the model
    #####
    # Detect if FSDP is enabled via TrainingArguments; if so, avoid device_map and let Trainer/FSDP place shards
    fsdp_enabled = bool(getattr(training_args, "fsdp", None)) and str(getattr(training_args, "fsdp")).strip().lower() not in ("", "none")

    if fsdp_enabled:
        # Load on CPU (or default device) and let FSDP handle placement/sharding.
        # 为了数值更稳定，这里使用 float32 作为默认 dtype（0.5B 模型在 12G 显存上仍然可以接受）。
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",       # flash_attention_2
            low_cpu_mem_usage=True,
        )
        # Ensure FSDP uses original parameters so param names remain stable for GMT and freezing logic
        fsdp_cfg = getattr(training_args, "fsdp_config", None)
        if fsdp_cfg is None:
            fsdp_cfg = {}
            training_args.fsdp_config = fsdp_cfg
        # Handle dict-like vs object-like config containers
        try:
            # dict path
            fsdp_cfg.setdefault("use_orig_params", True)
            fsdp_cfg.setdefault("state_dict_type", "FULL_STATE_DICT")
        except AttributeError:
            # object path
            if not hasattr(fsdp_cfg, "use_orig_params") or getattr(fsdp_cfg, "use_orig_params") is None:
                try:
                    setattr(fsdp_cfg, "use_orig_params", True)
                except Exception:
                    pass
            if not hasattr(fsdp_cfg, "state_dict_type") or getattr(fsdp_cfg, "state_dict_type") is None:
                try:
                    setattr(fsdp_cfg, "state_dict_type", "FULL_STATE_DICT")
                except Exception:
                    pass
    else:
        # 非 FSDP 场景下，同样使用 float32 提升训练稳定性，避免纯 fp16 造成的梯度 NaN。
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",         # NOTE: 禁用 flash_attention_2
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
        )

    ewc_obj = None
    lwf_obj = None

    # ============================================================
    # LA-SSU score collection mode
    # ============================================================
    # 这个模式只做一件事：
    #   用当前语言的 calibration data 计算所有 Linear weight 的 column-wise SSU/Wanda 分数，
    #   保存成 .pt 文件，然后直接 return，不进入训练。
    #
    # 用途：
    #   例如分别生成：
    #       ibo_scores.pt
    #       hau_scores.pt
    #       kir_scores.pt
    #       npi_scores.pt
    #       amh_scores.pt
    #
    # 注意：
    #   这里复用官方 SSU 的 calibration dataloader 逻辑。
    #   如果没有指定 lassu_calibration_dataset_path，就默认使用 calibration_dataset_path。
    if getattr(args, "lassu_collect_scores", False):
        print("=== LA-SSU Score Collection Mode Enabled ===")

        if getattr(args, "lassu_enable", False):
            raise ValueError("Do not use --lassu_collect_scores and --lassu_enable at the same time.")

        lassu_score_save_path = getattr(args, "lassu_score_save_path", None)
        if not lassu_score_save_path:
            raise ValueError("--lassu_collect_scores requires --lassu_score_save_path")

        # 优先使用 LA-SSU 专用 calibration path；
        # 如果没有，就复用原始 SSU 的 calibration_dataset_path。
        lassu_calib_path = getattr(args, "lassu_calibration_dataset_path", None)
        if lassu_calib_path is None:
            lassu_calib_path = getattr(args, "calibration_dataset_path", None)

        if lassu_calib_path is None:
            print("[LA-SSU] No calibration dataset path provided. Fallback to raw_train_dataset.")

        lassu_num_calib = getattr(args, "lassu_num_calibration_samples", None)
        if lassu_num_calib is None:
            lassu_num_calib = getattr(args, "num_calibration_samples", 128)

        lassu_top_ratio = getattr(args, "lassu_top_ratio", 0.5)
        lassu_skip_embeddings_and_head = getattr(args, "lassu_skip_embeddings_and_head", True)

        print(f"[LA-SSU] calibration path = {lassu_calib_path}")
        print(f"[LA-SSU] score save path = {lassu_score_save_path}")
        print(f"[LA-SSU] num calibration samples = {lassu_num_calib}")
        print(f"[LA-SSU] top ratio = {lassu_top_ratio}")

        calibration_data = create_calibration_dataloader(
            lassu_calib_path,
            lassu_num_calib,
            raw_train_dataset,
            tokenizer,
        )

        collect_lassu_column_scores(
            model=model,
            calibration_data=calibration_data,
            num_calibration_samples=lassu_num_calib,
            top_ratio=lassu_top_ratio,
            save_path=lassu_score_save_path,
            skip_embeddings_and_head=lassu_skip_embeddings_and_head,
        )

        print("[LA-SSU] Score collection finished. Exit without training.")
        return
    
    # ============================================================
    # Quick exclusivity enforcement: LA-SSU vs other mechanisms
    # ============================================================
    # 第一版 LA-SSU 先作为独立方法，不和 PEFT / HFT / GMT / LoTA / S2FT / EWC / LwF / GPM 混用。
    # 这样实验结论更干净：
    #   原始 SSU hard column freezing
    #   vs LA-SSU soft column scaling
    if getattr(args, "lassu_enable", False):
        print("=== LA-SSU Training Mode Enabled ===")

        if getattr(args, "peft_method", "none") != "none":
            print("[LA-SSU] Disabling PEFT because --lassu_enable is set.")
            args.peft_method = "none"

        if getattr(args, "do_hft", False):
            print("[LA-SSU] Disabling HFT/SSU freezing because --lassu_enable is set.")
            args.do_hft = False

        if getattr(args, "use_gmt", False):
            print("[LA-SSU] Disabling GMT because --lassu_enable is set.")
            args.use_gmt = False

        if getattr(args, "use_lota", False):
            print("[LA-SSU] Disabling LoTA because --lassu_enable is set.")
            args.use_lota = False

        if getattr(args, "use_s2ft", False):
            print("[LA-SSU] Disabling S2FT because --lassu_enable is set.")
            args.use_s2ft = False

        if getattr(args, "cl_method", "none") != "none":
            raise ValueError(
                "First version of LA-SSU should not be mixed with --cl_method "
                f"{args.cl_method}. Please set --cl_method none."
            )


    # Quick exclusivity enforcement: LoTA vs other baseline mechanisms
    if getattr(args, 'use_lota', False):
        # Disable conflicting methods
        if getattr(args, 'peft_method', 'none') != 'none':
            print("[LoTA] Disabling PEFT (LoRA/AdaLoRA) because --use_lota is set.")
            args.peft_method = 'none'
        if getattr(args, 'do_hft', False):
            print("[LoTA] Disabling HFT freezing strategies because --use_lota is set.")
            args.do_hft = False
        if getattr(args, 'use_gmt', False):
            print("[LoTA] Disabling GMT because --use_lota is set.")
            args.use_gmt = False
        if getattr(args, 'use_s2ft', False):
            print("[LoTA] Disabling S2FT because --use_lota is set.")
            args.use_s2ft = False
    
    # Quick exclusivity enforcement: S2FT vs other baseline mechanisms
    if getattr(args, 'use_s2ft', False):
        if getattr(args, 'peft_method', 'none') != 'none':
            print("[S2FT] Disabling PEFT (LoRA/AdaLoRA) because --use_s2ft is set.")
            args.peft_method = 'none'
        if getattr(args, 'do_hft', False):
            print("[S2FT] Disabling HFT freezing because --use_s2ft is set.")
            args.do_hft = False
        if getattr(args, 'use_gmt', False):
            print("[S2FT] Disabling GMT because --use_s2ft is set.")
            args.use_gmt = False
        if getattr(args, 'use_lota', False):
            print("[S2FT] Disabling LoTA because --use_s2ft is set.")
            args.use_lota = False
    
    # Optionally wrap with PEFT (LoRA/AdaLoRA) before any selective freezing
    if getattr(args, 'peft_method', 'none') != 'none':
        if not _PEFT_AVAILABLE:
            raise RuntimeError("peft library is not installed but peft_method was set. Install `peft`.")
        target_modules: Optional[List[str]] = None
        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
        bias = args.peft_bias
        # Ensure embeddings and lm_head are tuned with PEFT by saving these modules (kept trainable)
        modules_to_save = [
            'lm_head', 'embed_tokens', 'wte', 'word_embeddings', 'embeddings', 'token_embedding', 'output_projection'
        ]
        if args.peft_method == 'lora':
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias=bias,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, lora_cfg)
            print("Wrapped model with standard LoRA (PEFT)")
        elif args.peft_method == 'adalora':
            adalora_cfg = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias=bias,
                modules_to_save=modules_to_save,
                init_r=args.lora_r,
                target_r=args.adalora_target_r,
                tinit=args.adalora_tinit,
                tfinal=args.adalora_tfinal,
                deltaT=args.adalora_delta_t,
                beta1=args.adalora_beta1,
                beta2=args.adalora_beta2,
                orth_reg_weight=args.adalora_orth_reg_weight,
                total_step=args.adalora_total_step,
            )
            model = get_peft_model(model, adalora_cfg)
            print("Wrapped model with AdaLoRA (PEFT)")

        # PEFT is a baseline: do not combine with HFT, LSFT, or GMT
        if getattr(args, 'do_hft', False):
            print("PEFT baseline selected: disabling HFT freezing.")
            args.do_hft = False
        if getattr(args, 'use_gmt', False):
            print("PEFT baseline selected: disabling Gradient-Mask Tuning (GMT).")
            args.use_gmt = False

    # Check for mutual exclusivity between HFT and GMT
    if getattr(args, 'do_hft', False) and getattr(args, 'use_gmt', False):
        raise ValueError("Cannot use both HFT (--do_hft) and GMT (--use_gmt) simultaneously. Please choose one approach.")


    # Quick exclusivity enforcement: CL methods vs other baselines
    if getattr(args, "cl_method", "none") in ["replay", "ewc", "lwf","gpm"]:
        if getattr(args, "use_gmt", False):
            print("[CL] Disabling GMT because cl_method is set.")
            args.use_gmt = False
        if getattr(args, "do_hft", False):
            print("[CL] Disabling HFT because cl_method is set.")
            args.do_hft = False
        if getattr(args, "peft_method", "none") != "none":
            print("[CL] Disabling PEFT because cl_method is set.")
            args.peft_method = "none"
        if getattr(args, "use_lota", False):
            print("[CL] Disabling LoTA because cl_method is set.")
            args.use_lota = False
        if getattr(args, "use_s2ft", False):
            print("[CL] Disabling S2FT because cl_method is set.")
            args.use_s2ft = False
            
    # Optionally set up Lottery Ticket Adaptation (LoTA)
    lota_state = None
    if getattr(args, 'use_lota', False):
        print("=== Lottery Ticket Adaptation (LoTA) Enabled ===")
        print("[LoTA] Starting mask calibration phase...")
        
        # Build simple calibration dataloader (reuse train dataset; random shuffle)
        
        calib_batch_size = training_args.per_device_train_batch_size
        calib_loader = DataLoader(
            train_dataset,
            batch_size=calib_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            drop_last=False,
        )
        
        # Optimizer selection
        lr = training_args.learning_rate
        weight_decay = training_args.weight_decay
        if args.lota_optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif args.lota_optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif args.lota_optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported LoTA optimizer: {args.lota_optimizer}")
        lota_state = lota_calibrate_mask(
            model,
            calib_loader,
            optimizer,
            sparsity=args.lota_sparsity,
            calibration_steps=args.lota_calibration_steps,
            device=None,
            skip_embeddings_and_head=getattr(args, 'lota_skip_embeddings_and_head', False),
            grad_accum_steps=args.lota_grad_accum_steps,
            max_batches=args.lota_calibration_max_batches,
            verbose=args.lota_verbose,
        )
        
        # Prepare sparse adaptation phase
        lota_prepare_sparse_training(model, lota_state, verbose=True)
        
        # Display summary
        print(lota_parameter_summary(model))
    
    # Optionally set up S2FT
    if getattr(args, 'use_s2ft', False):
        print("=== S2FT (Structured Sparse Fine-Tuning) Enabled ===")
        print(f"S2FT config: ratio={args.s2ft_ratio:.2%}, strategy={args.s2ft_strategy}")
        o_ratio = args.s2ft_ratio if getattr(args, 's2ft_include_attn_output', False) else 0.0
        if o_ratio > 0.0:
            print("[S2FT] Including attention output heads (o_proj) with same ratio as FFN down.")
        model, selections = s2ft_enable(
            model,
            v_ratio=0.0,            # not selected in baseline
            o_ratio=o_ratio,        # optional heads
            u_ratio=0.0,            # only down_proj channels in baseline
            d_ratio=args.s2ft_ratio,
            seed=training_args.seed,
            gradient_checkpointing=getattr(training_args, 'gradient_checkpointing', False),
            make_gc_compatible_fn=None,
            freeze_bias=True,
            verbose=True,
        )
        print("[S2FT] Model conversion complete.")
        print(selections)
    
    # Decide which parameters to freeze or train for each module (HFT)
    if getattr(args, "lassu_enable", False):
        # ============================================================
        # LA-SSU training mode
        # ============================================================
        # 训练阶段不重新计算分数，而是读取已经保存好的 score 文件：
        #   old_score_paths     = 历史语言 score 文件
        #   current_score_path  = 当前语言 score 文件
        #
        # 然后根据 overlap 生成 soft scale，并注册 gradient hook。
        print("=== Applying LA-SSU column-wise soft scaling ===")

        current_score_path = getattr(args, "lassu_current_score_path", None)
        old_score_paths = getattr(args, "lassu_old_score_paths", "")

        if not current_score_path:
            raise ValueError("--lassu_enable requires --lassu_current_score_path")

        if isinstance(old_score_paths, str):
            old_score_paths = [p.strip() for p in old_score_paths.split(",") if p.strip()]

        print(f"[LA-SSU] current score path = {current_score_path}")
        print(f"[LA-SSU] old score paths = {old_score_paths}")

        english_score_path = getattr(args, "lassu_english_score_path", None)

        if english_score_path:
            print("=== Applying English-Anchored LA-SSU ===")
            print(f"[EA-LA-SSU] english score path = {english_score_path}")

            apply_lassu_english_anchor_from_score_files(
                model=model,
                english_score_path=english_score_path,
                old_score_paths=old_score_paths,
                current_score_path=current_score_path,
                en_current_shared_scale=getattr(args, "lassu_en_current_shared_scale", 0.7),
                en_past_shared_scale=getattr(args, "lassu_en_past_shared_scale", 0.1),
                en_only_scale=getattr(args, "lassu_en_only_scale", 0.0),
                others_scale=getattr(args, "lassu_others_scale", 1.0),
            )

        else:
            print("=== Applying history-only LA-SSU ===")

            apply_lassu_from_score_files(
                model=model,
                old_score_paths=old_score_paths,
                current_score_path=current_score_path,
                old_shared_threshold=getattr(args, "lassu_old_shared_threshold", 2),
                old_specific_scale=getattr(args, "lassu_old_specific_scale", 0.0),
                old_shared_scale=getattr(args, "lassu_old_shared_scale", 0.1),
                current_shared_scale=getattr(args, "lassu_current_shared_scale", 0.5),
                current_specific_scale=getattr(args, "lassu_current_specific_scale", 1.0),
                others_scale=getattr(args, "lassu_others_scale", 1.0),
            )

    elif args.do_hft:
        # Prepare calibration data for strategies that need it
        calibration_data = None
        if args.freeze_strategy in ["ssu_based", "ssu_elementwise", "ssu_rowwise"]:
            print("Preparing calibration data for SSU-based freezing...")
            calibration_data = create_calibration_dataloader(
                args.calibration_dataset_path,
                args.num_calibration_samples,
                train_dataset, tokenizer
            )
        
        # Apply chosen strategy
        if args.freeze_strategy == "random_based":
            strategy_desc = "random (neuron-level, structured)"
        elif args.freeze_strategy == "random_elementwise":
            strategy_desc = "random (element-wise)"
        elif args.freeze_strategy == "random_rowwise":
            strategy_desc = "random (row-wise, structured)"
        elif args.freeze_strategy == "hft_based":
            strategy_desc = "HFT-based (structured, using activation importance)"
        elif args.freeze_strategy == "magnitude_based":
            strategy_desc = "magnitude-based (freeze large weights, structured)"
        elif args.freeze_strategy == "magnitude_elementwise":
            strategy_desc = "magnitude-based (freeze large weights, element-wise)"
        elif args.freeze_strategy == "magnitude_rowwise":
            strategy_desc = "magnitude-based (row-wise, large rows frozen)"
        elif args.freeze_strategy == "ssu_based":
            strategy_desc = "SSU-based (structured, using activation importance)"
        elif args.freeze_strategy == "ssu_elementwise":
            strategy_desc = "SSU-based (element-wise, using activation importance)"
        elif args.freeze_strategy == "ssu_rowwise":
            strategy_desc = "SSU-based (row-wise, using activation importance)"
        elif args.freeze_strategy == "fisher_based":
            strategy_desc = "Fisher-based (structured, using gradient Fisher information)"
        elif args.freeze_strategy == "fisher_rowwise":
            strategy_desc = "Fisher-based (row-wise, using gradient Fisher information)"
        elif args.freeze_strategy == "fisher_elementwise":
            strategy_desc = "Fisher-based (element-wise, using gradient Fisher information)"
        elif args.freeze_strategy == "sgpt_based":
            strategy_desc = "SparseGPT-based (structured, E[x^2] input statistics)"
        elif args.freeze_strategy == "sgpt_rowwise":
            strategy_desc = "SparseGPT-based (row-wise aggregation of E[x^2])"
        elif args.freeze_strategy == "sgpt_elementwise":
            strategy_desc = "SparseGPT-based (element-wise, E[x^2])"
        else:
            strategy_desc = args.freeze_strategy

        if args.freeze_chat_template_tokens:
            strategy_desc += f" + chat template tokens (ratio: {args.chat_template_freeze_ratio})"

        print(f"Applying Half Fine-Tuning (HFT) with {args.freeze_ratio:.1%} {strategy_desc} parameter freezing...")
        freeze_random_parameters(
            model=model,
            freeze_ratio=args.freeze_ratio,
            seed=(
                args.freeze_seed
                if getattr(args, "freeze_seed", None) is not None
                else training_args.seed
            ),
            strategy=args.freeze_strategy,
            skip_embeddings_and_head=args.skip_embeddings_and_head,
            calibration_data=calibration_data,
            num_calibration_samples=args.num_calibration_samples,
            tokenizer=tokenizer,
            freeze_chat_template_tokens=args.freeze_chat_template_tokens,
            chat_template_freeze_ratio=args.chat_template_freeze_ratio,
        )

    else:
        print("Training all model parameters...")
        
    #####
    # Set up the trainer
    #####
    # ============================================================
    # Optional layer-range update
    # ============================================================
    # 放在 LA-SSU / HFT / LoTA / S2FT 之后，Trainer 创建之前。
    # 这样可以和 LA-SSU current-only top20 同时生效：
    #   LA-SSU: 控制哪些 neuron 有梯度
    #   LayerRange: 控制哪些 layer 可以更新
    if not apply_plnd_mask_from_env(model):
        if not apply_coupled_ffn_topk_from_env(model):
            apply_trainable_layer_range_from_env(model)
        
    #####
    # Set up the trainer
    #####
    callbacks = []
    exact_save_steps = os.environ.get("EXACT_SAVE_STEPS", "").strip()
    if exact_save_steps:
        callbacks.append(ExactSaveStepsCallback(
            [int(step) for step in exact_save_steps.split(",") if step.strip()]
        ))
        print(f"[Checkpoint] exact save steps: {exact_save_steps}")
    
    if args.cl_method == "lwf":
        print("=== Learning without Forgetting (LwF) Enabled ===")
 
        old_model = model  # 直接传给 LwF，由 LwF 内部 deepcopy 冻结 
        lwf_obj = LwF(
            teacher_model=old_model,
            cfg=LwFConfig(
                temperature=args.lwf_temperature,
                lambda_lwf=args.lwf_lambda,
            ),
            device=training_args.device,
        )
    
    if args.cl_method == "ewc":
        print("=== Elastic Weight Consolidation (EWC) Enabled ===")

        
        ewc_loader = DataLoader(
            raw_train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=base_collator
        )

        ewc_obj = EWC(
            model=model,
            device=training_args.device,
            cfg=EWCConfig(
                lambda_ewc=args.ewc_lambda,
                fisher_max_batches=args.ewc_fisher_max_batches,  # 可选，控制计算成本
                fisher_use_token_count=True,
                fisher_decay=args.ewc_fisher_decay,
            ),
        )

        # ===== 如果存在历史 EWC 状态，就加载（实现跨 task 生效）=====
        ewc_state_to_load = None

        # 1) 用户显式指定
        if getattr(args, "ewc_state_path", None):
            ewc_state_to_load = args.ewc_state_path
        else:
            # 2) 默认：尝试从“当前加载模型的目录”读 ewc_state.pt
            # 由于你脚本顺序训练时 MODEL_NAME 会变成上一次 output_dir，所以这里刚好能拿到上一任务状态
            if isinstance(args.model_name_or_path, str):
                cand = os.path.join(args.model_name_or_path, "ewc_state.pt")
                if os.path.isfile(cand):
                    ewc_state_to_load = cand

        if ewc_state_to_load is not None:
            print(f"[EWC] 发现历史状态，加载：{ewc_state_to_load}")
            ewc_obj.load(ewc_state_to_load, model)
        else:
            print("[EWC] 未发现历史状态：当前任务将作为第一个 task（EWC penalty 为 0）")


    task_id = getattr(args, "task_id", 0)
    # ===== GPM 初始化（在创建 trainer 之前）=====
    gpm_mgr = None
    gpm_loader = None
    if args.cl_method == "gpm":
        print("=== Gradient Projection Memory (GPM) Enabled ===")

        # 用 config 里的参数，不要写死
        kw = tuple([k.strip() for k in args.gpm_keywords.split(",") if k.strip()])

        cfg = GPMConfig(
            threshold_base=args.gpm_threshold_base,
            threshold_inc=args.gpm_threshold_inc,
            max_tokens_per_layer=args.gpm_max_tokens_per_layer,
            max_rank_per_layer=args.gpm_max_rank_per_layer,
            max_new_rank_per_task=args.gpm_max_new_rank_per_task,
            only_module_name_keywords=kw,
        )
        print(
            f"[GPM CONFIG] total_rank_cap={cfg.max_rank_per_layer} "
            f"new_rank_per_task_cap={cfg.max_new_rank_per_task} "
            f"tokens_per_layer={cfg.max_tokens_per_layer}"
        )
        
        gpm_device = next(model.parameters()).device
        gpm_mgr = GPMManager(model, device=gpm_device, cfg=cfg)


        # Explicit state path supports sharing the exact FFT task-1 model
        # while deriving GPM memory post hoc. Otherwise use the previous model dir.
        gpm_state_to_load = (
            args.gpm_state_path
            if getattr(args, "gpm_state_path", None)
            else os.path.join(args.model_name_or_path, "gpm_state.pt")
        )
        if os.path.isfile(gpm_state_to_load):
            print(f"[GPM] 加载历史状态：{gpm_state_to_load}")
            gpm_mgr.maybe_load(gpm_state_to_load)
        else:
            print("[GPM] 未发现历史状态：当前任务将作为第一个 task")

        # task 开始前构建投影矩阵
        gpm_mgr.before_task(task_idx=task_id)

        # 建立用于更新子空间的 dataloader（建议用 raw_train_dataset，不要混 replay）
        
        gpm_loader = DataLoader(
            raw_train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=base_collator,
            drop_last=False,
        )

        # 不要 trainer.add_callback（此时 trainer 还没创建），而是加入 callbacks 列表
        callbacks.append(GPMCallback(gpm_mgr))




    if getattr(args, 'use_gmt', False):
        print(
                f"[CL CONFIG] "
                f"Replay={args.cl_method == 'replay'}, "
                f"EWC={args.cl_method == 'ewc'}, "
                f"LwF={args.cl_method == 'lwf'}, "
                f"GMT={args.use_gmt}, "
                f"PEFT={args.peft_method != 'none'}, "
                f"LA-SSU={getattr(args, 'lassu_enable', False)}"
            )
        trainer = create_gmt_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            gmt_mask_ratio=args.gmt_mask_ratio,
            gmt_skip_embeddings_and_head=args.gmt_skip_embeddings_and_head,
            callbacks=callbacks,   
        )

    else:
        print(
                f"[CL CONFIG] "
                f"Replay={args.cl_method == 'replay'}, "
                f"EWC={args.cl_method == 'ewc'}, "
                f"LwF={args.cl_method == 'lwf'}, "
                f"GMT={args.use_gmt}, "
                f"PEFT={args.peft_method != 'none'}, "
                f"LA-SSU={getattr(args, 'lassu_enable', False)}"
            )

        # For ordinary FFT / single training, use the official standard Trainer.
        # Only use CLTrainer when EWC / LwF is actually enabled.
        if getattr(args, "cl_method", "none") == "none" and not assistant_only:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )
        else:
            trainer = CLTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
                ewc=ewc_obj,
                lwf=lwf_obj,
            )

    
    #####
    # Train the model
    #####
    resume_from_checkpoint = None
    if os.environ.get("AUTO_RESUME_FROM_CHECKPOINT", "0") == "1":
        from transformers.trainer_utils import get_last_checkpoint

        resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)
        if resume_from_checkpoint:
            print(f"[Trainer] Resuming from checkpoint: {resume_from_checkpoint}")
    if os.environ.get("BASELINE_MASK_AUDIT") == "1":
        from utils.baseline_mask_audit import collect
        from pathlib import Path
        mask_record = collect(model, Path(training_args.output_dir) / "selection")
        print("[BaselineMask] gradient eligible unique elements:", mask_record["gradient_eligible_elements"], flush=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Persist trained weights before recoverable task-boundary bookkeeping.
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    if ewc_obj is not None:
        print("[EWC] 任务结束：开始估计 Fisher 并保存状态...")

        # Training is complete, so Adam states are no longer needed. Releasing
        # them before the FP32 Fisher pass prevents a transient OOM on 1.5B LLMs.
        if trainer.optimizer is not None:
            trainer.optimizer.state.clear()
            trainer.optimizer = None
        trainer.lr_scheduler = None
        model.zero_grad(set_to_none=True)
        ewc_obj.offload_state_to_cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ewc_obj.estimate_fisher(model, ewc_loader)
        ewc_obj.update_ref_param(model)

        save_flag = getattr(args, "ewc_save_state", False)
        if save_flag:
            ewc_save_path = os.path.join(training_args.output_dir, "ewc_state.pt")
            ewc_obj.save(ewc_save_path)
            print(f"[EWC] 状态已保存：{ewc_save_path}")

    if args.cl_method == "gpm":
        print("[GPM] 任务结束：开始更新子空间并保存状态...")
        gpm_mgr.after_task_update_basis(
            task_idx=task_id,
            train_dataloader=gpm_loader,
            max_batches=args.gpm_update_max_batches
        )
        gpm_save_path = os.path.join(training_args.output_dir, "gpm_state.pt")
        gpm_mgr.save(gpm_save_path)
        print(f"[GPM] 状态已保存：{gpm_save_path}")


    #####
    # Save the model
    #####
    if getattr(args, "use_s2ft", False):
        replaced = convert_s2_modules_to_linear(model)
        print(f"[S2FT] Fused and replaced {replaced} S2 modules before saving.")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = CustomArgumentParser()
    args, training_args = parser.parse_args()
    main(args, training_args)
