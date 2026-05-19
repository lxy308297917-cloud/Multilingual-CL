import datasets
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer)

from torch.utils.data import DataLoader
# from utils.replay_buffer import ReplayBuffer
from transformers import TrainerCallback
from utils.ewc import EWC, EWCConfig
from utils.lwf import LwF, LwFConfig
import os
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


def main(args, training_args):
    #####
    # Load the dataset
    #####
    train_dataset = datasets.load_from_disk(args.dataset_path)
    # train_dataset = train_dataset.shuffle(seed=training_args.seed)
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
            # ===== 1. 标准 LM loss =====
            outputs = model(**inputs)
            lm_loss = outputs.loss
            loss = lm_loss

            if self.ewc is not None:
                ewc_loss = self.ewc.penalty(model)
                loss = loss + ewc_loss
            else:
                ewc_loss = None

            if self.lwf is not None:
                lwf_loss = self.lwf.kd_loss(
                    student_logits=outputs.logits,
                    inputs=inputs,
                    attention_mask=inputs.get("attention_mask", None),
                )
                loss = loss + lwf_loss
            else:
                lwf_loss = None

            step = int(getattr(self.state, "global_step", -1))
            loss_log_steps = max(int(getattr(self.args, "logging_steps", 10)), 1)
            should_log_extra = (self.ewc is not None) or (self.lwf is not None)
            if should_log_extra and (step <= 0 or (step % loss_log_steps == 0)):
                print(f"[LOSS][step={step}] lm_loss = {float(lm_loss.detach().cpu())}")
                if ewc_loss is not None:
                    print(f"[LOSS][step={step}] ewc_loss = {float(ewc_loss.detach().cpu())}")
                if lwf_loss is not None:
                    print(f"[LOSS][step={step}] lwf_loss = {float(lwf_loss.detach().cpu())}")
                print(f"[LOSS][step={step}] total_loss = {float(loss.detach().cpu())}")
                print(f"[LOSS][step={step}] lm_loss is nan? {torch.isnan(lm_loss).any().item()}")
                if ewc_loss is not None:
                    print(f"[LOSS][step={step}] ewc_loss is nan? {torch.isnan(ewc_loss).any().item()}")
                if lwf_loss is not None:
                    print(f"[LOSS][step={step}] lwf_loss is nan? {torch.isnan(lwf_loss).any().item()}")
                print(f"[LOSS][step={step}] total_loss is nan? {torch.isnan(loss).any().item()}")

            return (loss, outputs) if return_outputs else loss



    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
            torch_dtype=torch.bfloat16,             # float16 ? loss=0
            attn_implementation="eager",         # flash_attention_2
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
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )

    ewc_obj = None
    lwf_obj = None

    
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
    if getattr(args, "cl_method", "none") in ["replay", "ewc", "lwf", "gpm"]:
        if getattr(args, "use_gmt", False):
            print("[CL] Disabling GMT because cl_method is set.")
            args.use_gmt = False
        if getattr(args, "do_hft", False):
            print("[CL] Disabling HFT because cl_method is set.")
            args.do_hft = False
        if getattr(args, "use_lota", False):
            print("[CL] Disabling LoTA because cl_method is set.")
            args.use_lota = False   
        if getattr(args, "use_s2ft", False):
            print("[CL] Disabling S2FT because cl_method is set.")
            args.use_s2ft = False

        if getattr(args, "peft_method", "none") != "none":
            print(f"[CL] Keeping PEFT enabled with cl_method={args.cl_method}, peft_method={args.peft_method}")
                
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
    if args.do_hft:
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
            seed=training_args.seed,
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
    callbacks = []
    
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
                fisher_decay=0.0,
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
            only_module_name_keywords=kw,
        )
        
        gpm_device = next(model.parameters()).device
        gpm_mgr = GPMManager(model, device=gpm_device, cfg=cfg)


        # 默认从“当前加载模型目录”读取上一任务的 gpm_state.pt（跨 task 生效）
        gpm_state_to_load = os.path.join(args.model_name_or_path, "gpm_state.pt")
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
                    f"PEFT={args.peft_method != 'none'}"
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
                f"PEFT={args.peft_method != 'none'}"
            )
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

    # 打印一下
    batch = next(iter(DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)))
    labels = batch.get("labels", None)
    am = batch.get("attention_mask", None)

    print("input_ids:", batch["input_ids"].shape)
    print("labels:", None if labels is None else labels.shape)
    if labels is not None:
        valid = (labels != -100)
        print("labels!=-100 tokens:", valid.sum().item())
    if am is not None:
        print("attention_mask tokens:", am.sum().item())

    if labels is not None and am is not None:
        print("same mask?", (valid == am.bool()).all().item())



    trainer.train()


    if ewc_obj is not None:
        print("[EWC] 任务结束：开始估计 Fisher 并保存状态...")

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
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = CustomArgumentParser()
    args, training_args = parser.parse_args()
    main(args, training_args)
