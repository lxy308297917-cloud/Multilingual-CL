#!/usr/bin/env python3
"""LAPE collection and causal masking pre-experiment for Qwen2.5 MLPs."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


LANGUAGES = ("en", "ha", "ig", "ky")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="/root/models/Qwen2.5-1.5B-Instruct")
    common.add_argument(
        "--data-root", default="/root/autodl-tmp/eval_datasets_local/sum_ssu"
    )
    common.add_argument("--block-size", type=int, default=512)
    common.add_argument("--batch-size", type=int, default=2)
    common.add_argument("--device", default="cuda")

    collect = subparsers.add_parser("collect", parents=[common])
    collect.add_argument("--language", choices=LANGUAGES, required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--calibration-examples", type=int, default=400)
    collect.add_argument("--calibration-tokens", type=int, default=32768)

    select = subparsers.add_parser("select")
    select.add_argument("--activation-dir", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--top-rate", type=float, default=0.01)
    select.add_argument("--activation-quantile", type=float, default=0.95)
    select.add_argument("--seed", type=int, default=42)

    evaluate = subparsers.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--mask-file")
    evaluate.add_argument(
        "--conditions",
        default="none,en,ha,ig,ky,random_en,random_ha,random_ig,random_ky",
    )
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--eval-start", type=int, default=400)
    evaluate.add_argument("--eval-examples", type=int, default=100)
    evaluate.add_argument("--eval-tokens", type=int, default=8192)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)

    return parser.parse_args()


def load_jsonl_texts(path, start, count):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no < start:
                continue
            if len(rows) >= count:
                break
            obj = json.loads(line)
            pieces = [obj.get("title", "").strip(), obj.get("text", "").strip()]
            text = "\n".join(piece for piece in pieces if piece)
            if text:
                rows.append(text)
    if len(rows) != count:
        raise RuntimeError(f"Expected {count} examples in {path}, got {len(rows)}")
    return rows


def make_token_blocks(tokenizer, path, start, count, max_tokens, block_size):
    if max_tokens % block_size:
        raise ValueError("max_tokens must be divisible by block_size")
    texts = load_jsonl_texts(path, start, count)
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    tokens = []
    separator = tokenizer.eos_token_id
    for ids in encoded:
        tokens.extend(ids)
        if separator is not None:
            tokens.append(separator)
        if len(tokens) >= max_tokens:
            break
    if len(tokens) < max_tokens:
        raise RuntimeError(
            f"Only {len(tokens)} tokens available in {path}; need {max_tokens}"
        )
    return torch.tensor(tokens[:max_tokens], dtype=torch.long).view(-1, block_size)


def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "flash_attention_2" if device.startswith("cuda") else "sdpa",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except (ImportError, ValueError):
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def get_layers(model):
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise AttributeError("Expected a Qwen/LLaMA-style model.model.layers")
    return layers


@torch.inference_mode()
def collect_activations(args):
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    data_path = Path(args.data_root) / args.language / "test.jsonl"
    blocks = make_token_blocks(
        tokenizer,
        data_path,
        0,
        args.calibration_examples,
        args.calibration_tokens,
        args.block_size,
    )
    layers = get_layers(model)
    intermediate_size = layers[0].mlp.gate_proj.out_features
    counts = torch.zeros(
        len(layers), intermediate_size, dtype=torch.int64, device=args.device
    )
    handles = []

    for layer_id, layer in enumerate(layers):
        def hook(_module, _inputs, output, index=layer_id):
            counts[index].add_((output > 0).sum(dim=(0, 1)))

        handles.append(layer.mlp.act_fn.register_forward_hook(hook))

    loader = DataLoader(TensorDataset(blocks), batch_size=args.batch_size, shuffle=False)
    for step, (input_ids,) in enumerate(loader, start=1):
        model(input_ids=input_ids.to(args.device), use_cache=False)
        if step % 10 == 0 or step == len(loader):
            print(f"[collect:{args.language}] {step}/{len(loader)} batches", flush=True)

    for handle in handles:
        handle.remove()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "language": args.language,
            "counts": counts.cpu(),
            "n_tokens": int(blocks.numel()),
            "model": args.model,
            "block_size": args.block_size,
            "calibration_examples": args.calibration_examples,
        },
        output,
    )
    print(f"[collect:{args.language}] saved {output}")


def select_masks(args):
    if not 0 < args.top_rate <= 1:
        raise ValueError("top_rate must be in (0, 1]")
    objects = []
    for language in LANGUAGES:
        path = Path(args.activation_dir) / f"activation.{language}.pt"
        obj = torch.load(path, map_location="cpu")
        if obj["language"] != language:
            raise RuntimeError(f"Language mismatch in {path}")
        objects.append(obj)

    shape = objects[0]["counts"].shape
    if any(obj["counts"].shape != shape for obj in objects):
        raise RuntimeError("Activation tensors do not share the same shape")
    probs = torch.stack(
        [obj["counts"].double() / obj["n_tokens"] for obj in objects], dim=-1
    )
    normalized = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    entropy = -(normalized * normalized.clamp_min(1e-30).log()).sum(dim=-1)

    activation_bar = torch.quantile(probs.flatten(), args.activation_quantile)
    eligible = (probs > activation_bar).any(dim=-1)
    ranked_entropy = entropy.clone()
    ranked_entropy[~eligible] = torch.inf
    requested = max(1, round(entropy.numel() * args.top_rate))
    selected_count = min(requested, int(eligible.sum().item()))
    selected_flat = torch.topk(
        ranked_entropy.flatten(), k=selected_count, largest=False
    ).indices
    selected_layers = selected_flat // shape[1]
    selected_neurons = selected_flat % shape[1]

    language_masks = {language: [list() for _ in range(shape[0])] for language in LANGUAGES}
    selected_probs = probs[selected_layers, selected_neurons]
    assignments = selected_probs > activation_bar
    for row in range(selected_count):
        layer_id = int(selected_layers[row])
        neuron_id = int(selected_neurons[row])
        for lang_id, language in enumerate(LANGUAGES):
            if bool(assignments[row, lang_id]):
                language_masks[language][layer_id].append(neuron_id)

    generator = torch.Generator().manual_seed(args.seed)
    random_masks = {language: [] for language in LANGUAGES}
    for language in LANGUAGES:
        for layer_id, indices in enumerate(language_masks[language]):
            count = len(indices)
            excluded = set(indices)
            candidates = torch.tensor(
                [idx for idx in range(shape[1]) if idx not in excluded], dtype=torch.long
            )
            if count:
                order = torch.randperm(candidates.numel(), generator=generator)[:count]
                sampled = candidates[order].sort().values
            else:
                sampled = torch.empty(0, dtype=torch.long)
            random_masks[language].append(sampled)

    tensor_masks = {
        language: [torch.tensor(sorted(indices), dtype=torch.long) for indices in per_layer]
        for language, per_layer in language_masks.items()
    }
    output_obj = {
        "languages": LANGUAGES,
        "language_masks": tensor_masks,
        "random_masks": random_masks,
        "activation_probs": probs.float(),
        "entropy": entropy.float(),
        "meta": {
            "top_rate": args.top_rate,
            "activation_quantile": args.activation_quantile,
            "activation_bar": float(activation_bar),
            "requested_global_neurons": requested,
            "selected_global_neurons": selected_count,
            "eligible_neurons": int(eligible.sum()),
            "seed": args.seed,
            "n_tokens_per_language": {
                language: int(obj["n_tokens"])
                for language, obj in zip(LANGUAGES, objects)
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_obj, output)

    stats_path = output.with_suffix(".stats.json")
    stats = dict(output_obj["meta"])
    stats["per_language_total"] = {
        language: sum(mask.numel() for mask in tensor_masks[language])
        for language in LANGUAGES
    }
    stats["per_language_per_layer"] = {
        language: [mask.numel() for mask in tensor_masks[language]]
        for language in LANGUAGES
    }
    overlaps = {}
    for i, lang_a in enumerate(LANGUAGES):
        set_a = {
            (layer_id, int(neuron))
            for layer_id, mask in enumerate(tensor_masks[lang_a])
            for neuron in mask
        }
        for lang_b in LANGUAGES[i + 1:]:
            set_b = {
                (layer_id, int(neuron))
                for layer_id, mask in enumerate(tensor_masks[lang_b])
                for neuron in mask
            }
            union = set_a | set_b
            overlaps[f"{lang_a}-{lang_b}"] = len(set_a & set_b) / len(union) if union else 0.0
    stats["language_mask_jaccard"] = overlaps
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".layers.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", *LANGUAGES])
        for layer_id in range(shape[0]):
            writer.writerow(
                [layer_id, *[tensor_masks[lang][layer_id].numel() for lang in LANGUAGES]]
            )
    print(json.dumps(stats, indent=2))
    print(f"[select] saved {output}")


def resolve_condition(mask_obj, condition, device):
    if condition == "none":
        return None
    if condition.startswith("random_"):
        language = condition.removeprefix("random_")
        source = mask_obj["random_masks"]
    else:
        language = condition
        source = mask_obj["language_masks"]
    if language not in LANGUAGES:
        raise ValueError(f"Unknown condition: {condition}")
    return [indices.to(device) for indices in source[language]]


@torch.inference_mode()
def evaluate_ppl(args):
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    layers = get_layers(model)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    mask_obj = torch.load(args.mask_file, map_location="cpu") if args.mask_file else None
    if any(condition != "none" for condition in conditions) and mask_obj is None:
        raise ValueError("Non-baseline conditions require --mask-file")

    eval_blocks = {}
    for language in LANGUAGES:
        path = Path(args.data_root) / language / "test.jsonl"
        eval_blocks[language] = make_token_blocks(
            tokenizer,
            path,
            args.eval_start,
            args.eval_examples,
            args.eval_tokens,
            args.block_size,
        )

    active_mask = None
    handles = []
    for layer_id, layer in enumerate(layers):
        def hook(_module, _inputs, output, index=layer_id):
            if active_mask is not None and active_mask[index].numel():
                output.index_fill_(-1, active_mask[index], 0)
            return output

        handles.append(layer.mlp.act_fn.register_forward_hook(hook))

    rows = []
    for condition in conditions:
        active_mask = resolve_condition(mask_obj, condition, args.device) if mask_obj else None
        masked_count = sum(mask.numel() for mask in active_mask) if active_mask else 0
        for eval_language in LANGUAGES:
            loader = DataLoader(
                TensorDataset(eval_blocks[eval_language]),
                batch_size=args.batch_size,
                shuffle=False,
            )
            nll_sum = 0.0
            predicted_tokens = 0
            for (input_ids,) in loader:
                input_ids = input_ids.to(args.device)
                output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                count = input_ids.shape[0] * (input_ids.shape[1] - 1)
                nll_sum += float(output.loss) * count
                predicted_tokens += count
            mean_nll = nll_sum / predicted_tokens
            ppl = math.exp(min(mean_nll, 30.0))
            row = {
                "model": args.model,
                "condition": condition,
                "mask_language": condition.removeprefix("random_") if condition != "none" else "none",
                "is_random": condition.startswith("random_"),
                "eval_language": eval_language,
                "masked_neurons": masked_count,
                "tokens": predicted_tokens,
                "nll": mean_nll,
                "ppl": ppl,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    for handle in handles:
        handle.remove()
    write_rows(rows, Path(args.output))


def write_rows(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"[evaluate] saved {output}")


def merge_results(args):
    rows = []
    seen = set()
    for path in args.inputs:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["model"], row["condition"], row["eval_language"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    rows.sort(key=lambda row: (row["model"], row["condition"], row["eval_language"]))
    write_rows(rows, Path(args.output))


def main():
    args = parse_args()
    torch.manual_seed(42)
    random.seed(42)
    if args.command == "collect":
        collect_activations(args)
    elif args.command == "select":
        select_masks(args)
    elif args.command == "evaluate":
        evaluate_ppl(args)
    elif args.command == "merge":
        merge_results(args)


if __name__ == "__main__":
    main()
