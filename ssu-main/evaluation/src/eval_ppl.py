# -*- coding: utf-8 -*-
"""
eval_ppl_qwen_fineweb2_5langs.py


运行示例：
python eval_ppl_qwen_fineweb2_5langs.py \
  --data_root /d/fineweb2_cpt \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --batch_size 8 \
  --max_length 512 \
  --out_dir /d/eval_out
"""

import os
import json
import math
import time
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


LANGS_DEFAULT = ["swh_Latn", "amh_Ethi", "tel_Telu", "kir_Cyrl", "ibo_Latn"]


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def collate_batch(examples):
    """
    examples: List[dict], each has "input_ids" (list[int]) and maybe "labels"
    返回 torch.LongTensor: input_ids, attention_mask, labels
    """
    input_ids = [e["input_ids"] for e in examples]
    # 你的数据是 packing 后固定长度，一般无需 pad；这里仍做一次安全处理
    max_len = max(len(x) for x in input_ids)
    batch = torch.full((len(input_ids), max_len), fill_value=0, dtype=torch.long)
    attn = torch.zeros((len(input_ids), max_len), dtype=torch.long)

    for i, ids in enumerate(input_ids):
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, : len(ids)] = 1

    # labels：优先用数据里的 labels，否则用 input_ids
    if "labels" in examples[0]:
        labels_list = [e["labels"] for e in examples]
        labels = torch.full((len(labels_list), max_len), fill_value=-100, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
    else:
        labels = batch.clone()

    return batch, attn, labels


@torch.no_grad()
def eval_ppl_on_dataset(model, dataset, device, batch_size=8, dtype_autocast=None, log_interval=50):
    """
    返回：avg_nll, ppl, total_tokens
    """
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    total_nll = 0.0
    total_tokens = 0

    t0 = time.time()
    for step, (input_ids, attention_mask, labels) in enumerate(dl, start=1):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # 统计有效 token（labels != -100）
        valid_tokens = (labels != -100).sum().item()

        if dtype_autocast is None:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            with torch.autocast(device_type="cuda", dtype=dtype_autocast):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = out.loss  # 平均到 valid token 的交叉熵（标准 transformers 行为）
        total_nll += loss.item() * valid_tokens
        total_tokens += valid_tokens

        if step % log_interval == 0:
            elapsed = time.time() - t0
            cur_avg = total_nll / max(1, total_tokens)
            cur_ppl = math.exp(min(50, cur_avg))  # 防止极端溢出
            print(f"[{now_str()}]  进度：{step:>6} step | "
                  f"累计tokens={total_tokens:,} | 当前PPL≈{cur_ppl:.4f} | 用时={elapsed/60:.1f} 分钟")

    avg_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(min(50, avg_nll))
    return avg_nll, ppl, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="你的 fineweb2_cpt 根目录，例如 /d/fineweb2_cpt")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--langs", type=str, nargs="*", default=LANGS_DEFAULT)
    parser.add_argument("--split_dir", type=str, default="test",
                        help="每种语言下的 split 目录名，默认 test（即 {data_root}/{lang}/test）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="./eval_out")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="优先使用 bf16 autocast（需要支持 bf16 的 GPU）")
    parser.add_argument("--fp16", action="store_true", help="使用 fp16 autocast")
    parser.add_argument("--limit_batches", type=int, default=0,
                        help="调试用：只跑前 N 个 batch；0 表示全量")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("========================================")
    print(f"[{now_str()}] ✅ 开始评测 PPL")
    print(f"模型：{args.model_name}")
    print(f"数据根目录：{args.data_root}")
    print(f"语言：{args.langs}")
    print(f"batch_size：{args.batch_size}")
    print(f"max_length：{args.max_length}")
    print(f"输出目录：{args.out_dir}")
    print("========================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{now_str()}] 设备：{device}")

    # tokenizer 这里主要用于确保 pad_token 设置正确（尽管你数据是 packed）
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=(torch.float16 if (device == "cuda" and args.fp16) else None),
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    dtype_autocast = None
    if device == "cuda":
        if args.bf16:
            dtype_autocast = torch.bfloat16
        elif args.fp16:
            dtype_autocast = torch.float16

    results = []
    for lang in args.langs:
        test_dir = os.path.join(args.data_root, lang, args.split_dir)
        print("\n----------------------------------------")
        print(f"[{now_str()}] 🌍 语言：{lang}")
        print(f"[{now_str()}] 读取数据：{test_dir}")

        ds = load_from_disk(test_dir)

        # 可选：确保字段存在
        if "input_ids" not in ds.column_names:
            raise RuntimeError(f"{test_dir} 不包含 input_ids 字段，当前字段：{ds.column_names}")

        # 可选：调试只跑部分 batch
        if args.limit_batches and args.limit_batches > 0:
            n = min(len(ds), args.limit_batches * args.batch_size)
            print(f"[{now_str()}] ⚠️ 调试模式：只评测前 {n} 条 blocks")
            ds = ds.select(range(n))

        print(f"[{now_str()}] blocks 数量：{len(ds)}")
        t0 = time.time()
        avg_nll, ppl, total_tokens = eval_ppl_on_dataset(
            model=model,
            dataset=ds,
            device=device,
            batch_size=args.batch_size,
            dtype_autocast=dtype_autocast,
            log_interval=50,
        )
        mins = (time.time() - t0) / 60.0

        row = {
            "lang": lang,
            "blocks": len(ds),
            "total_tokens": int(total_tokens),
            "avg_nll": float(avg_nll),
            "ppl": float(ppl),
            "minutes": float(mins),
        }
        results.append(row)
        print(f"[{now_str()}] ✅ {lang} 完成：PPL={ppl:.4f} | avg_nll={avg_nll:.6f} | tokens={total_tokens:,} | 用时={mins:.2f} 分钟")

    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.out_dir, f"ppl_5langs_{ts}.json")
    out_csv = os.path.join(args.out_dir, f"ppl_5langs_{ts}.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 简单写 csv（不依赖 pandas）
    header = ["lang", "blocks", "total_tokens", "avg_nll", "ppl", "minutes"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in results:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    print("\n========================================")
    print(f"[{now_str()}] 🎉 全部语言评测完成！")
    print(f"结果已保存：\n- {out_json}\n- {out_csv}")
    print("========================================")


if __name__ == "__main__":
    main()
