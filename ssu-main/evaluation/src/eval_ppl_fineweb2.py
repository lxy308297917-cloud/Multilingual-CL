# -*- coding: utf-8 -*-

import os
import math
import argparse
import csv
from datetime import datetime

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


def p(msg):
    print(msg, flush=True)


@torch.no_grad()
def eval_dataset_ppl(model, dataset, batch_size=4, max_batches=None):
    """
    在一个 test dataset 上评测 PPL

    dataset:
        HuggingFace Dataset
        每条数据需包含:
            - input_ids
            - labels
            - (可选) attention_mask

    返回:
        avg_loss, ppl, 使用的 blocks 数
    """

    model.eval()

    total_loss = 0.0
    total_blocks = 0

    def collate_fn(batch):
        input_ids = torch.tensor(
            [x["input_ids"] for x in batch],
            dtype=torch.long
        )
        labels = torch.tensor(
            [x["labels"] for x in batch],
            dtype=torch.long
        )

        attention_mask = None
        if "attention_mask" in batch[0]:
            attention_mask = torch.tensor(
                [x["attention_mask"] for x in batch],
                dtype=torch.long
            )

        return input_ids, labels, attention_mask

    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"📦 测试集 blocks 总数：{num_samples}")
    print(f"🔢 batch_size = {batch_size}")
    print(f"🔁 预计 batch 数：{num_batches}")
    print("-" * 60)

    for batch_idx in range(num_batches):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"⏹ 已达到最大 batch 限制：{max_batches}，提前结束测试")
            break

        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)
        batch = [dataset[i] for i in range(start, end)]

        input_ids, labels, attention_mask = collate_fn(batch)

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss.item()
        bs = end - start

        total_loss += loss * bs
        total_blocks += bs

        # 每 20 个 batch 打印一次进度
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            avg_loss = total_loss / total_blocks
            ppl = math.exp(avg_loss)
            print(
                f"🧪 进度: {total_blocks}/{num_samples} blocks | "
                f"当前平均 Loss={avg_loss:.4f} | PPL={ppl:.2f}"
            )

    avg_loss = total_loss / max(1, total_blocks)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")

    return avg_loss, ppl, total_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="模型名称或本地 checkpoint 路径")
    parser.add_argument("--data_root", type=str, required=True,
                        help="FineWeb2 处理后的数据根目录")
    parser.add_argument("--langs", type=str, nargs="+", required=True,
                        help="要测试的语言列表，例如: swh_Latn amh_Ethi")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_test_blocks", type=int, default=None,
                        help="只测试前 N 个 blocks（调试用）")
    args = parser.parse_args()

    os.makedirs("eval_results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 只取模型路径的 basename，避免 Windows 下 ':' 等非法字符
    model_basename = os.path.basename(
        args.model_name.rstrip("/\\")
    )

    result_path = os.path.join(
    "eval_results",
    f"ppl_{model_basename}_{timestamp}.csv"
)
    
    
    p("=" * 80)
    p("🚀 开始大模型在 Test 集上的 PPL 测试")
    p(f"🧠 模型：{args.model_name}")
    p(f"📂 数据根目录：{args.data_root}")
    p(f"🌍 测试语言：{args.langs}")
    p(f"🔢 batch_size：{args.batch_size}")
    p(f"✂️ 最大测试 blocks：{args.max_test_blocks if args.max_test_blocks else '不限制'}")
    p("=" * 80)

    p("🔧 加载 tokenizer 和模型中...")
    p("✅ [1/6] 参数解析完成，准备加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    p("✅ [2/6] tokenizer 加载完成，准备加载 model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    p("✅ [3/6] model 加载完成，准备进入语言循环...")
    all_results = []

    for lang in args.langs:
        print("\n" + "#" * 80)
        print(f"🌍 开始测试语言：{lang}")

        test_dir = os.path.join(args.data_root, lang, "test")
        print(f"📂 test 路径：{test_dir}")

        if not os.path.isdir(test_dir):
            print("❌ test 目录不存在，跳过该语言")
            continue

        dataset = load_from_disk(test_dir)

        if args.max_test_blocks:
            dataset = dataset.select(
                range(min(args.max_test_blocks, len(dataset)))
            )
            print(f"✂️ 实际使用 blocks 数：{len(dataset)}")

        print("🧪 正式开始计算 PPL ...")
        avg_loss, ppl, used_blocks = eval_dataset_ppl(
            model,
            dataset,
            batch_size=args.batch_size
        )

        print(
            f"✅ 测试完成 | {lang} | "
            f"blocks={used_blocks} | "
            f"平均 Loss={avg_loss:.4f} | PPL={ppl:.2f}"
        )

        all_results.append((lang, used_blocks, avg_loss, ppl))

    print("\n" + "=" * 80)
    print("📊 所有语言测试完成，结果汇总：")

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "num_blocks", "avg_loss", "ppl"])

        for lang, blocks, loss, ppl in all_results:
            print(
                f"- {lang:10s} | blocks={blocks:6d} | "
                f"Loss={loss:.4f} | PPL={ppl:.2f}"
            )
            writer.writerow([lang, blocks, loss, ppl])

    print("=" * 80)
    print(f"✅ 测试结果已保存到：{result_path}")


if __name__ == "__main__":
    main()
