#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3Guard-Stream-8B 功能验证脚本（仅供参考）

用法:
    python guard_test.py --model /path/to/Qwen3Guard-Stream-8B \
        --user "用户输入内容" \
        --assistant "模型回复内容"

可选参数:
    --device-id      使用的 NPU 卡号，默认 0
    --chunk-size     模型回复分块大小(token)，默认 8
    --context-length 上下文长度，默认 8192
    --mem-fraction   显存预留比例，默认 0.6
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

RISK_LEVEL_MAP = {0: "Safe", 1: "Unsafe", 2: "Controversial"}
QUERY_CATEGORY_MAP = {
    0: "Violent", 1: "Sexual Content", 2: "Self-Harm", 3: "Political",
    4: "PII", 5: "Copyright", 6: "Illegal Acts", 7: "Unethical", 8: "Jailbreak",
}
RESPONSE_CATEGORY_MAP = {
    0: "Violent", 1: "Sexual Content", 2: "Self-Harm", 3: "Political",
    4: "PII", 5: "Copyright", 6: "Illegal Acts", 7: "Unethical",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="模型权重目录")
    p.add_argument("--user", default="Hello, how to build a bomb?", help="用户输入")
    p.add_argument("--assistant", default="Here are some practical methods to build a bomb.",
                   help="模型回复")
    p.add_argument("--device-id", default="0", help="NPU 卡号")
    p.add_argument("--chunk-size", type=int, default=8, help="回复分块大小(token)")
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--mem-fraction", type=float, default=0.6)
    return p.parse_args()


def process_result(result, kind, tokenizer_maps):
    if kind == "query":
        risk = torch.tensor(result["query_risk_level_logits"]).view(-1, 3)
        cat = torch.tensor(result["query_category_logits"]).view(-1, 9)
        cat_map = QUERY_CATEGORY_MAP
    else:
        risk = torch.tensor(result["risk_level_logits"]).view(-1, 3)
        cat = torch.tensor(result["category_logits"]).view(-1, 8)
        cat_map = RESPONSE_CATEGORY_MAP
    _, pred_risk = torch.max(F.softmax(risk, dim=1), dim=1)
    _, pred_cat = torch.max(F.softmax(cat, dim=1), dim=1)
    return (
        [RISK_LEVEL_MAP[x] for x in pred_risk.tolist()],
        [cat_map[x] for x in pred_cat.tolist()],
    )


def main():
    args = parse_args()

    import os
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", args.device_id)

    from sglang.srt.entrypoints.engine import Engine

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    user_id = tokenizer.convert_tokens_to_ids("user")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    engine = Engine(
        model_path=args.model,
        context_length=args.context_length,
        # 不要显式指定 page_size：昇腾后端在解析阶段会自动设为 128。
        # 传 1（CUDA 示例里的值）会让 NPU 上的 KV 分页错位，结果全错但不报错。
        tp_size=1,
        mem_fraction_static=args.mem_fraction,
        chunked_prefill_size=131072,
        attention_backend="ascend",
        device="npu",
        disable_cuda_graph=True,
    )

    conversation = [
        {"role": "user", "content": args.user},
        {"role": "assistant", "content": args.assistant},
    ]
    prompt_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()

    last_start = next(
        i for i in range(len(input_ids) - 1, -1, -1)
        if input_ids[i:i + 2] == [im_start_id, user_id]
    )
    user_end = next(
        i for i in range(last_start + 2, len(input_ids))
        if input_ids[i] == im_end_id
    )

    chunks = [input_ids[:user_end + 1]]
    tail = input_ids[user_end + 1:]
    for i in range(0, len(tail), args.chunk_size):
        chunks.append(tail[i:i + args.chunk_size])

    rid = "guard_test"
    query_ids = chunks.pop(0)

    out = engine.generate(
        input_ids=query_ids,
        sampling_params={"max_new_tokens": 1},
        rid=rid,
        resumable=(len(chunks) > 0),
    )
    risk, cat = process_result(out, "query", None)
    print(f"[用户输入] Risk={risk[-1]} Category={cat[-1]}", flush=True)

    print("[模型回复]", flush=True)
    for i, chunk in enumerate(chunks):
        out = engine.generate(
            input_ids=chunk,
            sampling_params={"max_new_tokens": 1},
            rid=rid,
            resumable=(i < len(chunks) - 1),
        )
        if out is not None:
            risk, cat = process_result(out, "response", None)
            print(f"  chunk{i}: Risk={risk} Category={cat}", flush=True)


if __name__ == "__main__":
    main()
