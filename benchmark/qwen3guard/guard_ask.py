#!/usr/bin/env python3
"""Qwen3Guard 真实分类：套对话模板后调 /generate，输出可读标签。

用法:
  python guard_ask.py "你的问题"
  python guard_ask.py "你的问题" "模型的回复"
"""
import argparse, json, urllib.request, uuid
from transformers import AutoTokenizer

RISK = {0: "Safe", 1: "Unsafe", 2: "Controversial"}
QCAT = {0: "Violent", 1: "Sexual Content", 2: "Self-Harm", 3: "Political",
        4: "PII", 5: "Copyright", 6: "Illegal Acts", 7: "Unethical", 8: "Jailbreak"}
RCAT = {k: v for k, v in QCAT.items() if k != 8}


def argmax_label(row, table):
    # 这些是 logits 不是概率；argmax 与 softmax 后 argmax 等价
    return table[max(range(len(row)), key=lambda i: row[i])]


def post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("user")
    p.add_argument("assistant", nargs="?", default="")
    p.add_argument("--model", default="/path/to/models/Qwen3Guard-Stream-8B")
    p.add_argument("--url", default="http://127.0.0.1:30111/generate")
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--rid", default="ask-" + uuid.uuid4().hex[:8])
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    conv = [{"role": "user", "content": a.user}]
    if a.assistant:
        conv.append({"role": "assistant", "content": a.assistant})
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors=None).input_ids

    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    user_id = tok.convert_tokens_to_ids("user")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    start = next(i for i in range(len(ids) - 1, -1, -1)
                 if ids[i:i + 2] == [im_start, user_id])
    end = next(i for i in range(start + 2, len(ids)) if ids[i] == im_end)
    query_ids, tail = ids[:end + 1], ids[end + 1:]

    chunks = [tail[i:i + a.chunk_size] for i in range(0, len(tail), a.chunk_size)]

    d = post(a.url, {"input_ids": query_ids,
                     "sampling_params": {"max_new_tokens": 1},
                     "rid": a.rid, "resumable": bool(chunks)})
    if "query_risk_level_logits" not in d:
        print(f"用户输入 出错: {json.dumps(d, ensure_ascii=False)[:300]}")
        return 1
    qr = d["query_risk_level_logits"][-1][-1]
    qc = d["query_category_logits"][-1][-1]
    print(f"[用户输入] {argmax_label(qr, RISK):<14} {argmax_label(qc, QCAT)}")

    for k, ch in enumerate(chunks):
        d = post(a.url, {"input_ids": ch,
                         "sampling_params": {"max_new_tokens": 1},
                         "rid": a.rid, "resumable": k < len(chunks) - 1})
        if "risk_level_logits" not in d:
            print(f"  块{k} 出错: {json.dumps(d, ensure_ascii=False)[:300]}")
            return 1
        rows = d["risk_level_logits"]
        c = d["category_logits"][-1][-1]
        r = rows[-1][-1]
        print(f"  块{k:<3} {argmax_label(r, RISK):<14} {argmax_label(c, RCAT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())