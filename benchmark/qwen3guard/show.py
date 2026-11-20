#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读 guard_bench_serving 的结果 JSON，打一行汇总 + 慢调用现场。

用法：  python3 show.py c100_chunk16.json [更多文件...]
"""
import json
import sys

if len(sys.argv) < 2:
    sys.exit("用法: python3 show.py <结果.json> [...]")

for path in sys.argv[1:]:
    try:
        j = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"{path}: 读不了 ({e})")
        continue
    a, s = j["实验参数"], j["汇总_按实际会话构成加权"]
    first, chunk = s["首次分类时延_ms"], s["分块分类时延_ms"]
    slo_key = next((k for k in s if k.startswith("首次分类超")), None)
    if slo_key is None:
        s["__noslo__"] = {"占比": "n/a"}
        slo_key = "__noslo__"
    rate = a["限速_token每秒每路"]
    budget = (a["分块大小"] / rate * 1000) if rate else None

    print(f"\n── {path}")
    print(f"   {a['并发会话数']} 路 | chunk {a['分块大小']} | rate {rate}"
          f"{f' | 单块预算 {budget:.0f}ms' if budget else ' | 背靠背'}"
          f" | {'失败' if '失败' in j else 'OK'}")
    print(f"   吞吐      {j['吞吐']['调用每秒']:>7.1f} 调用/s   "
          f"{j['吞吐']['被检查token每秒']:>8.1f} token/s   总耗时 {j['总耗时_s']:.1f}s")
    def line(label, d):
        if not d or "P50" not in d:
            print(f"   {label}  无样本（本轮没有成功的调用）")
            return
        print(f"   {label}  P50 {d['P50']:>7.1f}  P90 {d['P90']:>7.1f}  "
              f"P99 {d['P99']:>7.1f}  max {d['max']:>7.1f}"
              + (f"   超200ms {s[slo_key]['占比']}" if label == "首次分类" else ""))

    line("首次分类", first)
    line("分块分类", chunk)
    if budget and chunk and "P50" in chunk:
        print(f"   跟不跟得上 单次超预算 {s.get('单次超预算次数','?')} 次 "
              f"({s.get('单次超预算占比','?')})   "
              f"累积超期 {s.get('超期次数','?')} 次 / {s.get('超期会话数','?')} 路")
    ref = j["规模"].get("被准入控制拒绝_503", 0)
    total = j["规模"].get("业务问答数_即会话数", 0)
    if ref:
        tag = "全部被拒 —— 服务端拒单，本轮无有效数据" if total and ref >= total \
              else "准入控制在工作，不是故障"
        print(f"   被拒绝    {ref}/{total} 条流收到 503（{tag}）")
        for m in (j.get("被拒绝样例") or [])[:1]:
            print(f"      样例: {str(m)[:160]}")
    if j.get("失败"):
        for m in j["失败"]:
            print(f"   !! {m}")
    slow = j.get("最慢的调用") or []
    if slow:
        print(f"   慢调用 {len(slow)} 条（前 8）:")
        for x in slow[:8]:
            print(f"      {x['会话']:<10} {x['位置']:<10} "
                  f"{x['时延_ms']:>8.1f}ms  发出于 {x['发出时刻_s']:>6.2f}s")
    else:
        print("   慢调用  无")
