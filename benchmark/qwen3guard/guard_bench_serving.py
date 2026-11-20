#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3Guard-Stream-8B 在线压测（HTTP 客户端）

压测一个**已经启动**的 SGLang HTTP 服务。本脚本只通过 HTTP 访问服务，
不 import、不实例化 Engine。

协议依据（v0.5.20 + Qwen3Guard 适配分支，已核对源码）：
  * entrypoints/http_server.py  `async def generate_request(obj: GenerateReqInput, ...)`
    FastAPI 把请求 JSON 直接转成 GenerateReqInput，因此 rid / resumable 可从 body 传入。
  * managers/tokenizer_manager.py  Guard 响应为
    {"meta_info", "risk_level_logits", "category_logits",
     "query_risk_level_logits", "query_category_logits"}，全部是 list[float]。
  * managers/tokenizer_manager.py  请求入口处
    `if rid in self.rid_to_state: raise ValueError("Duplicate request ID detected")`
    —— 同一 rid 必须等上一次响应返回后才能再用，所以会话内严格串行，不能流水线。
  * Guard 不产出 token，stream=True 无意义，全部走非流式 POST。

用法：
    # 4 路 smoke
    python guard_bench_serving.py --base-url http://127.0.0.1:30000 \
        --tokenizer /path/to/Qwen3Guard-Stream-8B --concurrency 4

    # 100 路，背靠背（算力上限）
    python guard_bench_serving.py --base-url http://127.0.0.1:30000 \
        --tokenizer /path/to/Qwen3Guard-Stream-8B --concurrency 100 --rate 0

    # 100 路，主模型 15 token/s 节奏（真实场景）
    python guard_bench_serving.py --base-url http://127.0.0.1:30000 \
        --tokenizer /path/to/Qwen3Guard-Stream-8B --concurrency 100 --rate 15
"""

import argparse
import asyncio
import json
import random
import os
import statistics
import sys
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

# ---------------------------------------------------------------------------
# 复用 sglang.benchmark.serving 的客户端设施。
# 注意：不要用它的 get_request_headers()，那个函数引用模块级全局 args，
# 未经 set_global_args() 会 NameError。get_auth_headers() 是干净的。
# ---------------------------------------------------------------------------
try:
    from sglang.benchmark.serving import (  # noqa: F401
        _create_bench_client_session,
        get_auth_headers,
        wait_for_endpoint,
    )
    _REUSE = "sglang.benchmark.serving"
except Exception as _e:  # 没装 sglang 时退化为等价实现，便于在任意机器上自检
    _REUSE = f"内置等价实现（import sglang.benchmark.serving 失败: {type(_e).__name__}）"

    def _create_bench_client_session():
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
            read_bufsize=10 * 1024 ** 2,
        )

    def get_auth_headers():
        import os
        k = os.environ.get("OPENAI_API_KEY")
        if k:
            return {"Authorization": f"Bearer {k}"}
        k = os.environ.get("API_KEY")
        return {"Authorization": k} if k else {}

    def wait_for_endpoint(url, timeout_sec=60):
        import requests
        t0 = time.perf_counter()
        while True:
            try:
                if requests.get(url, headers=get_auth_headers(),
                                timeout=5).status_code == 200:
                    return True
            except Exception:
                pass
            if time.perf_counter() - t0 >= timeout_sec:
                return False
            time.sleep(1)


class StreamRefused(Exception):
    """服务端按准入控制拒绝了这条流（HTTP 503）。

    这是服务端的正常应答，不是故障：它恰恰说明保护机制在工作。所以被拒的流
    只是提前结束，不中断其余会话，也不让整轮标记为失败——否则"被拒了多少路"
    这个最该测的数字反而测不出来。
    """


CLASSES = ("simple", "medium", "complex")
CN = {"simple": "简单", "medium": "中等", "complex": "复杂"}


def parse_lens(s: str) -> Tuple[int, int]:
    a, _, b = s.partition(",")
    return int(a), int(b)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:30000",
                   help="已启动的 SGLang 服务地址")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer 路径（一般就是模型权重目录）。"
                        "留空进入自检模式：不加载 tokenizer，"
                        "用合成 token 验证客户端逻辑，不可用于性能结论")
    p.add_argument("--concurrency", type=int, default=100, help="并发会话数")
    p.add_argument("--mix", default="50,40,10",
                   help="简单,中等,复杂 三类会话的占比")
    p.add_argument("--len-simple", default="200,400",
                   help="简单会话的 输入token,回复token")
    p.add_argument("--len-medium", default="200,400",
                   help="中等会话的 输入token,回复token")
    p.add_argument("--len-complex", default="200,400",
                   help="复杂会话的 输入token,回复token")
    p.add_argument("--chunk-size", type=int, default=8, help="回复分块大小(token)")
    p.add_argument("--rate", type=float, default=0,
                   help="主模型产出速率(token/s)。0 = 背靠背发送，测算力上限")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=60,
                   help="单次 HTTP 调用超时(秒)")
    p.add_argument("--max-inflight", type=int, default=0,
                   help="全局在途请求上限，0 = 不限")
    p.add_argument("--warmup", type=int, default=2,
                   help="预热会话数，不计入统计")
    p.add_argument("--ramp", type=float, default=0,
                   help="会话启动时刻在这么多秒内随机铺开。0 表示全部同时开始——"
                        "那会造成人为的启动突发（N 路同时发首次分类），把首次分类时延"
                        "抬高一个数量级，不反映真实业务。高并发测首次分类务必设置")
    p.add_argument("--no-jitter", action="store_true",
                   help="关闭启动抖动。默认开启，避免 N 路完全同步造成人为尖峰")
    p.add_argument("--progress", type=float, default=5,
                   help="进度打印间隔(秒)，0 = 不打印")
    p.add_argument("--slow-ms", type=float, default=1000,
                   help="超过这个时延的调用会被单独记录现场（会话/位置/时间轴偏移），"
                        "用来判断尖刺是系统性的还是随机的")
    p.add_argument("--slo-ms", type=float, default=200,
                   help="首次分类时延目标(ms)，用于统计超标次数")
    p.add_argument("--out", default="guard_bench_serving_result.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 数据构造：必须符合 Qwen3Guard 的对话模板，裸随机 token 测的是另一条路径
# ---------------------------------------------------------------------------
def build_token_ids(tokenizer, input_len, output_len, rng):
    """构造 token 数精确的一轮对话。

    先用单 token 占位符拿到 chat template 的骨架，再把占位符替换成指定数量的
    随机 token。送进模型的结构与真实请求一致，只有内容是随机的。
    """
    ph_user, ph_asst = "\u0001", "\u0002"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": ph_user},
         {"role": "assistant", "content": ph_asst}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors=None).input_ids
    u_id = tokenizer(ph_user, add_special_tokens=False).input_ids
    a_id = tokenizer(ph_asst, add_special_tokens=False).input_ids
    assert len(u_id) == 1 and len(a_id) == 1, "占位符不是单 token，换一个占位符"
    u_pos, a_pos = ids.index(u_id[0]), ids.index(a_id[0])

    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    pool = [t for t in sorted(tokenizer.get_vocab().values())
            if isinstance(t, int) and t not in special]
    rand_u = [pool[rng.randrange(len(pool))] for _ in range(input_len)]
    rand_a = [pool[rng.randrange(len(pool))] for _ in range(output_len)]

    query_ids = ids[:u_pos] + rand_u + ids[u_pos + 1:a_pos]
    tail_ids = rand_a + ids[a_pos + 1:]
    return query_ids, tail_ids


def build_token_ids_selftest(input_len, output_len, rng):
    """自检模式：不加载 tokenizer，只为验证客户端逻辑。"""
    head, mid, tail = [1, 2, 3], [4, 5], [6, 7]
    rand = lambda n: [rng.randrange(1000, 90000) for _ in range(n)]
    return head + rand(input_len) + mid, rand(output_len) + tail


def split_mix(n: int, weights: List[float]) -> List[int]:
    """按权重把 n 个会话分到各类，最大余数法保证总数精确等于 n。"""
    total = sum(weights)
    raw = [n * w / total for w in weights]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True):
        if sum(out) >= n:
            break
        out[i] += 1
    return out


# ---------------------------------------------------------------------------
# 统计容器
# ---------------------------------------------------------------------------
@dataclass
class ClassStats:
    first_latency: List[float] = field(default_factory=list)
    chunk_latency: List[float] = field(default_factory=list)
    session_time: List[float] = field(default_factory=list)
    late_events: int = 0       # 累积漂移：完成时刻晚于下一块的计划到达时刻
    over_interval: int = 0     # 单次超预算：这一次的时延本身就超过了 chunk/rate
    late_sessions: int = 0
    sessions: int = 0
    refused: int = 0           # 被准入控制拒绝的流数（503，不是故障）
    calls: int = 0
    tokens: int = 0
    slow: List[dict] = field(default_factory=list)   # 慢调用现场


@dataclass
class Runtime:
    done: int = 0
    inflight: int = 0
    sessions_done: int = 0
    failures: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)


def pct(xs: List[float], q: int) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    # method="inclusive"：分位数不会外推到样本范围之外。默认的 "exclusive"
    # 在小样本下会算出比 max 还大的 P90，看起来像数据错乱。
    return statistics.quantiles(sorted(xs), n=100, method="inclusive")[min(q, 99) - 1]


def lat_block(xs: List[float]) -> Dict:
    if not xs:
        return {}
    return {
        "P50": round(pct(xs, 50) * 1000, 1),
        "P90": round(pct(xs, 90) * 1000, 1),
        "P95": round(pct(xs, 95) * 1000, 1),
        "P99": round(pct(xs, 99) * 1000, 1),
        "max": round(max(xs) * 1000, 1),
        "样本数": len(xs),
    }


# ---------------------------------------------------------------------------
# 单次 HTTP 调用
# ---------------------------------------------------------------------------
async def guard_call(session, url, headers, ids, rid, resumable,
                     rt: Runtime, sem, timeout, where):
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": 1},
        "rid": rid,
        "resumable": resumable,
    }

    async def _go():
        async with session.post(url, json=payload, headers=headers) as resp:
            body = await resp.read()
            if resp.status == 503:
                raise StreamRefused(
                    f"{where}: {body[:300].decode('utf-8', 'replace')}")
            if resp.status != 200:
                raise RuntimeError(
                    f"{where}: HTTP {resp.status} — {body[:400].decode('utf-8', 'replace')}")
            data = json.loads(body)
        # 协议自检：少了这个字段说明服务端不是 Guard 模型，或响应结构变了。
        # 不做这一步的话，一个空响应会被当成“超快”，直接污染性能结论。
        if "risk_level_logits" not in data:
            # 被服务端中止的请求，tokenizer_manager._handle_abort_req 一律回
            # {text, output_ids, meta_info}——这是“中止”，不是“模型不对”。
            # 中止原因在 meta_info.finish_reason 里，必须原样带出来。
            fr = (data.get("meta_info") or {}).get("finish_reason") or {}
            if isinstance(fr, dict) and fr.get("type") == "abort":
                mi = data.get("meta_info") or {}
                raise RuntimeError(
                    f"{where}: 服务端中止了这次调用（该块未被分类）。"
                    f"rid={rid} status={fr.get('status_code')} "
                    f"原因={fr.get('message')!r} "
                    f"e2e={mi.get('e2e_latency')}")
            raise RuntimeError(
                f"{where}: 响应里没有 risk_level_logits，"
                f"服务端可能不是 Qwen3Guard 模型。实际键: {sorted(data)[:8]} "
                f"meta_info={json.dumps(data.get('meta_info'), ensure_ascii=False)[:300]}")
        return data

    rt.inflight += 1
    try:
        if sem is None:
            return await asyncio.wait_for(_go(), timeout=timeout)
        async with sem:
            return await asyncio.wait_for(_go(), timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"分类调用超时 {timeout}s，卡在 {where}；"
            f"已完成 {rt.done} 次调用，在途 {rt.inflight} 个") from None
    finally:
        rt.inflight -= 1


# ---------------------------------------------------------------------------
# 一路会话
# ---------------------------------------------------------------------------
# 每次运行一个唯一前缀。radix 树活得比压测进程久，rid 复用会让新流命中上一轮
# 留下的前缀，那不是真实场景，也会干扰缓存命中率的读数。
RUN_ID = uuid.uuid4().hex[:8]

# 压测起跑时刻，慢调用记录相对它的偏移，便于看尖刺在时间轴上的分布
T_START = 0.0


async def run_session(session, url, headers, sid, kind, query_ids, tail_ids,
                      chunk_size, rate, cs: ClassStats, rt: Runtime,
                      sem, timeout, jitter, slo, slow_ms=1000.0):
    rid_base = f"guard-{RUN_ID}-{kind}-{sid}"

    def note_slow(where, lat, sent_at):
        if lat * 1000 >= slow_ms:
            cs.slow.append({
                "会话": f"{CN[kind]}{sid}", "位置": where,
                "时延_ms": round(lat * 1000, 1),
                "发出时刻_s": round(sent_at - T_START, 2),
            })
    chunks = [tail_ids[i:i + chunk_size]
              for i in range(0, len(tail_ids), chunk_size)]

    if jitter:
        await asyncio.sleep(jitter)

    try:
        await _run_session_body(
            session, url, headers, sid, kind, query_ids, tail_ids, chunks,
            chunk_size, rate, cs, rt, sem, timeout, slo, note_slow)
    except StreamRefused as e:
        cs.refused += 1
        rt.refused.append(str(e))
        return


async def _run_session_body(session, url, headers, sid, kind, query_ids,
                            tail_ids, chunks, chunk_size, rate, cs, rt, sem,
                            timeout, slo, note_slow):
    rid_base = f"guard-{RUN_ID}-{kind}-{sid}"
    interval = (chunk_size / rate) if rate > 0 else 0.0
    t0 = time.perf_counter()
    # 主模型第 k 块（0-based）产出完成的时刻
    planned = [t0 + (k + 1) * interval for k in range(len(chunks))]
    late_here = 0

    # 第一次：分类用户输入
    await guard_call(session, url, headers, query_ids, rid_base,
                     len(chunks) > 0, rt, sem, timeout,
                     f"{CN[kind]}会话{sid} 的用户输入分类")
    rt.done += 1
    cs.calls += 1
    cs.tokens += len(query_ids)
    c_first = time.perf_counter()
    cs.first_latency.append(c_first - t0)
    note_slow("首次分类", c_first - t0, t0)
    if interval and chunks and c_first > planned[0]:
        late_here += 1

    for k, chunk in enumerate(chunks):
        if interval:
            delay = planned[k] - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        t1 = time.perf_counter()
        await guard_call(session, url, headers, chunk, rid_base,
                         k < len(chunks) - 1, rt, sem, timeout,
                         f"{CN[kind]}会话{sid} 的第{k}个分块")
        rt.done += 1
        cs.calls += 1
        cs.tokens += len(chunk)
        c_k = time.perf_counter()
        cs.chunk_latency.append(c_k - t1)
        note_slow(f"第{k}块", c_k - t1, t1)
        if interval and (c_k - t1) > interval:
            cs.over_interval += 1
        if interval and k + 1 < len(chunks) and c_k > planned[k + 1]:
            late_here += 1

    cs.session_time.append(time.perf_counter() - t0)
    cs.sessions += 1
    cs.late_events += late_here
    if late_here:
        cs.late_sessions += 1
    rt.sessions_done += 1


async def progress_printer(rt: Runtime, total_calls, total_sessions, interval):
    t0 = time.perf_counter()
    last = -1
    while True:
        await asyncio.sleep(interval)
        flag = "   <-- 无进展" if rt.done == last else ""
        print(f"[{time.perf_counter() - t0:6.1f}s] 调用 {rt.done}/{total_calls}，"
              f"会话 {rt.sessions_done}/{total_sessions}，在途 {rt.inflight}{flag}",
              flush=True)
        last = rt.done


# ---------------------------------------------------------------------------
def bypass_proxy_for(base_url: str) -> Optional[str]:
    """把压测目标加进 no_proxy。

    容器里常设着公司代理（http_proxy/https_proxy），而 requests 会读它们，
    于是 http://127.0.0.1:port/health 会被送去代理服务器 —— 代理那边的
    127.0.0.1 是它自己，连不上，表现为“服务没就绪”。
    压测请求本来也不该经过代理，那会把代理的排队时间算进时延。
    （aiohttp 侧不受影响：ClientSession 没有 trust_env=True，默认不读环境代理。）
    """
    host = urllib.parse.urlsplit(base_url).hostname
    if not host:
        return None
    added = []
    for var in ("no_proxy", "NO_PROXY"):
        cur = os.environ.get(var, "")
        parts = [x.strip() for x in cur.split(",") if x.strip()]
        for h in (host, "127.0.0.1", "localhost"):
            if h not in parts:
                parts.append(h)
                if h not in added:
                    added.append(h)
        os.environ[var] = ",".join(parts)
    return ",".join(added) or "（本来就已豁免）"


async def main_async(args):
    rng = random.Random(args.seed)
    url = args.base_url.rstrip("/") + "/generate"
    headers = get_auth_headers()

    proxied = [v for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
               if os.environ.get(v)]
    np = bypass_proxy_for(args.base_url)
    if proxied:
        print(f"检测到代理环境变量 {', '.join(proxied)}；已把 {np} 加进 no_proxy，"
              f"压测流量不走代理")

    print(f"客户端设施来源：{_REUSE}")
    if not wait_for_endpoint(args.base_url.rstrip("/") + "/health", 60):
        sys.exit(f"服务 {args.base_url} 没有就绪，先把服务起起来")

    server_info = {}
    try:
        import requests
        r = requests.get(args.base_url.rstrip("/") + "/get_server_info",
                         headers=headers, timeout=10)
        if r.status_code == 200:
            info = r.json()
            for k in ("model_path", "attention_backend", "page_size",
                      "disable_radix_cache", "disable_overlap_schedule",
                      "chunked_prefill_size", "max_running_requests",
                      "mem_fraction_static", "context_length", "device"):
                if k in info:
                    server_info[k] = info[k]
                elif isinstance(info.get("server_args"), dict) and k in info["server_args"]:
                    server_info[k] = info["server_args"][k]
    except Exception as e:
        server_info = {"读取失败": f"{type(e).__name__}: {e}"}

    # ---- 会话构成 ----
    lens = {"simple": parse_lens(args.len_simple),
            "medium": parse_lens(args.len_medium),
            "complex": parse_lens(args.len_complex)}
    weights = [float(x) for x in args.mix.split(",")]
    counts = dict(zip(CLASSES, split_mix(args.concurrency, weights)))
    uniform = len(set(lens.values())) == 1

    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        mk = lambda i, o: build_token_ids(tok, i, o, rng)
    else:
        print("⚠  自检模式：未加载 tokenizer，token 不具备真实对话结构。"
              "只可用于验证客户端逻辑，不可用于性能结论。", flush=True)
        mk = lambda i, o: build_token_ids_selftest(i, o, rng)

    specs = {}
    for c in CLASSES:
        q, t = mk(*lens[c])
        specs[c] = (q, t)
        n_ch = -(-len(t) // args.chunk_size) if t else 0
        print(f"{CN[c]}：{counts[c]} 路，查询 {len(q)} token，回复 {len(t)} token，"
              f"每路 {1 + n_ch} 次调用")

    total_sessions = sum(counts.values())
    total_calls = sum(
        counts[c] * (1 + (-(-len(specs[c][1]) // args.chunk_size) if specs[c][1] else 0))
        for c in CLASSES)

    sem = asyncio.Semaphore(args.max_inflight) if args.max_inflight > 0 else None
    interval = (args.chunk_size / args.rate) if args.rate > 0 else 0.0

    async with _create_bench_client_session() as session:
        # ---- 预热 ----
        if args.warmup:
            warm_rt, warm_cs = Runtime(), ClassStats()
            try:
                await asyncio.gather(*[
                    run_session(session, url, headers, f"warm{i}", "simple",
                                *specs["simple"], args.chunk_size, 0,
                                warm_cs, warm_rt, sem, args.timeout, 0.0,
                                args.slo_ms, args.slow_ms)
                    for i in range(args.warmup)])
            except Exception as e:
                sys.exit(f"预热就失败了，正式测量没有意义，先查服务端：\n  {e}")
            print(f"=== 预热完成（{args.warmup} 路）===", flush=True)

        # ---- 正式 ----
        rt = Runtime()
        stats = {c: ClassStats() for c in CLASSES}
        tasks, sid = [], 0
        for c in CLASSES:
            for _ in range(counts[c]):
                # 启动铺开：真实业务里用户是陆续开口的，不是同一毫秒一起开口
                j = rng.uniform(0, args.ramp) if args.ramp > 0 else 0.0
                if interval and not args.no_jitter:
                    j += rng.uniform(0, interval)
                tasks.append(run_session(
                    session, url, headers, sid, c, *specs[c],
                    args.chunk_size, args.rate, stats[c], rt,
                    sem, args.timeout, j, args.slo_ms, args.slow_ms))
                sid += 1

        prog = None
        if args.progress > 0:
            prog = asyncio.create_task(
                progress_printer(rt, total_calls, total_sessions, args.progress))

        global T_START
        T_START = t_start = time.perf_counter()
        running = [asyncio.create_task(t) for t in tasks]
        try:
            done, pending = await asyncio.wait(
                running, return_when=asyncio.FIRST_EXCEPTION)
            err = next((d.exception() for d in done if d.exception()), None)
            if err is not None:
                # 干净收尾：取消其余会话，保留已有统计，把出事位置原样报出来
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                rt.failures.append(str(err))
            elif pending:
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            if prog:
                prog.cancel()
        wall = time.perf_counter() - t_start

    # ---- 汇总 ----
    all_first = [x for c in CLASSES for x in stats[c].first_latency]
    all_chunk = [x for c in CLASSES for x in stats[c].chunk_latency]
    all_sess = [x for c in CLASSES for x in stats[c].session_time]
    n_calls = sum(stats[c].calls for c in CLASSES)
    n_tokens = sum(stats[c].tokens for c in CLASSES)
    n_chunks_total = len(all_chunk)
    over = sum(1 for x in all_first if x * 1000 > args.slo_ms)

    per_class = {}
    for c in CLASSES:
        s = stats[c]
        if not (s.sessions or s.calls):
            continue
        o = sum(1 for x in s.first_latency if x * 1000 > args.slo_ms)
        per_class[CN[c]] = {
            "会话数": s.sessions,
            "HTTP调用数": s.calls,
            "输入token,回复token": list(lens[c]),
            "首次分类时延_ms": lat_block(s.first_latency),
            f"首次分类超{args.slo_ms:.0f}ms": {
                "次数": o,
                "占比": f"{o / max(len(s.first_latency), 1) * 100:.1f}%"},
            "分块分类时延_ms": lat_block(s.chunk_latency),
            "单路端到端_s": {"P50": round(pct(s.session_time, 50), 2),
                             "max": round(max(s.session_time), 2)} if s.session_time else {},
            "超期会话数": s.late_sessions,
            "超期次数": s.late_events,
            "被拒绝_503": s.refused,
        }

    result = {
        "实验参数": {
            "服务地址": args.base_url,
            "并发会话数": args.concurrency,
            "会话构成": {CN[c]: counts[c] for c in CLASSES},
            "各类长度_输入_回复": {CN[c]: list(lens[c]) for c in CLASSES},
            "长度画像": ("uniform-placeholder（三类长度相同，"
                          "简单/中等/复杂的真实长度需求方尚未给出，"
                          "本轮只验证 50/40/10 调度，不构成完整负载模型）"
                          if uniform else "per-class"),
            "分块大小": args.chunk_size,
            "限速_token每秒每路": args.rate,
            "启动抖动": bool(interval) and not args.no_jitter,
            "启动铺开_s": args.ramp or "无（全部同时开始，首次分类时延含人为突发）",
            "在途上限": args.max_inflight or "不限",
            "首次分类目标_ms": args.slo_ms,
            "随机种子": args.seed,
            "tokenizer": args.tokenizer or "未加载（自检模式，结果不可用于性能结论）",
        },
        "服务端参数": server_info,
        "规模": {
            "业务问答数_即会话数": sum(stats[c].sessions for c in CLASSES),
            "被准入控制拒绝_503": sum(stats[c].refused for c in CLASSES),
            "HTTP_Guard调用数": n_calls,
            "回复分块数": n_chunks_total,
            "被检查token数": n_tokens,
        },
        "总耗时_s": round(wall, 3),
        "吞吐": {
            "调用每秒": round(n_calls / wall, 1) if wall else 0,
            "被检查token每秒": round(n_tokens / wall, 1) if wall else 0,
        },
        "汇总_按实际会话构成加权": {
            "首次分类时延_ms": lat_block(all_first),
            f"首次分类超{args.slo_ms:.0f}ms": {
                "次数": over,
                "占比": f"{over / max(len(all_first), 1) * 100:.1f}%"},
            "分块分类时延_ms": lat_block(all_chunk),
            "单路端到端_s": {"P50": round(pct(all_sess, 50), 2),
                             "max": round(max(all_sess), 2)} if all_sess else {},
            "超期会话数": sum(stats[c].late_sessions for c in CLASSES),
            "超期次数": sum(stats[c].late_events for c in CLASSES),
            "超期占比_按分块": (f"{sum(stats[c].late_events for c in CLASSES) / n_chunks_total * 100:.1f}%"
                                if n_chunks_total else "n/a"),
            "单次超预算次数": sum(stats[c].over_interval for c in CLASSES),
            "单次超预算占比": (f"{sum(stats[c].over_interval for c in CLASSES) / n_chunks_total * 100:.1f}%"
                               if n_chunks_total else "n/a"),
            "分块超100ms次数": sum(1 for x in all_chunk if x * 1000 >= 100),
            "分块超100ms占比": (f"{sum(1 for x in all_chunk if x * 1000 >= 100) / max(len(all_chunk), 1) * 100:.1f}%"),
        },
        "最慢的调用": sorted(
            [x for c in CLASSES for x in stats[c].slow],
            key=lambda x: -x["时延_ms"])[:25],
        "分类统计": per_class,
        "口径说明": {
            "首次分类时延": "客户端发出该会话第一个待检测输入 → 收到第一份分类结果的耗时",
            "超期": "某会话第 n 块的响应完成时刻 晚于 第 n+1 块的计划到达时刻，计一次。"
                      "这是累积漂移：一次尖刺会让后续每一块都算超期，直到追平为止",
            "单次超预算": "某一次分块调用自身的时延 > chunk_size/rate，即这一次确实没跟上。"
                           "与上面的累积口径互补，两个数一起看",
            "最慢的调用": "时延超过 --slow-ms 的调用现场。位置集中在同一分块序号说明是"
                           "系统性的（如页边界），散在时间轴上说明是环境性的",
            "限速含义": "rate 是主模型的输出速率（负载参数），不是 Guard 自身的生成速度",
            "指标性质": "Guard 返回分类结果而非生成正文，以上不是 TTFT / ITL / output token/s",
            "被拒绝_503": "服务端准入控制主动拒绝的流。这是保护机制在工作，不是故障；"
                           "被拒的流提前结束，不影响其余会话，也不计入失败",
        },
    }
    if rt.failures:
        result["失败"] = rt.failures
    if rt.refused:
        result["被拒绝样例"] = rt.refused[:3]

    print("\n=== 结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.out}")
    n_ref = sum(stats[c].refused for c in CLASSES)
    if n_ref:
        print(f"\n· {n_ref} 条流被服务端准入控制拒绝（HTTP 503）。"
              f"这是保护机制在工作，不是故障；其余会话正常完成。", flush=True)
    if rt.failures:
        print("\n!! 本轮有失败，上面的数字只覆盖已完成的部分：", flush=True)
        for m in rt.failures:
            print("   " + m, flush=True)
        sys.exit(1)


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
