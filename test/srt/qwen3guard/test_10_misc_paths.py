"""Small guard wiring paths that the other test files do not reach.

Each of these is a handful of changed lines that need no NPU, only the right
object in hand.
"""

import types

import pytest

from conftest import make_guard_result, make_req
from sglang.srt.mem_cache.common import release_kv_cache


# ------------------------------------------------- arg declaration surfaces

def test_chunked_prefill_size_is_resolvable_by_model_overrides():
    """Qwen3ForGuardModel classifies a whole prefill in one pass, so chunking
    must be switchable off through MODEL_OVERRIDES."""
    from sglang.srt.arg_groups.fields.schedule import Schedule

    assert "chunked_prefill_size" in Schedule.__struct_fields__


# --------------------------------------------------------- io_struct plumbing

def test_batched_request_forwards_resumable_to_its_sub_requests():
    from sglang.srt.managers.io_struct import GenerateReqInput

    obj = GenerateReqInput(rid=["a", "b"], text=["x", "y"], resumable=True)
    # __getitem__ indexes every per-request field; production normalizes them
    # to lists first, so do the same here.
    for name in (
        "image_data",
        "video_data",
        "audio_data",
        "sampling_params",
        "return_logprob",
        "logprob_start_len",
        "top_logprobs_num",
        "token_ids_logprob",
        "return_sampling_mask",
    ):
        if getattr(obj, name, None) is None:
            setattr(obj, name, [None, None])
    sub = obj[0]
    assert sub.rid == "a"
    assert sub.resumable is True


# ------------------------------------------------- output dispatch

def test_stream_output_dispatches_to_the_guard_path(stub):
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )

    stub.is_stream_guard = True
    # stream_output dispatches onto self; bind the real implementation so the
    # dispatch itself is what gets exercised.
    stub._stream_output_guard = lambda reqs, result: (
        SchedulerOutputStreamer._stream_output_guard(stub, reqs, result)
    )
    req = make_req("r1", [1] * 4, extend_range=types.SimpleNamespace(length=2))
    req.return_logprob_len = 2
    req.finished_reason = types.SimpleNamespace(to_json=lambda: {"type": "stop"})

    SchedulerOutputStreamer.stream_output(
        stub, [req], False, None, result=make_guard_result(2)
    )

    out = stub.send_to_detokenizer.last
    assert len(out.risk_level_logits[0]) == 2


# ------------------------------------------------- detokenizer passthrough

def test_guard_output_needs_no_detokenization():
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager
    from sglang.srt.managers.io_struct import BatchStreamGuardOutput

    recv = BatchStreamGuardOutput(
        rids=["a"],
        http_worker_ipcs=[],
        finished_reasons=[],
        risk_level_logits=[[[0.0, 0.0, 0.0]]],
        category_logits=[[[0.0] * 8]],
        query_risk_level_logits=[[[0.0, 0.0, 0.0]]],
        query_category_logits=[[[0.0] * 9]],
        prompt_tokens=[3],
        cached_tokens=[0],
    )

    assert DetokenizerManager.handle_batch_stream_guard_out(None, recv) is recv


# ------------------------------------------------- lock ref handover ordering

@pytest.fixture
def policy():
    from sglang.srt.managers import schedule_policy as sp

    cls = next(
        obj
        for obj in vars(sp).values()
        if isinstance(obj, type) and hasattr(obj, "_req_inc_lock_ref")
    )
    self = types.SimpleNamespace()
    self.inc = []
    self.dec = []
    self.tree_cache = types.SimpleNamespace(
        inc_lock_ref=lambda node: types.SimpleNamespace(
            to_dec_params=lambda: ("receipt", node)
        ),
        dec_lock_ref=lambda node, receipt: self.dec.append((node, receipt)),
    )
    return self, cls


def test_new_lock_is_taken_before_the_parked_one_is_dropped(policy):
    """The prefix must never be unlocked between the two calls."""
    self, cls = policy
    req = make_req("s1", [1, 2, 3], resumable=True)
    req.last_node = "parked-node"
    req.guard_parked_lock = ("parked-node", "old-receipt")

    cls._req_inc_lock_ref(self, req)

    assert req.guard_parked_lock is None
    assert self.dec == [("parked-node", "old-receipt")]
    assert req.lock_receipt == ("receipt", "parked-node")


def test_nothing_is_released_without_a_parked_lock(policy):
    self, cls = policy
    req = make_req("s1", [1, 2, 3], resumable=True)
    req.last_node = "fresh-node"

    cls._req_inc_lock_ref(self, req)

    assert self.dec == []


# ------------------------------------------------- release restores the pair

def test_release_uses_the_parked_lock_not_the_fresh_match():
    """An aborted stream still holds only its parked lock; last_node may
    already point at a fresh match, and releasing that would drop the wrong
    lock."""
    node, receipt = object(), object()
    req = make_req("s1", [1, 2, 3], resumable=True)
    req.kv.req_pool_idx = None
    req.last_node = "fresh-node"
    req.lock_receipt = "fresh-receipt"
    req.guard_parked_lock = (node, receipt)

    tree_cache = types.SimpleNamespace(supports_mamba=lambda: True)
    release_kv_cache(req, tree_cache)

    assert req.last_node is node
    assert req.lock_receipt is receipt
    assert req.guard_parked_lock is None


# --------------------------------------- prefill branch: skipped req kinds

@pytest.fixture
def cache_calls(monkeypatch):
    from sglang.srt.managers.scheduler_components import batch_result_processor as brp

    calls = {"parked": [], "released": []}
    monkeypatch.setattr(
        brp,
        "maybe_cache_unfinished_req",
        lambda req, tree, **k: calls["parked"].append(req.rid),
    )
    monkeypatch.setattr(
        brp,
        "release_kv_cache",
        lambda req, tree, is_insert=True: calls["released"].append(req.rid),
    )
    return calls


def _run(processor, reqs):
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )

    batch = types.SimpleNamespace(reqs=reqs, return_logprob=False, prefill_stats=None)
    result = types.SimpleNamespace(copy_done=None, can_run_cuda_graph=False)
    SchedulerBatchResultProcessor.process_batch_result_prefill(processor, batch, result)


def test_retracted_request_is_skipped(processor, cache_calls):
    from conftest import make_prefill_req

    req = make_prefill_req("gone")
    req.is_retracted = True
    processor.stream_queue["gone"] = req

    _run(processor, [req])

    assert cache_calls["parked"] == []
    assert cache_calls["released"] == []


def test_partially_chunked_prefill_is_not_parked_yet(processor, cache_calls):
    from conftest import make_prefill_req

    req = make_prefill_req("mid", allocated=384, protected=256)
    req.inflight_middle_chunks = 2

    _run(processor, [req])

    assert req.inflight_middle_chunks == 1
    assert cache_calls["parked"] == []
    assert processor.tree_cache.freed_rows == []
