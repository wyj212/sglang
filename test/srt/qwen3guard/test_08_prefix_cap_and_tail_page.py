"""Point 8 - a paused stream gives back its trailing partial page, and the
prefix match is capped so a resumed call never re-matches its own new tokens.
"""

import types

import pytest

from conftest import make_prefill_req, make_req


@pytest.fixture
def caches(monkeypatch):
    """The two cache helpers have real side effects; record the calls."""
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


# ------------------------------------------------- trailing page is released

def test_trailing_page_of_a_paused_stream_is_freed(processor, caches):
    # 320 tokens allocated, 256 of them already safe to keep: the 64-token
    # tail is this pass's partial page and must not stay pinned while paused.
    req = make_prefill_req("s1", allocated=320, protected=256, labels=2)
    processor.stream_queue["s1"] = req

    _run(processor, [req])

    assert req.kv.kv_allocated_len == 256
    assert processor.tree_cache.freed_rows == [(req.kv.req_pool_idx, [(256, 320)])]
    assert caches["parked"] == ["s1"]


def test_each_paused_stream_frees_only_its_own_tail(processor, caches):
    a = make_prefill_req("a", allocated=320, protected=256, row=0)
    b = make_prefill_req("b", allocated=600, protected=512, row=1)
    processor.stream_queue.update({"a": a, "b": b})

    _run(processor, [a, b])

    freed = sorted(processor.tree_cache.freed_rows)
    assert freed == [(0, [(256, 320)]), (1, [(512, 600)])]


def test_stream_shorter_than_one_page_gives_its_row_back(processor, caches):
    """Nothing is worth keeping, so the pool row itself goes back."""
    req = make_prefill_req("s1", allocated=0, protected=0, labels=2, row=7)
    processor.stream_queue["s1"] = req

    _run(processor, [req])

    # free_kv_row is skipped (nothing above the protected length) and the row
    # returns to the pool, keeping (not holds_kv) == is_kv_released true.
    assert processor.tree_cache.freed_rows == []
    assert "s1" in processor.req_to_token_pool.freed
    assert caches["parked"] == ["s1"]


def test_page_aligned_stream_frees_nothing(processor, caches):
    """Already page-aligned: the tree owns everything, nothing is a tail."""
    req = make_prefill_req("s1", allocated=512, protected=512, labels=2)
    processor.stream_queue["s1"] = req

    _run(processor, [req])

    assert processor.tree_cache.freed_rows == []
    assert req.kv.kv_allocated_len == 512
    assert "s1" not in processor.req_to_token_pool.freed


def test_finished_stream_is_released_and_leaves_the_queue(processor, caches):
    req = make_prefill_req("s1", resumable=False, allocated=1024, protected=1024)
    req.finished_reason = types.SimpleNamespace(to_json=lambda: {"type": "stop"})
    processor.stream_queue["s1"] = req

    _run(processor, [req])

    assert caches["released"] == ["s1"]
    assert "s1" not in processor.stream_queue
    assert processor.tree_cache.freed_rows == []


# ------------------------------------------------------ prefix match is capped

def _cap(input_len, return_logprob_len=None, logprob_start_len=-1):
    """Call the real Req._compute_max_prefix_len on a request shaped like
    the one the scheduler hands it."""
    req = make_req("c", [1] * input_len)
    req.return_logprob = True
    req.logprob_start_len = logprob_start_len
    req.return_logprob_len = return_logprob_len
    return req._compute_max_prefix_len(input_len)


def test_first_call_caps_the_prefix_match_at_zero():
    """input_len - return_logprob_len == 0: everything was just sent."""
    assert _cap(input_len=8, return_logprob_len=8) == 0


def test_resumed_call_matches_everything_but_the_new_tokens():
    assert _cap(input_len=12, return_logprob_len=2) == 10


def test_cap_grows_with_each_resumed_chunk():
    caps = [_cap(input_len=n, return_logprob_len=1) for n in (4, 8, 16)]
    assert caps == sorted(caps)


def test_cap_is_never_negative():
    assert _cap(input_len=2, return_logprob_len=5) == 0


def test_cap_honours_logprob_start_len():
    assert _cap(input_len=16, return_logprob_len=1, logprob_start_len=4) == 4


def test_cap_leaves_room_for_one_token():
    """The match may never equal the whole input, or there is nothing to run."""
    assert _cap(input_len=4, return_logprob_len=0, logprob_start_len=99) == 3
