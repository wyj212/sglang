"""Point 9 - an unsatisfiable label request aborts one stream, not the process.

The prefill invariant used to be an ``assert``, which under ``-O`` and under a
shared scheduler means a bad client kills every other stream.  It is now a
``prepare_abort`` with a 500 plus ``resumable = False``, so the scheduler stays
alive and the client learns what happened.
"""

import types

import pytest

from conftest import make_prefill_req
from sglang.srt.disaggregation.utils import is_aborted


@pytest.fixture
def caches(monkeypatch):
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


def _status(req):
    if req.finished_reason is None:
        return None
    return req.finished_reason.to_json().get("status_code")


def test_normal_request_is_untouched(processor, caches):
    req = make_prefill_req("ok", origin_len=10, labels=2, computed=2)
    processor.stream_queue["ok"] = req

    _run(processor, [req])

    assert not is_aborted(req)
    assert caches["parked"] == ["ok"]


def test_impossible_label_request_is_aborted_not_asserted(processor, caches):
    """Asking for labels of more tokens than this pass computed leaves those
    labels unobtainable, so the one stream ends with a 500."""
    req = make_prefill_req("bad", origin_len=4, labels=9, computed=2)
    processor.stream_queue["bad"] = req

    _run(processor, [req])  # must not raise

    assert is_aborted(req)
    assert _status(req) == 500


def test_aborted_stream_is_no_longer_resumable(processor, caches):
    req = make_prefill_req("bad", origin_len=4, labels=9, computed=2, resumable=True)
    processor.stream_queue["bad"] = req

    _run(processor, [req])

    assert req.resumable is False


def test_label_request_longer_than_the_computed_range_does_not_raise(processor, caches):
    req = make_prefill_req("bad", origin_len=20, labels=5, computed=2)
    processor.stream_queue["bad"] = req

    _run(processor, [req])  # must not raise


def test_one_bad_request_does_not_stop_the_others(processor, caches):
    good = make_prefill_req("good", origin_len=10, labels=2, computed=2, row=0)
    bad = make_prefill_req("bad", origin_len=4, labels=9, computed=2, row=1)
    processor.stream_queue.update({"good": good, "bad": bad})

    _run(processor, [good, bad])  # must not raise

    assert is_aborted(bad)
    assert not is_aborted(good)
    assert caches["parked"] == ["good"]


def test_a_zeroed_label_count_does_not_shift_the_next_request(processor, caches):
    """output_streamer zeroes an out-of-range return_logprob_len, but the
    request still consumes its own tokens from the flat logits list."""
    from conftest import make_guard_result
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )

    from conftest import make_req

    bad = make_req("bad", [1] * 6, extend_range=types.SimpleNamespace(length=2))
    bad.return_logprob_len = 99
    bad.finished_reason = types.SimpleNamespace(to_json=lambda: {"type": "stop"})
    good = make_req("good", [2] * 4, extend_range=types.SimpleNamespace(length=3))
    good.return_logprob_len = 3
    good.finished_reason = types.SimpleNamespace(to_json=lambda: {"type": "stop"})
    processor.stream_queue.update({"bad": bad, "good": good})

    SchedulerOutputStreamer._stream_output_guard(
        processor, [bad, good], make_guard_result(5)
    )

    out = processor.send_to_detokenizer.last
    # bad asked for 99 but only computed 2, so it is zeroed - and its two
    # tokens are still taken off the front before good reads its three.
    assert len(out.risk_level_logits[0]) == 0
    assert len(out.risk_level_logits[1]) == 3
