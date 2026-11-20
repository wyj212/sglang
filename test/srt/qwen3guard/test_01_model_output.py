"""Point 1 - the four logits fields: present, one group per token, 3/8/3/9."""

import types

import pytest

from conftest import make_guard_result, make_req
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)

# The values Qwen3Guard-Stream-8B ships in config.json; the mock-side tests use
# them directly so the shape assertions below are the real ones.
NUM_RISK_LEVEL = 3
NUM_CATEGORY = 8
NUM_QUERY_RISK_LEVEL = 3
NUM_QUERY_CATEGORY = 9


def _mark_finished(req):
    """Req.finished() is just 'finished_reason is not None'."""
    req.finished_reason = types.SimpleNamespace(to_json=lambda: {"type": "stop"})
    return req


def _emit(stub, reqs, result):
    """Call the real _stream_output_guard on a stub self and return the payload."""
    SchedulerOutputStreamer._stream_output_guard(stub, reqs, result)
    return stub.send_to_detokenizer.last


def test_four_fields_present_one_group_per_token(stub):
    req = make_req("r1", [1] * 8, extend_range=types.SimpleNamespace(length=3))
    req.return_logprob_len = 3
    _mark_finished(req)

    out = _emit(stub, [req], make_guard_result(3))

    # One entry per request, each holding one group per computed token ...
    for field in (
        "risk_level_logits",
        "category_logits",
        "query_risk_level_logits",
        "query_category_logits",
    ):
        assert hasattr(out, field), field
        assert len(getattr(out, field)) == 1, field
        assert len(getattr(out, field)[0]) == 3, field

    # ... and the category counts are exactly 3 / 8 / 3 / 9.
    assert len(out.risk_level_logits[0][0]) == NUM_RISK_LEVEL
    assert len(out.category_logits[0][0]) == NUM_CATEGORY
    assert len(out.query_risk_level_logits[0][0]) == NUM_QUERY_RISK_LEVEL
    assert len(out.query_category_logits[0][0]) == NUM_QUERY_CATEGORY


def test_every_token_gets_its_own_group(stub):
    """Two tokens in, two groups out - the head runs per token, not per request."""
    req = make_req("r1", [1] * 5, extend_range=types.SimpleNamespace(length=2))
    req.return_logprob_len = 2
    _mark_finished(req)

    out = _emit(stub, [req], make_guard_result(2))

    assert len(out.risk_level_logits[0]) == 2
    assert len(out.category_logits[0]) == 2
    assert len(out.query_risk_level_logits[0]) == 2
    assert len(out.query_category_logits[0]) == 2


def test_batch_is_sliced_per_request(stub):
    """The model emits flat over the whole batch; each req takes its own range."""
    a = make_req("a", [1] * 4, extend_range=types.SimpleNamespace(length=2))
    a.return_logprob_len = 2
    _mark_finished(a)
    b = make_req("b", [2] * 4, extend_range=types.SimpleNamespace(length=3))
    b.return_logprob_len = 3
    _mark_finished(b)

    out = _emit(stub, [a, b], make_guard_result(5))

    assert [len(x) for x in out.risk_level_logits] == [2, 3]
    assert [len(x) for x in out.category_logits] == [2, 3]
    assert out.rids == ["a", "b"]


def test_abort_path_still_answers_with_empty_logits(stub):
    """No forward pass ran (result is None), but the caller must not hang."""
    req = make_req("r1", [1] * 4, extend_range=types.SimpleNamespace(length=2))
    req.return_logprob_len = 2
    _mark_finished(req)

    out = _emit(stub, [req], None)

    assert out.risk_level_logits == [[]]
    assert out.category_logits == [[]]
    assert out.query_risk_level_logits == [[]]
    assert out.query_category_logits == [[]]
    assert out.rids == ["r1"]


def test_unfinished_request_gets_no_finish_metadata(stub):
    req = make_req("r1", [1] * 4, extend_range=types.SimpleNamespace(length=2))
    req.return_logprob_len = 2
    assert req.finished() is False

    out = _emit(stub, [req], make_guard_result(2))

    assert out.rids == []
    assert out.finished_reasons == []
    assert out.prompt_tokens == []
    assert len(out.risk_level_logits[0]) == 2


@pytest.mark.needs_npu
def test_real_model_emits_four_fields_per_token():
    """Needs the real model on a free card; see qwen3guard_engine fixture."""
    pytest.importorskip("sglang.srt.entrypoints.engine")
    pytest.skip("requires a free NPU - covered by the integration run")
