"""Point 2 - same rid + resumable=True accumulates; resumable=False ends it."""

import types

from conftest import make_guard_result, make_req, make_trecv
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)


def _call(stub, rid, input_ids, resumable=True):
    """One client call, shaped the way production shapes it.

    scheduler.py builds a *fresh* Req per call carrying
    ``resumable=recv_req.resumable``; resume() then copies that flag onto the
    sequence it found in stream_queue.
    """
    return Scheduler.resume(
        stub,
        make_trecv(rid, input_ids, resumable=resumable),
        make_req(rid, input_ids, resumable=resumable),
    )


def _emit(stub, reqs, result):
    SchedulerOutputStreamer._stream_output_guard(stub, reqs, result)
    return stub.send_to_detokenizer.last


def test_first_call_registers_the_stream(stub):
    out = _call(stub, "s1", [1, 2, 3, 4])

    assert stub.stream_queue["s1"] is out
    # First call wants a label for every token it sent.
    assert out.return_logprob_len == 4


def test_resumed_call_appends_tokens_and_grows_prompt_tokens(stub):
    req = _call(stub, "s1", [1, 2, 3, 4])

    req = _call(stub, "s1", [5, 6])
    assert req.origin_input_ids == [1, 2, 3, 4, 5, 6]
    # Only the new tokens need labels on a resumed call.
    assert req.return_logprob_len == 2

    req = _call(stub, "s1", [7])
    assert req.origin_input_ids == [1, 2, 3, 4, 5, 6, 7]
    assert req.return_logprob_len == 1

    # prompt_tokens reports the whole accumulated sequence, not just this chunk.
    req.finished_reason = types.SimpleNamespace(to_json=lambda: {"type": "stop"})
    req.extend_range = types.SimpleNamespace(length=1)
    out = _emit(stub, [req], make_guard_result(1))
    assert out.prompt_tokens == [7]


def test_bos_token_is_not_appended_twice(stub):
    class Tok:
        bos_token_id = 1

    stub.tokenizer = Tok()
    req = _call(stub, "s1", [1, 2, 3])
    req = _call(stub, "s1", [1, 9])
    assert req.origin_input_ids == [1, 2, 3, 9]


def test_resumable_false_marks_the_stream_finished(stub):
    # Production builds the Req with resumable=recv_req.resumable (scheduler.py),
    # so the first call already carries it.
    req = _call(stub, "s1", [1, 2, 3])
    assert req.resumable is True

    req = _call(stub, "s1", [4], resumable=False)
    assert req.resumable is False


def test_resume_resets_per_call_output_state(stub):
    req = _call(stub, "s1", [1, 2, 3])
    req.finished_reason = types.SimpleNamespace(to_json=lambda: {"type": "stop"})

    req = _call(stub, "s1", [4])
    assert req.finished_reason is None
    assert list(req.output_ids) == []


def test_two_rids_do_not_share_a_sequence(stub):
    a = _call(stub, "a", [1, 2])
    b = _call(stub, "b", [8, 9])

    a = _call(stub, "a", [3])
    assert a.origin_input_ids == [1, 2, 3]
    assert b.origin_input_ids == [8, 9]
