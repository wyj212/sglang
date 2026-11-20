"""Point 3 - a second in-flight call for the same rid is rejected, not merged."""

import types

import pytest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class _BatchReq:
    """Minimal stand-in for a batched GenerateReqInput."""

    is_single = False

    def __init__(self, rids):
        self.rid = list(rids)
        self.received_time = 0.0
        self.external_trace_header = None
        self.bootstrap_room = None

    def __len__(self):
        return len(self.rid)

    def __getitem__(self, index):
        return self


def _tm():
    return types.SimpleNamespace(
        rid_to_state={},
        disaggregation_mode=list(DisaggregationMode)[0],
        enable_trace=False,
    )


def test_second_in_flight_call_with_the_same_rid_raises():
    tm = _tm()
    TokenizerManager._init_req_state(tm, _BatchReq(["dup"]))
    assert "dup" in tm.rid_to_state

    with pytest.raises(ValueError, match="Duplicate request ID detected"):
        TokenizerManager._init_req_state(tm, _BatchReq(["dup"]))


def test_duplicate_rid_inside_one_batch_raises():
    tm = _tm()
    with pytest.raises(ValueError, match="Duplicate request ID detected"):
        TokenizerManager._init_req_state(tm, _BatchReq(["a", "a"]))


def test_distinct_rids_are_all_registered():
    tm = _tm()
    TokenizerManager._init_req_state(tm, _BatchReq(["a", "b", "c"]))
    assert set(tm.rid_to_state) == {"a", "b", "c"}


def test_a_rid_is_reusable_once_released():
    """Sequential calls reuse the rid; only overlap is an error."""
    tm = _tm()
    TokenizerManager._init_req_state(tm, _BatchReq(["dup"]))
    tm.rid_to_state.pop("dup")
    TokenizerManager._init_req_state(tm, _BatchReq(["dup"]))
    assert "dup" in tm.rid_to_state
