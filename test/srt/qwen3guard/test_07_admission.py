"""Point 7 - admission control for new guard streams.

Two independent gates guard ``Scheduler.resume()``:

* a **count cap** - ``SGLANG_GUARD_MAX_STREAMS`` if set, else
  ``max_running_requests``;
* a **KV pressure gate** - refuse when
  ``1 - (available + evictable) / max_total `` exceeds
  ``SGLANG_GUARD_ADMIT_MAX_KV_USAGE`` (default 0.90).  Evictable cache counts
  as free, otherwise a full-but-idle pool would refuse work it can still run.

Both answer 503, and a refused stream must not disturb the ones already
running.
"""

import types

import pytest

from conftest import SchedulerStub, make_req, make_trecv
from sglang.srt.disaggregation.utils import is_aborted
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler


def _call(stub, rid, input_ids=(1, 2, 3), resumable=True):
    return Scheduler.resume(
        stub,
        make_trecv(rid, list(input_ids), resumable=resumable),
        make_req(rid, list(input_ids), resumable=resumable),
    )


def _status(req):
    """The HTTP status a refused request reports, or None if it was admitted."""
    if req.finished_reason is None:
        return None
    payload = req.finished_reason.to_json()
    return payload.get("status_code")


@pytest.fixture
def warnings(monkeypatch):
    """sglang's logger does not propagate to caplog, so record directly."""
    seen = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "warning",
        lambda msg, *a, **k: seen.append(msg % a if a else msg),
    )
    return seen


# ------------------------------------------------------------ count gate

def test_count_cap_refuses_the_next_new_stream(monkeypatch):
    monkeypatch.setenv("SGLANG_GUARD_MAX_STREAMS", "2")
    stub = SchedulerStub()

    _call(stub, "a")
    _call(stub, "b")
    third = _call(stub, "c")

    assert is_aborted(third)
    assert _status(third) == 503


def test_default_cap_comes_from_max_running_requests():
    stub = SchedulerStub(max_running_requests=2)

    _call(stub, "a")
    _call(stub, "b")
    third = _call(stub, "c")

    assert is_aborted(third)


def test_refusal_is_logged_with_the_cap(monkeypatch, warnings):
    monkeypatch.setenv("SGLANG_GUARD_MAX_STREAMS", "1")
    stub = SchedulerStub()

    _call(stub, "a")
    _call(stub, "b")

    assert warnings, "a refused stream must say why"
    assert any("1" in w for w in warnings)


# --------------------------------------------------------------- KV gate

def test_kv_usage_above_the_threshold_refuses_a_new_stream():
    # available == max_total means a completely full pool -> usage 100%.
    stub = SchedulerStub(max_total_num_tokens=1000, available=50, evictable=0)

    refused = _call(stub, "a")

    assert is_aborted(refused)
    assert _status(refused) == 503


def test_kv_usage_below_the_threshold_admits():
    stub = SchedulerStub(max_total_num_tokens=1000, available=500, evictable=0)

    admitted = _call(stub, "a")

    assert not is_aborted(admitted)
    assert stub.stream_queue["a"] is admitted


def test_evictable_cache_counts_as_free_capacity():
    """80 available + 150 evictable of 1000 is 77% used, not 92%."""
    stub = SchedulerStub(max_total_num_tokens=1000, available=80, evictable=150)

    admitted = _call(stub, "a")

    assert not is_aborted(admitted)


def test_the_kv_threshold_is_configurable(monkeypatch):
    # 60% used: fine at the 0.90 default, over the line at 0.50.
    stub = SchedulerStub(max_total_num_tokens=1000, available=400, evictable=0)
    assert not is_aborted(_call(stub, "a"))

    monkeypatch.setenv("SGLANG_GUARD_ADMIT_MAX_KV_USAGE", "0.50")
    stub = SchedulerStub(max_total_num_tokens=1000, available=400, evictable=0)

    refused = _call(stub, "a")

    assert is_aborted(refused)


# ------------------------------------------------- running streams are safe

def test_running_streams_are_never_refused():
    monkeypatch_cap = 1
    stub = SchedulerStub(max_running_requests=monkeypatch_cap)

    first = _call(stub, "a")
    assert not is_aborted(first)

    # The cap is already reached, but this is a *resume*, not a new stream.
    resumed = _call(stub, "a", [4])

    assert not is_aborted(resumed)
    assert resumed.origin_input_ids == [1, 2, 3, 4]


def test_both_gates_report_the_same_503(monkeypatch):
    # Count gate reached and pool full at the same time.
    monkeypatch.setenv("SGLANG_GUARD_MAX_STREAMS", "1")
    stub = SchedulerStub(max_total_num_tokens=1000, available=0, evictable=0)

    _call(stub, "a")
    by_count = _call(stub, "b")

    assert _status(by_count) == 503

    # Same status when only the KV gate can be blamed.
    monkeypatch.delenv("SGLANG_GUARD_MAX_STREAMS")
    stub2 = SchedulerStub(max_total_num_tokens=1000, available=0, evictable=0)
    by_kv = _call(stub2, "a")

    assert _status(by_kv) == 503
