"""Point 5 - a paused stream's tree lock is handed over, never silently dropped.

Fix 12 (1392ba85e2).  A guard stream that pauses keeps holding the lock its
prefix needs, otherwise the prefix becomes evictable while the client is
thinking between calls.  On the next resume the scheduler must therefore park
that receipt instead of releasing it, and drop it only once a fresh lock is in
place.
"""

import types

from conftest import make_req, make_trecv
from sglang.srt.managers.scheduler import Scheduler


def _call(stub, rid, input_ids, resumable=True):
    return Scheduler.resume(
        stub,
        make_trecv(rid, input_ids, resumable=resumable),
        make_req(rid, input_ids, resumable=resumable),
    )


def _node(name):
    return types.SimpleNamespace(name=name)


def _hold(req, node, receipt):
    """Put the stream in the state resume() requires to park a receipt:
    it must still hold its row (holds_kv is req_pool_idx is not None) and have
    a tree node to release."""
    req.kv.req_pool_idx = 0
    req.last_node = node
    req.lock_receipt = receipt


def test_paused_stream_lock_is_stashed_for_handover(stub):
    node = _node("paused")
    req = _call(stub, "s1", [1, 2, 3])
    _hold(req, node, "receipt-1")

    req = _call(stub, "s1", [4])

    # The stream keeps its own lock, but the receipt is parked so the next
    # resume can drop it only after taking a new one.
    assert req.guard_parked_lock is not None
    assert req.guard_parked_lock[0] is node
    assert req.guard_parked_lock[1] == "receipt-1"


def test_handover_is_replaced_by_the_newer_parked_lock(stub):
    first = _node("first")
    second = _node("second")

    req = _call(stub, "s1", [1, 2])
    _hold(req, first, "receipt-1")

    req = _call(stub, "s1", [3])
    assert req.guard_parked_lock[0] is first

    _hold(req, second, "receipt-2")

    req = _call(stub, "s1", [4])
    # The older receipt was consumed when the new lock was taken; only the
    # newest one may remain parked.
    assert req.guard_parked_lock[0] is second
    assert req.guard_parked_lock[1] == "receipt-2"


def test_no_handover_when_the_stream_holds_no_kv(stub):
    """holds_kv is req_pool_idx is not None - a parked stream has no row."""
    req = _call(stub, "s1", [1, 2, 3])
    req.last_node = _node("paused")
    req.lock_receipt = "receipt-1"
    req.kv.req_pool_idx = None

    req = _call(stub, "s1", [4])

    assert req.guard_parked_lock is None


def test_no_handover_without_a_tree_node(stub):
    req = _call(stub, "s1", [1, 2, 3])
    req.last_node = None
    req.lock_receipt = None

    req = _call(stub, "s1", [4])

    assert req.guard_parked_lock is None


def test_running_stream_never_records_a_handover(stub):
    """A stream that never paused still holds a row; nothing to park."""
    req = _call(stub, "fresh", [1, 2, 3])

    assert req.guard_parked_lock is None
