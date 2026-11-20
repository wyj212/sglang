"""Validates the fixtures themselves before the real assertions rely on them."""

from conftest import FakeDetokenizer, SchedulerStub, make_guard_result, make_req, make_trecv


def test_req_fixture_has_kv_state():
    req = make_req("a", [1, 2, 3])
    assert req.rid == "a"
    assert req.kv.holds_kv is False
    assert req.kv.cache_protected_len == 0


def test_guard_result_shape():
    result = make_guard_result(5)
    assert result.risk_level_logits.shape == (5, 3)
    assert result.category_logits.shape == (5, 8)
    assert result.query_risk_level_logits.shape == (5, 3)
    assert result.query_category_logits.shape == (5, 9)


def test_stub_and_trecv():
    stub = SchedulerStub(available=10, evictable=5)
    assert stub.token_to_kv_pool_allocator.available_size() == 10
    assert stub.tree_cache.evictable_size() == 5
    assert make_trecv("x", [1], False).resumable is False
    assert isinstance(stub.send_to_detokenizer, FakeDetokenizer)
