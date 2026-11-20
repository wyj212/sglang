"""Point 6 - every stream gets its own radix namespace.

Two guard streams that happen to start with the same tokens must not share a
KV prefix: stream B's chunk 2 says nothing about stream A's, so reusing A's
cached prefix would feed B's logits the wrong context.  The adaptation tags
each stream's requests with its own ``extra_key``, which the radix tree folds
into the node hash.

Inputs are compared as token *ids* here, so any tokenizer stand-in will do -
only the ids matter for the namespace.
"""

from conftest import make_req, make_trecv
from sglang.srt.managers.scheduler import Scheduler


class _Tok:
    bos_token_id = 1


def _call(stub, rid, input_ids, resumable=True):
    return Scheduler.resume(
        stub,
        make_trecv(rid, input_ids, resumable=resumable),
        make_req(rid, input_ids, resumable=resumable),
    )


def _fresh():
    from conftest import SchedulerStub

    stub = SchedulerStub(tokenizer=_Tok())
    return stub


def test_streams_with_an_identical_prefix_get_different_namespaces():
    stub = _fresh()
    a = _call(stub, "a", [1, 5, 6, 7])
    b = _call(stub, "b", [1, 5, 6, 7])

    assert a.extra_key is not None
    assert b.extra_key is not None
    # Same tokens, different namespace: the tree cannot share the prefix.
    assert a.extra_key != b.extra_key


def test_namespace_is_stable_across_resumes():
    stub = _fresh()
    first = _call(stub, "a", [1, 5, 6])
    key = first.extra_key

    for chunk in ([7], [8], [9]):
        first = _call(stub, "a", chunk)
        assert first.extra_key == key, "a stream must keep one namespace"


def test_three_streams_all_differ():
    stub = _fresh()
    keys = {_call(stub, rid, [1, 5, 6, 7]).extra_key for rid in ("a", "b", "c")}
    assert len(keys) == 3


def test_an_existing_extra_key_is_kept_as_a_prefix():
    """A caller that already set extra_key keeps its isolation; the guard
    namespace is appended rather than replacing it."""
    stub = _fresh()
    req = make_req("a", [1, 5])
    req.extra_key = "caller-owned"

    out = Scheduler.resume(stub, make_trecv("a", [1, 5]), req)

    assert out.extra_key == "caller-owned|guard:a"


def test_namespacing_still_allows_intra_stream_prefix_reuse():
    """Within one stream the prefix *must* still be reusable - that is the
    whole point of resuming instead of resending."""
    stub = _fresh()
    first = _call(stub, "a", [1, 5, 6, 7])
    second = _call(stub, "a", [8])

    assert second.extra_key == first.extra_key
    assert second.origin_input_ids[:4] == [1, 5, 6, 7]
    assert second.return_logprob_len == 1
