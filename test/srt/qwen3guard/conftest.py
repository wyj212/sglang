"""Fixtures and fakes for the Qwen3Guard-Stream adaptation tests.

Nothing in the unit tests needs an NPU: the scheduler-side logic is exercised by
calling the *real* methods unbound on a small stub ``self``, so no engine, no
device and no tensors are required.  The handful of tests that do need the real
model are marked ``needs_npu`` and skip themselves when no card is free.
"""

from __future__ import annotations

import types

import pytest


@pytest.fixture(autouse=True)
def guard_env(monkeypatch):
    """Guard env vars must never leak from one test into the next."""
    for name in ("SGLANG_GUARD_MAX_STREAMS", "SGLANG_GUARD_ADMIT_MAX_KV_USAGE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def sampling_params(**overrides):
    from sglang.srt.sampling.sampling_params import SamplingParams

    overrides.setdefault("max_new_tokens", 1)
    return SamplingParams(**overrides)


def make_req(rid="r0", input_ids=(1, 2, 3), **attrs):
    """A real ``Req`` so abort/park helpers behave exactly as in production."""
    from sglang.srt.managers.schedule_batch import Req

    req = Req(
        rid=rid,
        origin_input_text="text",
        origin_input_ids=list(input_ids),
        sampling_params=sampling_params(),
    )
    for key, value in attrs.items():
        setattr(req, key, value)
    return req


def make_trecv(rid="r0", input_ids=(1, 2, 3), resumable=True):
    """Stand-in for ``TokenizedGenerateReqInput``.

    ``resume()`` only reads ``rid`` / ``input_ids`` / ``resumable``; a real
    instance would drag in msgspec field bookkeeping for no benefit.
    """
    return types.SimpleNamespace(
        rid=rid, input_ids=list(input_ids), resumable=resumable
    )


class FakeDetokenizer:
    """Captures what the scheduler hands to the detokenizer."""

    def __init__(self):
        self.outputs = []

    def send_output(self, obj):
        self.outputs.append(obj)

    @property
    def last(self):
        return self.outputs[-1]


class SchedulerStub:
    """Just enough of ``Scheduler`` for the guard admission / resume logic."""

    def __init__(
        self,
        max_running_requests=4096,
        max_total_num_tokens=100_000,
        available=100_000,
        evictable=0,
        tokenizer=None,
        detokenizer=None,
    ):
        self.stream_queue = {}
        self.max_running_requests = max_running_requests
        self.max_total_num_tokens = max_total_num_tokens
        self.tokenizer = tokenizer
        self.req_to_token_pool = types.SimpleNamespace(freed=[], free=self._free)
        self.tree_cache = types.SimpleNamespace(
            evictable_size=lambda: self._evictable,
            free_kv_row=self._free_kv_row,
            freed_rows=[],
        )
        self.token_to_kv_pool_allocator = types.SimpleNamespace(
            available_size=lambda: self._available,
            free_group_begin=lambda: None,
            free_group_end=lambda: None,
        )
        self.send_to_detokenizer = detokenizer or FakeDetokenizer()
        self._available = available
        self._evictable = evictable

    # -- helpers the tests poke at -------------------------------------
    def set_available(self, value):
        self._available = value

    def set_evictable(self, value):
        self._evictable = value

    def _free(self, req):
        self.req_to_token_pool.freed.append(req.rid)

    def _free_kv_row(self, kv, ranges):
        self.tree_cache.freed_rows.append((kv.req_pool_idx, list(ranges)))


def make_guard_result(n_tokens, num_risk=3, num_category=8, num_query_risk=3, num_query_category=9):
    """A ``StreamGuardResult`` shaped like the one ``qwen3_guard`` produces."""
    import torch

    from sglang.srt.managers.utils import StreamGuardResult

    return StreamGuardResult(
        risk_level_logits=torch.zeros(n_tokens, num_risk),
        category_logits=torch.zeros(n_tokens, num_category),
        query_risk_level_logits=torch.zeros(n_tokens, num_query_risk),
        query_category_logits=torch.zeros(n_tokens, num_query_category),
        hidden_states=None,
        bid=None,
        can_run_cuda_graph=False,
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_npu: test needs a free NPU (skipped when none available)"
    )
    config.addinivalue_line(
        "markers", "slow_engine: test starts a real sglang Engine"
    )


@pytest.fixture
def stub():
    """A fresh scheduler stub with an empty stream queue and a roomy pool."""
    return SchedulerStub()


class ProcessorStub(SchedulerStub):
    """A batch-result processor whose guard-only path is reachable.

    For a guard ``is_generation`` is False and ``is_stream_guard`` is True, so
    process_batch_result_prefill skips the whole generation preamble and runs
    the guard branch alone - no tensors, no device.
    """

    is_generation = False
    is_stream_guard = True

    def __init__(self):
        super().__init__()
        self.stream_calls = []
        self.output_streamer = types.SimpleNamespace(
            stream_output=lambda reqs, return_logprob, skip, result=None: (
                self.stream_calls.append(
                    {
                        "reqs": list(reqs),
                        "return_logprob": return_logprob,
                        "skip": skip,
                        "result": result,
                    }
                )
            )
        )


@pytest.fixture
def processor():
    stub = ProcessorStub()
    stub.stream_queue = {}
    return stub


def make_prefill_req(
    rid="s",
    origin_len=10,
    resumable=True,
    allocated=384,
    protected=384,
    labels=2,
    computed=2,
    row=0,
):
    """A Req in the state the guard prefill branch expects to find it.

    SamplingParams reach the scheduler already normalized against the
    tokenizer; these unit tests have no tokenizer, so normalize the stop-string
    fields by hand.  ``labels`` is how many trailing tokens want logits and
    ``computed`` is how many this pass actually computed.
    """
    req = make_req(rid, [1] * origin_len, resumable=resumable)

    sp = req.sampling_params
    if getattr(sp, "stop_strs", None) is None:
        sp.stop_strs = []
    if getattr(sp, "stop_regex_strs", None) is None:
        sp.stop_regex_strs = []
    sp.stop_str_max_len = 0
    sp.stop_regex_max_len = 0
    sp.is_normalized = True

    req.kv.req_pool_idx = row
    req.kv.kv_allocated_len = allocated
    req.kv.cache_protected_len = protected
    req.inflight_middle_chunks = 0
    req.return_logprob_len = labels
    req.extend_range = types.SimpleNamespace(length=computed)
    return req
