"""Integration coverage for the points that need the real model.

These start a real sglang Engine on one NPU and speak the guard's resume
protocol end to end.  They skip when no card is free, which is the usual state
of a shared box - the mock suite in the other files runs everywhere.
"""

import os
import subprocess

import pytest

NPU = int(os.environ.get("QGUARD_NPU", "0"))
MODEL = os.environ.get("QGUARD_MODEL", "/path/to/models/Qwen3Guard-Stream-8B")

pytestmark = [pytest.mark.needs_npu, pytest.mark.slow_engine]


def card_is_free(index: int = NPU) -> bool:
    try:
        out = subprocess.run(
            ["npu-smi", "info", "-t", "proc-mem", "-i", str(index)],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "No process in device" in out


@pytest.fixture(scope="module")
def engine():
    if not card_is_free():
        pytest.skip("NPU %d is busy; integration tests need a free card" % NPU)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(NPU)
    from sglang.srt.entrypoints.engine import Engine

    eng = Engine(
        model_path=MODEL,
        attention_backend="ascend",
        mem_fraction_static=0.85,
        context_length=8192,
    )
    try:
        yield eng
    finally:
        eng.shutdown()


def _tell(engine, rid, tokens, resumable):
    return engine.generate(
        input_ids=list(tokens),
        sampling_params={"max_new_tokens": 1},
        rid=rid,
        resumable=resumable,
    )


def _token_usage(engine):
    info = engine.get_server_info()
    state = info["internal_states"][0]
    return state.get("token_usage", state.get("token_usage_server"))


# --------------------------------------------------------------- point 1

def test_real_model_returns_four_logit_fields_per_token(engine):
    out = _tell(engine, "it-1", [1] * 16, resumable=False)

    for field in (
        "risk_level_logits",
        "category_logits",
        "query_risk_level_logits",
        "query_category_logits",
    ):
        assert field in out, field

    assert len(out["risk_level_logits"][0][0]) == 3
    assert len(out["category_logits"][0][0]) == 8
    assert len(out["query_risk_level_logits"][0][0]) == 3
    assert len(out["query_category_logits"][0][0]) == 9


# --------------------------------------------------------------- point 2

def test_prompt_tokens_accumulate_until_resumable_false(engine):
    rid = "it-2"
    seen = []
    chunks = [[1] * 8, [5] * 4, [7] * 4]

    for index, chunk in enumerate(chunks):
        last = index == len(chunks) - 1
        out = _tell(engine, rid, chunk, resumable=not last)
        seen.append(out["prompt_tokens"][0])

    assert seen == sorted(seen)
    assert seen[-1] == sum(len(c) for c in chunks)
    assert out.get("finished_reasons") or out.get("meta_info")


# --------------------------------------------------------------- point 4

def test_kv_pool_returns_to_baseline_after_a_full_stream(engine):
    baseline = _token_usage(engine)

    for index in range(4):
        rid = "it-4-%d" % index
        _tell(engine, rid, [1] * 8, resumable=True)
        _tell(engine, rid, [5] * 4, resumable=False)

    after = _token_usage(engine)
    assert baseline is not None and after is not None
    # The pool may keep evictable radix prefixes, but nothing may stay held.
    assert after <= baseline + 0.05


# --------------------------------------------------------------- point 8

def test_paused_stream_does_not_accumulate_kv_across_chunks(engine):
    rid = "it-8"
    _tell(engine, rid, [1] * 8, resumable=True)
    first = _token_usage(engine)

    for _ in range(6):
        _tell(engine, rid, [5] * 4, resumable=True)
    later = _token_usage(engine)

    assert first is not None and later is not None
    # Each pause frees its trailing partial page, so usage must not climb by a
    # page per call.
    assert later - first < 0.05
