"""The intermediary layer, end to end.

    case -> Prompt A (raw answer shown to the clinician)
         -> Function A: structure the raw answer
         -> Function B: repeated Prompt B, Prompt C, contradiction,
                        completeness, missing info -> LOW / MODERATE / HIGH
         -> Function D: LLM-as-a-Judge (supplementary)
    Function C (clinician rationale) is the rendering of the structured
    fields in the interface; Function E (feedback) is captured by the app.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

import config
from layer.completeness import completeness_signal
from layer.consistency import consistency_signal, prompt_agreement_signal
from layer.contradiction import contradiction_signal
from layer.equivalence import check_equivalence
from layer.generation import generate_baseline, generate_critical, generate_samples
from layer.judge import run_judge
from layer.missing_info import (input_inconsistency_signal, missing_info_signal,
                                verbal_confidence_signal)
from layer.reliability import aggregate
from layer.schemas import LayerOutput
from llm.prompts import prompts_version


def run_layer(case: dict, client, *, raw_response: Optional[str] = None,
              use_injected: bool = True, n_samples: Optional[int] = None,
              progress: Callable[[str], None] = lambda msg: None) -> LayerOutput:
    calls_before = client.n_calls
    usage_before = client.usage_snapshot()
    n = n_samples or config.N_SAMPLES

    # 1. The answer the clinician sees (identical in both conditions).
    injected = False
    if raw_response is None and use_injected and case.get("injected_response"):
        raw_response = case["injected_response"]["text"]
        injected = True
    if raw_response is None:
        progress("Generating the baseline answer (Prompt A)")
        raw_response = generate_baseline(case, client)

    # 2. Function A - structuring.
    progress("Structuring the answer")
    structured = structure_response_safe(case, raw_response, client)

    # 3. Function B - reliability signals.
    progress(f"Generating {n} repeated answers (Prompt B)")
    samples = generate_samples(case, client, n)
    progress("Generating the critical-evaluation answer (Prompt C)")
    critical = generate_critical(case, client)

    progress("Comparing diagnoses by meaning")
    candidates = [x.primary_diagnosis or "(no answer)" for x in samples] + \
                 [critical.primary_diagnosis or "(no answer)"]
    same = check_equivalence(structured.primary_diagnosis, candidates, client)
    sample_agreement, agree_c = same[:n], same[n]

    b0 = samples[0] if samples and samples[0].parse_ok else None
    signals = [
        consistency_signal(sample_agreement, samples),
        prompt_agreement_signal(sample_agreement[0] if b0 else None,
                                agree_c if critical.parse_ok else None, b0, critical),
    ]
    progress("Checking for contradictions")
    contra_sig, conflicts = contradiction_signal(case, structured, client)
    signals += [
        contra_sig,
        completeness_signal(structured, samples),
        missing_info_signal(structured),
        input_inconsistency_signal(structured, critical),
        verbal_confidence_signal(structured, critical),
    ]
    reliability = aggregate(signals)

    # 4. Function D - automated quality check (supplementary).
    judge = None
    if config.USE_LLM_JUDGE:
        progress("Running the automated quality check")
        judge = run_judge(case, raw_response, client)

    return LayerOutput(
        case_id=case.get("id", "custom"), provider=client.provider, model=client.model,
        raw_response=raw_response, injected=injected, structured=structured,
        samples=samples, critical=critical, sample_agreement=sample_agreement,
        signals=signals, contradictions=conflicts, reliability=reliability, judge=judge,
        n_llm_calls=client.n_calls - calls_before,
        usage=_usage_delta(usage_before, client.usage_snapshot()),
        provenance={
            "prompts_version": prompts_version(), "n_samples": n,
            "sample_temperature": config.SAMPLE_TEMPERATURE, "baseline_temperature": config.BASELINE_TEMPERATURE,
            "consistency_pass": config.CONSISTENCY_PASS, "consistency_warn": config.CONSISTENCY_WARN,
            "llm_contradiction_check": config.USE_LLM_CONTRADICTION_CHECK, "llm_judge": config.USE_LLM_JUDGE,
            "llm_equivalence": config.USE_LLM_EQUIVALENCE,
        },
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _usage_delta(before: dict, after: dict) -> dict:
    """Usage attributable to this case: total, plus the baseline-only share
    (what Condition A would cost on its own)."""
    def diff(key):
        return {k: after[key].get(k, 0) - before[key].get(k, 0) for k in after[key]
                if after[key].get(k, 0) - before[key].get(k, 0)}
    calls, lat = diff("calls_by_task"), diff("latency_by_task")
    tokens = {k: [v[0] - before["tokens_by_task"].get(k, [0, 0])[0], v[1] - before["tokens_by_task"].get(k, [0, 0])[1]]
              for k, v in after["tokens_by_task"].items()}
    tokens = {k: v for k, v in tokens.items() if v[0] or v[1]}
    return {
        "calls": after["calls"] - before["calls"],
        "latency_s": round(after["latency_s"] - before["latency_s"], 3),
        "input_tokens": after["input_tokens"] - before["input_tokens"],
        "output_tokens": after["output_tokens"] - before["output_tokens"],
        "calls_by_task": calls,
        "latency_by_task": {k: round(v, 3) for k, v in lat.items()},
        "tokens_by_task": tokens,
        "baseline_latency_s": round(lat.get("baseline", 0.0), 3),
    }


def structure_response_safe(case, raw, client):
    from layer.structuring import structure_response
    return structure_response(case, raw, client)
