"""Function C: the clinician view always has the same sections and never leaks internals."""
import json

import config
from layer.inspectability import MAX_ITEMS, SECTIONS, build_rationale
from layer.pipeline import run_layer
from llm.client import get_client
from storage import load_cases

CASES = {c["id"]: c for c in load_cases()}


def _output(cid="S01"):
    return run_layer(CASES[cid], get_client("mock")).model_dump()


def test_fixed_sections_in_fixed_order():
    r = build_rationale(_output())
    assert [s.key for s in r.sections] == [k for k, _, _ in SECTIONS]


def test_empty_section_gets_placeholder():
    out = _output()
    out["structured"]["what_would_change"] = []
    sec = build_rationale(out).section("what_would_change")
    assert sec.is_empty and sec.placeholder


def test_items_are_capped_and_critical_first():
    out = _output("M01")
    out["structured"]["main_factors"] = [f"factor {i}" for i in range(9)]
    r = build_rationale(out)
    assert len(r.section("main_factors").items) == MAX_ITEMS
    assert r.section("main_factors").truncated == 9 - MAX_ITEMS
    assert r.section("missing_information").items[0].critical


def test_repeated_answers_never_reach_the_clinician_view():
    out = _output("A01")
    view = json.dumps(build_rationale(out).model_dump())
    assert "samples" not in view and "critical_c" not in view
    for s in out["samples"]:
        if s["primary_diagnosis"] != out["structured"]["primary_diagnosis"]:
            assert s["recommendation"] not in view


def test_reliability_level_unchanged_by_function_c():
    out = _output("E01")
    assert build_rationale(out).reliability_level == out["reliability"]["level"] == "LOW"


def test_manual_adjudication_survives_recompute(tmp_path, monkeypatch):
    import csv

    from scripts import precompute_outputs as pc
    monkeypatch.setattr(config, "ADJUDICATION_PATH", tmp_path / "adj.csv")
    pc.write_adjudication()
    rows = list(csv.DictReader(open(config.ADJUDICATION_PATH, encoding="utf-8")))
    rows[0]["output_correct"], rows[0]["adjudicated_by"] = "false", "researcher"
    with open(config.ADJUDICATION_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pc.ADJ_FIELDS)
        w.writeheader()
        w.writerows(rows)
    pc.write_adjudication()
    again = list(csv.DictReader(open(config.ADJUDICATION_PATH, encoding="utf-8")))
    assert again[0]["output_correct"] == "false" and again[0]["adjudicated_by"] == "researcher"
