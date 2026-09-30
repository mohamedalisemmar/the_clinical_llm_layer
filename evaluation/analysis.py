"""Analysis of the study: LLM only (A) vs LLM + layer (B).

Run:  python -m evaluation.analysis
Writes tables to results/ and figures to results/figures/.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from evaluation.metrics import (AGREEMENT, RELIANCE_CELLS, perceived_discrimination,  # noqa: E402
                                reliance_matrix, reliance_metrics, sus_score, tlx_raw)
from layer.inspectability import build_rationale  # noqa: E402
from storage import QUESTIONNAIRES_CSV, RESPONSES_CSV, SESSIONS_CSV, load_cases, outputs_meta  # noqa: E402
from ui.questionnaires import LAYER_COMPONENTS  # noqa: E402

COND_NAME = {"A": "LLM only", "B": "LLM + layer"}
COLORS = {"A": "#8A949C", "B": "#2F5D8A"}
LEVEL_ORDER = ["LOW", "MODERATE", "HIGH"]
TLX_COLS = ["tlx_mental", "tlx_physical", "tlx_temporal", "tlx_performance", "tlx_effort", "tlx_frustration"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def load_adjudication() -> pd.DataFrame:
    adj = _read(config.ADJUDICATION_PATH)
    if not adj.empty:
        adj["output_correct"] = adj["output_correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    return adj


# ---------------------------------------------------------------------------
def layer_calibration(adj: pd.DataFrame):
    """Automated evaluation: does the indicator separate correct from incorrect outputs?"""
    if adj.empty:
        return None, [], None
    tab = pd.crosstab(adj["layer_reliability"], adj["output_correct"].map({True: "correct", False: "incorrect"}))
    tab = tab.reindex(index=[lvl for lvl in LEVEL_ORDER if lvl in tab.index])
    for col in ("correct", "incorrect"):
        if col not in tab:
            tab[col] = 0
    tab = tab[["correct", "incorrect"]]
    tab["share_correct"] = (tab["correct"] / tab[["correct", "incorrect"]].sum(axis=1)).round(2)

    inc, cor = adj[~adj["output_correct"]], adj[adj["output_correct"]]
    lines = []
    if len(inc):
        lines.append(f"Incorrect outputs flagged LOW: {(inc['layer_reliability'] == 'LOW').sum()} of {len(inc)} "
                     f"({(inc['layer_reliability'] == 'LOW').mean():.0%}).")
        lines.append(f"Incorrect outputs rated HIGH (missed): {(inc['layer_reliability'] == 'HIGH').sum()} of {len(inc)}.")
    if len(cor):
        lines.append(f"Correct outputs rated HIGH: {(cor['layer_reliability'] == 'HIGH').sum()} of {len(cor)}; "
                     f"rated LOW (false alarm): {(cor['layer_reliability'] == 'LOW').sum()} of {len(cor)}.")
    missed = adj[(~adj["output_correct"]) & (adj["layer_reliability"] == "HIGH")]["case_id"].tolist()
    alarms = adj[adj["output_correct"] & (adj["layer_reliability"] == "LOW")]["case_id"].tolist()
    if missed:
        lines.append("Missed (HIGH but incorrect): " + ", ".join(missed) + ". Inspect these first.")
    if alarms:
        lines.append("False alarms (LOW but correct): " + ", ".join(alarms) + ".")
    stable_wrong = _consistently_wrong(adj)
    lines.append("Consistency is not correctness: incorrect outputs whose repeated answers were nevertheless "
                 f"consistent: {', '.join(stable_wrong) if stable_wrong else 'none'}.")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(tab))
    ax.bar(x - 0.18, tab["correct"], 0.36, label="Correct output", color="#1E6B52")
    ax.bar(x + 0.18, tab["incorrect"], 0.36, label="Incorrect output", color="#A3231B")
    ax.set_xticks(x, [lvl.capitalize() for lvl in tab.index])
    ax.set_ylabel("Number of cases")
    ax.set_title("Reliability indicator vs actual correctness")
    ax.legend(frameon=False)
    fig.tight_layout()
    return tab, lines, fig


def _load_outputs() -> dict:
    import json
    out = {}
    for p in config.OUTPUTS_DIR.glob("*.json"):
        if not p.name.startswith("_"):
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _consistently_wrong(adj: pd.DataFrame) -> list[str]:
    outs = _load_outputs()
    ids = []
    for cid in adj[~adj["output_correct"]]["case_id"]:
        o = outs.get(cid, {})
        sig = next((x for x in o.get("signals", []) if x["name"] == "consistency"), None)
        if sig and sig["status"] == "pass":
            ids.append(cid)
    return ids


def signal_breakdown(adj: pd.DataFrame) -> pd.DataFrame | None:
    """For each signal: how often it failed/warned on correct vs incorrect outputs.
    Shows which individual signals carry information (component-level analysis)."""
    outs = _load_outputs()
    if adj.empty or not outs:
        return None
    rows = []
    for _, r in adj.iterrows():
        for sig in outs.get(r["case_id"], {}).get("signals", []):
            rows.append({"signal": sig["label"], "correct": r["output_correct"],
                         "flagged": sig["status"] in ("warn", "fail")})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    t = df.pivot_table(index="signal", columns="correct", values="flagged", aggfunc="mean")
    t = t.rename(columns={True: "flagged on correct outputs", False: "flagged on incorrect outputs"})
    cols = [c for c in ["flagged on incorrect outputs", "flagged on correct outputs"] if c in t]
    return t[cols].round(2)


def cost_latency() -> tuple[pd.DataFrame | None, list[str]]:
    """Is the layer lightweight? Compare the baseline call (all Condition A needs)
    with the full layer, per case. Seeded-error cases are excluded because their
    shown answer is injected, not generated."""
    outs = {k: v for k, v in _load_outputs().items() if not v.get("injected") and v.get("usage")}
    if not outs:
        return None, []
    rows = []
    for o in outs.values():
        u = o["usage"]
        base_in, base_out = u.get("tokens_by_task", {}).get("baseline", [0, 0])
        rows.append({
            "only_calls": u.get("calls_by_task", {}).get("baseline", 1), "layer_calls": u["calls"],
            "only_latency": u.get("baseline_latency_s", 0.0), "layer_latency": u["latency_s"],
            "only_in": base_in, "layer_in": u["input_tokens"],
            "only_out": base_out, "layer_out": u["output_tokens"],
        })
    df = pd.DataFrame(rows).mean()
    price = lambda i, o: (i * config.PRICE_INPUT_PER_MTOK + o * config.PRICE_OUTPUT_PER_MTOK) / 1e6  # noqa: E731
    table = pd.DataFrame({
        "LLM only": [df.only_calls, df.only_latency, df.only_in, df.only_out, price(df.only_in, df.only_out)],
        "LLM + layer": [df.layer_calls, df.layer_latency, df.layer_in, df.layer_out,
                        price(df.layer_in, df.layer_out)],
    }, index=["LLM calls per case", "Generation time per case (s, sequential)", "Input tokens per case",
              "Output tokens per case", "Estimated cost per case (USD)"])
    table["Ratio (layer / only)"] = (table["LLM + layer"] / table["LLM only"].replace(0, np.nan)).round(1)
    table = table.round(4)
    tasks = pd.DataFrame([o["usage"].get("latency_by_task", {}) for o in outs.values()]).mean().round(2)
    notes = [f"Averaged over {len(outs)} generated cases. Prices: {config.PRICE_INPUT_PER_MTOK} / "
             f"{config.PRICE_OUTPUT_PER_MTOK} USD per million input / output tokens (set in .env).",
             "Mean seconds by step: " + ", ".join(f"{k} {v}" for k, v in tasks.items()) + ".",
             "Outputs are precomputed, so participants never wait for this; the repeated Prompt B calls "
             "are independent and could run in parallel in a deployed version."]
    return table, notes


def participants_summary(resp: pd.DataFrame) -> pd.DataFrame | None:
    sess = _read(SESSIONS_CSV)
    if sess.empty or resp.empty:
        return None
    sess = sess[sess["event"] == "start"].drop_duplicates("participant_id", keep="last")
    sess = sess[sess["participant_id"].isin(resp["participant_id"].unique())]
    both = resp.groupby("participant_id")["condition"].nunique().eq(2).sum()
    rows = {"Participants with data": resp["participant_id"].nunique(),
            "Completed both conditions": int(both)}
    for role, n in sess["role"].value_counts().items():
        rows[f"Role: {role}"] = int(n)
    if "experience_years" in sess:
        rows["Median years of experience"] = pd.to_numeric(sess["experience_years"], errors="coerce").median()
    if "group" in sess:
        for g, n in sess["group"].value_counts().sort_index().items():
            rows[f"Counterbalancing group {g}"] = int(n)
    return pd.DataFrame({"value": rows})


_KEYWORDS = {
    "Reliability level and its reason": ["reliab", "indicator", "low", "moderate", "high", "level"],
    "Main factors considered": ["factor", "reason"],
    "Missing information": ["missing", "absent", "not available", "lack"],
    "Alternative possibilities": ["alternative", "differential"],
    "What would change this recommendation": ["would change", "change this", "what would"],
    "Check before acting": ["check", "verify"],
    "Conflicting-data warning": ["conflict", "inconsisten", "contradict"],
    "Automated quality check": ["quality check", "judge", "automated"],
}


def component_usefulness(resp: pd.DataFrame, quest: pd.DataFrame) -> pd.DataFrame | None:
    """Which parts of the layer helped? (a) explicit choices after the layer block,
    (b) mentions in free-text comments during the layer block (simple keyword count,
    to be confirmed by reading the comments)."""
    rows = {c: {"chosen as helpful": 0, "mentioned in comments": 0} for c in LAYER_COMPONENTS}
    n_q = 0
    if not quest.empty and "useful_components" in quest:
        qb = quest[quest["condition"] == "B"]
        n_q = len(qb)
        for val in qb["useful_components"].fillna(""):
            for c in [x.strip() for x in str(val).split(";") if x.strip()]:
                if c in rows:
                    rows[c]["chosen as helpful"] += 1
    if not resp.empty:
        texts = resp[resp["condition"] == "B"][["comments", "missing_info_noted"]].fillna("").astype(str)
        joined = (texts["comments"] + " " + texts["missing_info_noted"]).str.lower()
        for comp, words in _KEYWORDS.items():
            rows[comp]["mentioned in comments"] = int(joined.apply(lambda t: any(w in t for w in words)).sum())
    df = pd.DataFrame(rows).T
    if df.values.sum() == 0:
        return None
    df["share of participants choosing it"] = (df["chosen as helpful"] / n_q).round(2) if n_q else np.nan
    return df.sort_values("chosen as helpful", ascending=False)


def rationale_profile() -> pd.DataFrame | None:
    """Descriptive: what the Function C view contained (never used in the rule)."""
    outs = _load_outputs()
    if not outs:
        return None
    rows = []
    cats = {c["id"]: c["category"] for c in load_cases()}
    for cid, o in outs.items():
        q = build_rationale(o).quality_log()
        rows.append({"category": cats.get(cid, "?"), "factors": q["n_factors"], "missing (critical)":
                     q["n_missing_critical"], "alternatives": q["n_alternatives"],
                     "has 'what would change'": q["has_what_would_change"],
                     "empty sections": len(q["empty_sections"])})
    return pd.DataFrame(rows).groupby("category").mean().round(2)


# ---------------------------------------------------------------------------
def main_comparison(resp: pd.DataFrame, quest: pd.DataFrame):
    rows, notes = {}, []
    for cond in ("A", "B"):
        d = resp[resp["condition"] == cond]
        if d.empty:
            continue
        m = reliance_metrics(d)
        q = quest[quest["condition"] == cond] if not quest.empty else pd.DataFrame()
        rows[COND_NAME[cond]] = {
            "Case responses": m["n"],
            "Error detection (reject | incorrect)": m["error_detection"],
            "Over-reliance (accept | incorrect)": m["commission_errors"],
            "Correct acceptance (accept | correct)": m["correct_acceptance"],
            "Under-reliance (reject | correct)": m["under_reliance"],
            "Appropriate reliance (overall)": m["appropriate_reliance"],
            "Perceived-reliability discrimination": perceived_discrimination(d),
            "Explanation usefulness (1-5)": pd.to_numeric(d["explanation_usefulness"]).mean(),
            "Decision confidence (1-5)": pd.to_numeric(d["decision_confidence"]).mean(),
            "Median time per case (s)": pd.to_numeric(d["time_seconds"]).median(),
            "SUS (0-100)": q["sus"].mean() if "sus" in q else np.nan,
            "NASA-TLX raw (0-100, lower better)": q["tlx"].mean() if "tlx" in q else np.nan,
            "Trust items (1-5)": q["trust"].mean() if "trust" in q else np.nan,
        }
    table = pd.DataFrame(rows).round(3)
    if {"LLM only", "LLM + layer"} <= set(table.columns):
        table["Difference (layer - only)"] = (table["LLM + layer"] - table["LLM only"]).round(3)
    b = resp[resp["condition"] == "B"]
    if not b.empty:
        ind = b["indicator_appropriateness"].value_counts(normalize=True).round(2).to_dict()
        notes.append("Condition B, was the reliability indicator appropriate? "
                     + ", ".join(f"{k}: {v:.0%}" for k, v in ind.items()))
    notes.append("Success is NOT higher acceptance. It is higher error detection and appropriate reliance "
                 "without a large increase in time or workload.")
    return table, notes


def reliance_matrices(resp: pd.DataFrame) -> pd.DataFrame:
    """The 2 x 2 decision outcome counts per condition (confusion-matrix view)."""
    cols = {}
    for cond in ("A", "B"):
        d = resp[resp["condition"] == cond]
        if not d.empty:
            m = reliance_matrix(d)
            cols[COND_NAME[cond] + " (n)"] = m
            cols[COND_NAME[cond] + " (%)"] = (m / m.sum() * 100).round(1)
    return pd.DataFrame(cols).reindex(RELIANCE_CELLS)


def by_category(resp: pd.DataFrame) -> pd.DataFrame:
    g = resp.assign(accepted=resp["decision"].eq("accept"))
    t = g.pivot_table(index="category", columns="condition", values="accepted", aggfunc="mean")
    return t.rename(columns=COND_NAME).round(2).rename_axis(None, axis=1).add_prefix("Acceptance rate: ")


def paired_tests(resp: pd.DataFrame, quest: pd.DataFrame):
    per = []
    for (pid, cond), d in resp.groupby(["participant_id", "condition"]):
        m = reliance_metrics(d)
        per.append({"participant_id": pid, "condition": cond, "appropriate_reliance": m["appropriate_reliance"],
                    "error_detection": m["error_detection"],
                    "time": pd.to_numeric(d["time_seconds"]).median()})
    per = pd.DataFrame(per)
    if not quest.empty:
        per = per.merge(quest[["participant_id", "condition", "sus", "tlx"]], on=["participant_id", "condition"],
                        how="left")
    wide = per.pivot(index="participant_id", columns="condition")
    out = []
    for metric in ["appropriate_reliance", "error_detection", "time", "sus", "tlx"]:
        if metric not in per:
            continue
        try:
            pair = wide[metric][["A", "B"]].dropna()
        except KeyError:
            continue
        n = len(pair)
        row = {"metric": metric, "n_pairs": n, "mean_A": pair["A"].mean(), "mean_B": pair["B"].mean()}
        if n >= 5 and not (pair["A"] - pair["B"]).eq(0).all():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w = stats.wilcoxon(pair["A"], pair["B"])
            row.update({"test": "Wilcoxon signed-rank", "statistic": w.statistic, "p_value": w.pvalue})
        else:
            row.update({"test": "not run (needs 5+ paired participants with differences)"})
        out.append(row)
    return pd.DataFrame(out).round(4) if out else None


def judge_vs_clinicians(resp: pd.DataFrame, adj: pd.DataFrame) -> list[str]:
    import json
    scores = {}
    for cid in resp["case_id"].unique():
        p = config.OUTPUTS_DIR / f"{cid}.json"
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8")).get("judge") or {}
            if j.get("overall") is not None:
                scores[cid] = j["overall"]
    if len(scores) < 5:
        return []
    agree = resp.assign(a=resp["agreement"].map(AGREEMENT)).groupby("case_id")["a"].mean()
    df = pd.DataFrame({"judge": pd.Series(scores), "clinician_agreement": agree}).dropna()
    lines = []
    if len(df) >= 5 and df["judge"].nunique() > 1 and df["clinician_agreement"].nunique() > 1:
        rho, p = stats.spearmanr(df["judge"], df["clinician_agreement"])
        lines.append(f"Spearman correlation, judge overall score vs mean clinician agreement: "
                     f"rho = {rho:.2f}, p = {p:.3f}, n = {len(df)} cases.")
    if not adj.empty:
        a = adj.set_index("case_id")["output_correct"]
        df2 = pd.DataFrame({"judge": pd.Series(scores), "correct": a}).dropna()
        if df2["correct"].nunique() == 2:
            lines.append(f"Mean judge score on correct outputs {df2[df2.correct]['judge'].mean():.2f} vs "
                         f"incorrect outputs {df2[~df2.correct.astype(bool)]['judge'].mean():.2f}.")
    return lines


# ---------------------------------------------------------------------------
def _figures(resp: pd.DataFrame, quest: pd.DataFrame, table: pd.DataFrame) -> dict:
    figs = {}
    conds = [c for c in ("A", "B") if c in resp["condition"].unique()]

    metrics = ["Error detection (reject | incorrect)", "Over-reliance (accept | incorrect)",
               "Correct acceptance (accept | correct)", "Appropriate reliance (overall)"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(metrics))
    for i, c in enumerate(conds):
        vals = [table.loc[m, COND_NAME[c]] for m in metrics]
        ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, label=COND_NAME[c], color=COLORS[c])
    ax.set_xticks(x, ["Error\ndetection", "Over-\nreliance", "Correct\nacceptance", "Appropriate\nreliance"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title("Reliance on the AI by condition")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["reliance_by_condition"] = fig

    mat = reliance_matrices(resp)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    cell_colors = {"Appropriate acceptance": "#1E6B52", "Appropriate rejection": "#7FB89F",
                   "Under-reliance": "#D8A657", "Over-reliance": "#A3231B"}
    for i, c in enumerate(conds):
        left = 0
        for cell in RELIANCE_CELLS:
            v = mat.loc[cell, COND_NAME[c] + " (%)"]
            ax.barh(i, v, left=left, color=cell_colors[cell], label=cell if i == 0 else None)
            if v >= 6:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
            left += v
    ax.set_yticks(range(len(conds)), [COND_NAME[c] for c in conds])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of decisions (%)")
    ax.set_title("Decision outcomes: AI correctness x clinician decision")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    figs["reliance_matrix"] = fig

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    data = [pd.to_numeric(resp[resp.condition == c]["time_seconds"]) for c in conds]
    ax.boxplot(data, tick_labels=[COND_NAME[c] for c in conds], widths=0.5)
    ax.set_ylabel("Seconds per case")
    ax.set_title("Time per case")
    fig.tight_layout()
    figs["time_by_condition"] = fig

    if not quest.empty:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        labels = ["SUS", "NASA-TLX"]
        for i, c in enumerate(conds):
            q = quest[quest.condition == c]
            ax.bar(np.arange(2) + (i - 0.5) * 0.36, [q["sus"].mean(), q["tlx"].mean()], 0.36,
                   label=COND_NAME[c], color=COLORS[c])
        ax.set_xticks(np.arange(2), labels)
        ax.set_ylim(0, 100)
        ax.set_title("Usability (higher better) and workload (lower better)", fontsize=9)
        ax.legend(frameon=False)
        fig.tight_layout()
        figs["questionnaires_by_condition"] = fig

    cat = by_category(resp)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(cat))
    for i, c in enumerate(conds):
        col = f"Acceptance rate: {COND_NAME[c]}"
        if col in cat:
            ax.bar(x + (i - 0.5) * 0.36, cat[col], 0.36, label=COND_NAME[c], color=COLORS[c])
    ax.set_xticks(x, [c.replace("_", "\n") for c in cat.index], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Acceptance by case category (seeded errors should be low)")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["acceptance_by_category"] = fig
    return figs


def run_analysis(save: bool = True) -> dict:
    adj = load_adjudication()
    calib, calib_lines, calib_fig = layer_calibration(adj)
    cost_table, cost_notes = cost_latency()
    res = {"layer_calibration": calib, "layer_summary": calib_lines, "figures": {}, "n_responses": 0,
           "n_participants": 0, "main_table": None, "by_category": None, "stats": None,
           "judge_agreement": [], "notes": [], "reliance_matrix": None, "participants": None,
           "signal_breakdown": signal_breakdown(adj), "cost_latency": cost_table, "cost_notes": cost_notes,
           "components": None, "rationale_profile": rationale_profile()}
    if calib_fig is not None:
        res["figures"]["reliability_vs_correctness"] = calib_fig

    resp, quest = _read(RESPONSES_CSV), _read(QUESTIONNAIRES_CSV)
    # If a participant repeated a case or questionnaire (e.g. restarted session), keep the last answer.
    if not resp.empty:
        resp = resp.drop_duplicates(["participant_id", "condition", "case_id"], keep="last")
    if not quest.empty:
        quest = quest.drop_duplicates(["participant_id", "condition"], keep="last")
    if not resp.empty:
        resp = resp.drop(columns=[c for c in ["output_correct"] if c in resp])
        resp = resp.merge(adj[["case_id", "output_correct"]], on="case_id", how="left")
        resp["output_correct"] = resp["output_correct"].fillna(True).astype(bool)
        if not quest.empty:
            quest["sus"] = quest[[f"sus_{i}" for i in range(1, 11)]].apply(lambda r: sus_score(list(r)), axis=1)
            quest["tlx"] = quest[TLX_COLS].apply(lambda r: tlx_raw(list(r)), axis=1)
            quest["trust"] = quest[["trust_1", "trust_2", "trust_3"]].mean(axis=1)
        table, notes = main_comparison(resp, quest)
        res.update(n_responses=len(resp), n_participants=resp["participant_id"].nunique(), main_table=table,
                   notes=notes, by_category=by_category(resp), stats=paired_tests(resp, quest),
                   judge_agreement=judge_vs_clinicians(resp, adj), reliance_matrix=reliance_matrices(resp),
                   participants=participants_summary(resp), components=component_usefulness(resp, quest))
        res["figures"].update(_figures(resp, quest, table))

    if outputs_meta().get("provider") == "mock":
        res["notes"].insert(0, "WARNING: AI outputs come from the mock simulator. These numbers are not findings.")

    if save:
        out = config.REPORT_DIR
        if calib is not None:
            calib.to_csv(out / "layer_calibration.csv")
        for key, fname in [("signal_breakdown", "signal_breakdown.csv"), ("cost_latency", "cost_latency.csv"),
                           ("reliance_matrix", "reliance_matrix.csv"), ("participants", "participants.csv"),
                           ("components", "component_usefulness.csv"),
                           ("rationale_profile", "rationale_profile.csv")]:
            if res[key] is not None:
                res[key].to_csv(out / fname)
        if res["main_table"] is not None:
            res["main_table"].to_csv(out / "results_table.csv")
            res["by_category"].to_csv(out / "acceptance_by_category.csv")
            if res["stats"] is not None:
                res["stats"].to_csv(out / "paired_tests.csv", index=False)
        (out / "report.md").write_text(build_report(res), encoding="utf-8")
        for name, fig in res["figures"].items():
            fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    return res


def _table(df) -> str:
    try:
        return df.to_markdown()
    except ImportError:  # 'tabulate' not installed
        return "```\n" + df.to_string() + "\n```"


def build_report(res: dict) -> str:
    """Full Markdown report: every table the Results page shows, in report order."""
    meta = outputs_meta()
    p = ["# Evaluation report: LLM only vs LLM + intermediary layer", ""]
    p += [f"- AI outputs: provider `{meta.get('provider', '?')}`, model `{meta.get('model', '?')}`, "
          f"generated {meta.get('updated_at', '?')}",
          f"- Participants: {res['n_participants']}, case responses: {res['n_responses']}", ""]
    p += [f"> {n}" for n in res["notes"]] + [""]
    sections = [
        ("1. Participants", res["participants"], []),
        ("2. Decision performance (LLM only vs LLM + layer)", res["main_table"], []),
        ("3. Trust calibration: decision outcomes", res["reliance_matrix"],
         ["Appropriate acceptance = correct AI accepted; under-reliance = correct AI rejected; "
          "over-reliance = incorrect AI accepted; appropriate rejection = incorrect AI rejected."]),
        ("4. Acceptance by case category", res["by_category"], []),
        ("5. Paired tests", res["stats"], []),
        ("6. Reliability indicator vs actual correctness", res["layer_calibration"], res["layer_summary"]),
        ("7. Which signals carry information", res["signal_breakdown"],
         ["Share of cases in which each signal warned or failed. A useful signal fires far more often on "
          "incorrect outputs than on correct ones."]),
        ("8. Which parts of the layer helped", res["components"], []),
        ("9. Content of the clinician rationale (Function C)", res["rationale_profile"], []),
        ("10. Cost and latency (is the layer lightweight?)", res["cost_latency"], res["cost_notes"]),
    ]
    for title, df, lines in sections:
        if df is None:
            continue
        p += [f"## {title}", ""]
        if df is not None:
            p += [_table(df), ""]
        p += [f"- {x}" for x in lines] + [""]
    if res["judge_agreement"]:
        p += ["## 11. Automated judge vs clinicians", ""] + [f"- {x}" for x in res["judge_agreement"]] + [""]
    p += ["## Figures", ""] + [f"- `results/figures/{n}.png`" for n in res["figures"]]
    return "\n".join(p)


if __name__ == "__main__":
    r = run_analysis(save=True)
    pd.set_option("display.width", 160)
    print("\n== Reliability indicator vs correctness ==")
    print(r["layer_calibration"])
    for line in r["layer_summary"]:
        print(" ", line)
    if r["main_table"] is None:
        print("\nNo participant responses yet (data/results/responses.csv).")
    else:
        print(f"\n== LLM only vs LLM + layer: {r['n_participants']} participants ==")
        print(r["main_table"])
        for n in r["notes"]:
            print(" ", n)
        print("\n== By category ==")
        print(r["by_category"])
        if r["stats"] is not None:
            print("\n== Paired tests ==")
            print(r["stats"])
        for line in r["judge_agreement"]:
            print(" ", line)
    print(f"\nTables in {config.REPORT_DIR}, figures in {config.FIGURES_DIR}")
