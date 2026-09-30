"""Research dashboard. Same analysis as `python -m scripts.run_evaluation`."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from evaluation.analysis import build_report, run_analysis  # noqa: E402
from storage import outputs_meta  # noqa: E402
from ui.components import research_notice, inject_css, mock_warning  # noqa: E402

st.set_page_config(page_title="Results", layout="wide")
inject_css()
research_notice()
st.title("Results")
meta = outputs_meta()
mock_warning(meta)
st.caption(f"AI outputs: {meta.get('provider', '?')} / {meta.get('model', '?')}, generated "
           f"{meta.get('updated_at', '?')}. Correctness labels come from data/outputs/adjudication.csv.")

res = run_analysis(save=True)
has_people = res["n_responses"] > 0


def fig(name):
    if name in res["figures"]:
        st.pyplot(res["figures"][name])


def waiting():
    st.info("Needs participant data. Complete sessions on the Study session page.")


t1, t2, t3, t4, t5, t6 = st.tabs(["Participants", "Decisions and trust calibration", "Usability",
                                  "Reliability signal", "Layer components", "Cost and latency"])

with t1:
    st.subheader("Participants")
    if res["participants"] is not None:
        st.dataframe(res["participants"])
    else:
        waiting()

with t2:
    st.subheader("Decision performance: LLM only vs LLM + layer")
    if not has_people:
        waiting()
    else:
        st.write(f"{res['n_participants']} participants, {res['n_responses']} case decisions.")
        st.dataframe(res["main_table"])
        for note in res["notes"]:
            st.caption(note)
        st.subheader("Decision outcomes (AI correctness x clinician decision)")
        st.dataframe(res["reliance_matrix"])
        st.caption("Over-reliance (accepting an incorrect AI answer) is the automation-bias measure. "
                   "The layer succeeds if over-reliance falls without a rise in under-reliance.")
        c1, c2 = st.columns(2)
        with c1:
            fig("reliance_matrix")
        with c2:
            fig("reliance_by_condition")
        st.subheader("Acceptance by case category")
        st.dataframe(res["by_category"])
        fig("acceptance_by_category")
        if res["stats"] is not None:
            st.subheader("Paired tests (per participant)")
            st.dataframe(res["stats"])

with t3:
    st.subheader("Usability cost")
    if not has_people:
        waiting()
    else:
        rows = [r for r in res["main_table"].index if any(k in r for k in ("SUS", "TLX", "time", "Trust"))]
        st.dataframe(res["main_table"].loc[rows])
        c1, c2 = st.columns(2)
        with c1:
            fig("time_by_condition")
        with c2:
            fig("questionnaires_by_condition")

with t4:
    st.subheader("Does Low / Moderate / High track actual correctness?")
    st.write("Computed from the AI outputs and adjudicated correctness. Needs no participants.")
    if res["layer_calibration"] is not None:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.dataframe(res["layer_calibration"])
            for line in res["layer_summary"]:
                st.write(line)
        with c2:
            fig("reliability_vs_correctness")
    if res["signal_breakdown"] is not None:
        st.subheader("Which individual signals carry information")
        st.dataframe(res["signal_breakdown"])
        st.caption("Share of cases where each signal warned or failed. A useful signal fires much more often "
                   "on incorrect outputs than on correct ones.")
    if res["judge_agreement"]:
        st.subheader("Automated judge vs clinicians")
        for line in res["judge_agreement"]:
            st.write(line)

with t5:
    st.subheader("Which parts of the layer helped")
    if res["components"] is not None:
        st.dataframe(res["components"])
        st.caption("Comment mentions are a keyword count; read the comments to confirm them.")
    else:
        waiting()
    if res["rationale_profile"] is not None:
        st.subheader("What the clinician rationale contained (Function C), by case category")
        st.dataframe(res["rationale_profile"])

with t6:
    st.subheader("Is the layer lightweight?")
    if res["cost_latency"] is not None:
        st.dataframe(res["cost_latency"])
        for line in res["cost_notes"]:
            st.caption(line)
    else:
        st.info("No usage data. Re-run `python -m scripts.precompute_outputs --force`.")

st.divider()
st.download_button("Download full report (Markdown)", build_report(res), file_name="report.md")
st.caption(f"Tables, report and figures were also written to {config.REPORT_DIR.name}/.")
