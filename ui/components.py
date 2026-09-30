"""Shared Streamlit rendering: the patient case, Condition A (raw answer),
Condition B (the intermediary layer = Functions A-D), and the signal table."""
from __future__ import annotations

import html

import streamlit as st

from layer.inspectability import LEFT_COLUMN, build_rationale
from llm.prompts import CASE_FIELDS

LEVEL_STYLE = {
    "HIGH": ("#1E6B52", "#E7F3EE", "High"),
    "MODERATE": ("#8A5300", "#FBF1E1", "Moderate"),
    "LOW": ("#A3231B", "#FBE9E7", "Low"),
}
STATUS_ICON = {"pass": "✓", "warn": "!", "fail": "✕", "na": "–"}
STATUS_COLOR = {"pass": "#1E6B52", "warn": "#8A5300", "fail": "#A3231B", "na": "#6B7680"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stRadio, .stTextInput, .stTextArea {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Serif', Georgia, serif; font-weight: 500; letter-spacing: -0.01em; }
.block-container { max-width: 1180px; padding-top: 2rem; }
.case-sheet { border-left: 3px solid #2F5D8A; padding: 0.2rem 0 0.2rem 1.1rem; }
.case-sheet .who { font-size: 1.15rem; font-weight: 600; margin-bottom: 0.6rem; }
.case-row { display: grid; grid-template-columns: 9.5rem 1fr; gap: 0.8rem; padding: 0.35rem 0;
            border-bottom: 1px solid #E4E9ED; line-height: 1.5; }
.case-row .k { color: #56626D; font-size: 0.9rem; }
.case-row .v { font-size: 0.97rem; }
.case-row .v.absent { color: #8A949C; font-style: italic; }
.raw-answer { background: #F7F9FA; border: 1px solid #DDE3E8; border-radius: 6px; padding: 1.1rem 1.3rem;
              white-space: pre-wrap; line-height: 1.6; font-size: 0.97rem; }
.rec { font-size: 1.12rem; line-height: 1.5; margin: 0.2rem 0 0.9rem 0; }
.rec .dx { font-weight: 600; display: block; font-size: 1.25rem; margin-bottom: 0.2rem; }
.rel { border-radius: 6px; padding: 0.85rem 1.1rem; margin: 0 0 1.1rem 0; display: flex; gap: 1.1rem; align-items: baseline; }
.rel .lvl { font-weight: 600; font-size: 1.05rem; min-width: 6.5rem; }
.rel .why { font-size: 0.95rem; line-height: 1.45; }
.verify { border: 1.5px solid #A3231B; color: #A3231B; border-radius: 6px; padding: 0.6rem 1rem;
          margin-bottom: 1rem; font-weight: 500; }
.field h4 { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.92rem; font-weight: 600; color: #34414D;
            margin: 0.4rem 0 0.3rem 0; }
.field ul { margin: 0 0 0.6rem 1.1rem; padding: 0; }
.field li { margin: 0.15rem 0; line-height: 1.45; font-size: 0.95rem; }
.crit { color: #A3231B; font-weight: 600; font-size: 0.8rem; margin-left: 0.35rem; }
.inconsistent { background: #FBF1E1; border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 0.8rem; font-size: 0.95rem; }
.sig { display: grid; grid-template-columns: 1.4rem 15rem 1fr; gap: 0.6rem; padding: 0.35rem 0;
       border-bottom: 1px solid #E4E9ED; font-size: 0.92rem; }
.sig .ic { font-weight: 700; text-align: center; }
.notice { border: 1px solid #C9D3DB; border-radius: 6px; padding: 0.45rem 0.9rem; font-size: 0.85rem;
          color: #46525D; margin-bottom: 1.2rem; }
.mockwarn { background: #FBF1E1; border-radius: 6px; padding: 0.5rem 0.9rem; font-size: 0.9rem; margin-bottom: 1rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _e(text) -> str:
    return html.escape(str(text or ""))


def render_case(case: dict) -> None:
    p = case.get("patient", {})
    rows = [f'<div class="who">{_e(p.get("age"))}-year-old {_e(str(p.get("sex", "")).lower())}</div>']
    for key, label in CASE_FIELDS:
        val = (case.get(key) or "").strip()
        cls, shown = ("v", _e(val)) if val else ("v absent", "Not available")
        rows.append(f'<div class="case-row"><div class="k">{label}</div><div class="{cls}">{shown}</div></div>')
    st.markdown(f'<div class="case-sheet">{"".join(rows)}</div>', unsafe_allow_html=True)


def render_raw(output: dict) -> None:
    """Condition A: the LLM answer exactly as generated."""
    st.markdown(f'<div class="raw-answer">{_e(output["raw_response"])}</div>', unsafe_allow_html=True)


def render_reliability(level: str, reason: str) -> None:
    color, bg, word = LEVEL_STYLE[level]
    st.markdown(
        f'<div class="rel" style="background:{bg};border-left:4px solid {color}">'
        f'<div class="lvl" style="color:{color}">Reliability: {word}</div><div class="why">{_e(reason)}</div></div>',
        unsafe_allow_html=True)


def render_verify_notice() -> None:
    st.markdown('<div class="verify">Verify independently before acting on this recommendation.</div>',
                unsafe_allow_html=True)


def _section_html(sec) -> str:
    if sec.is_empty:
        return f'<div class="field"><h4>{_e(sec.title)}</h4><ul><li><em>{_e(sec.placeholder)}</em></li></ul></div>'
    lis = "".join(f'<li>{_e(i.text)}{"<span class=crit>critical</span>" if i.critical else ""}</li>'
                  for i in sec.items)
    more = f'<li><em>{sec.truncated} more not shown</em></li>' if sec.truncated else ""
    return f'<div class="field"><h4>{_e(sec.title)}</h4><ul>{lis}{more}</ul></div>'


def render_layer(output: dict, *, show_judge: bool = True, show_signals: bool = True) -> None:
    """Condition B: recommendation (A), reliability (B), rationale (C), judge (D)."""
    r = build_rationale(output)
    st.markdown(f'<div class="rec"><span class="dx">{_e(r.diagnosis)}</span>{_e(r.recommendation)}</div>',
                unsafe_allow_html=True)
    render_reliability(r.reliability_level, r.reliability_reason)
    if r.verify_independently:
        render_verify_notice()
    if r.input_inconsistencies:
        st.markdown('<div class="inconsistent"><strong>The case record contains conflicting information:</strong> '
                    + _e("; ".join(r.input_inconsistencies)) + "</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    for sec in r.sections:
        with (c1 if sec.key in LEFT_COLUMN else c2):
            st.markdown(_section_html(sec), unsafe_allow_html=True)

    if show_signals:
        with st.expander("How the reliability signal was built"):
            render_signals(output)
    if show_judge and output.get("judge") and output["judge"].get("available", True):
        j = output["judge"]
        with st.expander("Automated quality check (supplementary, not a verdict)"):
            st.markdown(f"**Overall quality:** {j.get('overall', '–')} / 5. {_e(j.get('summary'))}")
            if j.get("safety_concern"):
                st.markdown(f"**Possible safety concern:** {_e(j.get('safety_note'))}")
            if j.get("scores"):
                st.markdown(", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in j["scores"].items()))


def render_signals(output: dict) -> None:
    rows = []
    for sig in output["signals"]:
        ic, col = STATUS_ICON[sig["status"]], STATUS_COLOR[sig["status"]]
        rows.append(f'<div class="sig"><div class="ic" style="color:{col}">{ic}</div>'
                    f'<div>{_e(sig["label"])}</div><div>{_e(sig["detail"])}</div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption("The level is set by rules, not by a score: a failed consistency check or a contradiction "
               "gives Low; High needs every core check to pass with nothing missing; everything else is Moderate.")


def mock_warning(meta: dict) -> None:
    if meta.get("provider") == "mock":
        st.markdown('<div class="mockwarn">These outputs were produced by the offline <strong>mock</strong> '
                    "simulator. Use them to test the workflow only. Run "
                    "<code>python -m scripts.precompute_outputs --force</code> with a real LLM before the study.</div>",
                    unsafe_allow_html=True)


def research_notice() -> None:
    """Safety boundary shown on every page."""
    st.markdown('<div class="notice">Research prototype for evaluation only. It does not provide validated '
                "clinical advice. All cases are synthetic, the AI answers are not checked against any medical "
                "source, and the clinician makes every decision.</div>", unsafe_allow_html=True)
