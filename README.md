# Clinical LLM intermediary layer

Research prototype for the internship *Explainability and Usability Evaluation of an AI-Assisted Clinical
Decision Support System* (ESI). It implements and evaluates the design proposed in Chapter 1.

> **Research prototype, not a clinical tool.** All cases are synthetic. The AI answers are not checked
> against any medical source. The clinician makes every decision.

## Research question

Can a lightweight intermediary layer improve the explainability, perceived reliability and usability of
ungrounded LLM outputs in clinical decision support, while keeping the clinician in control of the final
decision?

The success criterion is **calibrated reliance**, not higher trust: clinicians should reject more incorrect
AI answers without rejecting more correct ones, at an acceptable cost in time and workload.

## Motivation

An LLM can produce a fluent, confident clinical answer with no guarantee that it is correct. Retrieval (RAG)
is one fix, but a curated knowledge base is often unavailable. This project studies what can be achieved
**at the interface**, without retrieval, fine-tuning or access to model internals: structure the answer,
surface observable reliability signals, show an inspectable rationale, and record the clinician's judgment.

## Architecture

```
case ─► Prompt A ─► raw answer ───────────────────────────────────────────► Condition A: LLM only
                        │
                        ├─ A  Structuring          raw text ─► fixed fields (recommendation kept as written)
                        ├─ B  Reliability signal   Prompt B x5, Prompt C, contradiction, completeness,
                        │                          missing info, conflicting case data, stated confidence
                        │                          ─► rule ─► LOW / MODERATE / HIGH + plain-language reason
                        ├─ C  Inspectability       fixed, capped clinician rationale (no raw chain-of-thought)
                        └─ D  Automated check      LLM-as-a-Judge, shown separately, never part of the rule
                                                                     ─────────► Condition B: LLM + layer
clinician ─► E  Feedback: accept/reject, agreement, reliability rating, comments ─► CSV audit trail
```

## The five functions

| Function | Purpose | Code |
|---|---|---|
| A. Output structuring | Make the answer scannable | `layer/structuring.py`, `llm/prompts.py` |
| B. Reliability signal | Tell the clinician how cautiously to read it | `layer/consistency.py`, `contradiction.py`, `completeness.py`, `missing_info.py`, `equivalence.py`, `reliability.py` |
| C. Clinician-oriented inspectability | Show an output-level explanation in a fixed shape | `layer/inspectability.py`, `ui/components.py` |
| D. Automated quality check | Cheap supplementary screening | `layer/judge.py` |
| E. Clinician judgment and feedback | Keep the human as the correctness check | `pages/1_Study_session.py`, `storage.py` |

**Reliability rule** (`layer/reliability.py`, Chapter 1 Section 3.4.2):
**LOW** if fewer than 3 of 5 repeated answers reproduce the diagnosis, or a clear contradiction is found
(either alone; triggers "Verify independently"). **HIGH** only if consistency, prompt agreement, no
contradiction and completeness all pass, with no critical missing information, no conflicting case data and
no stated low confidence. **MODERATE** otherwise. No percentage is ever shown; the judge is not in the rule.

**Terminology.** The layer does not expose the model's internal reasoning. It provides an *output-level
explanation*: what the answer recommends, what it relies on, what is missing, and what would change it.

## Experimental design

Within-subject comparison of **Condition A (LLM only)** and **Condition B (LLM + layer)**. Both conditions
show the **same recommendation**; only the presentation differs. Four counterbalancing groups cross condition
order with two matched case sets, so no participant sees a case twice. After each block: SUS, raw NASA-TLX,
three trust items, and (after the layer block) which parts of the display helped.
Full protocol: `docs/EXPERIMENT_PROTOCOL.md`.

## Dataset

30 synthetic adult cases (`data/cases.json`), two balanced sets of 15.

| Category | n | Tests |
|---|---|---|
| straightforward | 10 | Correct answers should be accepted |
| ambiguous | 6 | Uncertainty is visible |
| missing_information | 5 | Critical gaps are flagged |
| contradictory_input | 3 | Conflicting records are flagged |
| seeded_error | 6 | Fluent, confidently wrong answers should be rejected |

Ground-truth fields are never sent to the LLM or shown to participants (tested). Cases must be reviewed by a
clinician before the study (`python -m scripts.export_case_review`).

## Evaluation metrics

| Dimension | Metrics |
|---|---|
| Trust calibration | Error detection (reject given incorrect), over-reliance (accept given incorrect), correct acceptance, under-reliance (reject given correct), appropriate reliance, the 2 x 2 decision-outcome table |
| Explainability | Explanation usefulness, recall of the main reason, perceived-reliability discrimination |
| Reliability signal | LOW / MODERATE / HIGH vs actual correctness; misses and false alarms; per-signal breakdown; "consistent but wrong" cases |
| Usability cost | Time per case, SUS, NASA-TLX, trust items |
| Layer components | Which parts participants found helpful; comment mentions; rationale content per category |
| Lightweight claim | LLM calls, tokens, generation time and estimated cost per case, LLM only vs layer |
| Automated judge | Correlation with clinician agreement and with correctness |
| Statistics | Wilcoxon signed-rank tests per participant (5 or more paired participants) |

## Repository structure

```
app.py, pages/            Streamlit: home, study session, layer explorer, results dashboard
layer/                    the intermediary layer (functions A to D) and its data schemas
llm/                      provider clients (Anthropic, OpenAI-compatible, offline mock) and all prompts
evaluation/               metrics, grading, analysis and report
scripts/                  precompute outputs, run evaluation, export case review, test connection
data/cases.json           the 30 synthetic cases
data/outputs/             cached AI outputs per case + adjudication.csv (correctness labels)
data/results/             participant data (CSV)
results/                  generated report, tables and figures
docs/                     protocol, consent form, chapter-writing guide, case review sheet
tests/                    21 automated tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
```

Set the LLM in `.env`: `anthropic` (Claude), `openai` (OpenAI, Groq, OpenRouter, or a free local Ollama
server via `OPENAI_BASE_URL=http://localhost:11434/v1`), or `mock` for offline testing. Check it with
`python -m scripts.test_llm_connection`. Set `PRICE_INPUT_PER_MTOK` and `PRICE_OUTPUT_PER_MTOK` to your
model's price for the cost table.

## Running the application

```bash
python -m scripts.precompute_outputs --force   # AI outputs for every case (identical for all participants)
streamlit run app.py
```

The folder ships with **mock** outputs so the app runs immediately. They are labelled everywhere and are
not results.

## Running the tests

```bash
python -m pytest -q
```

Covers the reliability rule, the pipeline, ground-truth isolation, the Function C view, preservation of
manual correctness labels, and the analysis.

## Running the evaluation

```bash
python -m scripts.run_evaluation               # analyse existing outputs and study data
python -m scripts.run_evaluation --precompute  # regenerate the AI outputs first
```

Produces `results/report.md` (all tables in report order), one CSV per table, figures in
`results/figures/`, and `results/run_info.json` (provider, model, prompt version, settings, test status).
Every output also stores its prompt version and settings, so results can be traced to the exact prompts.

## Current results

No participant study has been run yet. With real LLM outputs, the reliability-signal and cost tables can be
produced before recruiting anyone (Results page, tabs "Reliability signal" and "Cost and latency").
Numbers produced with the mock provider are not findings.

## Limitations

* Synthetic cases, one clinical task (adult diagnosis), one site, small sample.
* One LLM at a time; results may not transfer to other models.
* Correctness labels depend on researcher adjudication.
* Consistency is not correctness: a model can be consistently wrong (the analysis reports such cases).
* Comment keyword counts are approximate and must be checked by reading the comments.
* Generation time is measured with sequential calls; parallel calls would reduce it.

## Ethical considerations

No real patient data are used. Participants give informed consent (`docs/CONSENT_FORM.md`), are identified
by code only, and may withdraw at any time. Ethics approval or exemption is sought through the host
institution where required. The interface states on every page that it is a research prototype.

## Future work

* Test the layer with several LLMs and compare calibration.
* Parallelise the repeated generations to cut latency.
* Evaluate the layer with a retrieval-grounded model to see what grounding adds.
* Larger, clinician-authored case sets and a multi-site study.
