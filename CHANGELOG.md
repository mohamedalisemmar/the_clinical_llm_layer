# Changelog

## v1.1: evaluation and research polish (architecture unchanged)

The pipeline, the five functions, the Low/Moderate/High rule, the 30 cases, the A vs B design and the
questionnaires are unchanged.

**Evaluation**
- Main results table now has a `Difference (layer - only)` column and separates over-reliance
  (accept | incorrect) from under-reliance (reject | correct).
- New 2 x 2 decision-outcome table and figure (`reliance_matrix`).
- Reliability calibration now lists misses (HIGH but incorrect), false alarms (LOW but correct) and
  "consistent but wrong" cases.
- New per-signal breakdown: how often each signal fires on correct vs incorrect outputs.
- New component analysis: which parts of the layer participants chose as helpful, plus comment mentions.
- New cost and latency table: LLM calls, tokens, generation time and estimated cost per case, LLM only vs layer.
- New participants table (roles, experience, groups, completion).
- `python -m scripts.run_evaluation` reproduces everything and writes `results/report.md` and `run_info.json`.

**Layer**
- Function C made explicit: `layer/inspectability.py` builds the clinician view (fixed section order,
  capped lists, placeholders for empty sections). No LLM call; does not affect the reliability level.
- Each output now records usage (calls, tokens, time per step) and provenance (prompt version, settings).

**Study session**
- Suggests the next counterbalancing group and warns on duplicate participant IDs.
- After the layer block, asks which parts of the display helped.

**Interface and docs**
- Research-prototype notice on every page.
- Results page reorganised into research sections (participants, decisions and trust calibration,
  usability, reliability signal, layer components, cost and latency) with a report download.
- README restructured academically; chapter guide updated with terminology ("output-level explanation").

**Tests**: 15 → 21 (Function C view, manual labels survive recomputation, decision-outcome table).
