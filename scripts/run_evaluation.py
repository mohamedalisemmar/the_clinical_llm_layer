"""One command to reproduce every table and figure of the evaluation.

    python -m scripts.run_evaluation                 # analysis only (uses cached outputs + study CSVs)
    python -m scripts.run_evaluation --precompute    # regenerate AI outputs first (provider from .env)

Writes:
    results/report.md              full report, all tables in report order
    results/*.csv                  each table separately
    results/figures/*.png          all figures (200 dpi)
    results/run_info.json          provider, model, prompt version, settings, date, test status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from llm.prompts import prompts_version  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precompute", action="store_true", help="regenerate AI outputs before analysing")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    if args.precompute:
        subprocess.run([sys.executable, "-m", "scripts.precompute_outputs", "--force"], cwd=ROOT, check=True)

    tests = "skipped"
    if not args.skip_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
        tests = (r.stdout.strip().splitlines() or ["?"])[-1]
        print("Tests:", tests)

    from evaluation.analysis import run_analysis
    from storage import outputs_meta
    res = run_analysis(save=True)

    info = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outputs": outputs_meta(), "prompts_version": prompts_version(),
        "n_samples": config.N_SAMPLES, "consistency_thresholds": [config.CONSISTENCY_WARN, config.CONSISTENCY_PASS],
        "participants": res["n_participants"], "case_responses": res["n_responses"], "tests": tests,
    }
    (config.REPORT_DIR / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Participants: {res['n_participants']}, case responses: {res['n_responses']}")
    for n in res["notes"][:1]:
        print(n)
    print(f"Report:  {config.REPORT_DIR / 'report.md'}")
    print(f"Figures: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
