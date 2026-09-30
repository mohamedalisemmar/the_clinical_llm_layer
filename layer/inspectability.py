"""Function C - clinician-oriented inspectability.

Function A (structuring.py) EXTRACTS the fields. This module decides what the
clinician SEES: a fixed order of short sections, with length limits and a
neutral placeholder when a section is empty, so every Condition B screen has
the same shape. It makes no LLM call and never exposes raw chain-of-thought
or the repeated Prompt B / C answers. It does not affect the reliability level.

The rationale is an output-level explanation of the answer, not the model's
internal reasoning.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

MAX_ITEMS = 5
MAX_CHARS = 220

# (key, title, placeholder shown when the model gave nothing)
SECTIONS = [
    ("main_factors", "Main factors considered", "None stated. Ask which findings support this diagnosis."),
    ("missing_information", "Missing information", "No missing information identified."),
    ("alternatives", "Alternative possibilities", "None stated. Consider what else could explain the findings."),
    ("limitations", "Limitations", "None stated."),
    ("what_would_change", "What would change this recommendation",
     "None stated. Consider what finding would make this diagnosis unlikely."),
    ("verification_points", "Check before acting", "None stated."),
]
LEFT_COLUMN = {"main_factors", "missing_information", "alternatives"}


class RationaleItem(BaseModel):
    text: str
    critical: bool = False


class RationaleSection(BaseModel):
    key: str
    title: str
    items: list[RationaleItem] = Field(default_factory=list)
    placeholder: str = ""
    truncated: int = 0          # how many items were hidden by MAX_ITEMS

    @property
    def is_empty(self) -> bool:
        return not self.items


class ClinicianRationale(BaseModel):
    diagnosis: str
    recommendation: str
    reliability_level: str
    reliability_reason: str
    verify_independently: bool
    input_inconsistencies: list[str] = Field(default_factory=list)
    sections: list[RationaleSection] = Field(default_factory=list)

    def section(self, key: str) -> RationaleSection:
        return next(s for s in self.sections if s.key == key)

    def quality_log(self) -> dict:
        """Descriptive counts for analysis only (never used in the reliability rule)."""
        return {
            "n_factors": len(self.section("main_factors").items),
            "n_missing": len(self.section("missing_information").items),
            "n_missing_critical": sum(i.critical for i in self.section("missing_information").items),
            "n_alternatives": len(self.section("alternatives").items),
            "has_what_would_change": not self.section("what_would_change").is_empty,
            "empty_sections": [s.key for s in self.sections if s.is_empty],
        }


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1].rstrip() + "…"


def build_rationale(output: dict) -> ClinicianRationale:
    """Build the clinician-facing view from a LayerOutput (as a dict)."""
    s, rel = output["structured"], output["reliability"]
    sections = []
    for key, title, placeholder in SECTIONS:
        raw = s.get(key) or []
        if key == "missing_information":
            # critical items first
            raw = sorted(raw, key=lambda m: m.get("importance") != "critical")
            items = [RationaleItem(text=_clip(m["item"]), critical=m.get("importance") == "critical")
                     for m in raw if m.get("item")]
        else:
            items = [RationaleItem(text=_clip(x)) for x in raw if str(x).strip()]
        sections.append(RationaleSection(key=key, title=title, items=items[:MAX_ITEMS],
                                         placeholder=placeholder, truncated=max(0, len(items) - MAX_ITEMS)))
    return ClinicianRationale(
        diagnosis=_clip(s.get("primary_diagnosis", "")),
        recommendation=s.get("recommendation", "").strip(),
        reliability_level=rel["level"],
        reliability_reason=rel["explanation"].split(":", 1)[-1].strip(),
        verify_independently=bool(rel.get("verify_independently")),
        input_inconsistencies=[_clip(x) for x in s.get("input_inconsistencies", []) if str(x).strip()],
        sections=sections,
    )
