"""Validate an LLM `LogAnalysis` against the source log rows.

The model is asked to cite `LineId` values as evidence (see `models.py`). A
fluent answer is worthless if those citations are invented, so this module
checks that the evidence is *grounded*: every cited LineId must be one of the
lines the model was actually shown.

We grade each cited LineId into one of three buckets:

* **grounded**       — cited and present in the sampled batch (what we want).
* **out_of_sample**  — exists in the CSV, but wasn't in the lines we sent. The
                       model is referencing the file beyond what it was given.
* **unknown**        — not found anywhere in the CSV. A fabricated LineId.

An analysis is considered valid only when every citation is grounded.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import LogAnalysis


@dataclass
class ValidationResult:
    cited_line_ids: list[int]
    grounded_line_ids: list[int]
    out_of_sample_line_ids: list[int]
    unknown_line_ids: list[int]
    #: Whether the model had log-exploration tools. When True, citing a line from
    #: outside the sampled batch is legitimate (it was looked up), so out-of-sample
    #: citations no longer count against validity — only fabricated ones do.
    tools_used: bool = False

    @property
    def is_valid(self) -> bool:
        """True when the model cited evidence and none of it is fabricated.

        Without tools, an out-of-sample citation also fails (the model could only
        have seen the sampled lines). With tools, any line in the file is fair game.
        """
        if not self.cited_line_ids or self.unknown_line_ids:
            return False
        if self.out_of_sample_line_ids and not self.tools_used:
            return False
        return True

    @property
    def issues(self) -> list[str]:
        """Human-readable problems, empty when the analysis is grounded."""
        problems: list[str] = []
        if not self.cited_line_ids:
            problems.append("the analysis cited no evidence LineIds")
        if self.out_of_sample_line_ids and not self.tools_used:
            problems.append(
                "cited LineIds not in the batch shown to the model: "
                f"{self.out_of_sample_line_ids}"
            )
        if self.unknown_line_ids:
            problems.append(
                "cited LineIds not found anywhere in the log file (fabricated): "
                f"{self.unknown_line_ids}"
            )
        return problems


def _line_ids(rows: list[dict[str, str]]) -> set[int]:
    """Collect the integer LineId values from a set of CSV rows, skipping bad ones."""
    ids: set[int] = set()
    for row in rows:
        raw = row.get("LineId")
        if raw is None:
            continue
        try:
            ids.add(int(raw))
        except (TypeError, ValueError):
            continue
    return ids


def validate_analysis(
    analysis: LogAnalysis,
    sampled_rows: list[dict[str, str]],
    all_rows: list[dict[str, str]] | None = None,
    tools_used: bool = False,
) -> ValidationResult:
    """Check that the analysis's evidence citations are grounded in the source logs.

    `sampled_rows` are the lines actually sent to the model; `all_rows` is the
    full CSV (defaults to the sample) and is used to tell an out-of-sample
    reference apart from a fabricated LineId. `tools_used` relaxes grounding to
    the whole file when the model could look up lines beyond its sample.
    """
    sampled_ids = _line_ids(sampled_rows)
    corpus_ids = _line_ids(all_rows) if all_rows is not None else sampled_ids

    # Dedupe while preserving the order the model returned them.
    cited = list(dict.fromkeys(analysis.evidence_line_ids))

    grounded: list[int] = []
    out_of_sample: list[int] = []
    unknown: list[int] = []
    for line_id in cited:
        if line_id in sampled_ids:
            grounded.append(line_id)
        elif line_id in corpus_ids:
            out_of_sample.append(line_id)
        else:
            unknown.append(line_id)

    return ValidationResult(
        cited_line_ids=cited,
        grounded_line_ids=grounded,
        out_of_sample_line_ids=out_of_sample,
        unknown_line_ids=unknown,
        tools_used=tools_used,
    )
