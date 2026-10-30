"""Grade agent answers against DuckDB ground-truth values."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from shipping_data_agent.eval.questions import EvalQuestion, GradeKind

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])(-?\d+(?:[.,]\d+)?)(?![A-Za-z0-9_])")


def _normalize_number_token(token: str) -> float:
    cleaned = token.replace(",", "")
    return float(cleaned)


def extract_numbers(text: str) -> list[float]:
    """Pull numeric tokens from free-form agent text."""
    return [_normalize_number_token(m.group(1)) for m in _NUMBER_RE.finditer(text or "")]


def numbers_close(a: float, b: float, *, rel_tol: float = 1e-4, abs_tol: float = 0.02) -> bool:
    """Compare floats with a small absolute tolerance for rounded EUR/kg values."""
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def answer_contains_number(answer: str, expected: float) -> bool:
    return any(numbers_close(n, float(expected)) for n in extract_numbers(answer))


def answer_contains_token(answer: str, token: str) -> bool:
    return str(token).lower() in (answer or "").lower()


def sql_rows_to_expected(
    kind: GradeKind,
    columns: list[str],
    rows: list[tuple[Any, ...]] | list[list[Any]],
) -> dict[str, Any]:
    """Serialize SQL result into a grader-friendly expected payload."""
    if kind == "scalar":
        if not rows:
            raise ValueError("scalar ground truth query returned no rows")
        value = rows[0][0]
        return {"kind": kind, "value": value, "display": _format_value(value)}

    if kind == "discrepancy":
        # Prefer error_type + discrepancy_count columns when present.
        col_lower = [c.lower() for c in columns]
        try:
            type_i = col_lower.index("error_type")
        except ValueError as exc:
            raise ValueError("discrepancy SQL must include error_type") from exc
        count_i = None
        for name in ("discrepancy_count", "count", "n"):
            if name in col_lower:
                count_i = col_lower.index(name)
                break
        if count_i is None:
            # Fall back to second column.
            count_i = 1 if len(columns) > 1 else 0
        pairs = [
            {"error_type": str(row[type_i]), "count": int(row[count_i])}
            for row in rows
        ]
        return {
            "kind": kind,
            "pairs": pairs,
            "display": ", ".join(f"{p['error_type']}={p['count']}" for p in pairs),
        }

    # rows: treat first column as key / label when present, remaining as values.
    formatted_rows: list[dict[str, Any]] = []
    for row in rows:
        record = {columns[i]: row[i] for i in range(len(columns))}
        formatted_rows.append(record)
    display = "; ".join(
        ", ".join(f"{k}={_format_value(v)}" for k, v in rec.items())
        for rec in formatted_rows
    )
    return {"kind": kind, "rows": formatted_rows, "display": display}


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".") if "." in f"{value:.2f}" else f"{value:.2f}"
    return str(value)


@dataclass(frozen=True)
class GradeResult:
    passed: bool
    reason: str
    expected_display: str


def grade_answer(
    question: EvalQuestion,
    expected: dict[str, Any],
    answer: str,
) -> GradeResult:
    """Return whether the agent answer contains the SQL ground-truth values."""
    text = answer or ""
    display = str(expected.get("display") or "")

    if not text.strip():
        return GradeResult(False, "empty agent answer", display)

    kind = expected.get("kind") or question.kind

    if kind == "scalar":
        value = expected["value"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ok = answer_contains_number(text, float(value))
            return GradeResult(
                ok,
                "scalar match" if ok else f"missing expected number {value}",
                display,
            )
        ok = answer_contains_token(text, str(value))
        return GradeResult(
            ok,
            "scalar match" if ok else f"missing expected token {value}",
            display,
        )

    if kind == "discrepancy":
        missing: list[str] = []
        for pair in expected.get("pairs") or []:
            et = str(pair["error_type"])
            count = int(pair["count"])
            if not answer_contains_token(text, et):
                missing.append(f"label:{et}")
            if not answer_contains_number(text, float(count)):
                missing.append(f"count:{et}={count}")
        ok = not missing
        return GradeResult(
            ok,
            "discrepancy match" if ok else f"missing {', '.join(missing)}",
            display,
        )

    # rows: every cell value must appear (labels + numbers).
    missing_cells: list[str] = []
    for record in expected.get("rows") or []:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not answer_contains_number(text, float(value)):
                    missing_cells.append(f"{key}={value}")
            else:
                if not answer_contains_token(text, str(value)):
                    missing_cells.append(f"{key}={value}")
    ok = not missing_cells
    return GradeResult(
        ok,
        "row match" if ok else f"missing {', '.join(missing_cells)}",
        display,
    )
