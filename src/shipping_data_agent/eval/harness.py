"""Run the 15-question eval: SQL ground truth + optional live agent scoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from shipping_data_agent.eval.grading import GradeResult, grade_answer, sql_rows_to_expected
from shipping_data_agent.eval.questions import EvalQuestion, get_eval_questions
from shipping_data_agent.tools.warehouse_tools import DEFAULT_DB_PATH

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_PATH = ROOT / "data" / "eval" / "results.json"


@dataclass
class CaseResult:
    id: str
    question: str
    intent: str
    kind: str
    expected_display: str
    ground_truth_sql: str
    agent_answer: str | None = None
    passed: bool | None = None
    grade_reason: str | None = None
    retries: int | None = None
    agent_intent: str | None = None
    error: str | None = None
    skipped: bool = False


@dataclass
class EvalReport:
    ran_at: str
    warehouse: str
    llm_configured: bool
    live_agent_run: bool
    question_count: int
    answered: int = 0
    correct: int = 0
    failed: int = 0
    skipped: int = 0
    accuracy: float | None = None
    average_retries: float | None = None
    cases: list[CaseResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def compute_ground_truth(
    question: EvalQuestion,
    *,
    db_path: Path,
) -> dict[str, Any]:
    """Execute the question's SQL against DuckDB and return grader payload."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cursor = con.execute(question.sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    finally:
        con.close()
    return sql_rows_to_expected(question.kind, columns, rows)


def run_eval(
    *,
    db_path: Path | None = None,
    live: bool | None = None,
    questions: tuple[EvalQuestion, ...] | None = None,
) -> EvalReport:
    """Score the question bank.

    If ``live`` is True (or None and an LLM key is configured), run the agent.
    Otherwise only compute SQL ground truth and mark cases skipped.
    """
    from shipping_data_agent.agent.llm import llm_configured

    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Warehouse not found: {path}. Run `make etl` first."
        )

    bank = questions if questions is not None else get_eval_questions()
    keys_ok = llm_configured()
    do_live = keys_ok if live is None else bool(live)
    if do_live and not keys_ok:
        raise RuntimeError(
            "Live eval requested but no LLM API key found "
            "(GEMINI_API_KEY / GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY). "
            "Copy .env.example → .env and set a key."
        )

    report = EvalReport(
        ran_at=datetime.now(timezone.utc).isoformat(),
        warehouse=str(path),
        llm_configured=keys_ok,
        live_agent_run=do_live,
        question_count=len(bank),
        notes=(
            "Live agent scoring completed."
            if do_live
            else (
                "LLM keys missing or --ground-truth-only: shipped question bank + "
                "SQL ground truth only. Re-run with keys for accuracy / retries."
            )
        ),
    )

    ask_state = None
    if do_live:
        from shipping_data_agent.agent import ask_state as _ask_state

        ask_state = _ask_state

    retry_sum = 0
    retry_n = 0

    for q in bank:
        expected = compute_ground_truth(q, db_path=path)
        case = CaseResult(
            id=q.id,
            question=q.question,
            intent=q.intent,
            kind=q.kind,
            expected_display=str(expected.get("display") or ""),
            ground_truth_sql=_compact_sql(q.sql),
        )

        if not do_live:
            case.skipped = True
            report.skipped += 1
            report.cases.append(case)
            continue

        assert ask_state is not None
        try:
            state = ask_state(q.question, db_path=path)
            answer = str(state.get("answer") or "").strip()
            retries = int(state.get("retry_count") or 0)
            case.agent_answer = answer
            case.retries = retries
            case.agent_intent = str(state.get("intent") or "")
            grade: GradeResult = grade_answer(q, expected, answer)
            case.passed = grade.passed
            case.grade_reason = grade.reason
            report.answered += 1
            retry_sum += retries
            retry_n += 1
            if grade.passed:
                report.correct += 1
            else:
                report.failed += 1
        except Exception as exc:  # noqa: BLE001 — eval must continue on case failure
            case.error = str(exc)
            case.passed = False
            case.grade_reason = f"agent error: {exc}"
            report.answered += 1
            report.failed += 1

        report.cases.append(case)

    if report.answered:
        report.accuracy = report.correct / report.answered
    if retry_n:
        report.average_retries = retry_sum / retry_n

    return report


def save_results(report: EvalReport, path: Path | None = None) -> Path:
    """Write JSON results under data/eval/."""
    out = Path(path) if path is not None else DEFAULT_RESULTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return out


def format_readme_section(report: EvalReport) -> str:
    """Markdown block suitable for pasting into README.md."""
    lines = [
        "## Evaluation (Step 6)",
        "",
        "Fifteen fixed questions with answers computed directly in SQL against DuckDB. "
        "The harness runs the LangGraph agent (when LLM keys are set), grades free-text "
        "answers against those ground-truth values, and reports accuracy, failed cases, "
        "and average SQL retries.",
        "",
        "### How to run",
        "",
        "```bash",
        "cp -n .env.example .env   # set GEMINI_API_KEY (or ANTHROPIC/OPENAI)",
        "make etl                 # ensure warehouse exists",
        "make eval                # live agent scoring when keys present",
        "# or ground-truth only (no LLM):",
        ".venv/bin/python scripts/run_eval.py --ground-truth-only",
        "```",
        "",
        f"Results JSON: `data/eval/results.json` (last run: `{report.ran_at}`).",
        "",
        "### Latest metrics",
        "",
    ]

    if report.live_agent_run:
        acc = (
            f"{report.accuracy:.1%} ({report.correct}/{report.answered})"
            if report.accuracy is not None
            else "n/a"
        )
        avg_r = (
            f"{report.average_retries:.2f}"
            if report.average_retries is not None
            else "n/a"
        )
        lines.extend(
            [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Questions | {report.question_count} |",
                f"| Accuracy | {acc} |",
                f"| Failed | {report.failed} |",
                f"| Average SQL retries | {avg_r} |",
                f"| LLM configured | {report.llm_configured} |",
                "",
            ]
        )
        failed = [c for c in report.cases if c.passed is False]
        if failed:
            lines.append("#### Failed cases")
            lines.append("")
            for c in failed:
                lines.append(
                    f"- `{c.id}`: {c.question} — expected `{c.expected_display}`; "
                    f"reason: {c.grade_reason or c.error or 'unknown'}"
                )
            lines.append("")
        else:
            lines.append("All live cases passed.")
            lines.append("")
    else:
        lines.extend(
            [
                "Live agent scoring was **not** run (no Gemini / Anthropic / OpenAI "
                "key in `.env`). Question bank + SQL ground truth are shipped; "
                "re-run `make eval` after adding keys to fill accuracy / retries.",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Questions | {report.question_count} |",
                f"| Accuracy | _pending live run_ |",
                f"| Failed | _pending live run_ |",
                f"| Average SQL retries | _pending live run_ |",
                f"| LLM configured | {report.llm_configured} |",
                "",
            ]
        )

    lines.extend(
        [
            "### Question bank (SQL ground truth)",
            "",
            "| ID | Intent | Question | Expected (from DuckDB) |",
            "|----|--------|----------|------------------------|",
        ]
    )
    for c in report.cases:
        q = c.question.replace("|", "\\|")
        exp = c.expected_display.replace("|", "\\|")
        lines.append(f"| `{c.id}` | {c.intent} | {q} | `{exp}` |")
    lines.append("")
    return "\n".join(lines)


def _compact_sql(sql: str) -> str:
    return " ".join(line.strip() for line in sql.strip().splitlines() if line.strip())
