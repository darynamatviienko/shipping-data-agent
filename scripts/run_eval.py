#!/usr/bin/env python3
"""Run Step 6 evaluation: SQL ground truth + optional live agent scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

from shipping_data_agent.eval.harness import (
    DEFAULT_RESULTS_PATH,
    format_readme_section,
    run_eval,
    save_results,
)


def _replace_readme_section(readme: Path, section: str) -> None:
    """Insert or replace the ## Evaluation (Step 6) section in README.md."""
    text = readme.read_text() if readme.exists() else ""
    teaser = "Steps 6–7 (evaluation harness, Streamlit UI) come later."
    if teaser in text:
        text = text.replace(teaser + "\n", "").replace(teaser, "")

    start_marker = "## Evaluation (Step 6)"
    if start_marker in text:
        start_idx = text.index(start_marker)
        after_start = text[start_idx + len(start_marker) :]
        next_offset = None
        cursor = 0
        for line in after_start.splitlines(keepends=True):
            if line.startswith("## ") and not line.startswith(start_marker):
                next_offset = cursor
                break
            cursor += len(line)
        before = text[:start_idx].rstrip() + "\n\n"
        after = after_start[next_offset:] if next_offset is not None else ""
        text = before + section.rstrip() + "\n" + after
    else:
        text = text.rstrip() + "\n\n" + section.rstrip() + "\n"

    readme.write_text(text if text.endswith("\n") else text + "\n")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Evaluate the shipping agent on 15 SQL-backed questions."
    )
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Only compute DuckDB answers; do not call the LLM agent",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live agent scoring (fails if API keys are missing)",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Rewrite the Evaluation section in README.md",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"JSON output path (default: {DEFAULT_RESULTS_PATH})",
    )
    args = parser.parse_args(argv)

    if args.ground_truth_only and args.live:
        parser.error("use only one of --ground-truth-only / --live")

    live: bool | None
    if args.ground_truth_only:
        live = False
    elif args.live:
        live = True
    else:
        live = None  # auto: live when keys exist

    try:
        report = run_eval(live=live)
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    out = save_results(report, args.results)
    section = format_readme_section(report)
    if args.update_readme:
        _replace_readme_section(ROOT / "README.md", section)

    print(section)
    print(f"\nWrote {out}")
    if args.update_readme:
        print(f"Updated {ROOT / 'README.md'}")

    if report.live_agent_run and report.failed:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
