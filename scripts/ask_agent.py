#!/usr/bin/env python3
"""CLI for the LangGraph shipping agent (Step 5/7).

Examples:
  python scripts/ask_agent.py "How many shipments were delivered to Germany?"
  python scripts/ask_agent.py --interactive
  python scripts/ask_agent.py --print-diagram
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

from shipping_data_agent.agent import ask, export_graph_mermaid, llm_configured
from shipping_data_agent.agent.llm import LlmConfigError


KEY_HINT = (
    "No LLM API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
    "(LLM_PROVIDER=gemini), ANTHROPIC_API_KEY, or OPENAI_API_KEY in .env — "
    "see .env.example."
)


def _answer_one(question: str) -> int:
    try:
        print(ask(question))
    except LlmConfigError as exc:
        print(f"LLM config error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"{exc}\nRun `make etl` first.", file=sys.stderr)
        return 1
    return 0


def _interactive() -> int:
    print("Shipping Data Agent (interactive). Empty line or Ctrl-D to quit.")
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        code = _answer_one(question)
        if code != 0:
            return code


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Ask the shipping LangGraph agent a question (CLI)."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural-language question about shipments / invoices",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Read questions from stdin in a loop",
    )
    parser.add_argument(
        "--print-diagram",
        action="store_true",
        help="Print the agent Mermaid diagram and exit",
    )
    args = parser.parse_args(argv)

    if args.print_diagram:
        print(export_graph_mermaid())
        return 0

    if not args.question and not args.interactive:
        parser.error("provide a question, or use --interactive / --print-diagram")

    if not llm_configured():
        print(KEY_HINT, file=sys.stderr)
        return 2

    if args.interactive:
        if args.question:
            code = _answer_one(args.question)
            if code != 0:
                return code
        return _interactive()

    return _answer_one(args.question)


if __name__ == "__main__":
    raise SystemExit(main())
