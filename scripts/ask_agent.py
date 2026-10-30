#!/usr/bin/env python3
"""Ask the LangGraph shipping agent a single question (Step 5)."""

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


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Run one question through the shipping LangGraph agent."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural-language question about shipments / invoices",
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

    if not args.question:
        parser.error("question is required unless --print-diagram is set")

    if not llm_configured():
        print(
            "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "(and optionally LLM_PROVIDER) in .env — see .env.example.",
            file=sys.stderr,
        )
        return 2

    try:
        answer = ask(args.question)
    except LlmConfigError as exc:
        print(f"LLM config error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"{exc}\nRun `make etl` first.", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
