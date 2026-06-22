"""CLI entry point: `uv run log-analyzer` or `python -m log_analyzer.main`."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from .analyzer import run_analysis
from .config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample random log lines and get a structured LLM analysis."
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=["claude", "gemini", "nvidia"],
        help="Override the provider from config (switch models on the fly).",
    )
    parser.add_argument(
        "--min-size", type=int, help="Override sampling.min_size for this run."
    )
    parser.add_argument(
        "--max-size", type=int, help="Override sampling.max_size for this run."
    )
    args = parser.parse_args()

    load_dotenv()  # pull API keys from a local .env if present

    try:
        config = load_config(args.config)
        if args.min_size is not None:
            config.sampling.min_size = args.min_size
        if args.max_size is not None:
            config.sampling.max_size = args.max_size
        if config.sampling.max_size < config.sampling.min_size:
            raise ValueError("max-size must be >= min-size")

        run = run_analysis(config, provider_name=args.provider)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the CLI user
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"# provider={run.provider} model={run.model} "
        f"sampled={run.sample_count} line(s) "
        f"(LineIds: {', '.join(run.sampled_line_ids)})",
        file=sys.stderr,
    )
    print(json.dumps(run.analysis.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
