"""CLI entry point: `uv run log-analyzer` or `python -m log_analyzer.main`."""

from __future__ import annotations

import argparse
import json
import logging
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log each step of the analysis (incl. the Claude tool loop) to stderr.",
    )
    args = parser.parse_args()

    if args.verbose:
        # Show our package's per-step DEBUG logs, but keep noisy third-party
        # loggers (httpx, anthropic) quiet by leaving the root level at WARNING.
        logging.basicConfig(
            level=logging.WARNING,
            stream=sys.stderr,
            format="%(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger("log_analyzer").setLevel(logging.DEBUG)

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

    validation = run.validation
    print(
        f"# provider={run.provider} model={run.model} "
        f"tools={'on' if validation.tools_used else 'off'} "
        f"sampled={run.sample_count} line(s) "
        f"(LineIds: {', '.join(run.sampled_line_ids)})",
        file=sys.stderr,
    )

    status = "PASSED" if validation.is_valid else "FAILED"
    print(
        f"# validation={status} "
        f"grounded={len(validation.grounded_line_ids)}/{len(validation.cited_line_ids)} "
        f"cited evidence LineId(s)",
        file=sys.stderr,
    )
    for issue in validation.issues:
        print(f"#   - {issue}", file=sys.stderr)

    print(
        json.dumps(
            {
                "analysis": run.analysis.model_dump(),
                "validation": {
                    "is_valid": validation.is_valid,
                    "tools_used": validation.tools_used,
                    "cited_line_ids": validation.cited_line_ids,
                    "grounded_line_ids": validation.grounded_line_ids,
                    "out_of_sample_line_ids": validation.out_of_sample_line_ids,
                    "unknown_line_ids": validation.unknown_line_ids,
                    "issues": validation.issues,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
