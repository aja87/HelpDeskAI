from __future__ import annotations

import argparse
import logging
import os

from pathlib import Path

from helpdeskai.common.logging import init_logging
from helpdeskai.observability.config import LOG_FILE, VALID_ACTIONS, ObservabilityConfig
from helpdeskai.observability.workflow import run_observability_core


def _load_env_file(path: Path) -> None:
    """Load .env entries into process environment without overriding existing vars."""

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for observability workflows."""

    parser = argparse.ArgumentParser(description="Run the HelpDeskAI observability workflows")
    parser.add_argument("action", choices=sorted(VALID_ACTIONS), default="all", nargs="?")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--tracking-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--conversations-path", type=Path, default=None)
    parser.add_argument("--golden-path", type=Path, default=None)
    parser.add_argument("--monthly-budget", type=float, default=100.0)
    parser.add_argument("--continuous-sample-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> ObservabilityConfig:
    config = ObservabilityConfig(
        monthly_budget_usd=args.monthly_budget,
        continuous_sample_ratio=args.continuous_sample_ratio,
        seed=args.seed,
    )
    if args.tracking_uri:
        config.tracking_uri = args.tracking_uri
    if args.reports_dir is not None:
        config.reports_dir = args.reports_dir
        config.traces_dir = args.reports_dir / "traces"
    if args.conversations_path is not None:
        config.conversations_path = args.conversations_path
    if args.golden_path is not None:
        config.golden_path = args.golden_path
    return config


def main() -> None:
    """CLI entrypoint for local observability runs."""

    args = parse_args()
    _load_env_file(args.env_file)
    init_logging(log_file=LOG_FILE)

    payload = run_observability_core(_build_config(args), action=args.action)
    logging.info("Observability action completed: %s", payload["action"])


if __name__ == "__main__":
    main()
