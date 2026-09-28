"""Training entry point: python scripts/train.py --config configs/small.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.trainer import Trainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or resume a NanoGym experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None, help="override max_steps when extending a run")
    args = parser.parse_args()
    overrides = {"train": {"max_steps": args.max_steps}} if args.max_steps is not None else None
    config = load_config(args.config, overrides)
    if args.resume:
        frozen = Path(config.runs_dir) / config.model_name / "config.yaml"
        if frozen.exists():
            frozen_config = load_config(frozen)
            # The frozen run owns architecture/data settings. Only the explicit
            # extension knob is allowed to carry over from the new CLI config.
            frozen_config.train.max_steps = config.train.max_steps
            config = frozen_config
    Trainer(config, resume=args.resume).train()


if __name__ == "__main__":
    main()
