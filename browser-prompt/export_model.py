"""Load a NanoGym run's weights into this prompt-writer app.

This writes weights.bin, model.json and vocab.json into this folder (or --output),
replacing whichever model was exported before. For a generic standalone test
page for any run, use scripts/export.py instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export import write_model_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Run name under runs/, e.g. prompter-v0.4-Nano")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path; overrides --run")
    parser.add_argument("--vocab", type=Path, help="vocab.json path; overrides --run")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()

    if not args.checkpoint or not args.vocab:
        if not args.run:
            parser.error("pass --run <name> (a folder under runs/), or both --checkpoint and --vocab")
        run_dir = ROOT / "runs" / args.run
        args.checkpoint = args.checkpoint or run_dir / "ckpt_best.pt"
        args.vocab = args.vocab or run_dir / "vocab.json"

    source_run = args.run or args.checkpoint.resolve().parent.name
    metadata = write_model_files(args.checkpoint, args.vocab, args.output, source_run)
    print(f"Exported {metadata['parameter_count']:,} float32 parameters from {source_run} to {args.output / 'weights.bin'}")


if __name__ == "__main__":
    main()
