"""Export a run to a standalone browser test page under exports/<run>/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export import export_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a NanoGym run to a standalone browser test page")
    parser.add_argument("--run", required=True, help="run name under runs/, e.g. prompter-v0.4-Nano")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, help="output folder (default: exports/<run>/)")
    args = parser.parse_args()

    run_dir = args.runs_dir / args.run
    if not run_dir.is_dir():
        parser.error(f"no run folder at {run_dir}")
    result = export_run(run_dir, args.out or ROOT / "exports" / args.run)
    print(f"[nanogym] exported {result['parameters']:,} parameters from {result['run']} to {result['output']} ({result['bytes'] / 1e6:.1f} MB)")
    print("[nanogym] serve that folder with any static web server, e.g. `python -m http.server`, and open it in a browser")


if __name__ == "__main__":
    main()
