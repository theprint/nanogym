"""Launch the local dashboard server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.app import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the NanoGym dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7669, type=int)
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    parser.add_argument("--allow-control", action="store_true", help="allow the UI to launch resume subprocesses")
    args = parser.parse_args()
    app = create_app(Path(args.runs_dir), ROOT, allow_control=args.allow_control)
    print(f"[nanogym] dashboard http://{args.host}:{args.port} · controls {'enabled' if args.allow_control else 'view-only'}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
