"""Inspect a text corpus before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenizer import CharTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Report charset coverage and corpus statistics")
    parser.add_argument("path", type=Path)
    parser.add_argument("--extend-vocab", action="store_true", help="include out-of-base characters in the proposed vocab")
    args = parser.parse_args()
    text = args.path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text, extend_from_data=args.extend_vocab)
    report = tokenizer.coverage(text)
    report.update({"path": str(args.path.resolve()), "vocabulary_size_if_used": len(tokenizer), "lines": text.count("\n") + 1, "words_approx": len(text.split())})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

