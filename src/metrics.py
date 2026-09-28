"""Crash-tolerant JSONL metrics and sample logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class MetricsLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"
        self.events_path = self.run_dir / "events.log"

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()

    def metric(self, payload: dict[str, Any]) -> None:
        self._write(self.metrics_path, payload)

    def sample(self, payload: dict[str, Any]) -> None:
        self._write(self.samples_path, payload)

    def event(self, message: str, **fields: Any) -> None:
        stamp = __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")
        suffix = " " + json.dumps(fields, ensure_ascii=False, sort_keys=True) if fields else ""
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}{suffix}\n")
            handle.flush()


def read_jsonl(path: str | Path, since: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line_number < since:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(rows) >= limit:
                break
    return rows


def last_rows(path: str | Path, count: int = 1) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    return rows[-count:]

