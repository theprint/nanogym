"""Deterministic character tokenizer with a frozen, per-run vocabulary."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
BASE_CHARSET = (
    string.ascii_lowercase + string.ascii_uppercase + string.digits
    + " .,!?;:'\"-()[]{}_/\\|@#$%^&*+=<>~`\n\r\t"
)


class CharTokenizer:
    def __init__(self, tokens: Iterable[str]):
        self.tokens = list(tokens)
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("vocabulary must start with the four reserved special tokens")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary contains duplicate tokens")
        self.stoi = {token: idx for idx, token in enumerate(self.tokens)}
        self.itos = dict(enumerate(self.tokens))
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = range(4)

    @classmethod
    def from_text(cls, text: str, extend_from_data: bool = False) -> "CharTokenizer":
        chars = list(BASE_CHARSET)
        if extend_from_data:
            chars.extend(sorted(set(text) - set(chars)))
        return cls(SPECIAL_TOKENS + list(dict.fromkeys(chars)))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload["tokens"] if isinstance(payload, dict) else payload)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tokens": self.tokens, "size": len(self), "special_tokens": SPECIAL_TOKENS}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(char, self.unk_id) for char in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        special = set(SPECIAL_TOKENS)
        chars: list[str] = []
        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")
            if skip_special and token in special:
                continue
            chars.append(token)
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.tokens)

    def coverage(self, text: str) -> dict[str, object]:
        counts: dict[str, int] = {}
        for char in text:
            counts[char] = counts.get(char, 0) + 1
        outside = {char: count for char, count in counts.items() if char not in self.stoi}
        return {
            "characters": len(text), "unique_characters": len(counts), "outside_base_charset": outside,
            "coverage_fraction": 1.0 if not text else 1.0 - sum(outside.values()) / len(text),
            "top_characters": sorted(counts.items(), key=lambda item: item[1], reverse=True)[:30],
        }

