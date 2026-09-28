"""Small in-memory character dataset suitable for fast local experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .tokenizer import CharTokenizer


@dataclass
class DatasetStats:
    path: str
    characters: int
    train_characters: int
    val_characters: int
    vocabulary_size: int
    val_fraction: float


class CharDataset:
    def __init__(self, path: str | Path, tokenizer: CharTokenizer, val_fraction: float = 0.05):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"dataset not found: {self.path}")
        text = self.path.read_text(encoding="utf-8")
        if len(text) < 16:
            raise ValueError("dataset must contain at least 16 characters")
        split = max(1, min(len(text) - 1, int(len(text) * (1.0 - val_fraction))))
        encoded = np.asarray(tokenizer.encode(text), dtype=np.int64)
        self.train = torch.from_numpy(encoded[:split])
        self.val = torch.from_numpy(encoded[split:])
        self.stats = DatasetStats(str(self.path), len(text), len(self.train), len(self.val), len(tokenizer), val_fraction)

    def get_batch(self, split: str, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        if len(data) <= block_size:
            raise ValueError(f"{split} split has {len(data)} characters; need more than block_size={block_size}")
        starts = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[start : start + block_size] for start in starts])
        y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

