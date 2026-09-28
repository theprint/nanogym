"""Configuration loading and validation for NanoGym experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"n_layer": 4, "n_embd": 128, "n_head": 4},
    "small": {"n_layer": 6, "n_embd": 256, "n_head": 8},
    "medium": {"n_layer": 8, "n_embd": 384, "n_head": 12},
}


@dataclass
class VocabConfig:
    extend_from_data: bool = False


@dataclass
class DataConfig:
    path: str = "data/input.txt"
    val_fraction: float = 0.05
    vocab: VocabConfig = field(default_factory=VocabConfig)


@dataclass
class ModelConfig:
    preset: str = "small"
    block_size: int = 256
    n_layer: int | None = None
    n_embd: int | None = None
    n_head: int | None = None
    dropout: float = 0.15
    bias: bool = True

    def resolved(self) -> dict[str, Any]:
        if self.preset not in PRESETS:
            raise ValueError(f"unknown model preset {self.preset!r}; choose one of {sorted(PRESETS)}")
        values = dict(PRESETS[self.preset])
        for key in ("n_layer", "n_embd", "n_head"):
            value = getattr(self, key)
            if value is not None:
                values[key] = value
        values.update({"block_size": self.block_size, "dropout": self.dropout, "bias": self.bias})
        return values


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_steps: int = 20_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    lr_schedule: str = "cosine"
    warmup_steps: int = 200
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 10
    sample_interval: int = 500
    checkpoint_interval: int = 250
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    num_workers: int = 0


@dataclass
class LogConfig:
    attention_stats: bool = False
    prompts: list[str] = field(default_factory=lambda: ["", "<bos>"])
    sample_tokens: int = 300
    temperature: float = 0.8


@dataclass
class ExperimentConfig:
    model_name: str = "nanogym-run"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
    runs_dir: str = "runs"

    def validate(self) -> None:
        if not self.model_name or any(part in self.model_name for part in ("/", "\\", "..")):
            raise ValueError("model_name must be a simple folder name")
        if not 0 < self.data.val_fraction < 0.5:
            raise ValueError("data.val_fraction must be between 0 and 0.5")
        model = self.model.resolved()
        if model["n_embd"] % model["n_head"] != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        if self.model.block_size < 8:
            raise ValueError("model.block_size must be at least 8")
        if not 0 <= self.model.dropout < 1:
            raise ValueError("model.dropout must be between 0 (inclusive) and 1")
        if self.train.max_steps < 1 or self.train.batch_size < 1:
            raise ValueError("train.max_steps and train.batch_size must be positive")
        for name in ("eval_interval", "eval_iters", "log_interval", "sample_interval", "checkpoint_interval"):
            if getattr(self.train, name) < 1:
                raise ValueError(f"train.{name} must be positive")
        if self.train.lr <= 0 or self.train.min_lr < 0:
            raise ValueError("train.lr must be positive and train.min_lr cannot be negative")
        if self.train.lr_schedule not in {"cosine", "constant"}:
            raise ValueError("train.lr_schedule must be cosine or constant")
        if self.train.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("train.device must be auto, cpu, cuda, or mps")
        if self.train.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("train.dtype must be auto, float32, float16, or bfloat16")
        if not 0 < self.log.temperature:
            raise ValueError("log.temperature must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _default_dict() -> dict[str, Any]:
    return ExperimentConfig().to_dict()


def _build_config(values: Mapping[str, Any]) -> ExperimentConfig:
    data = values.get("data", {}) or {}
    model = values.get("model", {}) or {}
    train = values.get("train", {}) or {}
    log = values.get("log", {}) or {}
    cfg = ExperimentConfig(
        model_name=str(values.get("model_name", "nanogym-run")),
        runs_dir=str(values.get("runs_dir", "runs")),
        data=DataConfig(
            path=str(data.get("path", "data/input.txt")),
            val_fraction=float(data.get("val_fraction", 0.05)),
            vocab=VocabConfig(extend_from_data=bool((data.get("vocab", {}) or {}).get("extend_from_data", False))),
        ),
        model=ModelConfig(
            preset=str(model.get("preset", "small")),
            block_size=int(model.get("block_size", 256)),
            n_layer=None if model.get("n_layer") is None else int(model["n_layer"]),
            n_embd=None if model.get("n_embd") is None else int(model["n_embd"]),
            n_head=None if model.get("n_head") is None else int(model["n_head"]),
            dropout=float(model.get("dropout", 0.15)),
            bias=bool(model.get("bias", True)),
        ),
        train=TrainConfig(
            batch_size=int(train.get("batch_size", 64)), max_steps=int(train.get("max_steps", 20_000)),
            lr=float(train.get("lr", 3e-4)), min_lr=float(train.get("min_lr", 3e-5)),
            lr_schedule=str(train.get("lr_schedule", "cosine")), warmup_steps=int(train.get("warmup_steps", 200)),
            grad_clip=float(train.get("grad_clip", 1.0)), eval_interval=int(train.get("eval_interval", 250)),
            eval_iters=int(train.get("eval_iters", 100)), log_interval=int(train.get("log_interval", 10)),
            sample_interval=int(train.get("sample_interval", 500)), checkpoint_interval=int(train.get("checkpoint_interval", 250)),
            seed=int(train.get("seed", 1337)), device=str(train.get("device", "auto")),
            dtype=str(train.get("dtype", "auto")), num_workers=int(train.get("num_workers", 0)),
        ),
        log=LogConfig(
            attention_stats=bool(log.get("attention_stats", False)),
            prompts=[str(prompt) for prompt in log.get("prompts", ["", "<bos>"])],
            sample_tokens=int(log.get("sample_tokens", 300)), temperature=float(log.get("temperature", 0.8)),
        ),
    )
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config, filling missing keys from the documented defaults."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    values = _deep_merge(_default_dict(), supplied)
    if overrides:
        values = _deep_merge(values, overrides)
    cfg = _build_config(values)
    # Preset configs live in <project>/configs. For ad-hoc configs elsewhere,
    # resolve relative paths from the caller's working directory.
    project_root = path.parent.parent if path.parent.name.lower() == "configs" else Path.cwd()
    cfg.data.path = str((project_root / cfg.data.path).resolve()) if not Path(cfg.data.path).is_absolute() else cfg.data.path
    cfg.runs_dir = str((project_root / cfg.runs_dir).resolve()) if not Path(cfg.runs_dir).is_absolute() else cfg.runs_dir
    return cfg


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
