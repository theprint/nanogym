"""Export a NanoGym run into browser-friendly float32 weights.

The browser engine (src/web_export/model.js) reads three files:
  weights.bin  every tensor it needs, float32, concatenated
  model.json   architecture plus each tensor's offset/length/shape
  vocab.json   the run's frozen character vocabulary

``export_run`` also writes training.json: where the model came from (dataset
file name, training settings, checkpoint stats). It never includes local paths.

``export_run`` also copies a minimal standalone test page next to them, so the
exported folder works on any static web host with no Python at runtime.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.config import load_config

TEMPLATE_DIR = Path(__file__).resolve().parent / "web_export"
TEMPLATE_FILES = ("index.html", "model.js", "README.md")


def tensor_names(n_layer: int) -> list[str]:
    names = ["transformer.wte.weight", "transformer.wpe.weight"]
    for index in range(n_layer):
        prefix = f"transformer.h.{index}"
        names.extend([
            f"{prefix}.ln_1.weight", f"{prefix}.ln_1.bias",
            f"{prefix}.attn.c_attn.weight", f"{prefix}.attn.c_attn.bias",
            f"{prefix}.attn.c_proj.weight", f"{prefix}.attn.c_proj.bias",
            f"{prefix}.ln_2.weight", f"{prefix}.ln_2.bias",
            f"{prefix}.mlp.c_fc.weight", f"{prefix}.mlp.c_fc.bias",
            f"{prefix}.mlp.c_proj.weight", f"{prefix}.mlp.c_proj.bias",
        ])
    names.extend(["transformer.ln_f.weight", "transformer.ln_f.bias"])
    return names


def pick_checkpoint(run_dir: Path) -> Path:
    """Prefer the best-validation checkpoint, fall back to the latest one."""
    for name in ("ckpt_best.pt", "ckpt_latest.pt"):
        if (run_dir / name).exists():
            return run_dir / name
    raise FileNotFoundError(f"{run_dir.name} has no checkpoint to export")


def load_checkpoint(checkpoint: Path) -> dict[str, Any]:
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def write_model_files(checkpoint: Path, vocab: Path, output: Path, source_run: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write weights.bin, model.json and vocab.json into ``output``; return the metadata."""
    output.mkdir(parents=True, exist_ok=True)
    payload = payload if payload is not None else load_checkpoint(checkpoint)
    state = payload["model"]
    config = dict(payload["model_config"])
    if not config.get("bias", True):
        raise ValueError("the browser engine expects bias terms; this run was trained with bias: false")

    descriptors: dict[str, dict[str, Any]] = {}
    offset = 0
    with (output / "weights.bin").open("wb") as handle:
        for name in tensor_names(config["n_layer"]):
            values = state[name].detach().float().contiguous().numpy().reshape(-1)
            handle.write(values.tobytes())
            descriptors[name] = {"offset": offset, "length": int(values.size), "shape": list(state[name].shape)}
            offset += int(values.size)

    metadata = {**config, "parameter_count": offset, "source_run": source_run, "checkpoint": checkpoint.name, "tensors": descriptors}
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(vocab, output / "vocab.json")
    return metadata


def _last_run_started(run_dir: Path) -> dict[str, Any]:
    """Fields of the most recent run_started event (a resumed run logs one per start)."""
    found: dict[str, Any] = {}
    events = run_dir / "events.log"
    if events.exists():
        for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
            _, _, fields = line.partition("] run_started ")
            if fields:
                try:
                    found = json.loads(fields)
                except json.JSONDecodeError:
                    pass
    return found


def training_metadata(run_dir: Path, checkpoint: Path, payload: dict[str, Any], parameters: int) -> dict[str, Any]:
    """Describe how the exported model was trained. Only file names, never local paths."""
    config = load_config(run_dir / "config.yaml")
    started = _last_run_started(run_dir)
    dataset = started.get("dataset", {})
    model_config = payload["model_config"]
    groups = (payload.get("optimizer") or {}).get("param_groups") or [{}]
    best_val_loss = payload.get("best_val_loss")
    return {
        "run": run_dir.name,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exported_with": "NanoGym (https://github.com/theprint/nanogym)",
        "dataset": {
            "file": Path(str(config.data.path).replace("\\", "/")).name,
            "characters": dataset.get("characters"),
            "train_characters": dataset.get("train_characters"),
            "val_characters": dataset.get("val_characters"),
            "val_fraction": config.data.val_fraction,
            "vocabulary_size": model_config["vocab_size"],
            "extend_vocab_from_data": config.data.vocab.extend_from_data,
        },
        "model": {
            "preset": config.model.preset,
            "n_layer": model_config["n_layer"], "n_embd": model_config["n_embd"], "n_head": model_config["n_head"],
            "context_chars": model_config["block_size"], "dropout": model_config["dropout"], "bias": model_config["bias"],
            "parameters": parameters,
        },
        "training": {
            "max_steps": config.train.max_steps, "batch_size": config.train.batch_size,
            "learning_rate": config.train.lr, "min_learning_rate": config.train.min_lr, "lr_schedule": config.train.lr_schedule,
            "warmup_steps": config.train.warmup_steps, "grad_clip": config.train.grad_clip,
            "optimizer": "AdamW", "betas": list(groups[0].get("betas", [])), "weight_decay": max((group.get("weight_decay", 0.0) for group in groups), default=None),
            "seed": config.train.seed, "device": started.get("device") or config.train.device,
        },
        "checkpoint": {
            "file": checkpoint.name,
            "step": payload.get("step"),
            "best_val_loss": best_val_loss,
            "best_val_bits_per_char": None if best_val_loss is None else best_val_loss / math.log(2),
            "tokens_seen": payload.get("tokens_seen"),
            "training_seconds": payload.get("wall_time"),
        },
    }


def export_run(run_dir: Path, output: Path) -> dict[str, Any]:
    """Export a run's model, its training metadata and the standalone test page into ``output``."""
    run_dir = Path(run_dir)
    checkpoint = pick_checkpoint(run_dir)
    payload = load_checkpoint(checkpoint)
    metadata = write_model_files(checkpoint, run_dir / "vocab.json", output, run_dir.name, payload)
    training = training_metadata(run_dir, checkpoint, payload, metadata["parameter_count"])
    (output / "training.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    for name in TEMPLATE_FILES:
        shutil.copyfile(TEMPLATE_DIR / name, output / name)
    files = sorted(path.name for path in output.iterdir() if path.is_file())
    return {"run": run_dir.name, "output": str(output), "files": files, "bytes": sum((output / name).stat().st_size for name in files), "parameters": metadata["parameter_count"]}


def zip_folder(folder: Path, root_name: str) -> bytes:
    """Zip ``folder`` into memory with its files under ``root_name/``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                archive.write(path, f"{root_name}/{path.name}")
    return buffer.getvalue()
