"""Training loop with JSONL observability and resumable, atomic checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig, config_hash, save_config
from .dataset import CharDataset
from .metrics import MetricsLogger
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


def _replace_with_retry(source: Path, destination: Path, attempts: int = 40, delay: float = 0.05) -> None:
    """Atomically move ``source`` onto ``destination``, tolerating Windows sharing violations.

    On Windows ``os.replace`` fails with ``PermissionError`` (WinError 5) when another
    process has the destination open at that instant. The dashboard server polls
    ``status.json`` every second and reads checkpoints from the Playground, and fast
    (small) models rewrite these files many times per second, so a collision that would
    otherwise crash the whole run is likely. Retry briefly before giving up.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


class Trainer:
    def __init__(self, config: ExperimentConfig, resume: bool = False):
        self.config = config
        self.config.validate()
        self.resume = resume
        self.run_dir = Path(config.runs_dir) / config.model_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(self.run_dir)
        self.request_stop = False
        self.started_at = time.time()
        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        self.wall_time = 0.0
        random.seed(config.train.seed)
        np.random.seed(config.train.seed)
        torch.manual_seed(config.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.train.seed)

        data_path = Path(config.data.path)
        if (self.run_dir / "vocab.json").exists():
            self.tokenizer = CharTokenizer.load(self.run_dir / "vocab.json")
        else:
            text = data_path.read_text(encoding="utf-8")
            self.tokenizer = CharTokenizer.from_text(text, extend_from_data=config.data.vocab.extend_from_data)
            self.tokenizer.save(self.run_dir / "vocab.json")
        if not (self.run_dir / "config.yaml").exists():
            save_config(config, self.run_dir / "config.yaml")
        self.dataset = CharDataset(data_path, self.tokenizer, config.data.val_fraction)
        self.device = self._resolve_device(config.train.device)
        self.dtype, self.amp_enabled = self._resolve_dtype(config.train.dtype)
        model_values = config.model.resolved()
        self.model_config = GPTConfig(vocab_size=len(self.tokenizer), **model_values)
        self.model = GPT(self.model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self._parameter_groups(), lr=config.train.lr, betas=(0.9, 0.95))
        self._install_signal_handlers()
        if resume:
            self._load_checkpoint(self.run_dir / "ckpt_latest.pt")
        else:
            self.logger.event(
                "run_started", model_name=config.model_name, device=str(self.device),
                parameters=self.model.parameter_count(), dataset=self.dataset.stats.__dict__, config_hash=config_hash(config),
            )
        self._write_status("running")

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("train.device=cuda was requested but CUDA is unavailable")
        mps_backend = getattr(torch.backends, "mps", None)
        if requested == "mps" and (mps_backend is None or not mps_backend.is_available()):
            raise RuntimeError("train.device=mps was requested but MPS is unavailable")
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if mps_backend is not None and mps_backend.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested)

    def _resolve_dtype(self, requested: str) -> tuple[torch.dtype, bool]:
        if self.device.type != "cuda":
            return torch.float32, False
        if requested == "float16" or (requested == "auto" and torch.cuda.is_available()):
            return torch.float16, True
        if requested == "bfloat16":
            return torch.bfloat16, True
        return torch.float32, False

    def _parameter_groups(self) -> list[dict[str, Any]]:
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.dim() >= 2 and not name.endswith("bias"):
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        return [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}]

    def _install_signal_handlers(self) -> None:
        def handler(_signum: int, _frame: Any) -> None:
            if self.request_stop:
                raise KeyboardInterrupt
            self.request_stop = True
            self.logger.event("pause_requested", reason="SIGINT")

        signal.signal(signal.SIGINT, handler)

    def _write_status(self, state: str, **extra: Any) -> None:
        existing: dict[str, Any] = {}
        status_path = self.run_dir / "status.json"
        if status_path.exists():
            try:
                existing = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        payload = {
            **existing,
            "state": state, "step": self.step, "max_steps": self.config.train.max_steps,
            "best_val_loss": None if math.isinf(self.best_val_loss) else self.best_val_loss,
            "tokens_seen": self.tokens_seen, "parameters": self.model.parameter_count(),
            "device": str(self.device), "updated_at": time.time(), **extra,
        }
        temp = self.run_dir / "status.json.tmp"
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _replace_with_retry(temp, self.run_dir / "status.json")

    def _learning_rate(self, step: int) -> float:
        train = self.config.train
        if step < train.warmup_steps:
            return train.lr * (step + 1) / max(1, train.warmup_steps)
        if train.lr_schedule == "constant":
            return train.lr
        progress = min(1.0, (step - train.warmup_steps) / max(1, train.max_steps - train.warmup_steps))
        return train.min_lr + 0.5 * (train.lr - train.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _set_learning_rate(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True)

    @torch.no_grad()
    def evaluate(self) -> tuple[dict[str, float], dict[str, float]]:
        self.model.eval()
        results: dict[str, float] = {}
        attention_stats: dict[str, float] = {}
        for split in ("train", "val"):
            losses: list[float] = []
            entropy_values: list[float] = []
            for iteration in range(self.config.train.eval_iters):
                x, y = self.dataset.get_batch(split, self.config.train.batch_size, self.model_config.block_size, self.device)
                need_attention = self.config.log.attention_stats and iteration == 0
                with self._autocast():
                    _, loss, attentions = self.model(x, y, return_attentions=need_attention)
                losses.append(float(loss.detach()))
                if need_attention:
                    for layer, weights in enumerate(attentions):
                        entropy = -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(dim=-1).mean()
                        entropy_values.append(float(entropy))
            results[f"{split}_loss"] = float(np.mean(losses))
            if entropy_values:
                attention_stats[f"attention_entropy_{split}"] = float(np.mean(entropy_values))
        results["train_bpc"] = results["train_loss"] / math.log(2)
        results["val_bpc"] = results["val_loss"] / math.log(2)
        results["train_val_gap"] = results["val_loss"] - results["train_loss"]
        self.model.train()
        return results, attention_stats

    def _checkpoint_payload(self) -> dict[str, Any]:
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "step": self.step,
            "best_val_loss": self.best_val_loss, "tokens_seen": self.tokens_seen, "wall_time": self.wall_time + (time.time() - self.started_at),
            "model_config": self.model_config.__dict__, "vocab_size": len(self.tokenizer), "config_hash": config_hash(self.config),
            "rng_torch": torch.get_rng_state(), "rng_cuda": cuda_state, "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        }

    def save_checkpoint(self, best: bool = False) -> None:
        destination = self.run_dir / ("ckpt_best.pt" if best else "ckpt_latest.pt")
        temp = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self._checkpoint_payload(), temp)
        _replace_with_retry(temp, destination)
        self.logger.event("checkpoint_written", path=destination.name, step=self.step, best=best)

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        expected = self.model_config.__dict__
        if checkpoint.get("model_config") != expected or checkpoint.get("vocab_size") != len(self.tokenizer):
            raise ValueError("checkpoint architecture or vocabulary does not match the current run")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = int(checkpoint.get("step", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.tokens_seen = int(checkpoint.get("tokens_seen", 0))
        self.wall_time = float(checkpoint.get("wall_time", 0.0))
        # RNG state must live on the CPU as a ByteTensor. map_location above can
        # move it onto the training device (e.g. cuda), so coerce it back here;
        # otherwise resuming on the GPU fails with "RNG state must be a torch.ByteTensor".
        if checkpoint.get("rng_torch") is not None:
            torch.set_rng_state(checkpoint["rng_torch"].to("cpu", torch.uint8))
        if checkpoint.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.to("cpu", torch.uint8) for state in checkpoint["rng_cuda"]])
        if checkpoint.get("rng_python") is not None:
            random.setstate(checkpoint["rng_python"])
        if checkpoint.get("rng_numpy") is not None:
            np.random.set_state(checkpoint["rng_numpy"])
        self.logger.event("run_resumed", step=self.step)

    def _control_command(self) -> str | None:
        path = self.run_dir / "control.json"
        if not path.exists():
            return None
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("command"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_sample(self) -> None:
        self.model.eval()
        for prompt in self.config.log.prompts:
            seed = self.config.train.seed + self.step
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            encoded = self.tokenizer.encode(prompt, add_bos=True)
            context = torch.tensor([encoded], dtype=torch.long, device=self.device)
            generated = self.model.generate(context, self.config.log.sample_tokens, temperature=self.config.log.temperature, stop_id=self.tokenizer.eos_id)
            text = self.tokenizer.decode(generated[0].tolist())
            self.logger.sample({"step": self.step, "t": time.time(), "prompt": prompt, "text": text, "temperature": self.config.log.temperature})
        self.model.train()

    def train(self) -> None:
        self.model.train()
        amp_enabled = self.amp_enabled and self.dtype == torch.float16
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            except TypeError:
                scaler = torch.amp.GradScaler(enabled=amp_enabled)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        progress_interval = max(self.config.train.log_interval, min(self.config.train.eval_interval, 100))
        print(f"[nanogym] training {self.config.model_name} on {self.device} for {self.config.train.max_steps} steps", flush=True)
        try:
            while self.step < self.config.train.max_steps:
                command = self._control_command()
                if self.request_stop or command in {"pause", "stop"}:
                    state = "stopped" if command == "stop" else "paused"
                    self.logger.event(f"{state}_completed", step=self.step)
                    self.save_checkpoint()
                    self._write_status(state)
                    return
                x, y = self.dataset.get_batch("train", self.config.train.batch_size, self.model_config.block_size, self.device)
                lr = self._learning_rate(self.step)
                self._set_learning_rate(lr)
                start = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    _, loss, _ = self.model(x, y)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip))
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip))
                    self.optimizer.step()
                self.tokens_seen += x.numel()
                elapsed = max(1e-6, time.perf_counter() - start)
                self.step += 1
                loss_value = float(loss.detach())
                if self.step % self.config.train.log_interval == 0 or self.step == 1:
                    self.logger.metric({
                        "step": self.step, "t": time.time(), "split": "train", "loss": loss_value,
                        "bpc": loss_value / math.log(2), "lr": lr, "grad_norm": grad_norm,
                        "tok_per_sec": x.numel() / elapsed, "tokens_seen": self.tokens_seen,
                        "mem_mb": torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
                    })
                if self.step % progress_interval == 0 or self.step == 1:
                    print(f"[nanogym] step {self.step:>6}/{self.config.train.max_steps}  loss {loss_value:.4f}  bpc {loss_value / math.log(2):.3f}  lr {lr:.2e}  tok/s {x.numel() / elapsed:.0f}", flush=True)
                if self.step % self.config.train.eval_interval == 0 or self.step == self.config.train.max_steps:
                    values, attention = self.evaluate()
                    event = {"step": self.step, "t": time.time(), "split": "val", "loss": values["val_loss"], "bpc": values["val_bpc"], "train_loss": values["train_loss"], "train_bpc": values["train_bpc"], "train_val_gap": values["train_val_gap"], "lr": lr, "tokens_seen": self.tokens_seen}
                    event.update(attention)
                    self.logger.metric(event)
                    if values["val_loss"] < self.best_val_loss:
                        self.best_val_loss = values["val_loss"]
                        self.save_checkpoint(best=True)
                    print(f"[nanogym] eval  {self.step:>6}/{self.config.train.max_steps}  train {values['train_loss']:.4f}  val {values['val_loss']:.4f}  gap {values['train_val_gap']:+.4f}", flush=True)
                    self._write_status("running", val_loss=values["val_loss"], train_loss=values["train_loss"])
                if self.step % self.config.train.sample_interval == 0 or self.step == self.config.train.max_steps:
                    self._write_sample()
                if self.step % self.config.train.checkpoint_interval == 0:
                    self.save_checkpoint()
                self._write_status("running")
            self.save_checkpoint()
            self.logger.event("run_finished", step=self.step, best_val_loss=self.best_val_loss)
            self._write_status("finished", finished_at=time.time())
        except KeyboardInterrupt:
            self.logger.event("run_interrupted", step=self.step)
            self.save_checkpoint()
            self._write_status("paused")
            raise
        except Exception as exc:
            self.logger.event("run_failed", step=self.step, error=repr(exc))
            self._write_status("failed", error=str(exc))
            raise
