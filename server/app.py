"""Local FastAPI dashboard and inference API.

The server is intentionally read-mostly. Training owns the run folder; the only
write endpoints are the explicit pause/resume controls.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

import torch
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import load_config
from src.export import export_run, zip_folder
from src.metrics import read_jsonl
from src.model import GPT, GPTConfig
from src.tokenizer import CharTokenizer


class ControlRequest(BaseModel):
    command: str
    device: str | None = None


class StartRunRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    dataset: str
    preset: str = Field(default="tiny")
    block_size: int = Field(default=64, ge=16, le=2048)
    max_steps: int = Field(default=250, ge=1, le=1_000_000)
    batch_size: int = Field(default=8, ge=1, le=1024)
    learning_rate: float = Field(default=6e-4, gt=0.0, le=1.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=0.5)
    seed: int = Field(default=1337, ge=0)
    device: str = Field(default="auto")
    extend_vocab: bool = False


class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    checkpoint: str = "best"
    temperature: float = Field(default=0.8, gt=0.0, le=5.0)
    top_k: int = Field(default=0, ge=0, le=200)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_new_chars: int = Field(default=300, ge=1, le=2000)
    seed: int = Field(default=1337, ge=0)
    honor_tags: bool = True


# The character tokenizer stores <bos>/<eos> as literal text in the corpus, so the
# model emits them as ordinary characters rather than reserved token ids. Honoring
# them therefore means post-processing the generated *string*: begin the visible
# output at a generated <bos> and stop it at the first <eos>.
BOS_MARK = "<bos>"
EOS_MARK = "<eos>"


def _honor_span(text: str) -> tuple[int, int]:
    """Return the [start, end) slice of text bounded by a leading <bos> and a trailing <eos>."""
    end = text.find(EOS_MARK)
    end = len(text) if end == -1 else end
    start = text.find(BOS_MARK, 0, end)
    start = 0 if start == -1 else start + len(BOS_MARK)
    # An undertrained model may open several entries in a row; skip the empty ones.
    while (following := text.find(BOS_MARK, start, end)) != -1 and not text[start:following].strip():
        start = following + len(BOS_MARK)
    return start, end


def _honor_tags_text(text: str, honor: bool) -> str:
    if not honor:
        return text
    start, end = _honor_span(text)
    return text[start:end]


def _safe_name(name: str) -> str:
    if not name or name in {".", ".."} or Path(name).name != name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid run name")
    return name


def _safe_dataset(data_root: Path, dataset: str) -> Path:
    if not dataset or Path(dataset).is_absolute():
        raise HTTPException(status_code=400, detail="dataset must be a relative path under data/")
    candidate = (data_root / dataset).resolve()
    try:
        candidate.relative_to(data_root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="dataset must stay inside data/") from exc
    if candidate.suffix.lower() not in {".txt", ".text"} or not candidate.is_file():
        raise HTTPException(status_code=400, detail="dataset must be an existing .txt file under data/")
    return candidate


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_available()


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default or {}


def _tail_lines(path: Path, limit: int = 12) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-limit:]
    except OSError:
        return []


def _run_summary(run_dir: Path) -> dict[str, Any]:
    status = _read_json(run_dir / "status.json")
    config: dict[str, Any] = {}
    config_path = run_dir / "config.yaml" if (run_dir / "config.yaml").exists() else run_dir / "launch.yaml"
    try:
        config = load_config(config_path).to_dict() if config_path.exists() else {}
    except Exception as exc:
        config = {"error": str(exc)}
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    val_rows = [row for row in metrics if row.get("split") == "val"]
    train_rows = [row for row in metrics if row.get("split") == "train"]
    latest = val_rows[-1] if val_rows else (train_rows[-1] if train_rows else {})
    process_lines = _tail_lines(run_dir / "process.log", 20)
    process_error = next((line for line in reversed(process_lines) if "RuntimeError:" in line or "Error:" in line), None)
    if not status:
        if process_error or any("Traceback" in line for line in process_lines):
            status = {"state": "failed", "error": process_error or "training process exited with an error"}
        elif (run_dir / "launch.yaml").exists() and not (run_dir / "ckpt_latest.pt").exists():
            status = {"state": "starting"}
        elif (run_dir / "ckpt_latest.pt").exists():
            status = {"state": "finished" if latest.get("step", 0) >= config.get("train", {}).get("max_steps", 0) else "paused"}
        else:
            status = {"state": "created"}
    activity_times = [run_dir.stat().st_mtime]
    for activity_path in (run_dir / "status.json", run_dir / "metrics.jsonl", run_dir / "events.log", run_dir / "process.log"):
        if activity_path.exists():
            activity_times.append(activity_path.stat().st_mtime)
    return {
        "name": run_dir.name,
        "state": status.get("state", "unknown"),
        "step": status.get("step", latest.get("step", 0)),
        "max_steps": status.get("max_steps", config.get("train", {}).get("max_steps")),
        "best_val_loss": status.get("best_val_loss", latest.get("loss") if latest.get("split") == "val" else None),
        "tokens_seen": status.get("tokens_seen", latest.get("tokens_seen", 0)),
        "parameters": status.get("parameters"),
        "device": status.get("device"),
        "updated_at": status.get("updated_at", run_dir.stat().st_mtime),
        "last_activity_at": max(activity_times),
        "pid": status.get("pid"),
        "dataset": status.get("dataset"),
        "preset": status.get("preset"),
        "error": status.get("error"),
        "config": config,
        "has_best": (run_dir / "ckpt_best.pt").exists(),
        "has_latest": (run_dir / "ckpt_latest.pt").exists(),
        "metrics_count": len(metrics),
    }


def _load_model(run_dir: Path, checkpoint: str) -> tuple[GPT, CharTokenizer, torch.device]:
    if checkpoint not in {"best", "latest"}:
        raise ValueError("checkpoint must be best or latest")
    config = load_config(run_dir / "config.yaml")
    tokenizer = CharTokenizer.load(run_dir / "vocab.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(GPTConfig(vocab_size=len(tokenizer), **config.model.resolved())).to(device)
    checkpoint_path = run_dir / ("ckpt_best.pt" if checkpoint == "best" else "ckpt_latest.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path.name}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer, device


def _generate(request: GenerateRequest, run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(request.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(request.seed)
    model, tokenizer, device = _load_model(run_dir, request.checkpoint)
    ids = torch.tensor([tokenizer.encode(request.prompt, add_bos=True)], dtype=torch.long, device=device)
    output = model.generate(ids, request.max_new_chars, request.temperature, request.top_k or None, request.top_p, tokenizer.eos_id)
    text = _honor_tags_text(tokenizer.decode(output[0].tolist()), request.honor_tags)
    return {"text": text, "prompt": request.prompt, "checkpoint": request.checkpoint, "seed": request.seed, "model": request.model}


def _generate_details(request: GenerateRequest, run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(request.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(request.seed)
    model, tokenizer, device = _load_model(run_dir, request.checkpoint)
    ids = torch.tensor([tokenizer.encode(request.prompt, add_bos=True)], dtype=torch.long, device=device)
    details: list[dict[str, Any]] = []
    for _ in range(request.max_new_chars):
        token_id, alternatives = model.next_token(ids, request.temperature, request.top_k or None, request.top_p)
        ids = torch.cat((ids, torch.tensor([[token_id]], dtype=torch.long, device=device)), dim=1)
        chosen = next((item for item in alternatives if item["id"] == token_id), {"probability": 0.0})
        details.append({"char": tokenizer.decode([token_id], skip_special=False), "probability": chosen["probability"], "alternatives": alternatives})
        if token_id == tokenizer.eos_id:
            break
        if request.honor_tags and EOS_MARK in "".join(item["char"] for item in details):
            break
    generated = "".join(item["char"] for item in details)
    full = request.prompt + generated
    text = _honor_tags_text(full, request.honor_tags)
    # Keep the confidence tokens aligned with the visible text. Details cover only the
    # generated characters (one per step), so map the honored window of the full string
    # back into that region and slice to it.
    if request.honor_tags and all(len(item["char"]) == 1 for item in details):
        start, end = _honor_span(full)
        offset = len(request.prompt)
        details = details[max(0, start - offset):max(0, end - offset)]
    return {"text": text, "prompt": request.prompt, "tokens": details, "model": request.model, "seed": request.seed}


class SameOriginGuard:
    """Reject cross-site requests that could change state or use the model.

    The dashboard is served from the same origin as the API, so it never needs CORS.
    Browsers still *send* some cross-site requests (and never apply CORS to WebSockets),
    so any other web page open in the browser could otherwise start or stop runs on this
    machine. Requests without an Origin header (curl, scripts) are allowed.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = {key.decode("latin-1"): value.decode("latin-1") for key, value in scope.get("headers", [])}
            origin, host = headers.get("origin"), headers.get("host", "")
            unsafe = scope["type"] == "websocket" or scope.get("method") not in ("GET", "HEAD", "OPTIONS")
            if unsafe and origin and urlsplit(origin).netloc != host:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await send({"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"application/json")]})
                    await send({"type": "http.response.body", "body": b'{"detail":"cross-origin request blocked"}'})
                return
        await self.app(scope, receive, send)


def create_app(runs_dir: str | Path = "runs", project_root: str | Path | None = None, allow_control: bool = False) -> FastAPI:
    runs_dir = Path(runs_dir).resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    data_root = (project_root / "data").resolve()
    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(title="NanoGym dashboard", version="0.1.0")
    app.state.runs_dir = runs_dir
    app.state.project_root = project_root
    app.state.allow_control = allow_control
    app.add_middleware(SameOriginGuard)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def revalidate_ui_assets(request, call_next):
        # Make browsers revalidate the UI (cheap 304s via ETag) so updates show up without a hard refresh.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def get_run(name: str) -> Path:
        name = _safe_name(name)
        path = runs_dir / name
        if not path.is_dir():
            raise HTTPException(status_code=404, detail="run not found")
        return path

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        cuda_available = torch.cuda.is_available()
        return {
            "ok": True, "runs_dir": str(runs_dir), "allow_control": allow_control,
            "cuda_available": cuda_available, "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
            "mps_available": _mps_available(),
        }

    @app.get("/api/runs")
    async def runs() -> list[dict[str, Any]]:
        runs_dir.mkdir(parents=True, exist_ok=True)
        return [_run_summary(path) for path in sorted(runs_dir.iterdir()) if path.is_dir() and not path.name.startswith(".")]

    @app.get("/api/datasets")
    async def datasets() -> list[dict[str, Any]]:
        data_root.mkdir(parents=True, exist_ok=True)
        result = []
        for path in sorted(data_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".text"}:
                relative = path.relative_to(data_root).as_posix()
                result.append({"name": relative, "size_bytes": path.stat().st_size, "modified_at": path.stat().st_mtime})
        return result

    @app.get("/api/runs/{name}")
    async def run(name: str) -> dict[str, Any]:
        return _run_summary(get_run(name))

    @app.get("/api/runs/{name}/metrics")
    async def metrics(name: str, since: int = 0, limit: int = 5000) -> dict[str, Any]:
        if since < 0 or limit < 1 or limit > 20_000:
            raise HTTPException(status_code=400, detail="invalid since or limit")
        rows = read_jsonl(get_run(name) / "metrics.jsonl", since=since, limit=limit)
        return {"rows": rows, "since": since, "next": since + len(rows)}

    @app.get("/api/runs/{name}/samples")
    async def samples(name: str, limit: int = 200) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="invalid limit")
        rows = read_jsonl(get_run(name) / "samples.jsonl")
        return rows[-limit:]

    @app.get("/api/runs/{name}/activity")
    async def activity(name: str) -> dict[str, Any]:
        run_dir = get_run(name)
        return {
            "status": _read_json(run_dir / "status.json"),
            "process_log": _tail_lines(run_dir / "process.log"),
            "events": _tail_lines(run_dir / "events.log"),
        }

    @app.post("/api/runs/{name}/control")
    async def control(name: str, request: ControlRequest) -> dict[str, Any]:
        run_dir = get_run(name)
        if request.command not in {"pause", "resume", "stop"}:
            raise HTTPException(status_code=400, detail="command must be pause, resume, or stop")
        if request.command in {"resume", "stop"} and not allow_control:
            raise HTTPException(status_code=403, detail="this control is disabled; restart with --allow-control")
        if request.command == "resume" and _read_json(run_dir / "status.json").get("state") == "running":
            raise HTTPException(status_code=409, detail="run is already running")
        control_path = run_dir / "control.json"
        if request.command == "resume":
            config_path = run_dir / "config.yaml"
            if not config_path.exists():
                raise HTTPException(status_code=400, detail="run has no frozen config")
            if not (run_dir / "ckpt_latest.pt").exists():
                raise HTTPException(status_code=409, detail="run has no latest checkpoint; start a new experiment instead")
            if request.device is not None:
                if request.device not in {"auto", "cpu", "cuda", "mps"}:
                    raise HTTPException(status_code=400, detail="device must be auto, cpu, cuda, or mps")
                if request.device == "cuda" and not torch.cuda.is_available():
                    raise HTTPException(status_code=400, detail="CUDA is unavailable because this environment has CPU-only PyTorch; choose auto or cpu")
                if request.device == "mps" and not _mps_available():
                    raise HTTPException(status_code=400, detail="MPS is unavailable in this environment; choose auto or cpu")
                try:
                    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                    raw_config.setdefault("train", {})["device"] = request.device
                    config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
                except OSError as exc:
                    raise HTTPException(status_code=500, detail=f"could not update run device: {exc}") from exc
            try:
                resume_config = load_config(config_path)
                if resume_config.train.device == "cuda" and not torch.cuda.is_available():
                    raise HTTPException(status_code=409, detail="this run requires CUDA, but the installed PyTorch is CPU-only; start a new run with device=auto or cpu")
                if resume_config.train.device == "mps" and not _mps_available():
                    raise HTTPException(status_code=409, detail="this run requires MPS, but MPS is unavailable in this environment")
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"could not validate run config: {exc}") from exc
            control_path.write_text(json.dumps({"command": request.command, "requested_at": time.time()}), encoding="utf-8")
            status_path = run_dir / "status.json"
            status = _read_json(status_path)
            status.update({"state": "starting", "updated_at": time.time()})
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            command = [sys.executable, str(project_root / "scripts" / "train.py"), "--config", str(config_path), "--resume"]
            process_log = run_dir / "process.log"
            try:
                with process_log.open("a", encoding="utf-8") as log_handle:
                    process = subprocess.Popen(command, cwd=str(project_root), stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT)
            except OSError as exc:
                status.update({"state": "failed", "error": str(exc), "updated_at": time.time()})
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                raise HTTPException(status_code=500, detail=f"could not start resume process: {exc}") from exc
            status.update({"pid": process.pid, "updated_at": time.time()})
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            return {"ok": True, "command": request.command, "pid": process.pid}
        control_path.write_text(json.dumps({"command": request.command, "requested_at": time.time()}), encoding="utf-8")
        return {"ok": True, "command": request.command}

    @app.post("/api/experiments")
    async def start_experiment(request: StartRunRequest) -> dict[str, Any]:
        if not allow_control:
            raise HTTPException(status_code=403, detail="starting experiments is disabled; restart with --allow-control")
        try:
            name = _safe_name(request.name)
        except HTTPException:
            raise
        if request.preset not in {"tiny", "small", "medium"}:
            raise HTTPException(status_code=400, detail="preset must be tiny, small, or medium")
        if request.device not in {"auto", "cpu", "cuda", "mps"}:
            raise HTTPException(status_code=400, detail="device must be auto, cpu, cuda, or mps")
        if request.device == "cuda" and not torch.cuda.is_available():
            raise HTTPException(status_code=400, detail="CUDA is unavailable because this environment has CPU-only PyTorch; choose auto or cpu")
        if request.device == "mps" and not _mps_available():
            raise HTTPException(status_code=400, detail="MPS is unavailable in this environment; choose auto or cpu")
        dataset_path = _safe_dataset(data_root, request.dataset)
        run_dir = runs_dir / name
        if run_dir.exists():
            raise HTTPException(status_code=409, detail="a run with that name already exists; choose a new name to start over")
        run_dir.mkdir(parents=True, exist_ok=False)
        config_path = run_dir / "launch.yaml"
        config = {
            "model_name": name,
            "runs_dir": str(runs_dir),
            "data": {"path": str(dataset_path), "val_fraction": request.val_fraction, "vocab": {"extend_from_data": request.extend_vocab}},
            "model": {"preset": request.preset, "block_size": request.block_size, "dropout": request.dropout, "bias": True},
            "train": {"batch_size": request.batch_size, "max_steps": request.max_steps, "lr": request.learning_rate, "min_lr": request.learning_rate * 0.1, "lr_schedule": "cosine", "warmup_steps": min(200, max(1, request.max_steps // 10)), "grad_clip": 1.0, "eval_interval": max(1, min(250, request.max_steps // 10)), "eval_iters": 25, "log_interval": 10, "sample_interval": max(1, min(500, request.max_steps // 2)), "checkpoint_interval": max(1, min(250, request.max_steps // 10)), "seed": request.seed, "device": request.device, "dtype": "auto", "num_workers": 0},
            "log": {"attention_stats": False, "prompts": ["", "<bos>"], "sample_tokens": 200, "temperature": 0.8},
        }
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        process_log = run_dir / "process.log"
        command = [sys.executable, str(project_root / "scripts" / "train.py"), "--config", str(config_path)]
        status_path = run_dir / "status.json"
        status_path.write_text(json.dumps({"state": "starting", "step": 0, "max_steps": request.max_steps, "tokens_seen": 0, "dataset": request.dataset, "preset": request.preset, "device": request.device, "started_at": time.time(), "updated_at": time.time()}, indent=2), encoding="utf-8")
        try:
            with process_log.open("a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(command, cwd=str(project_root), stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT)
        except OSError as exc:
            status_path.write_text(json.dumps({"state": "failed", "error": str(exc), "updated_at": time.time()}, indent=2), encoding="utf-8")
            raise HTTPException(status_code=500, detail=f"could not start training process: {exc}") from exc
        status = _read_json(status_path)
        status.update({"pid": process.pid, "updated_at": time.time()})
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return {"ok": True, "name": name, "pid": process.pid, "dataset": request.dataset, "preset": request.preset}

    @app.post("/api/runs/{name}/export")
    def export(name: str) -> Response:
        # Sync route: FastAPI runs it in a worker thread, so loading the checkpoint doesn't block the event loop.
        run_dir = get_run(name)
        output = project_root / "exports" / run_dir.name
        try:
            result = export_run(run_dir, output)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(
            zip_folder(output, run_dir.name), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{run_dir.name}-web.zip"', "X-Export-Path": f"exports/{run_dir.name}/", "X-Export-Bytes": str(result["bytes"])},
        )

    @app.post("/api/generate")
    async def generate(request: GenerateRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(_generate, request, get_run(request.model))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/generate/details")
    async def generate_details(request: GenerateRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(_generate_details, request, get_run(request.model))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/attention")
    async def attention(request: GenerateRequest) -> dict[str, Any]:
        """Return a bounded last-layer attention slice for the playground heatmap."""
        try:
            model, tokenizer, device = await asyncio.to_thread(_load_model, get_run(request.model), request.checkpoint)
            encoded = tokenizer.encode(request.prompt, add_bos=True)[-min(model.config.block_size, 128):]
            context = torch.tensor([encoded], dtype=torch.long, device=device)
            _, _, layers = model(context, return_attentions=True)
            if not layers:
                return {"tokens": [tokenizer.decode([idx], skip_special=False) for idx in encoded], "heads": []}
            last = layers[-1][0].detach().cpu().tolist()
            return {"tokens": [tokenizer.decode([idx], skip_special=False) for idx in encoded], "heads": last}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.websocket("/ws/runs/{name}/metrics")
    async def metrics_socket(websocket: WebSocket, name: str) -> None:
        await websocket.accept()
        run_dir = get_run(name)
        try:
            cursor = max(0, int(websocket.query_params.get("since", "0")))
        except ValueError:
            cursor = 0
        try:
            while True:
                rows = read_jsonl(run_dir / "metrics.jsonl", since=cursor, limit=1000)
                cursor += len(rows)
                for row in rows:
                    await websocket.send_json(row)
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, asyncio.CancelledError):
            return

    @app.websocket("/ws/generate")
    async def generate_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            payload = await websocket.receive_json()
            request = GenerateRequest.model_validate(payload) if hasattr(GenerateRequest, "model_validate") else GenerateRequest.parse_obj(payload)
            result = await asyncio.to_thread(_generate, request, get_run(request.model))
            text = result["text"]
            # Honoring tags can trim the prompt off the front, so only treat the
            # prompt as an already-shown prefix when the output still starts with it.
            prefix = request.prompt if text.startswith(request.prompt) else ""
            await websocket.send_json({"type": "start", "prompt": prefix})
            for char in text[len(prefix):]:
                await websocket.send_json({"type": "token", "text": char})
            await websocket.send_json({"type": "done", "text": text})
        except (WebSocketDisconnect, asyncio.CancelledError):
            return
        except Exception as exc:
            await websocket.send_json({"type": "error", "error": str(exc)})
            await websocket.close()

    return app


app = create_app()
