"""Standalone generation CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


def load_run(run_dir: Path, checkpoint_name: str = "best") -> tuple[GPT, CharTokenizer, torch.device]:
    config = load_config(run_dir / "config.yaml")
    tokenizer = CharTokenizer.load(run_dir / "vocab.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(GPTConfig(vocab_size=len(tokenizer), **config.model.resolved())).to(device)
    checkpoint = run_dir / ("ckpt_best.pt" if checkpoint_name == "best" else "ckpt_latest.pt")
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer, device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a NanoGym run")
    parser.add_argument("--run", required=True, help="run folder, e.g. runs/shakespeare-small")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--max_new_chars", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer, device = load_run(Path(args.run), args.checkpoint)
    ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device)
    output = model.generate(ids, args.max_new_chars, args.temperature, args.top_k or None, args.top_p, tokenizer.eos_id)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()

