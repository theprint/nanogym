"""A compact GPT-style decoder with an inspectable manual attention path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("embedding dimension must be divisible by number of heads")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, time, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)

        weights: torch.Tensor | None = None
        if return_weights:
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            scores = scores.masked_fill(self.bias[:, :, :time, :time] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            y = weights @ v
        elif hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            scores = scores.masked_fill(self.bias[:, :, :time, :time] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            y = self.attn_dropout(weights) @ v
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(y)), weights


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, weights = self.attn(self.ln_1(x), return_weights=return_weights)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, weights


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer["wte"].weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        batch, time = idx.size()
        if time > self.config.block_size:
            raise ValueError(f"sequence length {time} exceeds block_size {self.config.block_size}")
        positions = torch.arange(0, time, dtype=torch.long, device=idx.device)
        x = self.transformer["drop"](self.transformer["wte"](idx) + self.transformer["wpe"](positions))
        attentions: list[torch.Tensor] = []
        for block in self.transformer["h"]:
            x, weights = block(x, return_weights=return_attentions)
            if weights is not None:
                attentions.append(weights)
        logits = self.lm_head(self.transformer["ln_f"](x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss, attentions

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: int | None = None, top_p: float = 1.0, stop_id: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            if 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, sorted_indices, sorted_logits)
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if stop_id is not None and bool((next_token == stop_id).all()):
                break
        return idx

    @torch.no_grad()
    def next_token(self, idx: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float = 1.0) -> tuple[int, list[dict[str, Any]]]:
        context = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
        logits, _, _ = self(context)
        logits = logits[:, -1, :] / temperature
        if top_k:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits[remove] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, min(5, probs.size(-1)), dim=-1)
        details = [{"id": int(token_id), "probability": float(prob)} for token_id, prob in zip(top_ids[0], top_probs[0])]
        return int(torch.multinomial(probs, 1).item()), details

    def parameter_count(self, trainable_only: bool = True) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad or not trainable_only)
