from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def jaccard_from_masks(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs = lhs.bool()
    rhs = rhs.bool()
    intersection = (lhs & rhs).sum(dim=-1).float()
    union = (lhs | rhs).sum(dim=-1).float().clamp_min(1.0)
    return intersection / union


def routing_entropy(routing: torch.Tensor) -> torch.Tensor:
    probs = routing.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1)


def confidence_keep_correlation(confidence: torch.Tensor, kept_tokens: torch.Tensor) -> float:
    if confidence.numel() < 2:
        return 0.0
    conf = confidence.float()
    kept = kept_tokens.float()
    conf = (conf - conf.mean()) / conf.std().clamp_min(1e-6)
    kept = (kept - kept.mean()) / kept.std().clamp_min(1e-6)
    return float((conf * kept).mean().item())


def estimate_vit_flops(
    image_size: int,
    patch_size: int,
    embed_dim: int,
    mlp_ratio: float,
    num_heads: int,
    token_counts: Iterable[int],
    num_classes: int,
) -> float:
    del num_heads
    patch_tokens = (image_size // patch_size) ** 2
    patch_embed = 3 * (patch_size ** 2) * embed_dim * patch_tokens
    total = patch_embed
    hidden_dim = int(embed_dim * mlp_ratio)
    for count in token_counts:
        qkv = 3 * count * embed_dim * embed_dim
        proj = count * embed_dim * embed_dim
        attn = 2 * count * count * embed_dim
        mlp = 2 * count * embed_dim * hidden_dim
        total += qkv + proj + attn + mlp
    total += embed_dim * num_classes
    return float(total)


def load_probe_jaccard(probe_dir: str | Path) -> float:
    files = sorted(Path(probe_dir).glob("epoch_*.pt"))
    if len(files) < 2:
        return 0.0
    scores = []
    prev = None
    for path in files:
        payload = torch.load(path, map_location="cpu")
        mask = payload["mask"]
        if prev is not None:
            scores.append(jaccard_from_masks(prev, mask).mean().item())
        prev = mask
    return sum(scores) / len(scores)
