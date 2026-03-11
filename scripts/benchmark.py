from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from pnn_vit.config import load_config
from pnn_vit.metrics import estimate_vit_flops
from pnn_vit.models.vit import build_model
from pnn_vit.utils.io import dump_json, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PNN-ViT.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = build_model(config.model).to(args.device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model)
    model.eval()
    images = torch.randn(args.batch_size, 3, config.dataset.image_size, config.dataset.image_size, device=args.device)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(images)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        token_counts = None
        for _ in range(args.iters):
            outputs = model(images)
            token_counts = outputs["token_counts"]
            records = outputs["records"]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    throughput = args.batch_size * args.iters / max(elapsed, 1e-6)
    flops = estimate_vit_flops(
        image_size=config.dataset.image_size,
        patch_size=model.patch_size,
        embed_dim=model.backbone.embed_dim,
        mlp_ratio=model.backbone.blocks[0].mlp.fc1.out_features / model.backbone.embed_dim,
        num_heads=model.backbone.blocks[0].attn.num_heads,
        token_counts=token_counts,
        num_classes=config.model.num_classes,
    )
    metrics = {
        "throughput": throughput,
        "flops": flops,
        "avg_kept_tokens": float(
            torch.stack([record.original_keep_mask.sum(dim=1) for record in records]).float().mean().item()
            if records
            else token_counts[-1] - 1
        ),
    }
    output_path = Path(args.output) if args.output else Path(args.checkpoint or args.config).with_name("benchmark.json")
    dump_json(output_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
