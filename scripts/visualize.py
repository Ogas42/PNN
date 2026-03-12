from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import torch

from pnn_vit.config import load_config
from pnn_vit.data.datasets import build_dataset
from pnn_vit.models.vit import build_model
from pnn_vit.utils.io import ensure_dir, load_checkpoint


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1)


def draw_token_overlay(image: torch.Tensor, keep_mask: torch.Tensor, grid_shape: tuple[int, int], output_path: Path) -> None:
    raw = denormalize(image).permute(1, 2, 0).numpy()
    heat = keep_mask.reshape(*grid_shape).float().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(raw)
    axes[1].imshow(heat, cmap="viridis", alpha=0.5, extent=(0, raw.shape[1], raw.shape[0], 0))
    axes[1].set_title("Kept Tokens")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_comparison_overlay(
    image: torch.Tensor,
    primary_mask: torch.Tensor,
    compare_mask: torch.Tensor,
    grid_shape: tuple[int, int],
    output_path: Path,
    primary_label: str,
    compare_label: str,
) -> None:
    raw = denormalize(image).permute(1, 2, 0).numpy()
    primary_heat = primary_mask.reshape(*grid_shape).float().numpy()
    compare_heat = compare_mask.reshape(*grid_shape).float().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(raw)
    axes[1].imshow(primary_heat, cmap="viridis", alpha=0.5, extent=(0, raw.shape[1], raw.shape[0], 0))
    axes[1].set_title(primary_label)
    axes[1].axis("off")
    axes[2].imshow(raw)
    axes[2].imshow(compare_heat, cmap="plasma", alpha=0.5, extent=(0, raw.shape[1], raw.shape[0], 0))
    axes[2].set_title(compare_label)
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_conductance_heatmaps(record, output_dir: Path, prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(record.D_hist[0, 0].numpy(), cmap="magma")
    axes[0].set_title("D0")
    axes[1].imshow(record.D_hist[0, -1].numpy(), cmap="magma")
    axes[1].set_title("DT")
    axes[2].imshow(record.A[0].numpy(), cmap="magma")
    axes[2].set_title("A")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_conductance.png")
    plt.close(fig)

    deltas = []
    for idx in range(record.D_hist.size(1) - 1):
        deltas.append((record.D_hist[0, idx + 1] - record.D_hist[0, idx]).abs().sum().item())
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(deltas) + 1), deltas, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|D(t+1)-D(t)|_1")
    ax.set_title("Conductance Convergence")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_convergence.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PNN-ViT routing.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--compare-config", type=str, default=None)
    parser.add_argument("--compare-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--samples", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = build_dataset(config.dataset, train=False)
    model = build_model(config.model)
    load_checkpoint(args.checkpoint, model)
    compare_model = None
    compare_method = None
    if args.compare_checkpoint:
        compare_config = load_config(args.compare_config or args.config)
        compare_model = build_model(compare_config.model)
        load_checkpoint(args.compare_checkpoint, compare_model)
        compare_method = compare_config.model.method
    output_dir = ensure_dir(args.output_dir or Path(args.checkpoint).with_name("visualizations"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if compare_model is not None:
        compare_model.to(device)
        compare_model.eval()
    for idx in range(min(args.samples, len(dataset))):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image, capture_router=True)
        if not outputs["records"]:
            continue
        record = outputs["records"][0]
        print(
            f"sample {idx}: "
            f"D0_mean={float(record.diagnostics['initial_mean']):.4f}, "
            f"DT_mean={float(record.diagnostics['final_mean']):.4f}, "
            f"flow_mean={float(record.diagnostics['flow_mean']):.4f}"
        )
        draw_conductance_heatmaps(record, output_dir, prefix=f"sample_{idx:03d}")
        draw_token_overlay(sample["image"], record.original_keep_mask[0], model.grid_shape, output_dir / f"sample_{idx:03d}_tokens.png")
        if compare_model is not None:
            with torch.no_grad():
                compare_outputs = compare_model(image, capture_router=True)
            if compare_outputs["records"]:
                compare_record = compare_outputs["records"][0]
                draw_comparison_overlay(
                    sample["image"],
                    record.original_keep_mask[0],
                    compare_record.original_keep_mask[0],
                    model.grid_shape,
                    output_dir / f"sample_{idx:03d}_comparison.png",
                    primary_label=config.model.method,
                    compare_label=compare_method or "compare",
                )
    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
