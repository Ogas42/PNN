from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from accelerate import Accelerator
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from pnn_vit.config import ExperimentConfig
from pnn_vit.data.datasets import build_loaders
from pnn_vit.metrics import accuracy, confidence_keep_correlation, load_probe_jaccard, routing_entropy
from pnn_vit.models.vit import build_model
from pnn_vit.utils.io import append_csv, dump_json, ensure_dir, save_checkpoint, set_seed


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _extract_batch(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch["image"], batch["label"], batch["index"]


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    capture_router: bool = False,
    probe_dir: Path | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_samples = 0
    all_confidence = []
    all_kept = []
    all_entropy = []
    probe_masks = []
    for batch in data_loader:
        images, labels, _ = _extract_batch(batch)
        outputs = model(images, capture_router=capture_router)
        logits = outputs["logits"]
        batch_size = images.size(0)
        total_loss += criterion(logits, labels).item() * batch_size
        total_top1 += accuracy(logits, labels) * batch_size
        total_samples += batch_size
        all_confidence.append(logits.softmax(dim=-1).max(dim=-1).values.detach().cpu())
        if outputs["records"]:
            kept = torch.stack([record.original_keep_mask.sum(dim=1) for record in outputs["records"]]).float().mean(dim=0)
            all_kept.append(kept.cpu())
            entropies = [routing_entropy(record.A).mean() for record in outputs["records"] if record.A.numel() > 0]
            if entropies:
                all_entropy.append(torch.stack(entropies).mean().unsqueeze(0).cpu())
            if capture_router:
                probe_masks.append(outputs["records"][0].original_keep_mask)
        else:
            all_kept.append(torch.full((batch_size,), outputs["token_counts"][-1] - 1, dtype=torch.float))
    kept_tensor = torch.cat(all_kept) if all_kept else torch.zeros(1)
    confidence_tensor = torch.cat(all_confidence) if all_confidence else torch.zeros(1)
    entropy_tensor = torch.cat(all_entropy) if all_entropy else torch.zeros(1)
    metrics = {
        "loss": total_loss / max(1, total_samples),
        "top1": total_top1 / max(1, total_samples),
        "avg_kept_tokens": float(kept_tensor.mean().item()),
        "routing_entropy": float(entropy_tensor.mean().item()),
        "confidence_keep_corr": confidence_keep_correlation(confidence_tensor, kept_tensor),
        "routing_stability": 0.0,
    }
    if capture_router and probe_dir is not None and epoch is not None and probe_masks:
        ensure_dir(probe_dir)
        torch.save({"mask": torch.cat(probe_masks, dim=0)}, probe_dir / f"epoch_{epoch:03d}.pt")
        metrics["routing_stability"] = load_probe_jaccard(probe_dir)
    return metrics


def train_experiment(config: ExperimentConfig, output_dir: str | Path) -> dict[str, float]:
    set_seed(config.optim.seed)
    output_dir = ensure_dir(output_dir)
    accelerator = Accelerator(
        mixed_precision="fp16" if config.optim.amp and torch.cuda.is_available() else "no",
        gradient_accumulation_steps=config.dataset.grad_accumulation,
    )
    train_loader, val_loader, probe_loader = build_loaders(config.dataset)
    model = build_model(config.model)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.optim.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / config.dataset.grad_accumulation)
    total_steps = steps_per_epoch * config.dataset.epochs
    warmup_steps = steps_per_epoch * config.optim.warmup_epochs
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    model, optimizer, train_loader, val_loader, probe_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, probe_loader, scheduler
    )
    best_top1 = -1.0
    probe_dir = output_dir / "probe_masks"
    for epoch in range(1, config.dataset.epochs + 1):
        model.train()
        progress = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            images, labels, _ = _extract_batch(batch)
            with accelerator.accumulate(model):
                outputs = model(images)
                logits = outputs["logits"]
                loss = criterion(logits, labels)
                loss = loss + config.model.pnn.sparse_lambda * outputs["aux_losses"]["sparse"]
                loss = loss + config.model.pnn.stable_lambda * outputs["aux_losses"]["stable"]
                accelerator.backward(loss)
                if config.optim.grad_clip is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % config.logging.log_interval == 0:
                progress.set_postfix(loss=float(loss.item()), top1=float(accuracy(logits.detach(), labels)))
        val_metrics = evaluate(accelerator, model, val_loader, criterion, capture_router=False)
        probe_metrics = evaluate(
            accelerator,
            model,
            probe_loader,
            criterion,
            capture_router=True,
            probe_dir=probe_dir if accelerator.is_local_main_process else None,
            epoch=epoch,
        )
        metrics = {
            "epoch": epoch,
            "train_lr": scheduler.get_last_lr()[0],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "avg_kept_tokens": val_metrics["avg_kept_tokens"],
            "routing_entropy": probe_metrics["routing_entropy"],
            "routing_stability": probe_metrics["routing_stability"],
            "confidence_keep_corr": probe_metrics["confidence_keep_corr"],
        }
        if accelerator.is_local_main_process:
            append_csv(output_dir / "history.csv", metrics)
        unwrapped = accelerator.unwrap_model(model)
        if val_metrics["top1"] > best_top1 and accelerator.is_local_main_process:
            best_top1 = val_metrics["top1"]
            save_checkpoint(output_dir / "best.pt", unwrapped, optimizer, scheduler, epoch, best_top1, {"name": config.name})
        if accelerator.is_local_main_process and (
            epoch % config.logging.checkpoint_interval == 0 or epoch == config.dataset.epochs
        ):
            save_checkpoint(output_dir / "last.pt", unwrapped, optimizer, scheduler, epoch, best_top1, {"name": config.name})
    summary = {
        "best_top1": best_top1,
        "routing_stability": load_probe_jaccard(probe_dir) if probe_dir.exists() else 0.0,
    }
    if accelerator.is_local_main_process:
        dump_json(output_dir / "train_summary.json", summary)
    return summary
