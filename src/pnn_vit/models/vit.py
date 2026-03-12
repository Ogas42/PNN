from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import torch
from torch import nn

from pnn_vit.config import ModelConfig, PNNConfig
from pnn_vit.models.router import PhysarumRouter, ScoreRouter


@dataclass
class BlockRecord:
    layer_idx: int
    keep_mask: torch.Tensor
    original_keep_mask: torch.Tensor
    patch_indices: torch.Tensor
    token_count: int
    D_hist: torch.Tensor
    Q_hist: torch.Tensor
    A: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    aux_losses: dict[str, torch.Tensor]


def _gather_tokens(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = mask.sum(dim=1)
    if int(counts.min().item()) != int(counts.max().item()):
        raise ValueError("All samples must keep the same number of tokens in a pruning block.")
    keep_count = int(counts[0].item())
    indices = mask.nonzero(as_tuple=False).reshape(mask.size(0), keep_count, 2)[:, :, 1]
    gather_index = indices.unsqueeze(-1).expand(-1, -1, tensor.size(-1))
    return tensor.gather(1, gather_index)


def _gather_indices(indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    keep_count = int(mask.sum(dim=1)[0].item())
    gather_ids = mask.nonzero(as_tuple=False).reshape(mask.size(0), keep_count, 2)[:, :, 1]
    return indices.gather(1, gather_ids)


def compute_cls_attention_scores(block: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    normalized = block.norm1(tokens)
    batch, total_tokens, channels = normalized.shape
    attn = block.attn
    qkv = attn.qkv(normalized).reshape(batch, total_tokens, 3, attn.num_heads, channels // attn.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    scores = (q * attn.scale) @ k.transpose(-2, -1)
    scores = scores.softmax(dim=-1)
    return scores[:, :, 0, 1:].mean(dim=1)


class PrunedBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layer_idx: int,
        method: str,
        keep_ratio: float,
        pnn_config: PNNConfig,
        score_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.method = method
        self.keep_ratio = keep_ratio
        embed_dim = block.attn.qkv.in_features
        if method == "pnn":
            local_config = PNNConfig(**vars(pnn_config))
            local_config.keep_ratio = keep_ratio
            self.router = PhysarumRouter(embed_dim, local_config)
        else:
            self.router = ScoreRouter(embed_dim, method=method, keep_ratio=keep_ratio, hidden_dim=score_hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        patch_indices: torch.Tensor,
        grid_shape: tuple[int, int],
        capture_router: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, BlockRecord]:
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        local_scores = compute_cls_attention_scores(self.block, tokens)
        if self.method == "pnn":
            router_output = self.router(
                patch_tokens,
                cls_token,
                grid_shape=grid_shape,
                patch_indices=patch_indices,
                local_scores=local_scores,
            )
        else:
            router_output = self.router(patch_tokens, cls_token, local_scores=local_scores)
        kept_tokens = _gather_tokens(tokens, router_output.keep_mask)
        kept_patch_indices = _gather_indices(patch_indices, router_output.patch_keep_mask)
        output_tokens = self.block(kept_tokens)
        original_keep_mask = torch.zeros(
            patch_indices.size(0),
            grid_shape[0] * grid_shape[1],
            dtype=torch.bool,
            device=patch_indices.device,
        )
        original_keep_mask.scatter_(1, kept_patch_indices, True)
        record = BlockRecord(
            layer_idx=self.layer_idx,
            keep_mask=router_output.patch_keep_mask.detach().cpu(),
            original_keep_mask=original_keep_mask.detach().cpu(),
            patch_indices=kept_patch_indices.detach().cpu(),
            token_count=output_tokens.size(1),
            D_hist=router_output.D_hist.detach().cpu() if capture_router else torch.empty(0),
            Q_hist=router_output.Q_hist.detach().cpu() if capture_router else torch.empty(0),
            A=router_output.A.detach().cpu() if capture_router else torch.empty(0),
            diagnostics={
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in router_output.diagnostics.items()
            },
            aux_losses=router_output.aux_losses,
        )
        return output_tokens, kept_patch_indices, record


class PNNVisionTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            drop_path_rate=config.drop_path_rate,
        )
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        grid_size = self.backbone.patch_embed.grid_size
        self.grid_shape = (int(grid_size[0]), int(grid_size[1]))
        if len(config.keep_ratios) != len(config.insert_layers):
            raise ValueError("keep_ratios must match insert_layers")
        ratio_map = {idx: ratio for idx, ratio in zip(config.insert_layers, config.keep_ratios)}
        self.pruned_blocks = nn.ModuleDict()
        for layer_idx in config.insert_layers:
            self.pruned_blocks[str(layer_idx)] = PrunedBlock(
                self.backbone.blocks[layer_idx],
                layer_idx=layer_idx,
                method=config.method,
                keep_ratio=ratio_map[layer_idx],
                pnn_config=config.pnn,
                score_hidden_dim=config.score_hidden_dim,
            )

    def forward(self, images: torch.Tensor, capture_router: bool = False) -> dict[str, Any]:
        x = self.backbone.patch_embed(images)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        batch = x.size(0)
        patch_indices = torch.arange(self.grid_shape[0] * self.grid_shape[1], device=x.device).unsqueeze(0).repeat(batch, 1)
        records: list[BlockRecord] = []
        aux_sparse = x.new_tensor(0.0)
        aux_stable = x.new_tensor(0.0)
        token_counts: list[int] = []
        for layer_idx, block in enumerate(self.backbone.blocks):
            if str(layer_idx) in self.pruned_blocks:
                x, patch_indices, record = self.pruned_blocks[str(layer_idx)](
                    x,
                    patch_indices,
                    grid_shape=self.grid_shape,
                    capture_router=capture_router,
                )
                aux_sparse = aux_sparse + record.aux_losses["sparse"]
                aux_stable = aux_stable + record.aux_losses["stable"]
                records.append(record)
            else:
                x = block(x)
            token_counts.append(x.size(1))
        x = self.backbone.norm(x)
        logits = self.backbone.forward_head(x, pre_logits=False)
        return {
            "logits": logits,
            "aux_losses": {"sparse": aux_sparse, "stable": aux_stable},
            "records": records,
            "token_counts": token_counts,
            "final_patch_indices": patch_indices.detach().cpu(),
        }


def build_model(config: ModelConfig) -> nn.Module:
    if config.method == "base":
        config = ModelConfig(**vars(config))
        config.insert_layers = []
        config.keep_ratios = []
    return PNNVisionTransformer(config)
