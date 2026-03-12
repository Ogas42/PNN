from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.amp import autocast

from pnn_vit.config import PNNConfig


@dataclass
class RouterOutput:
    keep_mask: torch.Tensor
    patch_keep_mask: torch.Tensor
    group_ids: torch.Tensor
    group_scores: torch.Tensor
    D_hist: torch.Tensor
    Q_hist: torch.Tensor
    A: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    aux_losses: dict[str, torch.Tensor]


def build_group_index(patch_indices: torch.Tensor, grid_shape: tuple[int, int], groups: int) -> torch.Tensor:
    grid_h, grid_w = grid_shape
    side = int(math.sqrt(groups))
    if side * side != groups:
        raise ValueError(f"groups must be a perfect square, got {groups}")
    row = patch_indices // grid_w
    col = patch_indices % grid_w
    group_h = math.ceil(grid_h / side)
    group_w = math.ceil(grid_w / side)
    group_row = torch.clamp(row // group_h, max=side - 1)
    group_col = torch.clamp(col // group_w, max=side - 1)
    return group_row * side + group_col


class PhysarumRouter(nn.Module):
    def __init__(self, embed_dim: int, config: PNNConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _offdiag_mask(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        eye = torch.eye(size, device=device, dtype=dtype).unsqueeze(0)
        return 1.0 - eye

    def _with_floor(self, matrix: torch.Tensor) -> torch.Tensor:
        mask = self._offdiag_mask(matrix.size(-1), matrix.device, matrix.dtype)
        floor = self.config.conductance_floor * mask
        return torch.maximum(matrix * mask, floor)

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (scores - mean) / std

    def _aggregate_groups(
        self,
        patch_tokens: torch.Tensor,
        group_ids: torch.Tensor,
        groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, patches, dim = patch_tokens.shape
        grouped = patch_tokens.new_zeros(batch, groups, dim)
        counts = patch_tokens.new_zeros(batch, groups, 1)
        index = group_ids.unsqueeze(-1).expand(-1, -1, dim)
        grouped.scatter_add_(1, index, patch_tokens)
        counts.scatter_add_(1, group_ids.unsqueeze(-1), patch_tokens.new_ones(batch, patches, 1))
        grouped = grouped / counts.clamp_min(1.0)
        return grouped, counts.squeeze(-1)

    def _pairwise_conductance(self, nodes: torch.Tensor) -> torch.Tensor:
        projected = self.proj(nodes)
        projected = torch.nn.functional.normalize(projected, dim=-1)
        distances = torch.cdist(projected, projected, p=2).pow(2)
        conductance = torch.exp(-distances / max(self.config.conductance_tau, 1e-6))
        conductance = self._with_floor(conductance)
        if self.config.symmetric:
            conductance = 0.5 * (conductance + conductance.transpose(-1, -2))
        return conductance

    def _solve_pressure(self, conductance: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # Some CUDA/MAGMA stacks crash on tiny batched solves. The system here is
        # very small (1 + G nodes, typically 17), so a CPU fp64 solve is cheap
        # and much more stable than relying on GPU batched LU.
        with autocast(device_type=conductance.device.type, enabled=False):
            conductance_fp64 = conductance.double().contiguous()
            source_fp64 = source.double().contiguous()
            degree = conductance_fp64.sum(dim=-1)
            laplacian = torch.diag_embed(degree) - conductance_fp64
            identity = torch.eye(
                conductance_fp64.size(-1),
                device=conductance_fp64.device,
                dtype=conductance_fp64.dtype,
            ).unsqueeze(0)
            system = (laplacian + self.config.eps * identity).contiguous()
            pressure = torch.linalg.solve(
                system.cpu(),
                source_fp64.unsqueeze(-1).cpu(),
            ).squeeze(-1)
        return pressure.to(device=conductance.device, dtype=conductance.dtype)

    def _evolve_conductance(
        self,
        conductance: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_hist = [conductance]
        q_hist = []
        current = conductance
        for _ in range(self.config.steps):
            if self.config.use_pressure:
                pressure = self._solve_pressure(current, source)
                delta = pressure.unsqueeze(-1) - pressure.unsqueeze(-2)
                flow = current * delta
            else:
                flow = current
            q_hist.append(flow)
            if self.config.use_dynamics:
                reinforce = flow.abs().pow(self.config.mu)
                reinforce = reinforce / (1.0 + reinforce)
                current = current + self.config.dt * (reinforce - self.config.gamma * current)
                current = self._with_floor(current.clamp_min(0.0))
                if self.config.symmetric:
                    current = 0.5 * (current + current.transpose(-1, -2))
                    current = self._with_floor(current)
            d_hist.append(current)
        return current, torch.stack(d_hist, dim=1), torch.stack(q_hist, dim=1)

    def forward(
        self,
        tokens: torch.Tensor,
        cls_token: torch.Tensor,
        grid_shape: tuple[int, int],
        patch_indices: torch.Tensor | None = None,
        local_scores: torch.Tensor | None = None,
    ) -> RouterOutput:
        batch, patches, _ = tokens.shape
        if patch_indices is None:
            patch_indices = torch.arange(patches, device=tokens.device).unsqueeze(0).repeat(batch, 1)
        group_ids = build_group_index(patch_indices, grid_shape, self.config.groups)
        grouped_tokens, _ = self._aggregate_groups(tokens, group_ids, self.config.groups)
        node_tokens = torch.cat([cls_token.unsqueeze(1), grouped_tokens], dim=1)
        initial = self._pairwise_conductance(node_tokens)
        source = tokens.new_full((batch, self.config.groups + 1), -1.0 / self.config.groups)
        source[:, 0] = 1.0
        final_d, d_hist, q_hist = self._evolve_conductance(initial, source)
        routing = final_d[:, 1:, 1:]
        routing = routing / routing.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        group_flow = q_hist[:, -1, 1:, :].abs().sum(dim=-1)
        group_scores = self._normalize_scores(group_flow).gather(1, group_ids)
        if local_scores is None:
            local_scores = tokens.new_zeros(batch, patches)
        local_scores = self._normalize_scores(local_scores)
        token_scores = (
            self.config.flow_score_weight * group_scores
            + self.config.local_score_weight * local_scores
        )
        keep_count = max(1, int(round(patches * self.config.keep_ratio)))
        topk = torch.topk(token_scores, k=keep_count, dim=1).indices
        patch_keep_mask = torch.zeros(batch, patches, dtype=torch.bool, device=tokens.device)
        patch_keep_mask.scatter_(1, topk, True)
        keep_mask = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=tokens.device), patch_keep_mask], dim=1)
        delta = d_hist[:, 1:] - d_hist[:, :-1]
        aux_losses = {
            "sparse": final_d[:, 1:, 1:].sum(dim=(-1, -2)).mean(),
            "stable": delta.abs().sum(dim=(-1, -2, -3)).mean(),
        }
        diagnostics = {
            "initial_mean": initial[:, 1:, 1:].mean(),
            "initial_max": initial[:, 1:, 1:].max(),
            "final_mean": final_d[:, 1:, 1:].mean(),
            "final_max": final_d[:, 1:, 1:].max(),
            "flow_mean": group_flow.mean(),
            "flow_max": group_flow.max(),
        }
        return RouterOutput(
            keep_mask=keep_mask,
            patch_keep_mask=patch_keep_mask,
            group_ids=group_ids,
            group_scores=group_scores,
            D_hist=d_hist,
            Q_hist=q_hist,
            A=routing,
            diagnostics=diagnostics,
            aux_losses=aux_losses,
        )


class ScoreRouter(nn.Module):
    def __init__(self, embed_dim: int, method: str, keep_ratio: float, hidden_dim: int = 96) -> None:
        super().__init__()
        self.method = method
        self.keep_ratio = keep_ratio
        if method in {"mlp", "dynamicvit_lite"}:
            input_dim = embed_dim if method == "mlp" else embed_dim * 2
            self.predictor = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.predictor = None

    def forward(
        self,
        patch_tokens: torch.Tensor,
        cls_token: torch.Tensor,
        local_scores: torch.Tensor | None = None,
    ) -> RouterOutput:
        batch, patches, _ = patch_tokens.shape
        if self.method == "random":
            scores = torch.rand(batch, patches, device=patch_tokens.device)
        elif self.method == "attention":
            scores = local_scores if local_scores is not None else torch.rand(batch, patches, device=patch_tokens.device)
        elif self.method == "mlp":
            scores = self.predictor(patch_tokens).squeeze(-1)
        elif self.method == "dynamicvit_lite":
            cls_expanded = cls_token.unsqueeze(1).expand_as(patch_tokens)
            scores = self.predictor(torch.cat([patch_tokens, cls_expanded], dim=-1)).squeeze(-1)
        else:
            raise ValueError(f"Unsupported score router method: {self.method}")
        keep_count = max(1, int(round(patches * self.keep_ratio)))
        topk = torch.topk(scores, k=keep_count, dim=1).indices
        patch_keep_mask = torch.zeros(batch, patches, dtype=torch.bool, device=patch_tokens.device)
        patch_keep_mask.scatter_(1, topk, True)
        keep_mask = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=patch_tokens.device), patch_keep_mask], dim=1)
        zeros = torch.zeros(batch, 1, 1, device=patch_tokens.device)
        return RouterOutput(
            keep_mask=keep_mask,
            patch_keep_mask=patch_keep_mask,
            group_ids=torch.zeros(batch, patches, dtype=torch.long, device=patch_tokens.device),
            group_scores=scores,
            D_hist=zeros.unsqueeze(1),
            Q_hist=zeros.unsqueeze(1),
            A=zeros,
            diagnostics={
                "initial_mean": scores.mean() * 0.0,
                "initial_max": scores.mean() * 0.0,
                "final_mean": scores.mean() * 0.0,
                "final_max": scores.mean() * 0.0,
                "flow_mean": scores.mean() * 0.0,
                "flow_max": scores.mean() * 0.0,
            },
            aux_losses={"sparse": scores.mean() * 0.0, "stable": scores.mean() * 0.0},
        )
