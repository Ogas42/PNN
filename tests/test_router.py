import torch

from pnn_vit.config import PNNConfig
from pnn_vit.models.router import PhysarumRouter, build_group_index


def test_group_index_supports_multiple_group_sizes() -> None:
    patch_indices = torch.arange(196).unsqueeze(0)
    for groups in (9, 16, 25):
        group_ids = build_group_index(patch_indices, (14, 14), groups)
        assert group_ids.min().item() == 0
        assert group_ids.max().item() == groups - 1


def test_physarum_router_properties() -> None:
    router = PhysarumRouter(embed_dim=32, config=PNNConfig(groups=16, keep_ratio=0.5))
    tokens = torch.randn(2, 196, 32)
    cls_token = torch.randn(2, 32)
    output = router(tokens, cls_token, grid_shape=(14, 14))
    assert output.keep_mask[:, 0].all()
    assert output.patch_keep_mask.sum(dim=1).min().item() >= 1
    assert torch.all(output.D_hist >= 0)
    final_d = output.D_hist[:, -1]
    assert torch.allclose(final_d, final_d.transpose(-1, -2), atol=1e-5)
    row_sum = output.A.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)
