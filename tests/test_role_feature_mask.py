"""Zero-masking of fingerprint descriptors in the role embedder (protocols/feature_ladder.md, A′4)."""

from __future__ import annotations

import pytest
import torch

from pm_foundation.data.roles import N_ROLE_FEATURES
from pm_foundation.models.role_encoder import RoleEmbedder, build_role_module

V = 12


def _graph(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    adj = (torch.rand(V, V, generator=g) < 0.3).float()
    mask = torch.ones(V, dtype=torch.bool)
    mask[:2] = False  # reserved rows
    return {"feats": torch.rand(V, N_ROLE_FEATURES, generator=g), "adj_in": adj.t().contiguous(),
            "adj_out": adj, "real_mask": mask}


def _pair(mask: list[int]) -> tuple[RoleEmbedder, RoleEmbedder]:
    torch.manual_seed(0)
    plain = RoleEmbedder(V, arch="gin")
    masked = RoleEmbedder(V, arch="gin", feature_mask=mask)
    masked.load_state_dict(plain.state_dict(), strict=True)  # mask buffer is non-persistent
    g = _graph()
    plain.set_graph(g)
    masked.set_graph(g)
    return plain, masked


def test_all_ones_mask_is_bit_identical_to_unmasked() -> None:
    plain, masked = _pair(list(range(N_ROLE_FEATURES)))
    assert torch.equal(plain(augment=False), masked(augment=False))
    torch.manual_seed(1)
    a = plain(augment=True)
    torch.manual_seed(1)
    b = masked(augment=True)
    assert torch.equal(a, b)


def test_masked_columns_receive_exactly_zero_gradient() -> None:
    kept = [0, 3, 7]
    _, masked = _pair(kept)
    masked(augment=True).pow(2).sum().backward()
    grad = masked.inp.weight.grad
    dropped = [j for j in range(N_ROLE_FEATURES) if j not in kept]
    assert torch.count_nonzero(grad[:, dropped]) == 0
    assert torch.count_nonzero(grad[:, kept]) > 0


def test_values_in_masked_columns_do_not_change_the_embedding() -> None:
    kept = [1, 8, 14]
    _, masked = _pair(kept)
    before = masked(augment=False)
    g = _graph()
    dropped = [j for j in range(N_ROLE_FEATURES) if j not in kept]
    g["feats"][:, dropped] = torch.randn(V, len(dropped)) * 100.0
    masked.set_graph(g)
    assert torch.equal(before, masked(augment=False))


def test_config_pass_through_and_empty_mask() -> None:
    cfg = {"role_arch": "gin", "role_dim": 64, "role_layers": 1, "role_feature_subset": "all"}
    assert build_role_module(cfg, V).input_mask is None
    enc = build_role_module({**cfg, "role_feature_mask": [2, 5]}, V)
    assert enc.inp.in_features == N_ROLE_FEATURES
    assert enc.input_mask.tolist() == [1.0 if j in (2, 5) else 0.0 for j in range(N_ROLE_FEATURES)]
    empty = build_role_module({**cfg, "role_feature_mask": []}, V)  # k = 0 of the ladder
    assert empty.input_mask.sum() == 0 and not empty.featureless
    assert "input_mask" not in enc.state_dict()


def test_invalid_masks_are_rejected() -> None:
    with pytest.raises(ValueError):
        RoleEmbedder(V, feature_mask=[0, 0])
    with pytest.raises(ValueError):
        RoleEmbedder(V, feature_mask=[N_ROLE_FEATURES])
    with pytest.raises(ValueError):
        RoleEmbedder(V, feature_subset="none", feature_mask=[])
