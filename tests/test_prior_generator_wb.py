import math

import torch

from stonco.core.wb_potentials import GeneratedSupportWBLoss, PriorSupportGenerator


def _toy_inputs():
    torch.manual_seed(7)
    h = torch.randn(18, 8)
    cancer = torch.tensor([0] * 6 + [1] * 6 + [2] * 6, dtype=torch.long)
    y = torch.tensor([0, 1, 0, 1, 0, 1] * 3, dtype=torch.long)
    return h, cancer, y


def _assert_prior_stats(stats, loss_key):
    assert bool(stats["valid"])
    assert stats["wb_active_cancers"] == 3.0
    assert stats["wb_active_spots"] == 18.0
    assert torch.isfinite(stats[loss_key])
    for key in (
        "wb_support_norm",
        "wb_support_std",
        "wb_h_norm",
        "wb_h_std",
        "wb_mean_gap",
        "wb_std_gap",
    ):
        assert key in stats
        assert math.isfinite(float(stats[key]))


def test_prior_support_generator_samples_requested_shape():
    generator = PriorSupportGenerator(in_dim=8, prior_dim=4, hidden=16, dropout=0.0)
    b = generator.sample(11)
    assert b.shape == (11, 8)
    assert b.requires_grad


def test_prior_generator_anchor_is_zero_for_all_supported_losses():
    h, cancer, y = _toy_inputs()
    prior = PriorSupportGenerator(in_dim=8, prior_dim=4, hidden=16, dropout=0.0)

    for loss_type, stat_key in (
        ("sinkhorn_divergence", "wb_sinkhorn"),
        ("sliced_wasserstein", "wb_sliced_wasserstein"),
        ("mmd", "wb_mmd"),
    ):
        torch.manual_seed(11)
        b = prior.sample(14)
        module = GeneratedSupportWBLoss(
            n_domains=3,
            in_dim=8,
            loss_type=loss_type,
            support_mode="prior_generator",
            support_size=12,
            min_cancers=2,
            min_spots=2,
            sinkhorn_iters=2,
            sw_num_projections=5,
            mmd_num_kernels=3,
        )
        loss, anchor, stats = module.model_loss(h=h, b=b, cancer_dom=cancer, y=y)

        assert loss.requires_grad
        assert torch.isfinite(loss)
        assert torch.equal(anchor.detach(), torch.zeros_like(anchor.detach()))
        _assert_prior_stats(stats, stat_key)


def test_mmd_is_rejected_for_generated_support_mode():
    try:
        GeneratedSupportWBLoss(n_domains=2, in_dim=4, loss_type="mmd")
    except ValueError as exc:
        assert "prior_generator" in str(exc)
    else:
        raise AssertionError("mmd must not be enabled for generated_support mode")


if __name__ == "__main__":
    test_prior_support_generator_samples_requested_shape()
    test_prior_generator_anchor_is_zero_for_all_supported_losses()
    test_mmd_is_rejected_for_generated_support_mode()
