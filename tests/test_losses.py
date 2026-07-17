"""Sanity tests for the distillation losses (run with pytest)."""
import torch

from distill import losses


def _rand(seed=0, b=2, s=8, v=50):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, s, v, generator=g)


MASK = torch.ones(2, 8)


def test_self_divergence_is_zero():
    x = _rand()
    assert losses.forward_kl_loss(x, x, MASK).item() == 0.0
    assert losses.reverse_kl_loss(x, x, MASK).item() == 0.0
    assert abs(losses.generalized_jsd_loss(x, x, MASK).item()) < 1e-5
    assert abs(losses.tvd_loss(x, x, MASK).item()) < 1e-6


def test_divergences_nonnegative():
    s, t = _rand(1), _rand(2)
    assert losses.forward_kl_loss(s, t, MASK).item() > 0
    assert losses.reverse_kl_loss(s, t, MASK).item() > 0
    assert losses.generalized_jsd_loss(s, t, MASK).item() > 0
    assert losses.tvd_loss(s, t, MASK).item() > 0


def test_jsd_between_bounds():
    # symmetric JSD is bounded by ln 2 (per position, before T^2 scaling)
    s, t = _rand(1), _rand(2)
    jsd = losses.generalized_jsd_loss(s, t, MASK, temperature=1.0).item()
    assert 0 < jsd < torch.log(torch.tensor(2.0)).item() + 1e-6


def test_tvd_bounded_by_one():
    s, t = _rand(1), _rand(2)
    assert 0 < losses.tvd_loss(s, t, MASK, temperature=1.0).item() <= 1.0


def test_masking_ignores_positions():
    s, t = _rand(1), _rand(2)
    mask = torch.zeros(2, 8)
    mask[:, :4] = 1.0
    full = losses.forward_kl_loss(s, t, MASK)
    half = losses.forward_kl_loss(s, t, mask)
    assert full.item() != half.item()

    # corrupting masked-out positions must not change the loss
    t2 = t.clone()
    t2[:, 4:] += 100.0
    assert torch.allclose(losses.forward_kl_loss(s, t, mask), losses.forward_kl_loss(s, t2, mask))


def test_gradients_flow_to_student_only():
    s, t = _rand(1).requires_grad_(True), _rand(2)
    loss = losses.forward_kl_loss(s, t, MASK)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
