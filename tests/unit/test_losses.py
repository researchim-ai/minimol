import torch

from minimol.training.losses import (
    boundary_focus_mask,
    cds_loss,
    inverse_variance_weights,
    masked_bce_with_logits,
    masked_huber,
    soft_bernoulli_kd,
)


def test_masked_positions_do_not_change_loss():
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    first = torch.tensor([[2.0, -3.0]])
    second = first.clone()
    second[0, 1] = 40.0
    assert torch.allclose(masked_huber(first, target, mask), masked_huber(second, target, mask))
    assert torch.allclose(
        masked_bce_with_logits(first, target, mask),
        masked_bce_with_logits(second, target, mask),
    )


def test_kd_is_zero_when_student_matches_teacher():
    logits = torch.tensor([0.2, -0.4, 1.5])
    mask = torch.ones_like(logits)
    loss = soft_bernoulli_kd(logits, logits, mask, temperature=2.0, scale_by_t2=False)
    assert float(loss) < 1e-6
    scaled = soft_bernoulli_kd(logits, logits + 0.5, mask, temperature=2.0, scale_by_t2=True)
    plain = soft_bernoulli_kd(logits, logits + 0.5, mask, temperature=2.0, scale_by_t2=False)
    assert torch.allclose(scaled, plain * 4)


def test_boundary_mask_marks_transitions_only():
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
    valid = torch.ones_like(target)
    focus = boundary_focus_mask(target, valid)
    assert float(focus[0, 0]) == 0.0
    assert float(focus[0, 2]) == 1.0
    assert float(focus[0, 1]) == 1.0
    total, parts = cds_loss(
        torch.zeros(1, 1, 5),
        target.unsqueeze(1),
        valid.unsqueeze(1),
        lambda_boundary=1.0,
    )
    assert torch.isfinite(total)
    assert "boundary" in parts


def test_inverse_variance_cap():
    weights = inverse_variance_weights([0.01, 1.0, 1.0, 1.0], cap_quantile=0.5)
    assert weights.max() <= weights[1:].max() + 1e-8 or weights[0] <= np_max_tail(weights)


def np_max_tail(weights):
    return float(weights[1:].max())
