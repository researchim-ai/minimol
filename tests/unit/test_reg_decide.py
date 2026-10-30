import torch

from minimol.data.reg_decisions import group_key, partition_group
from minimol.evaluation.reg_decide import answer_losses, combined_loss
from minimol.models.reg_decide import RegDecisionHead


def test_locus_partition_never_separates_variants():
    a, b = group_key('1:123:A:T:x'), group_key('1:123:A:G:y')
    assert a == b
    assert partition_group(a) == partition_group(b)


def test_typed_distribution_coherence_and_permutation():
    head = RegDecisionHead(8, 3, 16)
    mu, scale = head(torch.randn(4, 8), torch.randn(4, 3))
    boundaries = torch.tensor([0., .5, 1.])
    p = head.bin_probabilities(mu, scale, boundaries)
    torch.testing.assert_close(p.sum(-1), torch.ones(4, 3))
    event = torch.sigmoid(head.event_logits(mu, scale, boundaries))
    assert torch.all(event[..., 1:] <= event[..., :-1])
    permutation = torch.tensor([2, 0, 1])
    logits = head.assay_logits(mu, scale)
    swapped = head.assay_logits(mu[:, permutation], scale[:, permutation])
    torch.testing.assert_close(swapped, logits[:, permutation])
    a = head.pair_logits(mu[0], scale[0], mu[1], scale[1])
    b = head.pair_logits(mu[1], scale[1], mu[0], scale[0])
    torch.testing.assert_close(a, -b)


def test_losses_use_masks_and_train_on_answers():
    head = RegDecisionHead(8, 3, 16)
    features, base = torch.randn(6, 8), torch.randn(6, 3)
    y = torch.randn(6, 3)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[:, 2] = False
    mu, scale = head(features, base)
    first = combined_loss(answer_losses(mu, scale, y, mask))
    other = y.clone()
    other[:, 2] = 100
    second = combined_loss(answer_losses(mu, scale, other, mask))
    torch.testing.assert_close(first, second)
    first.backward()
    assert head.answer[-1].weight.grad.abs().sum() > 0
