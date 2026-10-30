import torch

from minimol.data.sequences import maybe_reverse_complement_batch, one_hot_sequence
from minimol.models.blocks import group_norm, masked_group_norm
from minimol.models.reg_cnn import MinimolReg


def test_full_mask_matches_group_norm():
    torch.manual_seed(0)
    norm = group_norm(32)
    hidden = torch.randn(2, 32, 17)
    mask = torch.ones(2, 17, dtype=torch.bool)
    torch.testing.assert_close(masked_group_norm(norm, hidden, mask), norm(hidden), atol=1e-5, rtol=1e-5)


def test_mask_aware_batch_matches_native_length():
    torch.manual_seed(0)
    model = MinimolReg("1m", n_assays=2, channels=16, n_blocks=2, dropout=0.0, padding_policy="mask_aware")
    model.eval()
    native_input = one_hot_sequence("ACGTACGTAC")
    native = model(native_input.unsqueeze(0))
    width = 30
    padded = torch.zeros(2, 5, width)
    padded[0, :, : native_input.shape[-1]] = native_input
    padded[1, :, :4] = one_hot_sequence("TTTT")
    mask = torch.zeros(2, width, dtype=torch.bool)
    mask[0, : native_input.shape[-1]] = True
    mask[1, :4] = True
    batched = model(padded, mask)
    torch.testing.assert_close(batched[0], native[0], atol=1e-4, rtol=1e-4)


def test_old_checkpoint_keeps_legacy_padding():
    model = MinimolReg.from_metadata({"preset": "1m", "n_assays": 2, "overrides": {"channels": 8, "n_blocks": 1}})
    assert model.padding_policy == "legacy_n_pad"


def test_reverse_complement_flips_the_pad_mask():
    encoded = torch.zeros(1, 5, 4)
    encoded[0, 0, 0] = 1.0
    pad_mask = torch.tensor([[True, True, False, False]])
    flipped, flipped_mask = maybe_reverse_complement_batch(
        encoded, True, coin=torch.zeros(1), pad_mask=pad_mask
    )
    assert flipped_mask.tolist() == [[False, False, True, True]]
    assert flipped[0, 3, -1] == 1.0
