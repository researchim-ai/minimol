from minimol.data.sequences import (
    normalize_sequence,
    one_hot_sequence,
    reverse_complement,
    reverse_complement_one_hot,
)


def test_rc_roundtrip_string():
    sequence = "ACGTNacgu"
    assert reverse_complement(reverse_complement(sequence)) == normalize_sequence(sequence)


def test_rc_roundtrip_one_hot():
    sequence = "AAACGTN"
    encoded = one_hot_sequence(sequence)
    restored = reverse_complement_one_hot(reverse_complement_one_hot(encoded))
    assert torch_equal(restored, encoded)
    assert torch_equal(reverse_complement_one_hot(encoded), one_hot_sequence(reverse_complement(sequence)))


def test_strand_swap_twice_is_identity():
    import numpy as np

    plus = np.array([0.1, 0.2, 0.3, 0.4])
    minus = np.array([0.5, 0.6, 0.7, 0.8])
    plus_rc, minus_rc = minus[::-1], plus[::-1]
    back_plus, back_minus = minus_rc[::-1], plus_rc[::-1]
    assert np.allclose(back_plus, plus)
    assert np.allclose(back_minus, minus)


def torch_equal(left, right):
    import torch

    return torch.equal(left, right)
