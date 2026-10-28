import numpy as np
import pytest

from minimol.errors import AlignmentError, RemoteCodeNotReviewed
from minimol.teachers.alignment import (
    expand_token_values_to_bases,
    id_to_sixmer,
    sixmer_token_id,
    tokenize_sixmers,
)
from minimol.teachers.carbon_a import assert_carbon_a_loadable, local_coordinate_smoke
from minimol.teachers.carbon_x import go_nogo, prompt_sensitivity, scores_to_pseudo_distribution


def test_sixmer_roundtrip_and_repeat_expansion():
    assert id_to_sixmer(sixmer_token_id("ATGCCC")) == "ATGCCC"
    assert sixmer_token_id("ATGNNN") == 4096
    bases, mask = expand_token_values_to_bases([[1.0, 2.0], [3.0, 4.0]], 13)
    assert np.allclose(bases[:6, 0], 1.0)
    assert np.allclose(bases[6:12, 1], 4.0)
    assert not mask[12]
    assert np.isnan(bases[12, 0])
    with pytest.raises(AlignmentError):
        expand_token_values_to_bases([1.0], 12)


def test_local_smoke_does_not_claim_upstream_verification():
    report = local_coordinate_smoke("ATGAAATTTAAACCCGGG")
    assert report["n_tokens"] == 3
    assert report["verified_against_upstream"] is False
    assert report["upstream_weights"] == "not_loaded"
    assert len(tokenize_sixmers("ACGTAC")) == 1


def test_carbon_loaders_refuse_unreviewed_code():
    with pytest.raises(RemoteCodeNotReviewed):
        assert_carbon_a_loadable("abc123", reviewed_remote_code=False)
    from minimol.teachers.carbon_x import assert_carbon_x_loadable

    with pytest.raises(RemoteCodeNotReviewed):
        assert_carbon_x_loadable("deadbeef", reviewed_remote_code=False)


def test_option_scores_are_not_called_calibrated():
    dist = scores_to_pseudo_distribution([-1.0, -1.0])
    assert dist["calibrated"] is False
    assert abs(sum(dist["pseudo_distribution"]) - 1) < 1e-8
    assert "not an event probability" in dist["meaning"]
    assert go_nogo(0.40, 0.45)["decision"] == "NO-GO"
    assert go_nogo(0.50, 0.45)["decision"] == "GO"
    assert prompt_sensitivity([[0.1, 0.2], [0.1, 0.8]]) > 0
