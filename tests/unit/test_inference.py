import numpy as np

from minimol.inference import predict_cds_probabilities
from minimol.models.cds_cnn import MinimolCDS


def test_overlap_add_keeps_sequence_length():
    model = MinimolCDS("5m", channels=16, n_blocks=2, dropout=0.0).eval()
    sequence = "ACGT" * 20
    direct = predict_cds_probabilities(model, sequence, window=len(sequence), overlap=8)
    stitched = predict_cds_probabilities(model, sequence, window=24, overlap=8)
    assert direct.shape == (2, len(sequence))
    assert stitched.shape == (2, len(sequence))
    assert np.isfinite(stitched).all()
