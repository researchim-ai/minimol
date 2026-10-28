import numpy as np

from minimol.data.normalize import LabelScaler, fit_scaler_train_only


def test_scaler_uses_train_rows_only():
    y = np.array([[0.0], [0.0], [1.0e6]])
    mask = np.ones_like(y, dtype=bool)
    splits = ["train", "train", "test"]
    scaler = fit_scaler_train_only(y, mask, splits)
    assert abs(scaler.mean[0]) < 1e-8
    contaminated = LabelScaler().fit(y, mask)
    assert contaminated.mean[0] > 1000


def test_inverse_roundtrip():
    y = np.array([[1.0, 5.0], [3.0, 7.0]])
    mask = np.ones_like(y, dtype=bool)
    scaler = LabelScaler().fit(y, mask)
    restored = scaler.inverse_numpy(scaler.transform(y))
    assert np.allclose(restored, y)
