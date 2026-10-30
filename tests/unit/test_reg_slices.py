import numpy as np

from minimol.evaluation.reg_slices import length_bin, slice_metrics


def test_length_bins_and_same_scale():
    rows = []
    yhat = []
    for index, length in enumerate((150, 180, 195, 200)):
        sequence = "A" * length
        y = np.array([float(index), float(index) + 0.5, 1.0])
        rows.append(
            {
                "sequence": sequence,
                "y": y,
                "mask": np.array([True, True, True]),
                "se": np.array([0.2, 0.8, 0.2]),
                "chromosome": "7" if index < 2 else "13",
            }
        )
        yhat.append(y + 0.1)
    report = slice_metrics(rows, np.stack(yhat), ["K562", "HepG2", "SK-N-SH"])
    assert report["scale"] == "raw_log2fc"
    assert report["by_length"]["<160"]["n_sequences"] == 1
    assert report["by_length"]["200"]["n_sequences"] == 1
    assert abs(report["per_assay"]["K562"]["mae"] - 0.1) < 1e-9
    assert set(report["by_chromosome"]) == {"7", "13"}
    assert length_bin(200) == "200"
