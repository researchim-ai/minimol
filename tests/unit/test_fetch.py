import pandas as pd
import pytest

from minimol.data.fetch_public import assert_local_cache, import_malinois, tdc_style_train_valid
from minimol.errors import MinimolError


def test_local_cache_must_stay_under_data_raw(tmp_path):
    allowed = tmp_path / "data" / "raw" / "malinois"
    assert assert_local_cache(allowed) == allowed
    with pytest.raises(MinimolError):
        assert_local_cache(tmp_path / "src" / "minimol")


def test_import_malinois_keeps_chromosome_splits_and_se_filter(tmp_path):
    raw = tmp_path / "data" / "raw" / "malinois" / "data"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "split": ["train", "test", "test"],
            "chromosome": ["1", "7", "7"],
            "sequence": ["ACGTACGT", "TTTTAAAA", "ACGTACGT"],
            "K562_log2FC": [0.2, 1.0, 0.3],
            "HepG2_log2FC": [0.1, 0.4, 0.2],
            "SKNSH_log2FC": [-0.2, 0.3, -0.1],
            "K562_lfcSE": [0.1, 2.0, 0.1],
            "HepG2_lfcSE": [0.2, 0.2, 0.2],
            "SKNSH_lfcSE": [0.2, 0.2, 0.2],
        }
    ).to_parquet(raw / "train-00000-of-00001.parquet")
    manifest = import_malinois(raw.parent, tmp_path / "out", revision="rev1")
    frame = pd.read_parquet(tmp_path / "out" / "data.parquet")
    assert manifest["n_sequences_quarantined"] == 1
    assert set(frame.loc[frame["sequence"] == "ACGTACGT", "split"]) == {"quarantine"}
    held_out = frame[(frame["group_id"] == "b") & (frame["assay"] == "K562")].iloc[0]
    assert held_out.split == "test"
    assert bool(held_out.valid_y) is False
    usable = frame[(frame["group_id"] == "b") & (frame["assay"] == "HepG2")].iloc[0]
    assert bool(usable.valid_y) is True


def test_tdc_scaffold_split_is_deterministic_and_disjoint():
    frame = pd.DataFrame(
        {
            "Drug": ["CCO", "c1ccccc1", "CCN", "c1ccncc1", "CC(=O)O", "c1ccc(O)cc1"],
            "Y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    first_train, first_valid = tdc_style_train_valid(frame, seed=1)
    second_train, second_valid = tdc_style_train_valid(frame, seed=1)
    assert first_train["Drug"].tolist() == second_train["Drug"].tolist()
    assert set(first_train["Drug"]).isdisjoint(set(first_valid["Drug"]))
    assert len(first_train) + len(first_valid) == len(frame)
    assert len(second_valid) == len(first_valid)
