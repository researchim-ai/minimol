import pytest

from minimol.data.split import (
    assert_disjoint_ids,
    assert_plinder_release_frozen,
    assert_test_ids_unchanged,
    make_stricter_split,
    scaffold_split,
    time_split,
)
from minimol.errors import MinimolError, SplitLeakageError
from minimol.teachers.cache import TeacherCache


def test_scaffold_assignment_is_disjoint_and_repeatable():
    ids = [f"s{i}" for i in range(12)]
    first = scaffold_split(ids, seed=3)
    second = scaffold_split(ids, seed=3)
    assert first == second
    train = {ids[i] for i, split in enumerate(first) if split == "train"}
    test = {ids[i] for i, split in enumerate(first) if split == "test"}
    assert train.isdisjoint(test)


def test_duplicate_molecule_and_scaffold_cross_split():
    import pandas as pd

    from minimol.data.homology import scan_leakage

    frame = pd.DataFrame(
        {
            "split": ["train", "test"],
            "inchikey": ["ABC", "ABC"],
            "scaffold_id": ["murcko:1", "murcko:1"],
        }
    )
    report = scan_leakage(frame)
    assert report.inchikey_cross_split
    assert report.scaffold_cross_split


def test_plinder_release_must_be_frozen():
    with pytest.raises(MinimolError):
        assert_plinder_release_frozen("UNPINNED")
    assert_plinder_release_frozen("2024-06")


def test_target_clusters_must_be_disjoint():
    with pytest.raises(SplitLeakageError):
        assert_disjoint_ids(["a", "b"], ["b"], what="target cluster")


def test_teacher_cache_never_contains_test_labels(tmp_path):
    cache = TeacherCache()
    cache.put("train-row", "train", {"p": 0.2})
    with pytest.raises(SplitLeakageError):
        cache.put("test-row", "test", {"p": 0.9})
    with pytest.raises(SplitLeakageError):
        cache.put("holdout", "release_holdout", {"p": 0.9})
    cache.save(tmp_path / "cache.json")
    restored = TeacherCache().load(tmp_path / "cache.json")
    assert not restored.contains_forbidden_split()


def test_stricter_split_quarantines_leaks_without_rewriting_official_test():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "group_id": ["g1", "g1", "g2", "g3"],
            "split": ["train", "test", "train", "train"],
        }
    )
    before = sorted(frame.loc[frame.split == "test", "sample_id"])
    stricter = make_stricter_split(frame, val_fraction=0.5, seed=0)
    assert list(stricter.loc[stricter.group_id == "g1", "strict_split"]) == ["quarantine", "quarantine"]
    assert_test_ids_unchanged(before, sorted(stricter.loc[stricter.split == "test", "sample_id"]))


def test_time_split_holds_out_the_latest_dates():
    result = time_split(["2020-01-01", None, "2024-01-01", "2021-01-01"], test_fraction=0.34)
    assert result["split"][2] == "test"
    assert 1 in result["missing_date_index"]
    assert result["split"][1] == "train"
