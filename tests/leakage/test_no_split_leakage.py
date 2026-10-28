import pandas as pd

from minimol.data.homology import NEAR_DUPLICATE_RULE, jaccard, scan_leakage
from minimol.data.sequences import reverse_complement, sequence_hash


def _frame(rows):
    return pd.DataFrame(rows)


def test_exact_and_group_leakage_fail():
    frame = _frame(
        [
            {"sequence": "ACGTACGT", "split": "train", "group_id": "g", "sequence_hash": sequence_hash("ACGTACGT")},
            {"sequence": "ACGTACGT", "split": "test", "group_id": "g", "sequence_hash": sequence_hash("ACGTACGT")},
        ]
    )
    report = scan_leakage(frame)
    assert not report.passed
    assert report.exact_cross_split
    assert report.group_cross_split
    assert "Jaccard" in NEAR_DUPLICATE_RULE


def test_reverse_complement_across_splits_fails():
    forward = "AAAAACCCCCGGGGGTTTTT"
    frame = _frame(
        [
            {"sequence": forward, "split": "train", "group_id": "a"},
            {"sequence": reverse_complement(forward), "split": "test", "group_id": "b"},
        ]
    )
    report = scan_leakage(frame)
    assert report.rc_cross_split
    assert not report.passed


def test_near_duplicate_threshold_flags_a_snp_but_not_random_dna():
    left = "ACGT" * 30
    right = "CCGT" + "ACGT" * 29
    random = "TTTT" * 30
    assert jaccard(left, right) >= 0.8
    assert jaccard(left, random) < 0.8
    frame = _frame(
        [
            {"sequence": left, "split": "train", "group_id": "a"},
            {"sequence": right, "split": "test", "group_id": "b"},
            {"sequence": random, "split": "test", "group_id": "c"},
        ]
    )
    report = scan_leakage(frame)
    pairs = {(item["i"], item["j"]) for item in report.near_cross_split}
    assert (0, 1) in pairs
    assert (0, 2) not in pairs


def test_same_split_variants_are_not_failures():
    left = "ACGT" * 20
    right = "CCGT" + "ACGT" * 19
    frame = _frame(
        [
            {"sequence": left, "split": "train", "group_id": "g"},
            {"sequence": right, "split": "train", "group_id": "g"},
        ]
    )
    assert scan_leakage(frame).passed
