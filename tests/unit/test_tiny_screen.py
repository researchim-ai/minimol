import pandas as pd
import torch

from minimol.data.inner_split import locus_group, partition_name, write_inner_split
from minimol.models.blocks import count_parameters
from minimol.models.tiny_screen import TinyScreen
from minimol.training.train_tiny_screen import RECIPES


def test_overlapping_windows_share_a_partition():
    left = locus_group("1:1000:A:C:A:wC")
    right = locus_group("1:1100:A:G:A:wC")
    assert left == right
    assert partition_name(left) == partition_name(right)
    assert partition_name(left) in {"inner_train", "development", "calibration"}


def test_inner_split_is_immutable_and_ignores_official_holdouts(tmp_path):
    rows = []
    for index, split in enumerate(("train", "train", "validation", "test")):
        rows.append(
            {
                "split": split,
                "sequence_hash": f"h{index}",
                "group_id": f"2:{1000 + index * 500}:A:C:A:wC",
                "sequence": "ACGT",
                "assay": "K562",
                "y_log2fc": 0.0,
                "y_se": 0.1,
                "valid_y": True,
            }
        )
    path = tmp_path / "data.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    first = write_inner_split(path, tmp_path / "split")
    second = write_inner_split(path, tmp_path / "split")
    assert first["assignments_sha256"] == second["assignments_sha256"]
    assert first["official_test_used"] is False
    assert first["n_sequences"] == 2
    assignments = pd.read_parquet(tmp_path / "split" / "assignments.parquet")
    assert set(assignments.sequence_hash) == {"h0", "h1"}


def test_tiny_recipes_match_native_and_padded_lengths():
    torch.manual_seed(0)
    sequence = torch.zeros(1, 5, 12)
    sequence[0, 0, :] = 1.0
    padded = torch.zeros(1, 5, 20)
    padded[:, :, :12] = sequence
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[:, :12] = True
    neighbor = torch.zeros(2, 5, 20)
    neighbor[0, :, :12] = sequence[0]
    neighbor[1, 1, :] = 1.0
    neighbor_mask = torch.zeros(2, 20, dtype=torch.bool)
    neighbor_mask[0, :12] = True
    neighbor_mask[1, :] = True
    for recipe in (*RECIPES, "positional_huber", "huber_matched"):
        model = TinyScreen(recipe, n_assays=3, dropout=0.0)
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.05)
        model.eval()
        native = model(sequence)
        batched = model(padded, mask)
        mixed = model(neighbor, neighbor_mask)
        left = native[0] if isinstance(native, tuple) else native
        right = batched[0] if isinstance(batched, tuple) else batched
        other = mixed[0][0] if isinstance(mixed, tuple) else mixed[0]
        torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(left[0], other, atol=1e-4, rtol=1e-4)
        assert count_parameters(model) < 1_500_000


def test_windows_across_a_bin_boundary_stay_together(tmp_path):
    from minimol.data.inner_split import write_inner_split_v2

    rows = []
    sequences = ("ACGT" * 50, "ACGT" * 49 + "TGCA", "TTTT" * 50)
    for index, position in enumerate((190, 210, 5000)):
        rows.append(
            {
                "split": "train",
                "sequence_hash": f"h{index}",
                "group_id": f"1:{position}:A:C:A:wC",
                "sequence": sequences[index],
                "assay": "K562",
                "y_log2fc": 0.0,
                "y_se": 0.1,
                "valid_y": True,
            }
        )
    path = tmp_path / "data.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    manifest = write_inner_split_v2(path, tmp_path / "split", giant_component=10)
    assignments = pd.read_parquet(tmp_path / "split" / "assignments.parquet")
    groups = dict(zip(assignments.sequence_hash, assignments.locus_group, strict=True))
    assert groups["h0"] == groups["h1"]
    assert groups["h0"] != groups["h2"]
    assert manifest["version"] == "malinois_inner_v2"


def test_factorial_recipes_are_allowed_and_selection_uses_predicted_scale():
    from minimol.models.tiny_screen import event_probability
    from minimol.training.train_tiny_screen import (
        ALLOWED_RECIPES,
        FACTORIAL,
        _selection_probability,
    )

    assert set(FACTORIAL) <= ALLOWED_RECIPES
    mu = torch.zeros(2, 3)
    scale = torch.full((2, 3), 0.1)
    thresholds = torch.tensor([1.0])
    chosen = _selection_probability((mu, scale), thresholds)
    torch.testing.assert_close(chosen, event_probability(mu, thresholds, scale))
    unit = event_probability(mu, thresholds, torch.ones_like(mu))
    assert float((chosen - unit).abs().max()) > 0.1


def test_scale_is_fit_on_calibration_only():
    from minimol.training.train_tiny_screen import _apply_scale, _constant_scale

    mu = torch.zeros(4, 2)
    y_cal = torch.tensor([[0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [2.0, 2.0]])
    mask = torch.ones(4, 2, dtype=torch.bool)
    fitted = _constant_scale(mu, y_cal, mask)
    applied = _apply_scale(torch.zeros(2, 2), fitted)
    assert torch.allclose(applied[0], fitted)
    assert float(fitted[0]) > 0.5
