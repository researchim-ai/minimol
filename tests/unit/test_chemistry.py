import pytest

from minimol.data.assays import parse_mutation
from minimol.data.deduplicate import aggregate_replicates
from minimol.data.download import assert_download_allowed, select_files_within_budget
from minimol.data.featurize import atom_feature_dim, bond_feature_dim, mol_to_graph
from minimol.data.standardize import standardize_smiles
from minimol.errors import LicenseError, MinimolError
from minimol.evaluation.guards import (
    assert_endpoints_not_pooled,
    overlapping_scaffolds,
    target_prior_flag,
)
from minimol.oracles.pocket_proxy import pocket_proxy_score


def test_stereo_is_retained_and_salts_are_recorded():
    stereo = standardize_smiles("C[C@H](N)C(=O)O")
    assert stereo["valid"]
    assert stereo["stereochemistry_status"] == "retained"
    assert "@" in stereo["smiles_canonical"]
    salt = standardize_smiles("CC(=O)[O-].[Na+]")
    assert salt["valid"] and salt["salt_stripped"]
    assert "Na" not in salt["smiles_canonical"]
    assert standardize_smiles("not_a_smiles")["valid"] is False
    isotope = standardize_smiles("[13C]O")
    assert isotope["isotope"] is True


def test_feature_layout_and_graph():
    assert atom_feature_dim() == 34
    assert bond_feature_dim() == 11
    graph = mol_to_graph("CCO")
    assert graph.x.shape[0] == 3
    assert graph.edge_index.shape[0] == 2
    lone = mol_to_graph("C")
    assert lone.edge_index.shape[1] == 1


def test_replicates_keep_variance_and_do_not_mix_relations():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "inchikey": ["A", "A", "A"],
            "assay_id": ["x", "x", "x"],
            "endpoint": ["Ki", "Ki", "Ki"],
            "relation": ["=", "=", "<"],
            "measurement": [1.0, 3.0, 0.2],
        }
    )
    aggregated = aggregate_replicates(frame)
    equals = aggregated[aggregated.relation == "="].iloc[0]
    assert equals.n_replicates == 2
    assert abs(equals.measurement - 2.0) < 1e-8
    assert equals.replicate_std > 0
    censored = aggregated[aggregated.relation == "<"].iloc[0]
    assert censored.aggregation == "not_averaged_censored"


def test_download_budget_and_license_gate():
    chosen = select_files_within_budget(
        [{"path": "a", "size": 30}, {"path": "b", "size": 30}, {"path": "c", "size": 50}],
        max_bytes=70,
    )
    assert [item["path"] for item in chosen] == ["a", "b"]
    with pytest.raises(LicenseError):
        assert_download_allowed("REVIEW_REQUIRED", "abc", "CC-BY-4.0")
    with pytest.raises(MinimolError):
        assert_download_allowed("APPROVED", "UNVERIFIED", "CC-BY-4.0")
    assert_download_allowed("LOCAL_USE", "e7310a37e760ebf85564e8f1a94c8bd0ef25e018", "Gosai2024-Nature-supplementary-terms")


def test_endpoint_pooling_and_pocket_proxy():
    import pandas as pd

    clean = pd.DataFrame({"endpoint": ["Ki", "IC50"], "assay_type": ["biochemical_ki", "cell_ic50"]})
    assert_endpoints_not_pooled(clean)
    mixed = pd.DataFrame({"endpoint": ["affinity", "affinity"], "assay_type": ["Ki", "IC50"]})
    with pytest.raises(MinimolError):
        assert_endpoints_not_pooled(mixed)
    assert overlapping_scaffolds(["s1"], ["s2"]) == []
    assert target_prior_flag(0.8, 0.79, 0.4) is True
    proxy = pocket_proxy_score("CCO")
    assert proxy["score"] is None
    assert proxy["is_binding_affinity"] is False
    assert parse_mutation("K12A") == ("K", 12, "A")
    with pytest.raises(ValueError):
        parse_mutation("not-a-mutation")
