import gc

import pytest
import torch

from minimol.models.blocks import conv_receptive_field, count_parameters, make_dilations
from minimol.models.cds_cnn import MinimolCDS
from minimol.models.gnn import DirectedMPNN, build_gine
from minimol.models.heads import OrdinalHead, PairwiseRanker, ordinal_loss
from minimol.models.presets import PARAM_BANDS, RANK_PRESETS
from minimol.models.protein_fusion import MinimolBind, MinimolPrecision
from minimol.models.reg_cnn import MinimolReg
from minimol.models.shared import SharedStem


def _expect(name, module):
    count = count_parameters(module)
    low, high = PARAM_BANDS[name]
    assert low <= count <= high, (name, count, low, high)
    del module
    gc.collect()


def test_parameter_budgets():
    for preset, key in (("1m", "reg_1m"), ("5m", "reg_5m"), ("10m", "reg_10m"), ("30m", "reg_30m")):
        _expect(key, MinimolReg(preset))
    for preset, key in (("5m", "cds_5m"), ("10m", "cds_10m"), ("30m", "cds_30m")):
        _expect(key, MinimolCDS(preset))
    for preset, key in (("5m", "admet_5m"), ("15m", "admet_15m"), ("30m", "admet_30m")):
        _expect(key, build_gine(34, 11, preset=preset, n_assays=4))
    _expect("rank_5m", PairwiseRanker(**RANK_PRESETS["5m"]))
    _expect("bind_60m", MinimolBind(34, 11))
    _expect("precision_100m", MinimolPrecision(34, 11))


def test_shapes_and_receptive_field():
    reg = MinimolReg("5m", channels=32, n_blocks=2, dropout=0.0)
    assert reg(torch.zeros(2, 5, 40)).shape == (2, 3)
    cds5 = MinimolCDS("5m")
    cds30 = MinimolCDS("30m")
    assert cds5.receptive_field_bp == conv_receptive_field(make_dilations(8))
    assert cds5.receptive_field_bp < cds30.receptive_field_bp
    assert cds30.attention_window >= 128
    shared = SharedStem(32, 2)
    assert shared(torch.zeros(1, 5, 24)).shape[1] == 32
    graph_x = torch.randn(4, 8)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    rev = torch.tensor([1, 0, 3, 2])
    attr = torch.randn(4, 4)
    batch = torch.tensor([0, 0, 0, 0])
    out = DirectedMPNN(8, 4, hidden=16, depth=2)(graph_x, edges, attr, rev, batch, 1)
    assert out.shape == (1, 16)
    lone = DirectedMPNN(8, 4, hidden=16, depth=2)(
        torch.randn(1, 8),
        torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, 4),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        1,
    )
    assert lone.shape == (1, 16)


def test_ordinal_probabilities_sum_to_one():
    head = OrdinalHead(8, n_classes=4)
    hidden = torch.randn(5, 8)
    probs = head.class_probabilities(hidden)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)
    labels = torch.tensor([0, 1, 2, 3, 1])
    loss = ordinal_loss(head.cumulative_logits(hidden), labels, n_classes=4)
    assert torch.isfinite(loss)


def test_decision_router_abstains():
    from minimol.models.decision import DecisionRequest, DecisionRouter

    router = DecisionRouter(known_endpoints=["logp_rdkit"])
    missing = router.answer(DecisionRequest("admet", "regression", endpoint="clinical_dose"))
    assert missing["abstain"] and missing["reason"] == "unsupported_endpoint"
    unloaded = router.answer(DecisionRequest("admet", "noul", endpoint="logp_rdkit", event={"gt": 1}))
    assert unloaded["reason"] == "weights_not_loaded"
    choice = router.answer(DecisionRequest("admet", "choice", endpoint="logp_rdkit"))
    assert choice["reason"] == "choice_requires_options"
    answered = router.answer(
        DecisionRequest("admet", "regression", endpoint="logp_rdkit"),
        prediction={"value": 1.2, "interval": [1.0, 1.4]},
    )
    assert answered["probability"] is None
    assert answered["value"] == 1.2
    assert any("не для клинического" in warning.casefold() for warning in answered["warnings"])


@pytest.mark.slow
def test_full_reg_forward():
    model = MinimolReg("10m")
    out = model(torch.zeros(1, 5, 32))
    assert out.shape == (1, 3)
    assert torch.isfinite(out).all()
