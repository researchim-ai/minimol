import logging

from fastapi.testclient import TestClient

from minimol.serving.app import create_app


def test_health_and_abstain_without_weights():
    # Close the client. Starlette's portal thread otherwise stays up and the next test can block.
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["research_use_only"] is True
        body = client.post("/v1/regulatory/predict", json={"sequence": "ACGTACGT", "assay": "K562"})
    assert body.status_code == 200
    payload = body.json()
    assert payload["abstain"] is True
    assert payload["predicted_log2fc"] is None
    assert "не для клинического применения" in " ".join(payload["warnings"]).casefold()


def test_bad_sequence_is_rejected_and_not_logged(caplog):
    with TestClient(create_app()) as client:
        secret = "ACGTXXXXSECRETBASES"
        with caplog.at_level(logging.INFO, logger="minimol.serving"):
            response = client.post("/v1/regulatory/predict", json={"sequence": secret, "assay": "K562"})
    assert response.status_code == 422
    assert secret not in caplog.text


def test_auth_and_rate_limit():
    with TestClient(create_app(token="secret", rate_limit_per_minute=2)) as client:
        assert client.get("/health").status_code == 200
        denied = client.post("/v1/rank", json={"candidates": ["CCO"], "objective": "qed"})
        assert denied.status_code == 401
        headers = {"Authorization": "Bearer secret"}
        first = client.post("/v1/rank", headers=headers, json={"candidates": ["CCO", "CCC"], "objective": "mw", "direction": "minimize"})
        second = client.post("/v1/rank", headers=headers, json={"candidates": ["CCO"], "objective": "qed"})
        third = client.post("/v1/rank", headers=headers, json={"candidates": ["CCO"], "objective": "qed"})
        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 429
        ranked = first.json()["ranked"]
    assert [item["smiles"] for item in ranked] == ["CCC", "CCO"]
    assert ranked[0]["score"] < ranked[1]["score"]
    assert first.json()["learned_model"] is False


def test_predict_uses_rdkit_for_exact_properties_and_abstains_otherwise():
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/predict",
            json={
                "molecules": [{"id": "a", "smiles": "CCO"}, {"id": "b", "smiles": "not_smiles"}],
                "questions": [
                    {"id": "q1", "type": "regression", "endpoint": "logp_rdkit"},
                    {"id": "q2", "type": "noul", "endpoint": "logp_rdkit", "event": {"gt": 5}},
                    {"id": "q3", "type": "regression", "endpoint": "herg_block"},
                ],
            },
        )
        payload = response.json()
        ethanol = payload["predictions"][0]["outputs"]
        assert ethanol[0]["value"] is not None
        assert ethanol[0]["probability"] is None
        assert ethanol[1]["probability"] == 0.0
        assert ethanol[2]["abstain"] is True
        assert payload["predictions"][1]["error"] == "unparseable_smiles"
        cds = client.post("/v1/cds/annotate", json={"sequence": "ATGCCC"})
        assert cds.json()["abstain"] is True
        assert cds.json()["strand_probabilities"] is None


def test_rank_rejects_unknown_objective():
    with TestClient(create_app()) as client:
        response = client.post("/v1/rank", json={"candidates": ["CCO"], "objective": "potency"})
    assert response.json()["abstain"] is True
