from minimol.cli import main


def test_cli_sample_audit_and_teacher(tmp_path):
    out = tmp_path / "reg"
    assert main(["data", "reg-sample", "--out", str(out), "--n", "8", "--length", "64", "--seed", "1"]) == 0
    assert main(["data", "reg-audit", "--data", str(out / "data.parquet")]) == 0
    assert main(["teacher", "carbon-a-smoke", "--sequence", "ATGCCCAAATTT"]) == 0
    assert main(["check-splits", "--manifest", str(out / "manifest.json")]) == 0
    assert main(["data", "build", "--dataset", "tdc_admet", "--out", str(tmp_path / "nope")]) == 1
