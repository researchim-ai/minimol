PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
export PYTHONPATH := src

.PHONY: env-check audit-licenses data-reg-sample data-cds-sample data-admet-sample check-splits \
	teacher-a-smoke teacher-x-smoke train-reg-ridge train-reg-smoke test release-check

env-check:
	$(PYTHON) -m minimol.cli env-check

audit-licenses:
	$(PYTHON) -m minimol.cli audit-licenses

data-reg-sample:
	$(PYTHON) -m minimol.cli data reg-sample --out data/processed/reg_sample --n 40

data-cds-sample:
	$(PYTHON) -m minimol.cli data cds-sample --out data/processed/cds_sample

data-admet-sample:
	$(PYTHON) -m minimol.cli data admet-sample --out data/processed/admet_sample

check-splits: data-reg-sample
	$(PYTHON) -m minimol.cli check-splits --manifest data/processed/reg_sample/manifest.json

teacher-a-smoke:
	$(PYTHON) -m minimol.cli teacher carbon-a-smoke

teacher-x-smoke:
	$(PYTHON) -m minimol.cli teacher carbon-x-smoke --with-metric 0.2 --without-metric 0.4

train-reg-ridge: data-reg-sample
	$(PYTHON) -m minimol.cli train reg-ridge --data data/processed/reg_sample/data.parquet --out artifacts/reg_ridge

train-reg-smoke: data-reg-sample
	$(PYTHON) -m minimol.cli train reg --config configs/smoke/reg.yaml --out artifacts/reg_smoke

train-reg-5m: data-reg-sample
	$(PYTHON) -m minimol.cli train reg --config configs/reg_5m.yaml --out artifacts/reg_5m

train-reg-10m: data-reg-sample
	$(PYTHON) -m minimol.cli train reg --config configs/reg_10m.yaml --out artifacts/reg_10m

train-reg-30m: data-reg-sample
	$(PYTHON) -m minimol.cli train reg --config configs/reg_30m.yaml --out artifacts/reg_30m

train-cds-5m: data-cds-sample
	$(PYTHON) -m minimol.cli train cds --config configs/cds_5m.yaml --out artifacts/cds_5m

test:
	$(PYTHON) -m pytest -q -m "not slow"

release-check:
	$(PYTHON) -m minimol.cli release-check

model-card:
	@echo "Pass metrics from a real run: python -m minimol.cli model-card --metrics artifacts/.../metrics.json --out artifacts/MODEL_CARD.md"
