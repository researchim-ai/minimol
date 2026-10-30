# Minimol

Очень маленькие исследовательские модели для регуляторной ДНК, кодирующих участков и молекулярных свойств.

Research use only. Not for clinical use, dosing, diagnosis, or regulatory drug-development decisions.

Только для исследований. Не для клинического применения.

Число параметров в названии — бюджет. В отчёт попадает `sum(p.numel())`. Дистилляция включается только если на отложенной выборке она лучше supervised-baseline. Синтетические таблицы в этом репозитории **не** являются измерениями MPRA, RefSeq, TDC или BindingDB.

## Установка

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,chem,serve,export,boost]"
```

Нужны Python 3.10+, PyTorch, RDKit. Версии фиксируются после смоук-теста окружения, а не заранее. `make env-check` пишет `artifacts/reports/system.json` и `artifacts/env.json`.

## Смоук

```bash
make audit-licenses
make data-reg-sample
make check-splits
make teacher-a-smoke
make train-reg-ridge
make train-reg-smoke
make test
```

`make test` гоняет `pytest -q -m "not slow"`. Полные пресеты 5M/10M/30M проверяются тестом бюджетов; обучение 30M на синтетике запускается отдельными целями `make train-reg-30m` и `make train-cds-5m`, когда есть данные и GPU.

Полезные команды:

```bash
minimol data reg-audit --data data/processed/reg_sample/data.parquet
minimol train reg --config configs/reg_5m.yaml --data data/processed/reg_sample/data.parquet --out artifacts/reg_5m
minimol train cds --config configs/cds_5m.yaml --data data/processed/cds_sample/data.parquet --out artifacts/cds_5m
minimol train admet --config configs/models/gine_5m.yaml --data data/processed/admet_sample/data.parquet --out artifacts/admet_5m
minimol train bind --config configs/smoke/bind.yaml --data data/processed/bind_sample/data.parquet --out artifacts/bind_smoke
minimol train precision --config configs/smoke/precision.yaml --data data/processed/precision_sample/data.parquet --out artifacts/precision_smoke
minimol eval controls --data data/processed/reg_sample/data.parquet --out artifacts/reg_controls
minimol eval ood --data data/processed/admet_sample/data.parquet --out artifacts/admet_ood
minimol eval motif --data data/processed/reg_sample/data.parquet --out artifacts/motif_baseline --motif TATAAA --motif CCAAT --motif GGGCGG
minimol generation tasks --out artifacts/tasks.jsonl --limit 20
minimol serve
```

Сервис: `GET /health`, `POST /v1/regulatory/predict`, `POST /v1/cds/annotate`, `POST /v1/predict`, `POST /v1/rank`. Без чекпоинта регуляторная и CDS-ручки отказываются отвечать числом. Точные свойства RDKit (logP, TPSA, QED, масса) считаются напрямую и помечаются как oracle, а не как обученная модель.

## Что уже в коде

- Семейства Reg 1M/5M/10M/30M и CDS 5M/10M/30M: свёртки, U-Net, локальное внимание.
- GINE 5M/15M/30M, направленный MPNN-baseline, ranker, fusion для bind/precision в бюджете до 100M.
- Проверка утечек: точные дубли, reverse complement, Jaccard 6-меров ≥ 0.80, группы, scaffold, InChIKey.
- Потери Huber, Bernoulli KD с необязательным множителем T², граничный член, RC-согласованность цепей.
- Реестр лицензий. Внешние корпуса в статусе `REVIEW_REQUIRED` и не скачиваются.
- Carbon-A / Carbon-X: выравнивание 6-меров и softmax опций. Веса не грузятся без просмотренного `trust_remote_code`.
- ONNX для свёрточных моделей последовательности и FastAPI.

## Чего код намеренно не делает

Не объявляет клиническую пригодность. Не подменяет экспериментальную метку ответом учителя. Не считает сжатие 1.2B до 10M успехом без независимого теста. Не выпускает коммерческую модель командой `release-check`: юридическая и научная проверка остаются `pending`.

## Regulatory decision pilot

Реализован локальный конвейер типизированных вопросов к ДНК: `noul`
(вероятность превышения порога log2FC), `score` (распределение по интервалам),
выбор из двух кандидатов и выбор assay. Один вызов encoder обслуживает несколько
вопросов. Это исследовательский прототип с фиксированной схемой вопросов.

```bash
make decide-prepare                  # нужны локальные Malinois и parent checkpoint
make decide-train DECIDE_SEED=11      # новый каталог; повторить для 23 и 47
make decide-summary
make decide-serve                    # http://127.0.0.1:8097/reg-decide
```

Существующие результаты не перезаписываются: для нового эксперимента задайте
`DECIDE_RUN=artifacts/new_run`. Продолжение нового checkpoint с полным состоянием:
`python -m minimol.cli decide train --config configs/decision/reg_pilot.yaml --cache artifacts/reg_decide_features_v1 --out artifacts/new_run --resume artifacts/new_run/checkpoint_last.pt`.
Старые checkpoint без полного состояния требуют отдельного warm-start эксперимента.

```python
from minimol.decide import RegDecider

model = RegDecider.load("artifacts/reg_decide_seed11")
result = model.predict({
    "candidates": [{"id": "candidate", "sequence": sequence}],
    "questions": [{"id": "q", "kind": "noul", "candidate_id": "candidate",
                   "assay": "K562", "threshold": 0.75}],
})
```

`sequence` — ваша локальная ДНК. POST `/v1/regulatory/decide` принимает ту же
структуру. По умолчанию используется `calibrated_regression`: три head-seed дали
смешанный результат. Экспериментальная голова включается явно через
`RegDecider.load(..., backend="trained")` или `serve --decide-backend trained`.
Порог abstention — пользовательская политика, не гарантия верхней границы риска.
Данные, признаки, предсказания и веса сохраняются в игнорируемых локальных каталогах;
ограничение Malinois `LOCAL_USE` сохраняется. Детали и результаты:
[Decision Model Next Stage](docs/Decision_Model_Next_Stage.md).
