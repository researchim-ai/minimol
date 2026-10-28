# Воспроизводимость

Каждый обучающий запуск пишет `metrics.json`, `checkpoint_best.pt` и хэш таблицы. В метаданных чекпоинта есть seed, git SHA (если репозиторий доступен) и `param_count`.

`set_seed` фиксирует Python, NumPy и PyTorch. На GPU бит-в-бит совпадение не обещается: детерминированные флаги cuDNN не покрывают все ядра.

BF16 включается только на CUDA, если ядра его поддерживают. Иначе обучение идёт в fp32. Память двух RTX 3090 не складывается в один пул: это два бюджета по 24 ГиБ.

Официальный test не переписывается. Teacher-кэш отвергает split `test` и `release_holdout`. Скалер меток учится только на train.

Команда `benchmark --suite tdc-admet` без `--official-tdc` завершается ошибкой. Синтетический прокси нельзя подписать как таблицу TDC.

Повторение смоук-прогона:

```bash
python -m minimol.cli data reg-sample --out data/processed/reg_sample --n 40
python -m minimol.cli data reg-audit --data data/processed/reg_sample/data.parquet
python -m minimol.cli train reg-ridge --data data/processed/reg_sample/data.parquet --out artifacts/reg_ridge
python -m minimol.cli train reg --config configs/smoke/reg.yaml --data data/processed/reg_sample/data.parquet --out artifacts/reg_smoke
pytest -q
```
