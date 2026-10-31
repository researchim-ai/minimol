# Minimol-Decision: отдельная LoRA-ветка и benchmark

> Исполняемый план следующего этапа для агента: [AGENT_EXECUTION_PLAN](AGENT_EXECUTION_PLAN.md).

31 октября 2026. Reg/ADMET/CDS и компактные специализированные encoder остаются
основной линейкой. Добавляется экспериментальный слой решений на pretrained LM.
Это Jev-подобный контракт state + question + allowed answers, не реализация
проприетарной архитектуры, training recipe или гарантий Jev.

## Что реализовано

`scripts/tune_decision_lora.py`: build-demo, validate, train, predict, baseline,
calibrate, score. Core/schema: `src/minimol/decision_lora.py`.

- LoRA и QLoRA NF4 double quantization; all-linear adapters, rank16/alpha32 по
  умолчанию, опциональный rSLora. Gradient checkpointing, mixed precision,
  один GPU или torchrun DDP. DDP реплицирует модель; две 24-GB карты не превращаются
  автоматически в единый 48-GB GPU. Для маленьких моделей удобно обучать разные
  seeds на разных картах; DDP ускорение нужно измерить.
- SFT loss только на answer-code и EOS, prompt/padding masked. Слишком длинные
  примеры отклоняются, а не обрезаются с потерей вопроса/ответа. Без hidden CoT.
- Inference сравнивает полные conditional log-likelihood допустимых кодов + EOS,
  включая multi-token codes. Возвращает restricted categorical distribution и
  выбранное типизированное значение. Нет свободной генерации JSON и parsing repair.
  Оценка кандидатов последовательная: это не Jev shared-state parallel execution;
  стоимость растёт с числом вариантов. Позже возможна dedicated classification head.
- Запрещены повторное использование run directory и чужой resume checkpoint.
  Resume проверяет подпись запуска/данных/кода; используется механизм HF Trainer,
  bit-exact DDP reproducibility не заявляется. Записываются версии библиотек,
  base revision, размер адаптера в параметрах, loaded params и GPU peak memory.
- Probabilities не объявляются calibrated confidence. Отдельная scalar temperature
  на calibration predictions; применение только к тому же model/adapter/data identity.
  Отказ — явный вариант `abstain`, который должен быть размечен в задаче.

## Какие способы тюнинга использовать

1. **SFT + LoRA (первый эксперимент).** Учим нужные решения по качественной разметке.
   Qwen3-0.6B, rank16, context2048, microbatch1, accumulation16, LR1e-4, 3 эпохи —
   исходная гипотеза, не оптимальный рецепт. Ранги8/16/32 проверять ограниченно.
2. **QLoRA.** Тот же objective и adapters поверх 4-bit base, экономия VRAM.
   Не обещает улучшить качество относительно LoRA; сравнить оба на одинаковых seeds.
3. **rSLora.** Включаемая абляция scaling, полезность измерить, не менять одновременно
   rank, данные и objective. DoRA/IA3/prompt tuning пока не нужны для первого screen.
4. **Full fine-tune** небольшой base — полезный контроль, если позволяет VRAM/бюджет;
   текущий скрипт реализует adapters, full fine-tune пока не добавлен.
5. **DPO** после SFT только если есть надёжные chosen/rejected решения, особенно
   near-misses и ошибки abstention. Не заменяет supervised ground truth.
6. **GRPO/RL с проверяемой reward** — позже для многошаговых/compositional задач,
   где есть независимый проверяющий. На простом finite classification часто лучше
   начинать с supervised learning. Reward за формат не доказывает био-качество.
7. Distillation в меньшую LM/encoder после подтверждения teacher. Модель не должна
   получать teacher leakage из development/calibration/test.

## Модели и оборудование

Первый base: Qwen/Qwen3-0.6B; второй — Qwen/Qwen3-1.7B. Обе model cards имеют
Apache-2.0. Это сотни миллионов/миллиарды полных параметров, а не 1M encoder:
маленький LoRA adapter не делает base маленьким при deployment.

Для одной RTX3090 24GB LoRA 0.6B при небольшом контексте — разумный старт;
1.7B LoRA/QLoRA — следующий пилот. Точный VRAM/throughput измерить на реальных
примерах. Не обещаем размер batch или часы на основании одной ёмкости GPU.
Вторую карту сначала использовать под независимый seed/base. На текущем хосте
проверены RTX3090Ti и RTX3090, обе около24GB. Проверки на случайных крошечных
моделях подтверждают pipeline, а не качество Qwen или биологические способности.
Новые pretrained веса в этой реализации не скачивались.

## Запуск

Из корня repo, PYTHONPATH=src (или editable install):

```bash
python -m pip install -e '.[decision-lora]'
export PYTHONPATH=src
python scripts/tune_decision_lora.py build-demo --out artifacts/decision_demo/benchmark.jsonl --groups 400
python scripts/tune_decision_lora.py validate --data artifacts/decision_demo/benchmark.jsonl
```

Hub revision обязательно фиксировать полным commit SHA. Получить актуальный SHA
можно через `HfApi().model_info('Qwen/Qwen3-0.6B').sha`, записать его в run config.
Указать его вместо `COMMIT_SHA` ниже. Первый реальный запуск скачает base с HF;
существующий local checkpoint можно передать путём.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tune_decision_lora.py train \
  --model Qwen/Qwen3-0.6B --revision COMMIT_SHA \
  --data artifacts/decision_demo/benchmark.jsonl --out artifacts/decision_lora_qwen06
```

Добавить `--qlora` для NF4, `--rslora` для соответствующей абляции.
Для двух GPU: `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2
scripts/tune_decision_lora.py train ...`; accumulation8 вместо16 сохраняет общий
batch16 при microbatch1. Без этого общий batch удвоится. Не использовать device_map=auto
как замену DDP. Resume: те же аргументы + `--resume RUN/checkpoint-N`.

```bash
python scripts/tune_decision_lora.py predict \
  --model Qwen/Qwen3-0.6B --revision COMMIT_SHA \
  --adapter artifacts/decision_lora_qwen06/adapter \
  --data artifacts/decision_demo/benchmark.jsonl --split calibration \
  --out artifacts/decision_lora_qwen06/calibration_predictions.json
python scripts/tune_decision_lora.py calibrate \
  --data artifacts/decision_demo/benchmark.jsonl \
  --predictions artifacts/decision_lora_qwen06/calibration_predictions.json \
  --out artifacts/decision_lora_qwen06/temperature.json
```

Аналогично получить development predictions через `predict --split development`,
затем `score --data ... --predictions ... --out ... --calibration .../temperature.json`.
Baseline без adapter: тот же `predict`, убрать `--adapter`.
Rule/uniform: `baseline --data ... --kind policy|uniform --out ...`.
Test использовать только после фиксации всех моделей/гиперпараметров.

## Benchmark v0: что есть и чего он не доказывает

Создан локальный `artifacts/decision_lora_demo/v2_benchmark.jsonl`: 1200 вопросов,
400 групп, train840 / development120 / calibration120 / test120 вопросов.
Три задачи: constrained choice, threshold boolean, ordinal score; отсутствующие
данные требуют abstain в constrained choice. Answer-code assignments перемешаны.
Все вопросы одного состояния/кандидатов остаются в одном split.

Это **синтетическая policy compliance**, не реальные MPRA/ADMET измерения.
Rule solver знает правила и должен иметь accuracy100%; нейросеть не нужна для
точного исполнения этих простых правил. Его смысл — проверка ceiling и сложности,
а не слабый соперник, которого мы ожидаем победить. Uniform — chance control.
Benchmark не содержит настоящего human-labelled interpretation challenge;
обучение только на нём не даст готовую научную decision-модель.

Record JSONL: id, group_id, entity_ids, split, task, track, source,
license_status, state, question, options[{code,value,description}], answer.
`validate` проверяет id, варианты/метки, disjoint groups/entities/identical states.
Перед созданием пар importer обязан объединить связанные сущности/локусы/
scaffolds/studies в split units: валидатор не обнаруживает неизвестные ему
гомологию и chemical-scaffold similarity. IDs должны быть глобальными, не
просто 'candidate1' во всех исследованиях. Gold answer/provenance/split не входят
в prompt. `track` отделяет synthetic_policy, evidence_decision, sequence_prediction.
Метрики считаются отдельно по track/task: accuracy, multiclass Brier, log loss,
число примеров/групп. Не сравнивать этот multiclass Brier с Reg binary-event Brier.

## Benchmark v1: следующий обязательный объём работы

A. **Policy/robustness**: unseen question templates, смена units/порогов/числа
кандидатов, order permutations, missing/conflicting evidence, near-ties,
неподдерживаемые assays, irrelevant fields и prompt-injection внутри state.
Фиксировать accuracy, constraint violations, consistency и false confident answers.
Нынешний v0 не имеет held-out templates; это отдельное расширение.

B. **Real evidence decisions**: реальные разрешённые измерения + вопросы о выборе
эксперимента/приоритетах, разметка с provenance. Если state содержит прогнозы
Minimol, получать их out-of-fold для train и от чистого encoder для holdout;
финальные biological outcomes не включать в state, если они являются целевой
неизвестной. Разделять measured-evidence reasoning и prediction of unknown outcomes.

C. **Raw sequence/scaffold**: отдельный track DNA/SMILES→результат. Для LM без
биопредобучения здесь нет гарантированного преимущества; специализированные
Minimol encoder обязательные соперники. Не выдавать evidence reasoning за
предсказание активности последовательности.

D. **Comparison protocol**: base LM zero/few-shot, LoRA, QLoRA, explicit rules,
специализированный encoder, hybrid evidence+LM; одинаковые доступные evidence.
Три seeds, paired bootstrap по biological units, coverage/risk и calibration,
per-domain/task metrics, complete latency + peak VRAM + полный deployment size.
Не использовать LLM-судью как единственный источник biological truth.

E. **Release gates**: superiority по заранее выбранной task metric против
сильнейшего подходящего baseline; независимый test, устойчивость, права на base,
данные и производные adapters. LOCAL_USE исходных Malinois не превращается в
разрешение публиковать adapter. Публикация/Hub push скриптом выключены.

## Источники

- Jev contract: https://docs.typesafe.ai/introduction и https://docs.typesafe.ai/models
- Base cards: https://huggingface.co/Qwen/Qwen3-0.6B и https://huggingface.co/Qwen/Qwen3-1.7B
- LoRA/QLoRA: https://huggingface.co/docs/peft/developer_guides/quantization
- SFT: https://huggingface.co/docs/trl/sft_trainer
- DPO/GRPO: https://huggingface.co/docs/trl/dpo_trainer и https://huggingface.co/docs/trl/grpo_trainer
