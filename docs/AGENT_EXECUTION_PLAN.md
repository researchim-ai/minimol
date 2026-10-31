# Minimol — исполняемый план для агента

Версия 1, 31 октября 2026. Статус: **задание на следующую реализацию**.
Этот файл определяет порядок работы; это не отчёт о выполнении перечисленных этапов.
При расхождении старых планов с этим документом ближайшую очередь брать отсюда.

## 1. Цель проекта и состав результата

Сохранить и развивать две линии:

1. **Специализированные модели** Reg, ADMET, позднее CDS/Rank: компактные
   task-specific encoder, ориентировочно 0.3–5M параметров, локальный inference.
2. **Minimol-Decision**: Jev-подобный интерфейс state + typed questions → typed
   answers/distributions, первоначально pretrained LM + LoRA/QLoRA. Это отдельный
   продукт; не выдавать его за воспроизведение Jev или за encoder на 1M.

Первый полезный вертикальный сценарий: пользователь передаёт кандидатов,
доступные измерения/прогнозы и ограничения, задаёт вопросы о сравнении и выборе
следующего эксперимента. Система выдаёт допустимые ответы, вероятности, provenance
и явный отказ при неподдерживаемой постановке. Прогноз неизвестного свойства из
DNA/SMILES и рассуждение по уже известным измерениям оцениваются раздельно.

Обязательный результат первой очереди: чистый воспроизводимый benchmark,
исправленный Reg Tiny, реально обученные и оценённые Decision adapters,
сопоставимые baselines, локальный SDK/API/demo и release candidate с честной
model/data card. Публичный выпуск не является автоматическим шагом этого задания.
Нельзя гарантировать superiority до эксперимента. Честный отрицательный результат
с воспроизводимыми артефактами — допустимый исход исследования, но не успешный
quality gate выпуска модели.

## 2. Проверенная исходная точка

- Tiny screen: `artifacts/tiny_screen/screen_summary.json`, шесть рецептов,
  seed11, 6 эпох. Positional: 699139 параметров, development Brier0.0956687,
  Huber0.0992231; относительный выигрыш3.58%, к distribution1.17%.
- До подтверждения устранить calibration leakage, positional padding dependency,
  негарантированные interval splits, train/eval threshold overlap и normalization.
- `src/minimol/decision_lora.py` + `scripts/tune_decision_lora.py` реализуют
  contract/validation, synthetic demo, LoRA/QLoRA/rSLora, finite-option inference,
  temperature calibration и scoring. Это стартовая инфраструктура, не готовая LM.
- `artifacts/decision_lora_demo/v2_benchmark.jsonl`: 1200 синтетических вопросов,
  400 групп; это policy smoke benchmark. Ему не приписывать bio-качество.
- Проверены CPU smoke, NF4 QLoRA на одной GPU, LoRA DDP на двух GPU на случайных
  маленьких моделях. Полноценный Qwen в этой ветке ещё не обучался.
- На последней проверке109 тестов passed, один slow deselected. Агент должен
  заново проверить актуальный код; это исходный ориентир, не ожидаемый вечный count.
- GPU хоста: RTX3090Ti + RTX3090, примерно24GB каждая. Не останавливать чужие jobs.
- Локальные Malinois/производные веса имеют LOCAL_USE. Новая LM/base license
  не отменяет ограничения данных. Сырые данные, adapters, checkpoints, predictions
  и caches должны оставаться в игнорируемых локальных каталогах.

Контекст: `Decision_Model_Next_Stage.md`, `Decision_LoRA_and_Benchmark.md`, оба
master plan. Сохранять существующие незакоммиченные изменения.

## 3. Как агент ведёт работу

- Выполнять этапы по зависимостям; не запускать длинное обучение до data/code gates.
- До начала записать `artifacts/execution/status.json`: plan_version, git HEAD,
  dirty diff hash, runtime versions, этап, очередь, completed/failed/pending jobs.
  Обновлять атомарно после этапа/job; сохранять resume command и причины остановки.
- Для каждого task ниже фиксировать pending/running/passed/failed/deferred,
  ссылки на артефакты и доказательство acceptance. Не отмечать done без проверки.
- Рабочие результаты — уникальные versioned каталоги, старые runs не перезаписывать.
- Не запускать бесконечный sweep. Сначала профиль, затем заранее записанная матрица.
  При недостатке ресурсов уменьшать очередь с записью причины, не менять test.
  Стартовые локальные лимиты: profiling суммарно до1 GPU-hour; E4 screen до12
  GPU-hours; подтверждающая Decision серия до24 GPU-hours; Reg R1 до24 GPU-hours.
  GPU-hours суммируются по занятым картам (2 карты × 1 час = 2 GPU-hours).
  Это верхние границы первого прохода, не обещание конвергенции. После лимита
  checkpoint+report; автоматическое расширение бюджета запрещено. Облачные/платные
  ресурсы не использовать. При явном бюджете пользователя он имеет приоритет.
- Если внешний ресурс/права/экспертная разметка отсутствуют, завершить независимую
  работу, записать точный blocker. Не выдумывать разрешения или reviewed labels.
- В конце каждого этапа обновлять этот checklist или отдельный status-report.
  Итог: что реализовано, реальные результаты, какой gate пройден, что ещё открыто.

## 4. Зависимости и очередь

E0 → E1 → E2 → E3 → E4 → E5 → E6 → E7 → E8.
Reg fixes R0 делать после E0; Reg confirmation R1 требует E2 и R0.
E4 и R1 можно исполнять независимыми jobs на разных GPU после общего data gate.
ADMET A1 начинать после первого работающего Reg/Decision вертикального сценария.
CDS/Rank остаются в roadmap, не блокируют первый выпуск и не объявляются готовыми.

### E0 — инвентаризация, environment, воспроизводимость

- [ ] Прочитать local AGENTS.md, если он есть; изучить git status, активные процессы,
  доступную VRAM/диск и артефакты. Не коммитить/удалять чужую работу.
- [ ] Проверить `pyproject.toml`, зафиксировать реально работающие версии torch,
  transformers, peft, accelerate, bitsandbytes, CUDA/driver и GPU.
- [ ] Зависимости устанавливать в проектное окружение; не обновлять глобальный Python.
- [ ] Добавить preflight: schema/tokenizer/dtype/chat template/GPU/disk/sequence lengths.
  Для Hub base фиксировать commit SHA, для local base — хэши config/tokenizer/weights.
- [ ] Сделать runner smoke воспроизводимым: генерация маленького local random base,
  один optimizer step, save/reload, predict/calibrate/score; отдельно single GPU
  QLoRA и DDP. GPU checks optional в CPU CI, но обязательны перед GPU-серией.

Acceptance: tests + smoke проходят, dependency snapshot и источник base фиксированы,
нет зависимости от секретов/путей другой машины. Артефакты: `artifacts/execution/env/`.

### E1 — довести Decision runner до безопасной научной серии

Файлы: `src/minimol/decision_lora.py`, `scripts/tune_decision_lora.py`, новые
конфиги в `configs/decision_lora/`, специализированные unit/integration tests.

- [ ] Разделить public request и labelled benchmark record. Inference принимает
  только state/question/options; gold labels и partition вообще не требуются.
- [ ] Добавить явный question kind: boolean, choice, ordinal_score. Проверять типы
  option.value, порядок ordinal bins, distinct codes, обязательные units/schema version.
  Сначала finite score bins; continuous score не обещать и не подменять free text.
- [ ] Поддержать request с несколькими независимыми вопросами к одному state;
  вернуть ответы с question_id в исходном порядке. Не заявлять shared encoder/KV
  optimization до её реализации и benchmark.
- [ ] Сделать конфигурации исполняемыми: YAML loader + resolved_config.json;
  тот же backend CLI, без двух расходящихся источников defaults. Validator должен
  принимать явный набор required splits: train/dev/cal валидируются без test labels;
  существующий validate требует все четыре split, это изменить в рамках E1.
- [ ] Сохранять base/tokenizer/chat-template hashes, adapter hashes, training dataset
  и split hashes, source snapshot, actual trainable/full params, context, precision,
  quantization, global batch, seeds, peak allocated/reserved VRAM, time/throughput.
  Quantized numel не выдавать за точное исходное число параметров base.
- [ ] Проверить DDP directory creation/resume guards на гонки; согласовать ошибки
  между rank, сохранять файлы одним rank. Interrupted job не считать завершённым.
- [ ] Resume разрешать только с совпадающими data/code/model/config signatures;
  проверить 1GPU continuity, DDP не объявлять bit-exact без отдельного доказательства.
- [ ] Prediction artifact обязан фиксировать base/tokenizer/adapter identity,
  scoring policy/EOS/temperature/max length и data hash. При смене этих условий
  старую calibration отклонять. Не применять calibration к другой quantization
  без отдельной проверки/повторного fit на calibration.
- [ ] Устранить silent truncation и silent task skipping; invalid examples — отдельный
  report с причиной, научная серия падает до train при невалидном наборе.

Acceptance: реальные tokenizer examples проверены, prompt/pad не участвуют в loss,
полный option likelihood включает все answer tokens, вероятности конечны и sum=1,
labels никогда не входят в public inference. Artifact reload воспроизводит predict.

### E2 — immutable benchmark v1, три независимых трека

Реализовать `src/minimol/data/decision_benchmark.py` и CLI builder/validator.
Это **новые интерфейсы**, их сначала создать и документировать; не считать
существующим CLI старый inner-split v1 с проблемной группировкой.

**Track P: synthetic policy + robustness.**

- [ ] Предлагаемый старт: 10000 train states, 2000 development, 2000 calibration,
  2000 held-out test states; несколько вопросов на состояние, stats по states
  и questions отдельно. Это целевые размеры генератора, не число измерений.
- [ ] До вопросов разбить группы кандидатов. Развести template families между
  train/development/test; шаблоны и словари версионировать. Calibration не смешивать
  с development. Gold получаем независимым точным evaluator правил.
- [ ] Добавить constraint choice, boolean thresholds, ordinal scores, ties,
  multiple/no feasible options, missing/conflicting evidence, unit conversions,
  unsupported assay/schema, irrelevant fields, перестановки вариантов и state.
- [ ] Adversarial strings внутри state не должны менять контракт. Prompt instruction
  сам по себе не гарантирует устойчивость; нужны отдельные cases и метрики.
- [ ] Exact rule solver — ceiling для формализованной задачи. Он может быть лучше LM;
  не ставить цель «обойти 100%». Uniform и answer-frequency priors — controls.

**Track E: реальные evidence decisions.**

- [ ] Сначала Reg: реальные допустимые measured inputs и задачи выбора/сравнения.
  Для reasoning по известным измерениям явно отметить, что target не неизвестен.
- [ ] Для prediction будущего результата убрать его из state. Входом могут быть
  training-clean encoder forecasts, uncertainty, assay и ограничения пользователя.
  Teacher forecasts на train — out-of-fold; outer holdout не видел teacher training.
- [ ] Разметка: формальные вопросы — из измерений по заранее заданным правилам;
  неоднозначные вопросы — reviewed labels с disagreement/adjudication. Агент не
  может сам объявить свою разметку экспертно проверенной. Такие cases не включать
  в подтверждающий benchmark без review.
- [ ] Источники/assay/study/units/репликаты/SE и лицензии сохранять в provenance,
  без подсказок ответа. Реальные наблюдения не дублировать как независимые примеры.

**Track S: raw DNA/SMILES prediction.**

- [ ] Первая задача — Reg DNA→threshold/choice, разметка реальными outcomes.
  Для LM явно измерить цену sequence tokenization; запретить утечки target/teacher.
- [ ] Сравнивать с task-trained Reg encoder. Не считать победу в Track P доказательством
  умения читать DNA. SMILES/ADMET добавляется в A1 с molecular scaffold split.

**Общее разделение и integrity.**

- [ ] Reg: выяснить смысл координат, строить interval components/blocked split с purge,
  аудит exact/RC duplicates и похожих sequences. Огромные connected components
  не разрезать случайно ради заданного процента. Изменения выпускать новым manifest.
- [ ] Пары/варианты/все вопросы одного объекта назначать только после group split.
  Molecular series — scaffold; связанные studies/assays учитывать в external evaluation.
- [ ] Поверх этого — frozen benchmark manifest: data/source/license hashes, assignments,
  group sizes, train/dev/cal/test counts, distribution/label masks, task/template IDs.
- [ ] Держать labels закрытого test отдельно от train runner. Builder/validator может
  проверять структуру, но train/tuning не должны загружать test labels.
- [ ] Проверить доступность достаточного независимого test после exclusion overlaps.
  Если честного test не остаётся — ограничить claims и завести внешний holdout.

Acceptance: zero prohibited intersections по заданному grouping; проверенные cases
на обеих сторонах bin boundary; missing/duplicate IDs не проходят; manifest immutable.
P/E/S metrics не усредняются в одну победную цифру. Артефакты:
`artifacts/decision_benchmark_v1/{manifest,split_audit,label_audit}.json`, локальные
JSONL и отдельный закрытый test labels файл. Политика доступа документирована.

### E3 — evaluator и baselines до тюнинга

- [ ] Baselines: exact policy solver для формальных задач, uniform/train-priors,
  base Qwen zero-shot и fixed few-shot из train, специализированный Reg encoder,
  hybrid encoder+rules. Не сравнивать разные доступные evidence как одну постановку.
- [ ] Сохранить одинаковые public requests для всех подходящих соперников.
  Few-shot examples и prompt фиксировать по development до закрытой оценки.
- [ ] Metrics: accuracy, multiclass Brier, log loss, per-task/per-domain; ordinal MAE
  по смысловому порядку, constraint violation, valid-answer rate, false answer на
  abstain-required cases, risk/coverage50/75/100%, ECE/reliability diagnostics.
- [ ] Отдельно измерять confidence threshold и явный semantic abstain: это разные
  механизмы. Abstain не считать автоматически правильным ответом на обычной задаче.
- [ ] Добавить paired bootstrap по независимым groups на одних и тех же objects,
  fixed seed, не менее2000 replicates; mean/std по model seeds отдельно.
  Для сложных зависимостей cluster unit задавать по provenance, не по вопросу.
- [ ] Температуру fit только на calibration после выбора checkpoints. Порог отказа
  выбирать там же с указанной неопределённостью; risk гарантии не обещать без
  отдельного статистического протокола/независимой выборки.
- [ ] Добавить сравнение option permutations/перефразировок на одном состоянии,
  threshold monotonicity и stability under irrelevant fields. Структурная валидность
  finite answers не означает семантическую правильность.
- [ ] Измерять полный request latency p50/p95, warmup, batch1, tokens/context/число
  questions/options, CPU/GPU, peak memory, base+adapter deployment bytes.

Acceptance: gold baseline даёт ожидаемый ceiling; метрики проверены вручную на
маленьком fixture; отсутствующие predictions или mismatch hashes — ошибка, не skip.
Артефакты `artifacts/decision_baselines_v1/`, evaluator report и tests.

### R0 — исправить Reg Tiny до повторного screen

- [ ] Huber per-assay scale fit только calibration и перенос без refit на development.
- [ ] Positional bins по индивидуальной valid length. Проверить encoder + nonzero
  mu/scale head для native/padded/смешанного batch, включая разные длины.
- [ ] Train/eval threshold sets разнести; четыре порога screen уже seen, не считать unseen.
- [ ] Event loss нормировать по valid events и явно задать lambda. Зафиксировать
  masking, assay/object weights, shift units и probability temperature.
- [ ] Выбор checkpoint согласовать с заранее заданной метрикой, не менять после
  просмотра результатов. Regression MAE selector допустим как явно отдельный control.
- [ ] Использовать manifest E2; не ремонтировать старые показатели задним числом.

Acceptance: regression tests на конкретные ошибки, schema/hash/provenance,
короткий train/eval smoke, новые versioned артефакты. Старые runs сохраняются.

### R1 — подтвердить маленький специализированный Reg

- [ ] Матрица: global/positional × Huber/distribution, плюс capacity-matched global
  control при существенной разнице параметров. Screen seed11, максимум20 эпох,
  patience5, одинаковые splits/data exposure/schedule policy.
- [ ] Установить конечный compute budget после профиля до старта серии. Записать
  LR и scheduler; максимум два заранее заданных LR на development при необходимости.
- [ ] Лучший positional, лучший global и regression control повторить seeds11/23/47
  с тем же полным budget. Старый v1 seed не считать новым повтором.
- [ ] На этом же split получить сопоставимую5M: historical encoder, видевший весь
  official train, не использовать как чистый comparator на inner holdout.
- [ ] Только после устойчивости расширять победивший рецепт на ~1M/~2M, затем
  при измеренной пользе5M. Фактические параметры пересчитать для новой головы.

Gate к финальной оценке: >=2% relative improvement primary Brier против сильнейшего
сопоставимого regression baseline на development, paired CI delta<0, устойчивость
по seeds, assay-choice drop не более1п.п.; per-assay деградации показать отдельно.
Это исследовательский gate, не утверждение, что требования уже выполнены.
Неуспех → одна диагностическая абляция data/optimization, затем честный report;
не масштабировать бесконечно ради прохождения gate.

### E4 — настоящий LoRA/QLoRA screen

Base candidates: Qwen3-0.6B и Qwen3-1.7B, immutable SHA, проверка model card/license
в момент загрузки. Rank16, alpha32, dropout0.05, all-linear, context2048,
microbatch1, global batch16, LR1e-4, warmup0.05, 3epochs — стартовый рецепт.
Значения зафиксировать в executable config, а не разносить по shell-командам.

- [ ] Профиль каждого base на реалистичной длине: tokenizer stats, 20 optimizer
  steps + inference с 2/4/8 options. Записать throughput и peak memory.
- [ ] Smoke pretrained0.6B на малом train subset: убедиться, что loss падает,
  adapters получают gradient и reload predictions совпадают в пределах precision.
  Tiny random-base smoke этого не заменяет.
- [ ] Начальная матрица seed11: 0.6B LoRA, 0.6B QLoRA, 1.7B QLoRA.
  Добавить1.7B LoRA только если профиль помещается и есть бюджет.
- [ ] На первом этапе Track P + допустимые train Track E; веса task mixtures
  фиксировать, считать examples/groups, не подменять hard tasks простыми synthetic.
  Raw sequence Track S — отдельная абляция, не обязательная смесь с первого run.
- [ ] Первую серию ограничить заранее записанными epochs/max updates/GPU-hours;
  первоначальные границы определить профилем. После превышения — save+stop,
  report о недоученности; не сравнивать это как окончательный capacity limit.
- [ ] Выбирать recipe по заранее заданным development task metrics, а не по test.
  Completion loss использовать как training diagnostic; для early stopping/выбора
  эпох добавить development metric callback либо сохранить все epoch adapters
  и оценить их одинаковым uncalibrated evaluator. Не fit temperature каждую эпоху.
- [ ] Победивший recipe и ближайший сильный neural baseline — seeds11/23/47.
  При rank bottleneck одна rank8/16/32 абляция; rSLora отдельно, не полный cross-product.

Две GPU предпочтительно занять независимыми jobs. DDP использовать только при
измеренном выигрыше и необходимости; global batch учитывать world_size.
OOM → по порядку уменьшить microbatch, увеличить accumulation, включить checkpointing/
QLoRA. Context уменьшать только после повторной проверки examples, без silent truncation.

Acceptance: реальные pretrained adapters обучены, есть все declared seeds или
явно отмеченные failed runs, base-vs-adapter сравнение, versioned predictions,
calibration и таблица результатов. GPU memory не прогнозировать из random smoke.

### E5 — подтвердить пользу и выбрать минимальный полезный размер

- [ ] Выделить primary task заранее: для formal policies нужен высокий compliance,
  а не превосходство над exact rules. Для real evidence comparison заранее
  выбрать primary metric, например decision accuracy/regret или Brier.
- [ ] До серии записать практический эффект. Стартовая цель для semantic choice:
  +2п.п. accuracy к сильнейшему сопоставимому neural baseline или >=2% relative
  Brier reduction, выбрать **одну** primary до run; CI соответствующей delta
  исключает ноль. Нельзя после run выбрать выигравшую метрику.
- [ ] Policy gate: target>=99% на корректно формализованных cases, отдельно
  reporting ошибок и robustness subsets. Если solver точен и быстрее, продукт
  должен использовать solver в этой области, не заменять его худшей LM.
- [ ] Проверить per-task/per-assay guardrails, abstention-required correctness,
  calibration и позиционный bias verbalizers. Проверить restricted probabilities
  относительно выбора других допустимых label codes.
- [ ] Выбрать минимальную модель в пределах заранее заданного допуска от best
  (предлагаемый1% relative Brier) при выполнении guardrails. Сравнить fp/bnb отдельно.
- [ ] Если adapter не лучше base на реальных задачах — диагноз data/labels/task mix,
  одна ограниченная правка и повтор с новым experiment ID. Если снова нет эффекта,
  оставить экспериментальным, не скрывать провал масштабированием или новым prompt.

Артефакт: `artifacts/decision_experiments_v1/summary.json` и читаемый report:
все runs, mean/std, paired CI, constraints, memory/latency, рекомендации,
known limitations. Winning seed не выдавать за весь продукт.

### E6 — SDK/API и локальный продукт

- [ ] `MinimolDecision.load(base, adapter, calibration)`; `.predict(request)` без
  gold labels; typed response с model/calibration versions, probabilities,
  chosen value, semantic abstain и policy rejection reasons.
- [ ] Отдельный endpoint `/v1/decision`, сохранить существующие Reg/CDS endpoints.
  Подключить текущие auth/rate-limit, payload/token budgets и validation.
- [ ] Не позволять state менять system instructions/schema; unsupported kind/domain
  возвращает понятный отказ. No free-form repair loop.
- [ ] Локальная demo: supplied measurements/forecasts + constraints + typed questions.
  Показать evidence provenance, результат и ограничения уверенности. Сырые DNA/SMILES
  режимы пометить и оценивать отдельно; не обещать неизвестное без предиктора.
- [ ] Тесты: SDK/API parity, option reorder, несколько вопросов/одно состояние,
  unsupported inputs, отказ, limits, save/load и корректность calibration identity.
- [ ] Model-state caching и batched options добавлять после профиля; ускорение
  должно сохранять probability semantics. Shared-prefix optimization отдельная задача.

Acceptance: один локальный end-to-end пример из real-evidence track и один
synthetic smoke работают; нет необходимости знать training record format.
Никакой внешней публикации или отправки пользовательских данных в облачный API.

### E7 — закрытая оценка

- [ ] До открытия test создать freeze manifest: data/split/model/adapter/tokenizer/
  calibration hashes, all candidate IDs/seeds, preprocessing, thresholds,
  metric/gates/comparator, quantization, software, статистический протокол.
- [ ] Проверить, что каждый comparator/teacher не видел test, в том числе при
  pretraining/fine-tuning если provenance позволяет это установить. Неизвестное
  пересечение указать как limitation, не утверждать отсутствие contamination.
- [ ] Прогнать заранее замороженный набор на закрытом test одним evaluation cycle.
  Не выбирать новый рецепт/temperature/threshold по результату test.
- [ ] Сохранить raw predictions локально, group-level intervals, отрицательные
  результаты, все seeds, per-track отчёт. Для множества secondary comparisons
  отметить exploratory статус/применить заранее заданную correction.
- [ ] Затем независимый разрешённый внешний/study holdout для переносимости.
  При отсутствии такого набора переносимость не заявлять.

Acceptance: freeze предшествует test predictions, нет post-test tuning,
репродукция агрегатов из сохранённых predictions. Если gate провален — честный
NO-GO; модель остаётся experimental, работа над infrastructure не пропадает.

### E8 — release candidate, без автоматической публикации

- [ ] Model/data cards: задача, base и полные параметры, adapter rank/bytes,
  training data/labels, splits, baselines, metrics/CI, calibration, OOD/abstention,
  hardware/latency, ограничения и provenance.
- [ ] Отдельно проверить права base, inputs/labels, teacher outputs и производных
  adapters. Флаг LOCAL_USE не снимать без подтверждающего основания.
- [ ] Лицензионно готовые компоненты отделить от локальных артефактов; не считать
  release-check автоматическим юридическим разрешением на weights.
- [ ] Веса/конфиги + install/run instructions + regression fixtures + checksum manifest.
  Прогнать существующие release проверки, убедиться в отсутствии raw data/secrets
  и больших артефактов в git. Не коммитить адаптеры в source tree.
- [ ] Если качество не достигнуто, выпускать только честно обозначенный research
  toolkit/benchmark spec, а не «superior biological decision model».

## 5. Следующие направления после первого вертикального сценария

### A1 — ADMET

- [ ] Воспроизвести corrected-primary-metric XGBoost/GINE baselines на одинаковых
  scaffold splits, missing labels и hyperparameter budget.
- [ ] Сформировать real evidence decisions по свойствам/ограничениям; исходные
  токсикологические endpoint не переводить в клинические рекомендации.
- [ ] Проверить полезность hybrid: fingerprint/GNN/XGBoost forecasts → Decision LM.
  Для purely numeric constraints использовать solver ceiling.
- [ ] Отдельный adapter/domain head, отдельные model/data cards; объединять adapters
  только после проверки cross-domain regressions.

### A2 — CDS/Rank и дальнейший тюнинг

- [ ] CDS требует реальных аннотаций и gene/genome-aware splits, не synthetic smoke.
- [ ] Rank требует query/context + candidate sets + реальных relevance/outcome labels;
  сначала split queries/entities, потом pair construction.
- [ ] DPO только после SFT и trustworthy preference pairs; GRPO только с независимой
  проверяемой reward и budget. Reward форматирования не равно bio-эффективности.
- [ ] Distillation/quantization после подтверждённого teacher, не как обход test gate.
  Общий adapter и routed adapters сравнивать на одинаковых данных/полном размере.

## 6. Минимальный набор команд для старта

Существующие интерфейсы; они не заменяют E1/E2:

```bash
export PYTHONPATH=src
.venv/bin/python -m pytest -m 'not slow'
.venv/bin/python scripts/tune_decision_lora.py --help
.venv/bin/python scripts/tune_decision_lora.py validate \
  --data artifacts/decision_lora_demo/v2_benchmark.jsonl
```

После фиксации base SHA и benchmark v1 (пути/SHA подставить из manifest):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/tune_decision_lora.py train \
  --model Qwen/Qwen3-0.6B --revision BASE_COMMIT_SHA \
  --data artifacts/decision_benchmark_v1/train_dev_cal.jsonl \
  --out artifacts/decision_experiments_v1/qwen06_lora_seed11 \
  --rank 16 --epochs 3 --batch-size 1 --accumulation 16 --lr 0.0001 --seed 11
```

Для NF4 добавить `--qlora`. Для seed23/47 — новый out и seed. Для DDP:
`CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --standalone
--nproc_per_node=2 scripts/tune_decision_lora.py train ... --accumulation 8`.

Не запускать эти длинные команды до E0–E3. Для benchmark E2 нужен train/dev/cal
файл без test labels и отдельный evaluation artifact. Примеры здесь показывают
существующий backend; YAML orchestration/config loader нужно реализовать в E1.

## 7. Формат итоговой передачи агентом

1. Список завершённых task IDs и ссылки на commits/diff/локальные артефакты.
2. Команды воспроизведения environment/data/train/eval/serve, включая resume.
3. Полная матрица фактически выполненных экспериментов и failed/deferred jobs.
4. Results по P/E/S, Reg/ADMET отдельно, baselines/seeds/CI/latency/memory.
5. GO/NO-GO для каждого quality/release gate, без подмены infrastructure readiness.
6. Непройденные критерии, блокирующие внешние зависимости, следующий конкретный job.

Агент начинает с **E0, затем E1/E2 и R0**. Сначала исправить доверие к измерениям,
потом тратить основной GPU budget. Плановая задача этого документа не означает,
что training, новый benchmark, API или release gates уже выполнены.
