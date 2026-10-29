# Minimol: техническое задание и пошаговый runbook

**Версия:** 1.0 · **Дата:** 2026-10-23  
**Цель:** построить коммерчески пригодную линейку очень маленьких моделей (целевые бюджеты **5M / 10M / 30M параметров**) для конкретных задач биоинформатики. Использовать **Carbon-A-1.2B** и **Carbon-X** как потенциальных *teacher*, но подтверждённые экспериментальные и референсные метки — как главный источник истины.  
**Железо:** 2 × NVIDIA RTX 3090 (24 GiB каждая; VRAM не объединяется автоматически).  
**Статус:** план экспериментов, не заявление о достигнутых метриках.  
**Принцип:** заимствовать у Needle стратегию узкой специализации, размера, latency и качественного benchmark; **не предполагать**, что архитектура Needle оптимальна для геномных задач.

---

## 0. Ожидаемые артефакты и критерии успеха

Репозиторий должен содержать работающий пайплайн от открытого датасета до воспроизводимой модели и benchmark:

1. **Minimol-Reg-5M/10M/30M** — предсказание количественной активности коротких ДНК-последовательностей для конкретных MPRA assays; для нескольких cell types — multi-output regression. Первый приоритет.
2. **Minimol-CDS-5M/10M/30M** — nucleotide-level вероятности coding sequence отдельно на `+/-` strands, с teacher Carbon-A. Второй приоритет.
3. **Minimol-Core** — опциональный общий DNA encoder (10–30M) с заменяемыми task heads. Тестируется **после** независимых моделей; совместное обучение не должно снижать качество каждой задачи.
4. **Minimol-Decision** — опциональный JEV-подобный интерфейс (`choice`, `noul`, `score`) поверх task-specific моделей. Это **не** копия проприетарной TypeSafe Jev; слово «JEV-подобный» относится к типизированному вероятностному интерфейсу.
5. Публичные **model cards**, **dataset cards / manifest**, eval-отчёты, inference API, экспорт ONNX (если операторная совместимость позволяет), бенчмарк на CPU/GPU и скрипты повторения.

**Запреты на интерпретацию результатов:** не утверждать клиническую пригодность; не приравнивать teacher confidence к вероятности биологической истины; не считать «сжатие 1.2B до 10M» успешным без честного независимого теста. Честно публиковать negative results.

**Decision gates:**
- **G0:** происхождение данных, лицензии и условия коммерческого переиспользования проверены; train/test изолированы.
- **G1:** сильный supervised baseline обучен и измерен, прежде чем запускать дорогую teacher distillation.
- **G2:** distillation повышает метрику / уменьшает data requirement на held-out тесте; иначе teacher исключается.
- **G3:** 5M/10M/30M — реальный Pareto-front качества, размера и latency; иначе выпускается меньше моделей.
- **G4:** reproducibility: минимум 3 seed для финальных кандидатов; независимые test sets, confidence intervals и latency на заявленном железе.

---

## 1. Проверенные upstream-источники

| Источник | URL | Обоснованное применение | Осторожность |
|---|---|---|---|
| Needle (Cactus Compute) | https://github.com/cactus-compute/needle | Образец разработки специализированных tiny models и on-device deployment | Tool-use архитектура ≠ гарантированно подходящая архитектура DNA |
| Carbon-A-1.2B | https://huggingface.co/HuggingFaceBio/Carbon-A-1.2B | CDS teacher: вероятности по каждому bp на 2 strands, 6-mer tokenizer | 98,304 bp full-window context; student на маленьком окне может проиграть на границах |
| Carbon-A training data | https://huggingface.co/buckets/HuggingFaceBio/Carbon-A-training-data | Референсные аннотации и геномные окна после аудита источников | Очень большой объём; **не скачивать целиком** для MVP |
| Carbon-X | https://huggingface.co/HuggingFaceBio/Carbon-X | Второстепенный teacher для typed biological decisions / QA | Text likelihood ≠ calibrated event probability; кастомная DNA-токенизация |
| Malinois MPRA regression | https://huggingface.co/datasets/HuggingFaceBio/malinois-mpra-regression | Главный ground-truth для Minimol-Reg | Сохранять assay, split и экспериментальные нормировки; избежать близких последовательностей в разных частях |
| Random Promoter DREAM 2022 | https://huggingface.co/datasets/HuggingFaceBio/random-promoter-dream-2022 | Возможный внешний тест / дополнительная задача | Проверить совместимость assay и лицензии |
| JASPAR | https://jaspar.elixir.no/ | Контрольный baseline для мотивов | Не выдавать PWM baseline за новый нейросетевой метод |
| RefSeq | https://www.ncbi.nlm.nih.gov/refseq/ | Независимые CDS annotations / holdout genomes | Уточнить сборку, версию и provenance |

**Факт по Carbon-A:** модель на HF обозначена как 1.2B, предсказывает CDS на уровне баз по обеим цепям и использует неперекрывающиеся 6-mers; максимальная последовательность 16,384 токена = 98,304 bp. **Факт по Carbon-X:** модель — генеративная text+DNA, не непосредственный labeler MPRA. Перед кодированием перепроверить актуальные model cards и pinned revisions.

### 1.1. Legal/provenance checklist — перед скачиванием и перед релизом

- Записать `repository`, `revision/commit`, `URL`, `license`, `upstream license`, `attribution requirements`, `redistribution eligibility`, `download date`, `SHA256`, `data steward` для **каждого** исходного корпуса.
- Записать отдельно лицензии весов Carbon-A, Carbon-X, всех датасетов и сторонних обучающих кодов: лицензия teacher **не покрывает автоматически** upstream datasets.
- Если лицензионный статус конкретного набора неоднозначен, использовать в pilot только внутренне, без перераспространения, до правового разрешения.
- Убедиться, что биологические последовательности и метаданные не содержат личной или чувствительной генетической информации. Для персональной геномики в ЕС: отдельная GDPR/privacy оценка.
- Публичная карточка модели должна явно указывать использование Carbon-A/Carbon-X, тип дистилляции и источники тестов.
- Выполнять `trust_remote_code` только после pin revision и просмотра удалённого Python-кода.

---

## 2. Продуктовая и научная постановка

### Track R: Minimol-Reg (первый коммерческий эксперимент)

**Input:** `sequence` длиной порядка 73–200 nt в исходном MPRA corpus (размеры и поля сверить с выбранной revision), `assay/cell_type`, возможно парная `reference/alternate` при доказанном соответствии наблюдений.  
**Output v1:** вещественная регуляторная активность (`log2FC` в той шкале, которую действительно использует данный dataset), стандартная ошибка/интервал при наличии корректной uncertainty модели.  
**Output v2:** категориальное решение decrease / neutral / increase **только если** определены биологически обоснованные threshold и валидированная calibration.

**Метрики:** Spearman, Pearson, MAE, RMSE по каждой cell line (K562, HepG2, SK-N-SH, если присутствуют в выбранном split); ранг по assay; uncertainty calibration / coverage. Контроль: mean-predictor, ridge/k-mer features, CNN baseline, frozen public DNA encoder + head. Для paired variants оцениваем предсказание `delta`, только если есть корректные paired experiment records; **не имитировать референс по случайным последовательностям**.

### Track C: Minimol-CDS (пилот сильного сжатия teacher)

**Input:** геномные окна (начать 2,048–8,192 bp; далее пробовать 16–32 kb, если память/скорость позволяют); `strand` и координатное происхождение.  
**Output:** per-base `p_cds_plus`, `p_cds_minus` с вероятностями [0,1].  
**Метрики:** nucleotide-level F1/precision/recall/MCC/AUPRC, boundary F1 ±{3,6,12} bp, exon/CDS IoU, метрики по видам, held-out species, latency per Mbp. Не называть непосредственно выдачу CDS вероятностей полноценной gene annotation (в GFF требуется дополнительный decoder).

**Ground truth:** RefSeq/Carbon-A dataset annotations. **Teacher:** soft per-base Carbon-A scores. **Критично:** сохранить соответствие 6-mer → bp и обеих цепей; использовать точное выравнивание output на уровне баз и маски на padding/неизвестных позициях.

### Track Q: JEV-style Minimol decision interface (только после R/C)

- `choice`: categorical normalized distribution + optional abstain;
- `noul`: Bernoulli probability для сформулированного бинарного вопроса;
- `score`: явно определённая ordinal scale, не произвольное число, смешанное с probability;
- `regression`: отдельный тип для количественного MPRA результата; не выдавать значения регрессии за Jev `score`.

Использовать tool routing / thin typed adapter без «универсальной» tiny language model на первом этапе.

---

## 3. Стратегия: 3 уровня дистилляции, только после supervised baseline

### 3.1. Для Carbon-A — soft-output distillation

1. Выбрать **только train genomes/windows**; holdout определяется **до** inference teacher.
2. Carbon-A inference → per-base `[p_plus, p_minus]`; сохранять оригинальные logits если API отдаёт их и semantics однозначны.
3. Перепроверить предсказания на known references. Не превращать все soft labels в hard ground truth.
4. Обучать student при наличии ground truth:

   `L = lambda_gt * BCEWithLogits(student, y, valid_mask) + lambda_kd * (T*T) * KL_Bernoulli(sigmoid(z_teacher/T), sigmoid(z_student/T), mask) + lambda_boundary * L_boundary`.

   Начать с `lambda_gt=1.0, lambda_kd=0.25`, `T=1`, без boundary loss; потом искать веса сеткой. Формула с `T²` — опция, должна быть протестирована, а не механически применяться к Bernoulli KL при всех T.
5. Проверить: **GT only**, **KD only**, **GT + KD**, **GT + KD + feature**, **quantized student**.
6. Feature distillation — только если совпадение слоёв/координат/форматов проверено и хранения скрытых состояний достаточно; брать несколько пробных окон, вместо materialize всего корпуса.

**Важный failure mode:** учитель видит десятки килобаз; tiny-model видит пару тысяч — при контекстной зависимости целевые soft labels могут быть недостижимы. Сначала измерить sensitivity-to-context на teacher при crop длинах 2k / 8k / 32k bp, затем решить требуемый receptive field student.

### 3.2. Для Carbon-X — conditional sequence scoring вместо генерации рассуждений

Использовать **только там, где известна чёткая постановка вопроса и конечный набор допустимых ответов**. Для каждой опции `o_i` вычислить teacher conditional token log-likelihood при **одинаковом шаблоне контекста**, полном ответе и чётко зафиксированных правилах длины. Преобразовать log scores в softmax-псевдораспределение, оценить prompt sensitivity, order bias и calibration against ground truth.

**Не делать**: брать 0.8 из text softmax и называть «80% вероятность патогенности». Не обучать Reg на ответах Carbon-X «активный промотор» вместо реальных MPRA измерений.

**Правило принятия:** если Carbon-X псевдометки не повышают held-out метрику относительно `ground_truth-only`, исключить из данной задачи. Carbon-X может пригодиться позже для QA, routing, species/motif typed decision tasks и human-readable объяснений через отдельный сервис.

### 3.3. Представления (feature KD)

`L_feature = 1 - cosine(project(h_student), normalized(h_teacher))` на **проверенных позициях**. Использовать только при совместимом aligning семплов, crop и token-to-bp mapping. Адаптационный проектор обучаем вместе со student; при inference его можно отбросить, если не участвует в задаче.

---

## 4. Минимальная архитектура family 5M/10M/30M

**Не обещать точное число параметров без `sum(p.numel())`.** Размеры — бюджеты, задаются конфигурацией и проверяются скриптом.

### Minimol-Reg

- Baseline 0: one-hot / k-mer ridge и маленький 1D CNN.
- Model A (5M): one-hot DNA encoder + stem Conv1d + 6–10 residual/dilated conv blocks + attention pooling + cell-type embedding + 3 regression outputs.
- Model B (10M): wider dilated CNN, skip connections; учёт reverse complement как augmentation *с учётом биологического смысла assay*.
- Model C (30M): hybrid CNN + small Transformer (short sequence), compare against 10M.
- Loss: masked Huber/MSE per assay; опционально inverse-variance weighted loss **только** при корректно определённой measurement SE и с cap веса, чтобы избежать доминирования шумных примеров.

### Minimol-CDS

- 5M: local multi-resolution CNN with dilated residual blocks + small decoder upsampling to per-base head.
- 10M: CNN/U-Net variant with long-range convolution/SSM candidate.
- 30M: hybrid conv + local/window attention with overlap-add for larger windows.
- Два strand outputs **без** искусственного mutual exclusivity — coding regions могут перекрываться на противоположных цепях.
- RC equivariance: `p_plus(RC(x))` должен примерно совпадать с reverse(`p_minus(x)`), и наоборот. Добавить consistency penalty лишь после корректных coordinate transforms.
- Для сильного genomic baseline сравнивать nucleotide resolution, не только средний score окна.

### Общий encoder?

Не строить как обязательную часть v1: короткий MPRA (~200 nt) и CDS (~kb–100kb) предъявляют разные требования к receptive field. В Phase 4 экспериментировать shared low-level stem или distillation into common trunk; сохранять task-specific frontend. Shared encoder выпускается только при Pareto-непроигрыше независимым моделям.

---

## 5. Структура репозитория

```text
minimol/
├── README.md
├── LICENSES/                # licenses, notices, SPDX, upstream manifests
├── pyproject.toml
├── uv.lock                  # после воспроизводимой установки
├── configs/
│   ├── reg_5m.yaml
│   ├── reg_10m.yaml
│   ├── reg_30m.yaml
│   ├── cds_5m.yaml
│   ├── cds_10m.yaml
│   ├── cds_30m.yaml
│   ├── carbon_a_teacher.yaml
│   └── carbon_x_teacher.yaml
├── src/minimol/
│   ├── data/{registry,download,normalize,split,homology,manifest}.py
│   ├── teachers/{carbon_a,carbon_x,cache,alignment}.py
│   ├── models/{reg_cnn,cds_cnn,shared,heads,quantization}.py
│   ├── train/{train_reg,train_cds,losses,checkpoint}.py
│   ├── eval/{reg,cds,calibration,latency,bootstrap}.py
│   ├── serving/{schemas,app,export}.py
│   └── cli.py
├── tests/{unit,integration,leakage,alignment}/
├── scripts/{check_machine,download_sample,smoke_teacher,train_all,bench}/
├── data/{raw,interim,processed,manifests}/ # в .gitignore
├── artifacts/{predictions,checkpoints,reports}/ # в .gitignore
└── docs/{DATA_CARD,MODEL_CARD,REPRODUCIBILITY,SECURITY,RELEASE}.md
```

**Data schema Reg (`parquet`):** `sample_id`, `sequence`, `source_id`, `data_revision`, `assay`, `split`, `group_id`, `chromosome`, `y_log2fc`, `y_se`, `valid_y`, `sequence_hash`, `license_id`. Хранить оригинальные CSV/parquet поля и отдельно normalized view.

**Data schema CDS:** `sample_id`, `assembly_accession`, `species_taxid`, `chromosome`, `start_0based`, `end_exclusive`, `genome_build`, `sequence`, `gt_plus`/`gt_minus` (compressed bitsets/arrays), `teacher_p_plus`/`teacher_p_minus` (optional float16), `valid_mask`, `split`, `sequence_hash`, `teacher_revision`, `teacher_context_bp`, `upstream_license`. Не создавать огромные JSON массивы меток; хранить Arrow/Parquet sharded или zarr с manifest.

**Обязательные тесты:** `RC(RC(x))==x`; roundtrip token-to-base coordinates; strand swap under RC; soft labels `[0,1]`, lengths match; padding masked; no overlap of train/test groups/homology clusters; deterministic checkpoint resume; identical predictions after ONNX export within declared tolerance.

---

## 6. Воспроизводимое окружение и аппаратный baseline

Ubuntu Linux + NVIDIA driver / CUDA подходящих версий. **Не фиксировать непроверенные версии PyTorch/Transformers до локального smoke test; после успеха pin в lockfile.**

```bash
mkdir minimol && cd minimol
git init
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Затем установить подходящий torch согласно https://pytorch.org/get-started/locally/
pip install transformers datasets accelerate safetensors huggingface_hub \
  numpy pandas pyarrow scikit-learn scipy pydantic pyyaml tqdm \
  tensorboard pytest fastapi uvicorn
nvidia-smi
python - <<'PY'
import torch
print('CUDA', torch.cuda.is_available(), 'GPUs', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

- `scripts/check_machine.sh` собирает model GPU, CUDA, driver, PyTorch, свободную RAM/disk, результаты `nvidia-smi` в `artifacts/reports/system.json`.
- Seed фиксировать для Python/NumPy/PyTorch; документировать, что deterministic kernels не всегда дают бит-в-бит идентичный throughput и output.
- Планировщик: teacher inference на GPU0, student trials на GPU1, **только если** это не создаёт OOM. Для 30M / длинных CDS делать DDP на обеих GPU по отдельному запуску.
- Для training: mixed precision BF16 если проверено; `torch.autocast`, grad accumulation, gradient clipping, activation checkpointing при необходимости; профиль p95 peak memory.

---

## 7. Пошаговый pipeline Reg — сначала без teacher

### R0. Data ingest и audit

1. Выбрать pinned HF dataset revision. Скачать только нужные shards/splits.
2. Просмотреть реальные поля и значения, проверить `sequence` (ACGTN), дубликаты, `chromosome`, `variant_class`, метрики assay, missing/NaN, распределение z-score.
3. **Уважать официальный train/validation/test**, не перемешивать всё в общий массив. Дополнительно проверить leakage: одинаковые и близкие `sequence`, один исходный genomic locus или группы вариантов. При обнаружении leakage создать **отдельный stricter grouped test**, не подгонять по нему модель.
4. Удалить переменные с прямой утечкой target; train normalization fit **only train**.
5. Сохранить manifest по всем семплам и hash split.

### R1. Baselines

- `mean-per-cell-type` baseline.
- k-mer counts (k=3..6) + Ridge/ElasticNet.
- 1D CNN до ~1M параметров.
- Обязательно повторить baseline для исходного и stricter grouped test.

### R2. Обучение 5M / 10M / 30M

Начальные hyperparameters **для sweep** (не утверждения об оптимуме):

```yaml
experiment: reg_10m_v0
model:
  kind: dilated_cnn
  param_budget: 10000000
  rc_augment: false
train:
  seed: 42
  batch_size_per_gpu: 128
  accumulation_steps: 1
  precision: bf16
  optimizer: adamw
  lr: 0.0003
  weight_decay: 0.01
  epochs: 35
  scheduler: cosine
  early_stopping_patience: 7
  grad_clip: 1.0
loss:
  type: huber
  delta: 1.0
eval:
  metrics: [spearman, pearson, mae, rmse]
  split: validation
```

- LR sweep `[1e-3, 3e-4, 1e-4]`, weight decay `[0,0.01]`, seeds `[17,42,123]` **для победивших конфигураций**.
- Per-assay reporting, residual plots, outlier audit; inference-time resizing after quantization.
- `R2` стоп-критерий: выбранный student заметно улучшает простые baselines на **stricter grouped validation**, а не только на случайных строках.

### R3. Опционально Carbon-X

1. Выбрать только случаи с понятными categorical questions (например, направление эффекта относительно конкретного assay threshold). Назвать это **auxiliary teacher task**, не MPRA regression.
2. Получить sequence-conditioned option scores с несколькими order-preserving prompt templates. Проверить корреляцию с истинной MPRA разметкой.
3. Обучить classification auxiliary head на unlabeled sequences без test contamination.
4. Сравнить `Reg GT only` vs `Reg + Carbon-X auxiliary`. При отсутствии выигрыша teacher убрать. **Не генерировать десятки миллионов ответов без подтверждения пользы**.

---

## 8. Пошаговый pipeline CDS — Carbon-A teacher

### C0. Составить подвыборку

- Выбрать ~50–200 genome assembly для пилота с балансом taxonomic classes; выделить unseen species **до** генерации teacher outputs.
- Для начала ~50k–200k окон train (количество — ориентир, CPU/disk budget проверить), lengths 2,048 / 8,192 bp, отдельно long-context benchmark.
- Group split by species/assembly/homology clusters; перекрывающиеся окна одного генома всегда в одной части.
- Записывать число positives per strand и distribution of exon boundaries.

### C1. Teacher smoke test (Carbon-A)

Использовать официальный HF интерфейс, pinned revision, протестировать **на одном коротком окне** правильность входа/выхода и маппинга 6-mer. HF model card описывает `AutoModelForTokenClassification` с `trust_remote_code=True`. **Не строить батчевый soft label extractor до unit-test координат.**

```python
from transformers import AutoModelForTokenClassification
# Только после review upstream custom code и pin revision:
model = AutoModelForTokenClassification.from_pretrained(
    'HuggingFaceBio/Carbon-A-1.2B',
    trust_remote_code=True,
    revision='REPLACE_WITH_VERIFIED_COMMIT',
)
# Токенизация, входы, shape и token-to-bp mapping — по официальному
# примеру model card/репозитория. Не предполагать, что logits уже per-base.
```

- Сначала на CPU или GPU0 записать входную длину, shapes, dtype, как обрабатывается reverse strand.
- Сравнить per-base teacher output с reference output официального gene annotation script, если возможно.
- Проводить infer chunks; сохранять predictions float16/uint16, компрессию и диагностическую статистику, избегать ненужных huge logits.
- Для каждого crop `context_bp` фиксировать отдельно: label от full context и short context *не взаимозаменяемы*.

### C2. Students

- `CDS-5M`: simple dilated CNN; ограниченный receptive field.
- `CDS-10M`: multi-resolution U-Net/CNN; длинные окна.
- `CDS-30M`: hybrid local attention/SSM prototype после R0/C0 benchmark.
- Loss variants: `GT only`, `soft KD only`, `GT+KD`. Сравнить strand-balanced BCE, позитивные веса только по train statistics, boundary-focused auxiliary loss.
- Требовать **full assembly evaluation** и агрегировать окна перекрытием; отдельно анализировать оверлап и padding.

### C3. Multi-resolution evaluation

- Train/validation для 2k, 8k, 16k bp и измерить performance по расстоянию от границ окон.
- Показать performance by taxon, GC content, read length, high/low CDS density, exon boundaries.
- Carbon-A benchmark на тех же входах; сравнивать accuracy, F1, throughput per Mbp и p95 memory.

---

## 9. Losses, calibration, uncertainty

**Reg:** `MaskedHuber(pred, measured_log2FC)` плюс optional heteroscedastic `GaussianNLL(mean, sigma)` только если измерительная неопределённость и estimator validation позволяют. Для prediction intervals применить split conformal calibration на независимой calibration subset; оценить coverage и width.

**CDS:** `MaskedBCEWithLogits(student_logits, y)`; teacher KL — Bernoulli per bp со стабильным clipping. Документировать, какие регионы исключаются: N-rich, padded, annotation missing.

**Decision heads:** CE/soft KL для `choice`, BCE для `noul`, ordinal loss для `score`; только на task-specific честных метках. Temperature scaling на отдельных calibration данных, ECE + Brier/NLL на test. **Raw entropy ≠ confidence.**

**Отрицательные контроли:** shuffled targets, random teacher labels, only-GC-content baseline, sequence reverse/complement coordinate stress. Если эти контроли показывают почти то же качество — вероятна утечка/артефакт.

---

## 10. Реальные границы вычислений на 2 × 3090

- GPU0 и GPU1 по 24 GiB: запуск teacher и student одновременно возможен только если каждый помещается в 24 GiB.
- Carbon-A 1.2B в half precision весит порядка 2.4 GB, но **активации при 98kb input могут доминировать памятью**. Проверять batch/context в smoke test.
- Carbon-X 14.8B в BF16 на 1×3090 не помещается по весам; понадобится 4-bit/CPU offload/tensor parallel при совместимости. Не планировать continuous online KD от Carbon-X на каждом minibatch.
- Сначала offline teacher cache 10k examples → скорость/sec, VRAM и качество; затем 50k, 200k только если видно улучшение.
- Для Minimol-Reg 5–30M на коротких входах 1×3090 вероятно достаточно; проверить измерением. Для CDS контекст/активации определяют предел batch.

**Бюджетировать по измерениям:** `teacher_examples/sec`, `train_tokens_or_bp/sec`, p50/p95 latency, GPU memory peak, disk per 100k samples, energy/experiment. Все цифры в отчёте, не догадки.

---

## 11. Экспериментальная матрица и ablations

| ID | Track | Size | Training recipe | Test hypothesis |
|---|---|---:|---|---|
| R00 | Reg | Ridge | GT only | минимальный baseline |
| R01 | Reg | ~1M | CNN GT | нейросеть лучше k-mer ridge? |
| R05 | Reg | 5M | GT only | tiny качество |
| R10 | Reg | 10M | GT only | scaling benefit |
| R30 | Reg | 30M | GT only | diminishing returns? |
| RX10 | Reg | 10M | GT + X auxiliary | Carbon-X помогает? |
| C05 | CDS | 5M | GT only | baseline |
| C05K | CDS | 5M | GT + A soft | KD improves? |
| C10K | CDS | 10M | GT + A soft | scaling |
| C30K | CDS | 30M | GT + A soft | best quality vs latency |
| C10F | CDS | 10M | GT + A features | representation KD useful? |
| M10 | Multitask | 10M | shared trunk + heads | transfer without regression? |

**Выбор модели:** отбирать по заранее объявленной метрике (например, macro Spearman across assays для Reg; species-macro nucleotide F1 + boundary F1 для CDS) с constraints по latency и RAM. Не выбирать разные модели по test; test открываем только для финального репорта.

---

## 12. API v1 и on-device deployment

**Нужны два task-specific endpoints** прежде чем строить общую JEV-оболочку:

```http
POST /v1/regulatory/predict
Content-Type: application/json
{
  "sequence": "ACGT...",
  "assay": "K562",
  "model": "minimol-reg-10m-v1"
}
```

```json
{
  "model_version": "minimol-reg-10m-v1",
  "assay": "K562",
  "predicted_log2fc": 0.37,
  "prediction_interval": null,
  "ood_warning": false
}
```

```http
POST /v1/cds/annotate
Content-Type: application/json
{
  "sequence": "ACGT...",
  "model": "minimol-cds-10m-v1"
}
```

```json
{
  "model_version": "minimol-cds-10m-v1",
  "coordinate_system": "0-based-half-open",
  "strand_probabilities": {
    "+": [0.1, 0.2],
    "-": [0.3, 0.4]
  },
  "example_only": true
}
```

Примеры **вымышленные, не реальные предсказания**. Реальный CDS API возвращает массив длиной `len(sequence)` по каждой цепи (или удобное compressed/file format для длинных последовательностей), а не две вероятности.

**Требования:** explicit input limit, ACGTN validation, assay/organism provenance, streaming/chunked handling длинной ДНК, model_version, request ID, optional abstention/OOD warning, auth/rate limit, health endpoint, batch inference. Не отправлять клиентскую персональную ДНК в публичный logging; минимизировать хранение.

**Экспорт:** PyTorch → ONNX → ONNX Runtime CPU benchmark (float32/float16/int8 если поддерживается). INT4 только при наличии реально работающего backend: таблица 5M~2.5MB — арифметика весов без overhead, **не гарантия рабочего tiny-файла**. Сравнить результаты PyTorch/ONNX на фиксированных test vectors.

---

## 13. Репродуцируемый запуск / Make targets

Разработчик должен реализовать команды (псевдоинтерфейс; перечисленные CLI **ещё не существуют**):

```bash
make env-check
make audit-licenses
make data-reg-sample
make data-cds-sample
make check-splits
make teacher-a-smoke
make teacher-x-smoke
make train-reg-5m
make train-reg-10m
make train-reg-30m
make train-cds-5m
make distill-cds-10m
make evaluate-all
make export-onnx
make benchmark-cpu
make model-card
make release-check
```

Для каждого запуска сохранять `config.yaml`, git commit, data manifest hash, teacher revision, random seed, metrics.json, checkpoints, hardware info и logs. По умолчанию offline/immutable final test; агрегация для 3 seeds.

---

## 14. Roadmap по неделям: конкретные deliverables

### Неделя 1 — G0: legal + reproducibility + split design

- [ ] Развернуть репозиторий, виртуальное окружение, CI/pytest.
- [ ] Проверить upstream licenses и записи в SPDX/NOTICE manifest.
- [ ] Скачать небольшой MPRA subset; изучить реальные поля.
- [ ] Спроектировать strict group holdouts и leakage scanner.
- [ ] Задокументировать baseline metrics и test protocols.
- **Готово если:** `make env-check`, `make audit-licenses`, `make check-splits` выполняются без ошибок и есть data card draft.

### Неделя 2 — R1/R2: регуляторные tiny models

- [ ] Ridge, 1M CNN baselines.
- [ ] Reg-5M и Reg-10M, первые runs.
- [ ] Spearman/MAE по K562/HepG2/SKNSH и grouped validation.
- [ ] Reg-30M только если 10M видно что может выиграть.
- **Готово если:** сравнимые метрики и inference speed, записан Pareto frontier.

### Неделя 3 — C0/C1: teacher Carbon-A и CDS dataset

- [ ] Сформировать stratified genome sample и held-out assemblies/species.
- [ ] Запустить Carbon-A smoke, сравнить strand/coordinate alignment.
- [ ] Получить 10k–50k soft-label windows и проверить качество.
- [ ] Начать CDS-5M GT-only baseline.
- **Готово если:** тесты координат и reverse complement проходят; teacher cache воспроизводим.

### Неделя 4 — CDS compression

- [ ] CDS-5M/10M train GT-only и GT+KD.
- [ ] Проверить влияние crop 2k/8k/16k и boundary F1.
- [ ] Сравнить Carbon-A vs students по held-out species и runtime.
- [ ] Принять решение об увеличении до 30M.
- **Готово если:** есть хотя бы один честный quality/speed trade-off и подробные ошибки.

### Неделя 5 — Carbon-X feasibility + shared encoder research

- [ ] Carbon-X smoke с корректным tokenizer.
- [ ] На небольшом наборе MPRA проверить scoring categorical effect prompts и насколько scores полезны вне тренировочной выборки.
- [ ] `Reg GT only` vs `Reg+Carbon-X auxiliary` на grouped validation.
- [ ] Решить, стоит ли shared encoder экспериментировать или оставить family независимой.
- **Готово если:** сформулирован **go/no-go по Carbon-X**; при отсутствии эффекта дистилляция X не масштабируется.

### Неделя 6 — Final training / quantization / API

- [ ] 3-seed retrain финальных кандидатов.
- [ ] ONNX/int8 exports при поддержке, parity tests.
- [ ] p50/p95 inference latency и throughput на CPU/3090.
- [ ] FastAPI with input validation; model cards / dataset cards.
- [ ] Legal/privacy/release audit.
- **Готово если:** reproduce `train→eval→export→serve` из README, финальные метрики, changelog, model card.

**При нехватке времени:** prioritise Reg-5M/10M + Carbon-A CDS-5M; не пытаться закончить 6 моделей и общий encoder любой ценой.

---

## 15. Release criteria для стартапа

Каждая модель, объявленная production/research preview, обязана иметь:

- Правомерность коммерческого использования **каждого** источника и модели, корректные notices.
- Разделённые supervised labels и teacher pseudo labels в документации.
- Зафиксированные splits, включая harder out-of-domain или species holdout.
- Отчёт по accuracy + calibration + robustness + latency + file size **с точным железом**.
- Comparison against baseline + teacher (если teacher решает *ту же* задачу).
- Non-clinical research-only labeling до соответствующей валидации.
- Версионирование модели, training data provenance, reproducibility commands, rollback.
- Проверки на некорректные/несовместимые входы и OOD warnings.

**Экономический критерий:** модель должна обеспечить **либо** измеримое улучшение в выбранной научной задаче, **либо** близкое качество при существенном снижении стоимости/latency/памяти. Если обе величины проигрывают сильному baseline, не выпускать такую версию только ради красивого числа параметров.

---

## 16. Известные риски и конкретные mitigation

| Риск | Как обнаружить | Что делать |
|---|---|---|
| Student слишком короткий для CDS | boundary F1 падает у crop edges | увеличить receptive field, overlap inference, hybrid long-context |
| MPRA разметка не переносится между assay | poor per-cell-type metrics / OOD | отдельные heads и label normalization; не смешивать шкалы |
| Carbon-X не знает quantitative MPRA | teacher correlation низкая | отказаться от X KD для Reg |
| Teacher ошибается уверенно | calibration и discordant samples | hard labels primary, teacher weighting/abstain |
| Homology leakage | sequence similarity train↔test | group/cluster holdout, near-duplicate scan |
| Shared model деградирует | ablation vs independent models | разделить encoders или применять task-specific adapters |
| Tiny model не быстрее на CPU | measured latency despite fewer params | выбрать kernel-friendly convolutions, ONNX benchmark |
| Учебная лицензия/неясные источники | legal audit | замена корпуса / ограничение релиза |
| OOD biological use | high uncertainty / failure per species | scope model cards, reject/abstain |

---

## 17. Что поручить AI coding agent первым PR (очень конкретно)

**PR #1: Repository skeleton, Reg ingest, leakage checking, first baseline.**

**Задачи:**
1. Создать Python-пакет `minimol` (src layout), configs, pytest, Ruff, Makefile.
2. Реализовать `download_reg.py` по pinned HF dataset revision с проверкой metadata/sha256 и без автоматического скачивания гигабайт за пределами выбранного объёма.
3. Реализовать data validation (sequence alphabet, nonempty label, consistent assay metadata) и leakage scanner exact duplicates + near duplicates (с объяснённым similarity threshold) по split.
4. Реализовать baseline `Ridge(k-mer)` + скрипт `evaluate_reg.py` c Spearman/Pearson/MAE/RMSE per cell type.
5. Записать `test_no_split_leakage.py`, `test_rc.py`, `test_label_normalization.py`; создать CLI `minimol data reg-audit`.
6. Создать CI workflow для unit tests без GPU и optional local GPU smoke.

**Definition of Done:** `pip install -e .`, `pytest -q`, `minimol data reg-audit`, `minimol train reg-ridge`, `minimol eval reg-ridge` работают на малом sample и генерируют `metrics.json` + `manifest.json`; нет сквозной утечки split; README описывает воспроизведение.

**PR #2:** Reg-CNN (5M/10M), reproducible train/eval, checkpoints, GPU profiling.  
**PR #3:** Carbon-A adapter, strand mapping tests, soft-label cache.  
**PR #4:** CDS-student GT/KD and held-out genome evaluation.  
**PR #5:** Carbon-X option scoring trial, ablations, automatic go/no-go report.  
**PR #6:** export/serving/model cards/licensing release checks.

---

## 18. Источники, которые нужно закрепить в README и citation manifest

- [Needle — code and documentation](https://github.com/cactus-compute/needle)
- [Carbon-A-1.2B — official model card](https://huggingface.co/HuggingFaceBio/Carbon-A-1.2B)
- [Carbon-A training data bucket](https://huggingface.co/buckets/HuggingFaceBio/Carbon-A-training-data)
- [Carbon-X — official model card](https://huggingface.co/HuggingFaceBio/Carbon-X)
- [Malinois MPRA dataset card](https://huggingface.co/datasets/HuggingFaceBio/malinois-mpra-regression)
- [Random Promoter DREAM 2022](https://huggingface.co/datasets/HuggingFaceBio/random-promoter-dream-2022)
- [JASPAR](https://jaspar.elixir.no/)
- [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/refseq/)

**Итог:** начни с **Reg-5M/10M на настоящих MPRA**, в параллели — **CDS-5M/10M с Carbon-A distillation**. Carbon-X — экспериментальная добавка, которую масштабируем только при доказанном выигрыше. Это лучший способ проверить Needle-подобную гипотезу с 2×3090 без многонедельной бессмысленной генерации synthetic data.


## Актуализация задания по результатам обучения — 29 октября 2026

Обязательный ближайший этап: [исправление и улучшение обучения](Training_Recovery_Plan.md).
Документ фиксирует обнаруженную ошибку головы Reg, исправление накопления
градиентов, совместимость старых чекпоинтов, повторное обучение Reg-5M,
диагностику ADMET по endpoint и критерии перехода к большим моделям.
Чеклист и критерии приёмки из этого дополнения входят в настоящее ТЗ.
