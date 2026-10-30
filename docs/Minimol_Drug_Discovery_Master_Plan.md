# Minimol Drug Discovery — Master Implementation Plan

**Версия:** 1.0  
**Дата:** 2026-10-28  
**Язык:** русский  
**Статус:** проектное ТЗ / исследовательский план, не результаты экспериментов  
**Вычисления:** две NVIDIA RTX 3090 (24 GB VRAM каждая; память НЕ объединена автоматически)  
**Целевой релиз:** серия компактных моделей с каждой deployable моделью ≤100 млн параметров, измеренной скоростью, доверительными интервалами и честными ограничениями.

> **Принцип:** experimental ground truth > proxy oracle > teacher prediction > бездоказательные объяснения LLM. Дистилляция допускается, только если её польза доказана ablation. Модель не объявляется пригодной для клинических решений, выбора доз или реального применения лекарств.

---

## 0. Executive decision / ключевые решения

1. Не обучать с нуля универсальную молекулярную LLM. Начать с дешёвых сильных baseline (ECFP/RDKit + XGBoost, Chemprop-like D-MPNN, GINE).
2. Первый продукт: **Minimol-ADMET** (5M/15M/30M), затем **Minimol-Tox**; **Minimol-Bind** после проверки данных, **Minimol-Rank** как decision layer; **Minimol-Precision** как рискованный научный флагман.
3. JEV-like означает **неавторегрессионные typed decisions** (`choice`, `noul`, `score`, `regression`) + uncertainty + abstain; это собственная архитектура, **не реплика закрытой Jev**.
4. Carbon-A — teacher для CDS/геномной разметки; Carbon-X — возможный auxiliary teacher для genomic target/variant reasoning; ни один не заменяет SMILES/ADMET/binding teacher.
5. От статьи о synthetic task scaling используем **cheap-to-expensive curriculum**, BRICS ограничения, валидируемые задачи, tool-budget, контрфактическое ранжирование; **не предполагаем**, что оптимизация RDKit-дескрипторов создаёт истинно эффективные лекарства.
6. На двух RTX 3090 один исследовательский трек за раз; допускается параллелизм ETL на CPU и GPU training, но GPU memory budget обязательно профилируется.

### Чего не делаем на старте
- Не называем предсказанную docking score доказательством binding affinity или potency.
- Не смешиваем training labels разных assay без assay ID, единиц и условий.
- Не используем тестовые комплексы/скелеты для генерации train-задач или teacher-query.
- Не обучаем JEV-головы на выдуманных «confidence» из LLM.
- Не предполагаем, что коммерческое использование всех данных автоматически разрешено лицензией базовой модели.

## 1. Системная архитектура

```text
SMILES / molecular graph + optional assay context
            │
            ▼
Molecular encoder (GINE/MPNN 5–30M)
            │
            ├── ADMET heads (continuous/binary)
            ├── Tox heads (endpoint-specific)
            ├── Typed option-conditioned decision head (choice/noul/score)
            └── uncertainty / applicability domain / abstain

protein sequence / pocket / variant  ──► frozen protein embeddings + fusion (30–100M)
                                              ├── Bind affinity head
                                              └── variant-aware delta head (research-only)

BRICS/fragment search or optional small SMILES generator
            │ candidates
            ▼
validity → constraints → Minimol predictions → diversity → independent oracle
            │
            └── ranked shortlist + provenance, not a claim of therapeutic efficacy
```

**Ограничение ≤100M:** число параметров каждого публичного inference-пакета считается вместе с применяемыми в runtime энкодерами. Если большой protein encoder используется офлайн, это явно указано; модель не поддерживает произвольные новые мишени без preprocessing и отдельно измеренного compute.

## 2. Семейство моделей и выходные контракты

| Артефакт | Размер-цель | Входы | Выходы | Релизный приоритет |
|---|---:|---|---|---|
| Minimol-ADMET-5M | 3–7M | molecular graph, assay | 5–10 свойств, regression / calibrated binary | P0 |
| Minimol-ADMET-15M | 10–20M | graph, assay | multih ead ADMET | P0 |
| Minimol-ADMET-30M | 20–35M | graph/SMILES | multitask + domain estimate | P0 |
| Minimol-Tox-10M | 5–20M | graph, endpoint | конкретные assay-specific риски | P1 |
| Minimol-Rank-5M | 1–10M | набор кандидатов, objective, constraints | pairwise ranking, abstain | P1 |
| Minimol-Bind-60M | 30–80M | molecule + target embedding | pKi/pKd/pIC50 с указанием assay | P2 |
| Minimol-Precision-100M | 50–100M | molecule + WT/mutant target context | delta experimental activity и interval | P3 / R&D |

**Обязательное поле** у любого score: endpoint_id, label_definition, units, assay_domain, model_version, calibration_version, uncertainty, is_out_of_distribution, provenance.

### Typed decision semantics
- `noul`: Bernoulli по *явно определённому* событию (например: экспериментальный logS > заранее заданного порога при конкретном протоколе).
- `choice`: взаимоисключающие классы; softmax только внутри полностью определённого набора вариантов.
- `score`: порядковые категории с определёнными границами, ordinal likelihood, не произвольная «оценка из десяти».
- `regression`: измеряемая величина и predictive interval, отдельно от probability of threshold.
- `abstain`: недостаточно близких аналогов / нет поддержки нужного assay / высокий epistemic risk.

## 3. Источники и реестр лицензий

**До загрузки массовых данных** создать `data_registry.csv`: `source,version,url,citation,license,commercial_use,derivative_redistribution,PII_risk,assay_type,download_date,checksum,reviewer,decision`.

| Источник | Назначение | Непременное условие |
|---|---|---|
| TDC ADMET group (22 benchmark datasets) | supervised ADMET + official splits | соблюдать per-dataset лицензии; фиксировать версии |
| ChEMBL | bioactivity, targets, assay context | проверять CC BY-SA и отдельные source restrictions; harmonization |
| BindingDB | protein–ligand experimental binding | разделять Ki/Kd/IC50 и censoring, не усреднять несопоставимое |
| MoleculeNet / Tox21 / ClinTox | tox and molecular property baselines | лицензия upstream, assay-specific labels |
| ZINC20 | molecular inputs и недорогие synthetic tasks | проверить доступность/лицензии на выбранный subset |
| PLINDER | protein–ligand complexes and benchmark splits | freeze конкретный release; train/test pocket+ligand leakage controls |
| PubChem BioAssay (отобранные assay) | расширение экспериментальной разметки | assay и compound normalization, quality filtering |
| Carbon-A-1.2B | CDS annotations | Genomic-only auxiliary track, protein target annotation not molecule property |
| Carbon-X | genomic variant / target context | standalone evaluation teacher; avoid unverified chemistry labels |
| Опционально ChemBERTa, pretrained GNN / Chemprop | альтернативные molecular encoders/teachers | лицензия и честные model-specific baselines |
| Опционально ESM-style protein encoder | protein representations | runtime cost/license; offline embeddings for fixed target list |

**Licensing gate:** NOT APPROVED → не включать в train до проверки. Хранить каждое происхождение на уровне строки, включая synthetic tasks и teacher outputs.

Официальные источники:
- https://tdcommons.ai/benchmark/admet_group/overview/
- https://www.ebi.ac.uk/chembl/
- https://www.bindingdb.org/
- https://zinc20.docking.org/
- https://github.com/plinder-org/plinder
- https://huggingface.co/HuggingFaceBio/Carbon-A-1.2B
- https://huggingface.co/HuggingFaceBio/Carbon-X
- https://arxiv.org/abs/2609.04735

## 4. Репозиторий / структура

```text
minimol/
  README.md  LICENSE  CITATIONS.md  SECURITY.md  pyproject.toml  uv.lock
  configs/
    data/admet.yaml  data/binding.yaml  models/gine_5m.yaml
    models/gine_15m.yaml  models/gine_30m.yaml  models/bind_60m.yaml
    training/admet.yaml  training/distill.yaml  training/rank.yaml
    evaluation/splits.yaml  evaluation/calibration.yaml
  src/minimol/
    data/{registry,download,standardize,deduplicate,assays,splits,featurize}.py
    oracles/{rdkit_properties,brics,validity,pocket_proxy}.py
    teachers/{carbon_x,carbon_a,molecular_teacher,cache}.py
    models/{gnn,smiles_encoder,protein_fusion,heads,uncertainty}.py
    training/{pretrain,train_admet,train_tox,train_bind,distill,ranker}.py
    generation/{task_factory,candidate_search,reward,rollout_store}.py
    evaluation/{benchmark,scaffold_audit,affinity_metrics,calibration,ood,latency}.py
    serving/{schemas,api,export}.py
  tests/{unit,integration,leakage,scientific_invariants}/
  scripts/{env_check,make_manifest,smoke_test,train_sweep,benchmark_all}.sh
  data/{raw,interim,processed,manifests}/
  models/  artifacts/  results/  docs/{DATA_CARD,MODEL_CARD,RELEASE_CHECKLIST}.md
```

**Dependency strategy:** Python 3.11; PyTorch CUDA version determined by local driver; RDKit, PyG or DGL, Transformers/PEFT optional, scikit-learn, pandas/pyarrow, Hydra, MLflow, FastAPI, pytest. Lock exact versions only after environment smoke test; avoid pretending that untested `pip install` combinations are guaranteed.

## 5. PR-00: инфраструктура и проверка железа (1–3 дня)

Deliverables: Linux/WSL2 environment, GPU diagnostics, CUDA+RDKit smoke test, determinism report, git+data version tracking.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python --version
mkdir -p minimol && cd minimol
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
# Подобрать подходящий PyTorch CUDA wheel по официальному selector, затем:
python -m pip install rdkit pandas pyarrow numpy scikit-learn pydantic pytest mlflow
python - <<'PY'
import torch
from rdkit import Chem
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
print(Chem.MolFromSmiles('CCO') is not None)
PY
```

- `scripts/env_check.sh`: free VRAM, RAM, NVMe, CUDA, GPU interconnect, installed packages; write `artifacts/env.json`.
- All runs: seed (Python/NumPy/Torch), git SHA, dataset manifest hash, config YAML, checkpoint SHA.
- Mixed precision BF16 only if measured stable on target kernels; tune accumulation and activation checkpointing.
- GPU0/GPU1 independent jobs initially. DDP after verifying single-GPU throughput and loss parity.

**PASS:** 100 batches train without NaN/OOM; all packages import; checkpoint save+resume reproducible.

## 6. PR-01: data ingestion + chemistry standardization

### Chemical input policy
1. Parse and sanitize SMILES through RDKit.
2. Define treatment for disconnected salts, counterions, isotopes, explicit hydrogens, stereochemistry, tautomer identity, protonation: store *original* and *standardized* forms. Do not silently erase stereochemistry of assay-specific compounds.
3. Canonical identifiers: canonical isomeric SMILES, InChIKey (where supported), molecule graph hash, Bemis–Murcko scaffold.
4. Remove exact duplicate measurements only with traceable aggregation rules; retain replicate variance / interval-censored labels.
5. Flag unsupported elements, metallocomplexes, mixtures, unusual valence, exceptionally large compounds, and missing assay metadata.
6. Regression normalization only fit on train; track transformation and inverse transformation per endpoint.

### Label policy
- Label must include `target_name`, `assay_id`, `assay_type`, `value`, `units`, `operator` (`=`, `<`, `>`, interval), `organism`, `conditions`, `quality_flag`, `source_id`.
- Convert concentrations to molar/log units only when source semantics permit.
- Do not pool Ki/Kd/IC50 without explicit task/context; do not infer comparable therapeutic potency.
- Clinical labels, animal toxicity, in vitro assay toxicity and computed descriptors are separate tasks.

Example schema (`examples.parquet`):

```json
{
  "sample_id":"sha256:...", "smiles_original":"CCO",
  "smiles_canonical":"CCO", "scaffold_id":"...",
  "assay_id":"example_assay", "endpoint":"logS", "units":"log_mol_per_L",
  "measurement":-1.2, "relation":"=", "source":"dataset_revision",
  "split":"train", "label_quality":"experimental", "license_id":"registry_014",
  "stereochemistry_status":"retained"
}
```

`data build` writes `manifest.json`: row count, error categories, unique molecules, scaffolds, endpoints, label missingness, license distribution. No split yet until de-duplication links are built.

## 7. PR-02: Split firewall — защищаем benchmark до обучения

### Основной TDC benchmark
- Использовать **официальные scaffold splits** TDC ADMET, где они есть. Никогда не переопределять публичный test по удобству.
- Сохранять immutable test IDs под version control, закрыть test-set training/inference distillation (no teacher labeling of test for supervised training).
- Внутри train создать scaffold-disjoint validation для hyperparameter search.

### Вторичные OOD-проверки
- Time-based split по assay date при наличии дат; target family and protein sequence cluster splits для binding.
- Проверять одинаковый canonical compound и scaffold между splits, synonyms, salt forms, stereoisomers.
- Для PLINDER freeze release и official train/val/test; учитывать similarity pocket/protein/ligand, not merely PDB ID.
- Не использовать ZINC-структуры из reserved test наборов для supervised synthetic property objectives, если оценивается generalization на novelty; for broad unsupervised pretraining explicitly report exposure audit.
- Разделить **development test** (можно использовать для выбора разработки ограниченно) и финальный **one-shot release holdout**.

### Unit tests for leakage
`test_duplicate_molecule_cross_split`, `test_scaffold_rules`, `test_plinder_release_frozen`, `test_target_cluster_disjoint`, `test_label_scaler_fit_train_only`, `test_teacher_cache_never_contains_test_labels`.

**PASS:** воспроизводимые manifest/split hashes; 0 запрещённых пересечений; отчёт `reports/leakage_audit.md`.

## 8. PR-03: Baselines прежде нейросетевой дистилляции

Каждый endpoint:
- RDKit ~100-200 descriptors → median imputation trained on train → LightGBM/XGBoost.
- Morgan fingerprint ECFP4 (2048 bits) → Random Forest, XGBoost, Tanimoto kNN.
- Simple MPNN/Chemprop-like from scratch; обязательно model configs and seeds.
- Pretrained ChemBERTa-type encoder + linear probe if license OK.
- Always dummy majority/mean baseline.

Метрики: classification AUROC/AUPRC/Brier/NLL/ECE plus prevalence; regression MAE/RMSE/Spearman and uncertainties. Three/five seeds, paired bootstrap at molecule/scaffold group, CI 95%.

**Гейт:** if tiny GNN не превосходит ECFP+XGBoost по meaningful target, no claim; investigate features and label noise first.

## 9. PR-04: Tiny molecular encoder family

### Base encoder
- Input graph: atom features element, formal charge, aromatic, chirality, valence, hydrogen count; bond features order, conjugation, stereo, ring; optional standardized descriptors.
- GINE or directed MPNN, 4–10 layers, residuals, layer norm, dropout, global pooling (mean+attention); configurable embedding width 128/256/384.
- Learned assay context embeddings only for tasks with supported training examples; disable OOD extrapolation to unknown assay IDs.
- Multitask masked heads; no label leakage via assay-only shortcuts.

### Param budgets (confirm with `sum(p.numel())`)
- Tiny ~5M; Small ~15M; Medium ~30M; binding fusion ≤100M.
- Early stop on scaffold validation primary metric, patience 15–30 evals; min epochs + keep best checkpoint.

### Initial search space
```yaml
encoder:
  type: gine
  hidden_dim: [128, 256, 384]
  depth: [4, 6, 8]
  dropout: [0.0, 0.1, 0.2]
training:
  precision: [bf16, fp32]
  optimizer: adamw
  lr: [0.0001, 0.0003, 0.001]
  weight_decay: [0.0, 0.01]
  batch_size: [32, 64, 128]
  seeds: [11, 23, 47]
  early_stopping: scaffold_val
```
Start with 5M single task; then multitask 15/30M. These are search proposals, not verified optimum.

## 10. PR-05: Generated cheap tasks (inspired by molecular-design Qwen paper)

Paper: **Training Large Language Models for Small-Molecule Design with Synthetic Task Scaling** (2026), https://arxiv.org/abs/2609.04735.

### What the paper actually demonstrates
- Tier-1: 200,000 tasks from ZINC seed compounds, 116 RDKit/OpenEye descriptors, one optimize target + 2–4 constraints + BRICS-preserved scaffold.
- Tier-2: 999 PLINDER protein–ligand train systems; ligand/pocket observations converted into tool-readable text, candidate assessment with structure-based proxy.
- Final evaluation: 40 held-out PLINDER test systems with different more expensive SQM scoring.
- RL recipe at 35B scale; cannot copy compute recipe directly onto 2×3090.
- Proprietary OpenEye components may require a license. Our default pipeline uses fully available RDKit descriptors and public tools, with separately audited alternatives.

### Our task generator (100k → 200k examples only after validation)
1. Sample permitted ZINC molecule, standardize and compute RDKit descriptor vector, store descriptor version.
2. Choose one objective among QED, logP, TPSA, MW, HBD, HBA, rotatable bonds, ring metrics, synthetic accessibility *proxy* (flag computed, not experimental).
3. Choose 2–4 constraints and a BRICS-derived scaffold/substructure. Require feasibility check with known candidate or mark `unsolved`.
4. Construct candidate pool through BRICS recombination, matched molecular pairs, rule-based neighborhood, modest random edits with validity filters.
5. Compute **exact deterministic labels** for RDKit quantities; construct pairwise candidate preference or best-of-N ranking.
6. Include hard negatives: valid but constraint-violating, same-scaffold but worse target, tempting high reward but poor diversity, invalid and tautomer-equivalent.
7. Keep `task_id`, seed, source license, descriptor computation version, generator seed, candidate list, oracle outputs, failure reasons.
8. Ensure all candidates for a given scaffold neighborhood share one split and don't reuse benchmark holdouts.

**Do not confuse** predicted QED/SA/logP rewards with experimentally measured ADMET; synthetic tasks can train constraint-following and ranking, *not prove pharmaceutical quality*.

### Training usage
- For **Minimol-Rank**: pairwise/listwise ranking, filter invalid candidates, predict which candidate passes exact constraints; train on 100k–200k synthetic tasks.
- For molecular encoder: self-supervised descriptor prediction as auxiliary pretraining, but ablate vs starting from experimental labels to ensure it does not hurt scientific generalization.
- For optional small generator: supervised successful edits → short-budget RL or best-of-N rejection sampling; do not require 35B prime-rl training.

## 11. PR-06: Distillation — teacher selection and ablation

**Hierarchy:** known labels (experimental) first, reliable computation second, teacher probabilities third, unverified freeform explanation last.

### Molecular teacher candidates
- ECFP+boosting and strong GNN ensemble can serve as soft-label teachers; calibrated held-out probabilities only.
- Well-validated pretrained ChemBERTa/graph foundation model for embeddings, only if license and benchmark contamination checks pass.
- For binding, experimentally grounded multitask models trained exclusively on train compounds/targets; proxy docking scores are separate noisy tasks.

### Carbon-X and Carbon-A roles (DO NOT overclaim)
- **Carbon-A:** 1.2B strand-specific genomic CDS probability maps. Use only in gene annotation or target-gene feature experiments. Not for SMILES ADMET.
- **Carbon-X:** DNA & biological reasoning model. Option scoring of explicit genomic variant/target questions; evaluate against independent genomic labels first. Not a chemically calibrated affinity oracle.
- For target variant branch, store DNA build, variant positions, protein consequence, reference/alternate context, source; ensure no inference of biochemical effects from CDS-only predictions.

### Loss recipes
Multitask labeled:
`L_supervised = Σ_t (mask_t * weight_t * BCE/Huber/GaussianNLL)_t / Σ_t mask_t`.
Soft teacher binary:
`L_distill = T² * KL(Bern(p_teacher^T) || Bern(p_student^T))`, mask unreliable/OOD teacher.
Regression:
`L = α*Huber(y_true,y_student) + β*w_teacher*Huber(teacher_pred,student_pred)` with β=0 baseline and tuned validation.
Feature matching:
`L_feat = cosine(project(student_embedding), stopgrad(teacher_embedding))`, **only** if embedding alignment and precompute cost justify it.
Calibration trained after model selection, on untouched calibration split; NEVER calibrate using test.

### Mandatory experiment grid
A. supervised only; B. synthetic pretraining then supervised; C. teacher soft labels then supervised; D. A+C; E. A+B+C; F. frozen encoder + heads; G. full finetune. Same splits/seeds/budgets. If distillation hurts, ship the non-distilled model.

## 12. PR-07: Binding model (research stage)

- Scope: predict one defined experimental endpoint at a time (e.g. biochemical Ki with quality filters). IC50 only with assay-aware treatment, avoid treating it as universal binding constant.
- Molecular encoder + precomputed protein embeddings (or tiny sequence encoder) + cross-features / bilinear fusion / cross-attention bounded to ≤100M runtime parameters.
- Training labels from BindingDB/ChEMBL with relation operators, assay context; retain censored data; assay-level normalization.
- Dedicated tests: unseen compounds, unseen target proteins, combined novelty split, time split, target family leave-out.
- Required baselines: ligand-only, protein-only, simple concatenation MLP, matrix factorization, sequence+ligand baseline; if protein-only performs strangely well, audit target priors.
- PLINDER for structural **representation and ranking** tasks, not a substitute for experimental affinity labels. Use PLINDER official test restrictions.
- Independent docking rescoring can be research analysis, not primary ground truth. Record pose validity/failure rates and avoid choosing scores on test.

## 13. PR-08: Minimol-Precision (high-risk R&D)

**Question:** given a ligand and WT/mutant target, can the model predict *measured change in affinity/activity* for a particular assay?

- Data requirement: matched WT/mutant target measurements for same ligand and matched assay; all triplets carry sequence positions and conditions.
- Avoid training on computational mutation scores alone; define auxiliary proxy labels separately.
- Encode molecular graph, WT and mutant protein representation; interaction uses Δprotein embedding, molecular embedding, assay ID.
- Train paired ranking/regression and direction class (decrease/no change/increase) with confidence interval.
- Ensure leakage split by mutation site, ligand scaffold, protein family; report all three OOD scenarios.
- Baselines: zero-delta, nearest mutant neighbor, sequence-only, docking-only, target+ligand no mutation.
- Gate: no market claims until independent matched-mutation experiments and prospective lab validation.

## 14. PR-09: Minimol-Rank decision engine

### Input
- Task specification: endpoint thresholds, hard constraints, candidate structures, optional target, available verification budget.
- No arbitrary question gets a numeric probability; reject unsupported endpoint/domain.

### Architecture
- Encode each candidate with frozen/shared molecular encoder.
- Condition on typed query + normalized constraints; pairwise/listwise scoring MLP or shallow attention.
- For known deterministic constraints call RDKit **directly**, do not learn facts a code checker can compute exactly.
- Use prediction heads only for difficult measured properties and attach uncertainty; Pareto filtering before ranking.
- Implement `recommend_next_to_measure`: uncertainty + diversity + predicted upside, optional active learning.

### Benchmark
- Static property ordering accuracy, feasibility, top-k enrichment, precision@k, NDCG, hypervolume, chemical novelty/diversity, invalidity rate, calibration and abstention quality, oracle calls per accepted candidate, runtime.
- Compare to random, greedy search, RDKit rule-based/BRICS search, Bayesian optimization, simple MMP edit search, and pretrained generator if present.
- Rule: measure performance with fixed candidate budget and same oracle calls.

## 15. PR-10: Optional molecular generator — separate from JEV

This is not a JEV model. It is a proposal engine whose candidates are verified and ranked by JEV-like heads.

Options in ascending complexity:
1. **BRICS + matched molecular pair edits + beam/best-of-N** (best initial baseline).
2. Small graph-edit policy with constrained edit actions.
3. Small SMILES autoregressive model with supervised edit examples; optionally LoRA on suitable compact chemical-aware LLM.
4. RL with cheap deterministic objectives only after strong imitation baseline; prefer offline preference/DPO or bandit optimization first on two GPUs.

### Cheap-to-expensive curriculum
Tier 0: syntactic validity + exact RDKit constraints.
Tier 1: property-descriptor optimization with strict fixed budget.
Tier 2: experimentally trained ADMET proxy + uncertainty & diversity constraints.
Tier 3: structure-informed proxy on PLINDER **train only**; reserve expensive independent scoring for final evaluation, not reward leakage.
Tier 4: independent lab measurements through partner (future).

### Reward against exploitation
`reward = verified_feasibility * normalized_objective - λ_ood * ood_penalty - λ_unc * uncertainty - λ_dup * duplicate_penalty - λ_cost * oracle_cost`, with hard invalidity rejection; every reward component logged.

**Red-team evaluations:** property gaming, novelty collapse, tautomer tricks, invalid molecules, extreme MW, stereochemistry destruction, shortcut through assayed compound memorization, excessive oracle calls.

## 16. PR-11: Scientific evaluation and benchmark protocol

### Benchmark matrix

| Track | Source & split | Main metrics | Negative controls |
|---|---|---|---|
| ADMET | TDC official scaffold splits (22 datasets) | AUROC/AUPRC, MAE/Spearman per endpoint; Brier/ECE | ECFP+XGB, RF, Chemprop-like, pretrained encoder |
| Tox | Tox21 etc. endpoint-aware | AUPRC, ROC, calibration & sensitivity at fixed specificity | simple fingerprint classifier |
| Bind | BindingDB/ChEMBL strict assay filters, target/scaffold/time splits | MAE/RMSE on log units, correlation, ranking, calibration | ligand-only, protein-only, regression baseline |
| Pocket/pose | PLINDER pinned official test | pose validity, RMSD / appropriate structural metrics, ranking | docking baseline, physics proxy |
| Rank | held-out synthetic objectives + external experimental candidates | feasibility/top-k/hypervolume/diversity/oracle calls | greedy BRICS, BO, random, MMP |
| Precision | independently measured matched WT/mutant pairs | Δ-assay MAE, directional accuracy, CI coverage | no-change, target-only |

### Statistical/reporting rules
- Pre-register primary metric for each endpoint, secondary metrics, target minimum improvement, limits on hyperparameter trials.
- Report mean±std over ≥3 training seeds; paired bootstrap 95% CI, clustered by scaffold/target/system where appropriate; compare per endpoint, not only grand average.
- Submit full TDC group where feasible, don't cherry-pick endpoints.
- For imbalanced classification use AUPRC and base rate; for small data include CIs and reliability warnings.
- For regression report MAE in physical units after inverse standardization, plus Pearson/Spearman, interval coverage and sharpness.
- Calibration: temperature / Platt / isotonic on dedicated calibration subset; ECE+reliability curves+Brier score, threshold-specific performance.
- OOD: scaffold novelty, Tanimoto distance, target sequence similarity, assay shift; selective risk coverage curves for abstention.
- Latency: CPU (single and batched), GPU (batch sizes 1/32/256), peak RSS/VRAM, throughput, warm/cold start, runtime artifact size; include pre-processing and protein embedding lookup cost.
- Avoid using only simulated scores as real-world efficacy claims.

### Success gates (provisional decision targets, not predictions)
- P0: ADMET-15M wins ≥3 carefully chosen endpoints against Chemprop-like baseline with CI excluding zero **or** maintains equivalent quality with ≥5× lower measured CPU inference cost; no severe regression elsewhere.
- P1: calibrated probability performance materially improves Brier/NLL on held-out endpoints and abstention reduces error as coverage decreases.
- P2: successful end-to-end candidate ranking beats best simple BRICS/BO baseline at fixed oracle budget on independent evaluation.
- P3: external collaborator confirms reproducible improvement in prospective assay prioritization.

## 17. Calibrated uncertainty / reliability

- Binary heads: logits → separate calibration on a calibration split; multiclass: vector/temperature scaling under adequate sample sizes; continuous: residual/conformal or distributional model with group-aware validation.
- Distinguish epistemic and aleatoric uncertainty where possible (ensemble of 3 seeds, heteroscedastic noise head).
- OOD proxy: nearest scaffold similarity, latent density/distance, conformal applicability intervals; calibrated score only within supported domain.
- Prediction report includes `calibration_n`, source assay, coverage, known failure modes.
- No claiming medical safety based on `P(toxic)<0.1` absent correct endpoint, assay and validation.

## 18. Synthetic data schema and storage

```json
{
  "task_id":"sha256:...",
  "source_smiles":"...",
  "source_dataset":"ZINC-subset-vX",
  "scaffold":"BRICS fragment hash",
  "objective":{"property":"qed","direction":"maximize"},
  "constraints":[{"property":"tpsa","max":100.0}],
  "candidates":[{"smiles":"...","valid":true,"oracle":{"qed":0.61,"tpsa":81.2}}],
  "evaluation_method":"rdkit-<pinned version>",
  "provenance":{"source_record":"...","split":"train","license_id":"..."},
  "verification":{"all_candidates_recomputed":true,"hard_constraints_enforced":true}
}
```

Parquet shards target 128–512 MB compressed; content-addressed cache for teacher outputs; hash of model and input normalization; resume with exactly-once sample ID processing; store failures and retry budgets, not just successes.

For teacher output record `teacher_model_commit`, `tokenizer_commit`, inference precision/quantization, decoding config, calibration status, confidence derivation. Do not mistake LLM token likelihoods for experimental probabilities.

## 19. Training orchestration on 2×3090

### GPU plan
- GPU 0: 5–30M graph training or one compact chemical model; GPU 1: separate experiment / molecular teacher inference; use both GPUs for DDP only after throughput proof.
- Carbon-X 14.8B may need quantization or 2-GPU sharding; use only sporadically for genomic branch to avoid blocking molecular tasks.
- Carbon-A 1.2B can be run more readily, but long-context memory can dominate; benchmark windows before assuming full context will fit.
- Cache protein embeddings offline when permitted; document the model generating them and total preprocessing footprint.

### Run sequence (do not launch all at once)
1. Train ECFP+XGB per ADMET endpoint (CPU) and baseline MPNN (1 GPU).
2. Tiny/Small GINE 5M/15M three seeds each; evaluate on scaffold validation.
3. 30M for tasks with evidence of scaling benefit only.
4. Synthetic task generation on CPU and Minimol-Rank on GPU.
5. Distillation ablations and calibration.
6. Binding and precision only after data gates.

### Profiling record
`run_id, GPU, effective_batch, atom_count, model_params, step_time, it/s, allocated/reserved VRAM, dataloader_time, eval_time, total_gpu_hours, energy estimate (optional)`.

Checkpoints: every N steps and best val, max 3–5 checkpoints per experiment, auto-resume; emergency handling for OOM (reduce batch, gradient accumulation, enable checkpointing; log exact change).

## 20. Commands / CLI contract (to be implemented)

**These commands are *specifications for repo scripts*, not tools already available:**

```bash
python -m minimol.data.registry validate --config configs/data/admet.yaml
python -m minimol.data.build --dataset tdc_admet --out data/processed/admet
python -m minimol.data.split_audit --manifest data/processed/admet/manifest.json
python -m minimol.training.baselines --config configs/training/admet.yaml
python -m minimol.training.train_admet --config configs/models/gine_5m.yaml --seed 11
python -m minimol.training.train_admet --config configs/models/gine_15m.yaml --seed 23
python -m minimol.generation.task_factory --config configs/data/synthetic.yaml --limit 1000
python -m minimol.training.train_ranker --config configs/training/rank.yaml
python -m minimol.training.distill --config configs/training/distill.yaml
python -m minimol.evaluation.benchmark --suite tdc-admet --out results/admet_v1
python -m minimol.evaluation.calibration --run results/admet_v1
python -m minimol.serving.export --run results/admet_v1 --format onnx
pytest -q
```

## 21. API format

Example POST `/v1/predict`:

```json
{
  "model":"minimol-admet-15m@1.0.0",
  "molecules":[{"id":"candidate_01","smiles":"CCO"}],
  "questions":[
    {"id":"q1","type":"regression","endpoint":"aqsol_logS"},
    {"id":"q2","type":"noul","endpoint":"aqsol_logS","event":{"gt":-3.0}}
  ]
}
```

Output structure (numbers are placeholders, never clinical facts):

```json
{
  "predictions":[{"id":"candidate_01","outputs":[
    {"question_id":"q1","value":null,"units":"log_mol/L","interval":null},
    {"question_id":"q2","probability":null,"calibration_version":"pending"}
  ],"ood_flag":null}],
  "model_version":"1.0.0","warnings":["research use only"]
}
```

Unknown endpoint, unparseable SMILES, out-of-scope element or missing target embedding → typed error/abstention rather than fake probability. Batched inference deduplicates molecules and uses cache.

## 22. Team, milestones and budgets

### Phase 0 — weeks 1–2: evidence and baseline
- PR-00 env; PR-01 data registry/standardization; PR-02 leakage firewall; PR-03 ECFP+XGB and MPNN.
- Deliver: `reports/baseline_v0.md`, dataset/benchmark manifests, resource profile.
- STOP if data licensing or label harmonization unsolved.

### Phase 1 — weeks 3–5: tiny ADMET models
- PR-04 5/15/30M; three seeds, full TDC run; quantization trial; calibration.
- Deliver: first candidate model and reproducible benchmarking scripts.
- STOP if simple baseline is better and no efficiency advantage.

### Phase 2 — weeks 6–8: synthetic curriculum / decision heads
- PR-05 task factory, 1k smoke→20k→100k→200k if QC passes; PR-09 ranker; PR-06 distillation ablations.
- Deliver: deterministic task dataset with provenance + Minimol-Rank baseline.
- STOP if reward hacking or diversity collapse.

### Phase 3 — weeks 9–12: tox, independent OOD and product preview
- Extend assay-specific Tox heads; create API/ONNX; latency/CPU profiling; model/data cards; external benchmark.
- Deliver: reproducible research preview, not medical device.

### Phase 4 — months 4–6: binding R&D
- Curate BindingDB/ChEMBL assay-specific data and PLINDER; train ≤80M fusion; target novelty evaluation.
- Deliver: binding proof-of-concept only if it beats target/ligand-only baselines.

### Phase 5 — months 6+: precision and independent validation
- Matched mutation-specific data, external collaborator, blinded prospective measurement, model iteration.
- Deliver: research results; no claims of clinical outcome.

### Staff assumption
One ML engineer + part-time computational chemist/biologist for label validity and evaluation; independent review for high-impact commercial claims. More collaborators improve scientific reliability more than simply adding model size.

## 23. PR-by-PR acceptance checklist

- [ ] PR-00: deterministic smoke, env manifest, checkpoint resume.
- [ ] PR-01: chemistry standardization tests incl. stereo/salts, source/licence registry.
- [ ] PR-02: no prohibited train/val/test leakage, audit report.
- [ ] PR-03: ECFP+XGB + Chemprop-like strong baselines, all endpoint results.
- [ ] PR-04: 5/15/30M models, parameter counts, seeds, reproducible checkpoints.
- [ ] PR-05: 1k/20k/100k task QA, valid candidates, property recomputation, source tracking.
- [ ] PR-06: supervised/distill ablation complete; teacher not assumed superior.
- [ ] PR-07: binding clean assays and target generalization baselines.
- [ ] PR-08: matched mutation data and independent mutation/target splits.
- [ ] PR-09: calibrated typed decision outputs and abstain tests.
- [ ] PR-10: optional generator evaluated vs simple fragment search at fixed budget.
- [ ] PR-11: external benchmarks, grouped CIs, latency, model cards.
- [ ] Legal review: source/weight/code/teacher terms, attribution, redistribution decisions.
- [ ] Science review: endpoint definitions, limits, independent lab relevance.

## 24. Research kill-switches and failure diagnostics

**If validation AUROC improves but OOD degrades:** investigate scaffold memorization, similarity-conditioned metrics, target shortcuts.  
**If multitask hurts small tasks:** train specialized heads/encoders and gate by assay.  
**If teacher distillation hurts:** stop teacher; direct supervised baseline wins.  
**If model confident OOD:** require abstention/caution; test calibration by similarity buckets.  
**If generative policy exploits RDKit proxies:** keep separate experimental/structural oracle and penalty for unrealistic chemistry; no artificial efficacy claims.  
**If no useful labels for Minimol-Precision:** do not fabricate target-mutation labels from Carbon; postpone.  
**If small model underperforms simple tools:** ship deterministic pipeline instead of a model.

## 25. Publication and commercialization package

Every released artifact must ship:
- Versioned weights, executable inference, pinned dependencies, signed SHA checksums.
- Dataset card, model card, train manifest hashes and exclusions, benchmark suite scripts and leaderboard results.
- License and NOTICE per code/weights/data; provenance and relevant upstream license review.
- Clearly delimited endpoints and applicability domain; uncertainty/calibration caveats and exclusions.
- CPU/GPU latency benchmarks and memory footprint with measured hardware/software environment.
- Intended use: exploratory research candidate prioritization, not medical advice or clinically validated prediction.
- Safety guardrails for unsupported molecules, endpoints, and uncertainty, audit logs of inference and model version.

## 26. Source bibliography (retrieved/checked 2026-10-28)

1. Hu et al. (2026), *Training Large Language Models for Small-Molecule Design with Synthetic Task Scaling*. https://arxiv.org/html/2609.04735v1
2. Official TDC ADMET group: https://tdcommons.ai/benchmark/admet_group/overview/
3. PLINDER releases, evaluation/splits: https://github.com/plinder-org/plinder
4. PrimeRL: https://github.com/PrimeIntellect-ai/prime-rl — enterprise-scale async RL; optional reference, not required for our 2×3090 prototype.
5. ZINC20: https://zinc20.docking.org/
6. RDKit: https://www.rdkit.org/docs/
7. ChEMBL: https://www.ebi.ac.uk/chembl/
8. BindingDB: https://www.bindingdb.org/
9. Carbon-A: https://huggingface.co/HuggingFaceBio/Carbon-A-1.2B
10. Carbon-X: https://huggingface.co/HuggingFaceBio/Carbon-X
11. Needle reference: https://github.com/cactus-compute/needle
12. SMDD-Bench: https://smddbench.com/ — evaluate applicability of its tasks, budget controls, licensing and contamination before any claimed results.

---

**Final go/no-go:** release candidate accepted only if (a) legal/data gates pass, (b) endpoint and split correctness verified, (c) measurable scientific or economic advantage against baseline survives independent test, (d) calibrated abstention and clear limitations, (e) fully repeatable on stated hardware. Otherwise publish as exploratory research, not commercial drug-discovery predictor.


## Актуализация задания по результатам обучения — 29 октября 2026

Обязательный ближайший этап: [исправление и улучшение обучения](Training_Recovery_Plan.md).
Документ фиксирует обнаруженную ошибку головы Reg, исправление накопления
градиентов, совместимость старых чекпоинтов, повторное обучение Reg-5M,
диагностику ADMET по endpoint и критерии перехода к большим моделям.
Чеклист и критерии приёмки из этого дополнения входят в настоящее ТЗ.


## Следующий этап после Reg: обучаемые решения

Актуальные результаты, восстановленная оценка checkpoint и задание на
question-conditioned Reg Decide: [Decision_Model_Next_Stage.md](Decision_Model_Next_Stage.md).
