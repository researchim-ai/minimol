"""Command line for the Minimol research pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minimol.errors import LicenseError, MinimolError, SplitLeakageError


def _fail(exc: Exception) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 1


def cmd_env_check(_args) -> int:
    from minimol.systeminfo import write_system_info

    info = write_system_info(Path.cwd())
    print(f"python {info['python']} cuda {info.get('cuda_available')}")
    return 0


def cmd_audit_licenses(args) -> int:
    from minimol.data.registry import validate_registry

    errors = validate_registry(args.registry)
    if errors:
        print("error: " + "; ".join(errors), file=sys.stderr)
        return 1
    print(f"registry ok: {args.registry}")
    return 0


def cmd_fetch_malinois(args) -> int:
    from minimol.data.fetch_public import download_malinois, import_malinois
    from minimol.data.registry import load_registry

    payload = download_malinois(Path(args.registry), Path(args.raw))
    row = load_registry(args.registry)
    revision = str(row.loc[row["source"] == "malinois_mpra", "version"].iloc[0])
    manifest = import_malinois(Path(args.raw), Path(args.out), revision=revision)
    print(f"malinois rows {manifest['n_rows']} sha256 {manifest['sha256']} revision {payload['revision']}")
    return 0


def cmd_fetch_tdc(args) -> int:
    from minimol.data.fetch_public import download_tdc_admet

    payload = download_tdc_admet(Path(args.registry), Path(args.raw), Path(args.out), seed=args.seed)
    print(f"tdc datasets {len(payload['datasets'])} revision {payload['revision']}")
    return 0


def cmd_reg_sample(args) -> int:
    from minimol.data.synthetic import write_reg_sample

    manifest = write_reg_sample(Path(args.out), n_groups=args.n, length=args.length, seed=args.seed)
    print(f"rows {manifest['n_rows']} sha256 {manifest['sha256']}")
    return 0


def cmd_cds_sample(args) -> int:
    from minimol.data.synthetic import write_cds_sample

    manifest = write_cds_sample(Path(args.out), per_species=args.per_species, length=args.length, seed=args.seed)
    print(f"rows {manifest['n_rows']}")
    return 0


def cmd_admet_sample(args) -> int:
    from minimol.data.synthetic import write_admet_sample

    manifest = write_admet_sample(Path(args.out), seed=args.seed)
    print(f"rows {manifest['n_rows']}")
    return 0


def cmd_bind_sample(args) -> int:
    from minimol.data.synthetic import write_bind_sample

    manifest = write_bind_sample(Path(args.out), seed=args.seed)
    print(f"rows {manifest['n_rows']}")
    return 0


def cmd_precision_sample(args) -> int:
    from minimol.data.synthetic import write_precision_sample

    manifest = write_precision_sample(Path(args.out), seed=args.seed)
    print(f"rows {manifest['n_rows']}")
    return 0


def cmd_build(args) -> int:
    from minimol.data.build import build_dataset

    manifest = build_dataset(args.dataset, Path(args.out), limit=args.limit, seed=args.seed)
    print(f"rows {manifest['n_rows']}")
    return 0


def cmd_reg_audit(args) -> int:
    import pandas as pd

    from minimol.data.homology import render_leakage_markdown, scan_leakage
    from minimol.data.normalize import audit_reg_frame

    frame = pd.read_parquet(args.data)
    audit = audit_reg_frame(frame)
    leakage = scan_leakage(frame)
    report = Path(args.report) if args.report else Path(args.data).parent / "leakage_audit.md"
    report.write_text(render_leakage_markdown(leakage))
    if not audit["passed"] or not leakage.passed:
        print("audit failed", file=sys.stderr)
        return 1
    print("audit passed")
    return 0


def cmd_split_audit(args) -> int:
    from minimol.data.split_audit import split_audit

    report = getattr(args, "report", None)
    split_audit(Path(args.manifest), Path(report) if report else None)
    print("split audit passed")
    return 0


def cmd_train_ridge(args) -> int:
    from minimol.training.baselines import run_reg_baselines

    payload = run_reg_baselines(Path(args.data), Path(args.out), k_max=args.k_max)
    print(f"ridge macro MAE {payload['ridge']['macro_mae']}")
    return 0


def cmd_decide_prepare(args) -> int:
    from minimol.data.reg_decisions import prepare_cache

    prepare_cache(Path(args.checkpoint), Path(args.data), Path(args.out),
                  train_limit=args.train_limit, validation_limit=args.validation_limit, device=args.device)
    return 0


def cmd_decide_train(args) -> int:
    from minimol.config import load_yaml
    from minimol.training.train_reg_decide import train_reg_decide

    config = load_yaml(args.config)
    if args.seed is not None:
        config.setdefault("train", {})["seed"] = args.seed
    if args.resume is not None:
        config.setdefault("train", {})["resume_from"] = args.resume
    report = train_reg_decide(config, Path(args.cache or config["cache"]), Path(args.out))
    print(report["decision"])
    return 0


def cmd_decide_summary(args) -> int:
    from minimol.evaluation.decide_experiment import question_controls, summarize_decide_runs

    report = summarize_decide_runs([Path(run) for run in args.runs], Path(args.out))
    if args.cache:
        from minimol.jsonutil import write_json
        report["controls"] = question_controls(Path(args.cache), Path(args.runs[0]))
        write_json(args.out, report)
    print(report["decision"], report["recommended_backend"])
    return 0


def cmd_decide_predict(args) -> int:
    import json

    from minimol.decide import RegDecider
    from minimol.jsonutil import write_json

    request = json.loads(Path(args.request).read_text())
    result = RegDecider.load(args.run, backend=args.backend).predict(request)
    write_json(args.out, result)
    return 0


def cmd_train_reg(args) -> int:
    from minimol.config import load_yaml
    from minimol.training.train_reg import train_reg

    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    if getattr(args, "seed", None) is not None:
        config.setdefault("train", {})["seed"] = args.seed
    data = args.data or config.get("data", {}).get("path")
    if getattr(args, "resume", None):
        config.setdefault("train", {})["resume_from"] = args.resume
    train_reg(config, Path(data), Path(args.out))
    return 0


def cmd_train_cds(args) -> int:
    from minimol.config import load_yaml
    from minimol.training.train_cds import train_cds

    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    data = args.data or config.get("data", {}).get("path")
    train_cds(config, Path(data), Path(args.out))
    return 0


def cmd_train_admet(args) -> int:
    from minimol.config import load_yaml
    from minimol.training.train_admet import train_admet

    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    if args.seed is not None:
        config.setdefault("train", {})["seed"] = args.seed
    data = args.data or config.get("data", {}).get("path")
    train_admet(config, Path(data), Path(args.out))
    return 0


def _train_from_config(train_fn, args) -> int:
    from minimol.config import load_yaml

    config = load_yaml(args.config)
    if getattr(args, "max_steps", None) is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    data = args.data or config.get("data", {}).get("path")
    train_fn(config, Path(data), Path(args.out))
    return 0


def cmd_train_bind(args) -> int:
    from minimol.training.train_bind import train_bind

    return _train_from_config(train_bind, args)


def cmd_train_precision(args) -> int:
    from minimol.training.train_precision import train_precision

    return _train_from_config(train_precision, args)


def cmd_train_tox(args) -> int:
    from minimol.training.train_tox import train_tox

    return _train_from_config(train_tox, args)


def cmd_train_admet_baseline(args) -> int:
    from minimol.training.baselines import run_chem_baselines

    payload = run_chem_baselines(Path(args.data), Path(args.out))
    print(f"backend {payload['backend']} endpoints {len(payload['endpoints'])}")
    return 0


def cmd_eval(args) -> int:
    from minimol.evaluation.reload import evaluate_cds_checkpoint, evaluate_reg_checkpoint

    if args.task == "reg":
        payload = evaluate_reg_checkpoint(Path(args.checkpoint), Path(args.data), Path(args.out))
        print(f"macro_spearman {payload['metrics'].get('macro_spearman')}")
    else:
        payload = evaluate_cds_checkpoint(Path(args.checkpoint), Path(args.data), Path(args.out))
        print(f"macro_f1 {payload['metrics'].get('macro_f1')}")
    return 0


def cmd_bind_baselines(args) -> int:
    from minimol.evaluation.bind_baselines import write_binding_baselines

    report = write_binding_baselines(Path(args.data), Path(args.out))
    print(f"target_prior_risk {report['target_prior_risk']}")
    return 0


def cmd_precision_baselines(args) -> int:
    from minimol.evaluation.precision_baselines import write_precision_baselines

    report = write_precision_baselines(Path(args.data), Path(args.out))
    print(f"zero_delta MAE {report['scores']['zero_delta']['mae']}")
    return 0


def cmd_controls(args) -> int:
    from minimol.evaluation.controls import write_regulatory_controls

    report = write_regulatory_controls(Path(args.data), Path(args.out), seed=args.seed)
    print(f"leakage_suspect {report['leakage_suspect']}")
    return 0


def cmd_aggregate(args) -> int:
    import json

    from minimol.evaluation.aggregate import write_aggregate

    records = [json.loads(Path(path).read_text()) for path in args.metrics]
    report = write_aggregate(records, args.field, Path(args.out))
    print(f"runs {report['n_runs']}")
    return 0


def cmd_ood(args) -> int:
    from minimol.evaluation.ood_report import write_applicability

    report = write_applicability(Path(args.data), Path(args.out), threshold=args.threshold)
    print(f"fraction_ood {report['fraction_ood']}")
    return 0


def cmd_motif(args) -> int:
    from minimol.evaluation.motif_baseline import write_motif_baseline

    report = write_motif_baseline(Path(args.data), Path(args.out), args.motif)
    k562 = report["motif"]["per_target"].get("K562", {})
    print(f"K562 MAE {k562.get('mae')}")
    return 0


def cmd_rank_benchmark(args) -> int:
    from minimol.evaluation.rank_benchmark import fixed_budget_rank_report
    from minimol.generation.task_factory import make_tasks
    from minimol.jsonutil import write_json

    tasks = make_tasks(limit=args.limit, seed=args.seed)
    report = fixed_budget_rank_report(tasks, k=args.k, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", report)
    print(f"oracle NDCG {report['oracle_mean_ndcg']} random NDCG {report['random_mean_ndcg']}")
    return 0


def cmd_train_rank(args) -> int:
    import json

    from minimol.config import load_yaml
    from minimol.generation.task_factory import make_tasks
    from minimol.training.train_ranker import train_ranker

    config = load_yaml(args.config) if args.config else {"model": {"preset": "5m", "overrides": {"fp_dim": 128, "hidden": 64, "depth": 1, "dropout": 0.0}}}
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    if args.tasks:
        tasks = [json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    else:
        tasks = make_tasks(limit=args.limit, seed=args.seed)
    train_ranker(tasks, Path(args.out), config)
    return 0


def cmd_tasks(args) -> int:
    from minimol.generation.task_factory import make_tasks, write_tasks

    tasks = make_tasks(limit=args.limit, seed=args.seed)
    write_tasks(Path(args.out), tasks)
    print(f"tasks {len(tasks)}")
    return 0


def cmd_teacher_a(args) -> int:
    from minimol.teachers.carbon_a import assert_carbon_a_loadable, local_coordinate_smoke

    if args.load_upstream:
        assert_carbon_a_loadable(args.revision, args.i_reviewed_remote_code)
        print("loader gate passed; weights are not fetched by the default smoke")
        return 0
    report = local_coordinate_smoke(args.sequence)
    print(
        f"tokens {report['n_tokens']} valid_bases {report['valid_bases']} "
        f"upstream {report['upstream_weights']} verified {report['verified_against_upstream']}"
    )
    return 0


def cmd_teacher_x(args) -> int:
    from minimol.teachers.carbon_x import (
        assert_carbon_x_loadable,
        go_nogo,
        scores_to_pseudo_distribution,
    )

    if args.load_upstream:
        assert_carbon_x_loadable(args.revision, args.i_reviewed_remote_code)
    if args.with_metric is not None and args.without_metric is not None:
        report = go_nogo(args.with_metric, args.without_metric)
        print(report["decision"], report["delta"])
        return 0 if report["decision"] == "NO-GO" else 0
    demo = scores_to_pseudo_distribution([-1.0, -1.2, -2.0])
    print(demo["meaning"])
    return 0


def cmd_export(args) -> int:
    from minimol.serving.export import export_checkpoint

    report = export_checkpoint(Path(args.run), args.format, args.length)
    print(report["path"], "parity", report["parity"])
    return 0


def cmd_benchmark(args) -> int:
    from minimol.evaluation.benchmark import run_benchmark

    paths = [Path(item) for item in args.metrics]
    run_benchmark(args.suite, paths, Path(args.out), official_tdc=args.official_tdc)
    print(args.out)
    return 0


def cmd_model_card(args) -> int:
    import json

    from minimol.reporting import render_model_card

    metrics = json.loads(Path(args.metrics).read_text())
    render_model_card(metrics, Path(args.out))
    print(args.out)
    return 0


def cmd_release(args) -> int:
    from minimol.reporting import release_check

    report = release_check(Path(args.registry), Path(args.model_card) if args.model_card else None)
    print(
        f"automated {report['passed_automated']} commercial_release {report['commercial_release']} "
        f"legal {report['human_legal_review']}"
    )
    return 0 if report["passed_automated"] else 1


def cmd_serve(args) -> int:
    import uvicorn

    from minimol.serving.app import create_app, load_cds_checkpoint, load_reg_checkpoint

    reg = load_reg_checkpoint(args.reg) if args.reg else None
    cds = load_cds_checkpoint(args.cds) if args.cds else None
    decider = None
    if getattr(args, "decide_run", None):
        from minimol.decide import RegDecider
        decider = RegDecider.load(args.decide_run, backend=args.decide_backend)
    app = create_app(decider=decider, reg=reg, cds=cds, token=args.token, rate_limit_per_minute=args.rate_limit)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minimol")
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env-check")
    env.set_defaults(func=cmd_env_check)
    audit = sub.add_parser("audit-licenses")
    audit.add_argument("--registry", default="LICENSES/data_registry.csv")
    audit.set_defaults(func=cmd_audit_licenses)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="action", required=True)
    fetch_malinois = data_sub.add_parser("fetch-malinois")
    fetch_malinois.add_argument("--registry", default="LICENSES/data_registry.csv")
    fetch_malinois.add_argument("--raw", default="data/raw/malinois")
    fetch_malinois.add_argument("--out", default="data/processed/malinois")
    fetch_malinois.set_defaults(func=cmd_fetch_malinois)
    fetch_tdc = data_sub.add_parser("fetch-tdc")
    fetch_tdc.add_argument("--registry", default="LICENSES/data_registry.csv")
    fetch_tdc.add_argument("--raw", default="data/raw/tdc_admet")
    fetch_tdc.add_argument("--out", default="data/processed/tdc_admet")
    fetch_tdc.add_argument("--seed", type=int, default=1)
    fetch_tdc.set_defaults(func=cmd_fetch_tdc)
    reg_sample = data_sub.add_parser("reg-sample")
    reg_sample.add_argument("--out", required=True)
    reg_sample.add_argument("--n", type=int, default=40)
    reg_sample.add_argument("--length", type=int, default=96)
    reg_sample.add_argument("--seed", type=int, default=7)
    reg_sample.set_defaults(func=cmd_reg_sample)
    cds_sample = data_sub.add_parser("cds-sample")
    cds_sample.add_argument("--out", required=True)
    cds_sample.add_argument("--per-species", type=int, default=8)
    cds_sample.add_argument("--length", type=int, default=192)
    cds_sample.add_argument("--seed", type=int, default=11)
    cds_sample.set_defaults(func=cmd_cds_sample)
    admet_sample = data_sub.add_parser("admet-sample")
    admet_sample.add_argument("--out", required=True)
    admet_sample.add_argument("--seed", type=int, default=5)
    admet_sample.set_defaults(func=cmd_admet_sample)
    bind_sample = data_sub.add_parser("bind-sample")
    bind_sample.add_argument("--out", required=True)
    bind_sample.add_argument("--seed", type=int, default=3)
    bind_sample.set_defaults(func=cmd_bind_sample)
    precision_sample = data_sub.add_parser("precision-sample")
    precision_sample.add_argument("--out", required=True)
    precision_sample.add_argument("--seed", type=int, default=3)
    precision_sample.set_defaults(func=cmd_precision_sample)
    build = data_sub.add_parser("build")
    build.add_argument("--dataset", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--limit", type=int)
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(func=cmd_build)
    reg_audit = data_sub.add_parser("reg-audit")
    reg_audit.add_argument("--data", required=True)
    reg_audit.add_argument("--report")
    reg_audit.set_defaults(func=cmd_reg_audit)
    split_audit = data_sub.add_parser("split-audit")
    split_audit.add_argument("--manifest", required=True)
    split_audit.add_argument("--report")
    split_audit.set_defaults(func=cmd_split_audit)

    check = sub.add_parser("check-splits")
    check.add_argument("--manifest", required=True)
    check.set_defaults(func=cmd_split_audit)

    train = sub.add_parser("train")
    train_sub = train.add_subparsers(dest="action", required=True)
    ridge = train_sub.add_parser("reg-ridge")
    ridge.add_argument("--data", required=True)
    ridge.add_argument("--out", required=True)
    ridge.add_argument("--k-max", type=int, default=6)
    ridge.set_defaults(func=cmd_train_ridge)
    for name, func in (("reg", cmd_train_reg), ("cds", cmd_train_cds), ("admet", cmd_train_admet)):
        item = train_sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--data")
        item.add_argument("--out", required=True)
        item.add_argument("--max-steps", type=int)
        if name in {"admet", "reg"}:
            item.add_argument("--seed", type=int)
        if name == "reg":
            item.add_argument("--resume")
        item.set_defaults(func=func)
    for name, func in (
        ("bind", cmd_train_bind),
        ("precision", cmd_train_precision),
        ("tox", cmd_train_tox),
    ):
        item = train_sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--data")
        item.add_argument("--out", required=True)
        item.add_argument("--max-steps", type=int)
        item.set_defaults(func=func)
    chem_base = train_sub.add_parser("admet-baseline")
    chem_base.add_argument("--data", required=True)
    chem_base.add_argument("--out", required=True)
    chem_base.set_defaults(func=cmd_train_admet_baseline)
    rank = train_sub.add_parser("rank")
    rank.add_argument("--config")
    rank.add_argument("--tasks")
    rank.add_argument("--out", required=True)
    rank.add_argument("--limit", type=int, default=12)
    rank.add_argument("--seed", type=int, default=0)
    rank.add_argument("--max-steps", type=int)
    rank.set_defaults(func=cmd_train_rank)

    tasks = sub.add_parser("generation")
    tasks_sub = tasks.add_subparsers(dest="action", required=True)
    make_tasks = tasks_sub.add_parser("tasks")
    make_tasks.add_argument("--out", required=True)
    make_tasks.add_argument("--limit", type=int, default=20)
    make_tasks.add_argument("--seed", type=int, default=0)
    make_tasks.set_defaults(func=cmd_tasks)

    teacher = sub.add_parser("teacher")
    teacher_sub = teacher.add_subparsers(dest="action", required=True)
    carbon_a = teacher_sub.add_parser("carbon-a-smoke")
    carbon_a.add_argument("--sequence", default="ATGAAATTTAAACCCGGGTTTAAA")
    carbon_a.add_argument("--load-upstream", action="store_true")
    carbon_a.add_argument("--revision", default="UNVERIFIED")
    carbon_a.add_argument("--i-reviewed-remote-code", action="store_true")
    carbon_a.set_defaults(func=cmd_teacher_a)
    carbon_x = teacher_sub.add_parser("carbon-x-smoke")
    carbon_x.add_argument("--load-upstream", action="store_true")
    carbon_x.add_argument("--revision", default="UNVERIFIED")
    carbon_x.add_argument("--i-reviewed-remote-code", action="store_true")
    carbon_x.add_argument("--with-metric", type=float)
    carbon_x.add_argument("--without-metric", type=float)
    carbon_x.set_defaults(func=cmd_teacher_x)

    evaluate = sub.add_parser("eval")
    eval_sub = evaluate.add_subparsers(dest="task", required=True)
    for task in ("reg", "cds"):
        item = eval_sub.add_parser(task)
        item.add_argument("--checkpoint", required=True)
        item.add_argument("--data", required=True)
        item.add_argument("--out", required=True)
        item.set_defaults(func=cmd_eval, task=task)
    for name, func in (
        ("bind-baselines", cmd_bind_baselines),
        ("precision-baselines", cmd_precision_baselines),
        ("controls", cmd_controls),
    ):
        item = eval_sub.add_parser(name)
        item.add_argument("--data", required=True)
        item.add_argument("--out", required=True)
        if name == "controls":
            item.add_argument("--seed", type=int, default=0)
        item.set_defaults(func=func)
    aggregate = eval_sub.add_parser("aggregate")
    aggregate.add_argument("--metrics", nargs="+", required=True)
    aggregate.add_argument("--field", nargs="+", required=True)
    aggregate.add_argument("--out", required=True)
    aggregate.set_defaults(func=cmd_aggregate)
    ood = eval_sub.add_parser("ood")
    ood.add_argument("--data", required=True)
    ood.add_argument("--out", required=True)
    ood.add_argument("--threshold", type=float, default=0.4)
    ood.set_defaults(func=cmd_ood)
    motif = eval_sub.add_parser("motif")
    motif.add_argument("--data", required=True)
    motif.add_argument("--out", required=True)
    motif.add_argument("--motif", action="append", required=True)
    motif.set_defaults(func=cmd_motif)
    rank_bench = eval_sub.add_parser("rank-benchmark")
    rank_bench.add_argument("--out", required=True)
    rank_bench.add_argument("--limit", type=int, default=8)
    rank_bench.add_argument("--k", type=int, default=3)
    rank_bench.add_argument("--seed", type=int, default=0)
    rank_bench.set_defaults(func=cmd_rank_benchmark)

    decide = sub.add_parser("decide")
    decide_sub = decide.add_subparsers(dest="decide_action", required=True)
    prepare = decide_sub.add_parser("prepare")
    prepare.add_argument("--checkpoint", required=True)
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--train-limit", type=int, default=40000)
    prepare.add_argument("--validation-limit", type=int, default=20000)
    prepare.add_argument("--device", default="auto")
    prepare.set_defaults(func=cmd_decide_prepare)
    decision_train = decide_sub.add_parser("train")
    decision_train.add_argument("--config", required=True)
    decision_train.add_argument("--cache")
    decision_train.add_argument("--out", required=True)
    decision_train.add_argument("--seed", type=int)
    decision_train.add_argument("--resume")
    decision_train.set_defaults(func=cmd_decide_train)
    summary = decide_sub.add_parser("summary")
    summary.add_argument("--runs", nargs="+", required=True)
    summary.add_argument("--out", required=True)
    summary.add_argument("--cache")
    summary.set_defaults(func=cmd_decide_summary)
    predict = decide_sub.add_parser("predict")
    predict.add_argument("--run", required=True)
    predict.add_argument("--request", required=True)
    predict.add_argument("--out", required=True)
    predict.add_argument("--backend", choices=["trained", "calibrated_regression"], default="calibrated_regression")
    predict.set_defaults(func=cmd_decide_predict)

    export = sub.add_parser("export")
    export.add_argument("--run", required=True)
    export.add_argument("--format", default="onnx")
    export.add_argument("--length", type=int)
    export.set_defaults(func=cmd_export)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--suite", required=True)
    benchmark.add_argument("--metrics", nargs="+", required=True)
    benchmark.add_argument("--out", required=True)
    benchmark.add_argument("--official-tdc", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark)

    card = sub.add_parser("model-card")
    card.add_argument("--metrics", required=True)
    card.add_argument("--out", required=True)
    card.set_defaults(func=cmd_model_card)

    release = sub.add_parser("release-check")
    release.add_argument("--registry", default="LICENSES/data_registry.csv")
    release.add_argument("--model-card", default="docs/MODEL_CARD.md")
    release.set_defaults(func=cmd_release)

    serve = sub.add_parser("serve")
    serve.add_argument("--decide-run")
    serve.add_argument("--decide-backend", choices=["trained", "calibrated_regression"], default="calibrated_regression")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reg")
    serve.add_argument("--cds")
    serve.add_argument("--token")
    serve.add_argument("--rate-limit", type=int, default=0)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (MinimolError, LicenseError, SplitLeakageError, ValueError) as exc:
        return _fail(exc)


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
