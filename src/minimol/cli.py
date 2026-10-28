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


def cmd_train_reg(args) -> int:
    from minimol.config import load_yaml
    from minimol.training.train_reg import train_reg

    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    data = args.data or config.get("data", {}).get("path")
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
    app = create_app(reg=reg, cds=cds, token=args.token, rate_limit_per_minute=args.rate_limit)
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
        if name == "admet":
            item.add_argument("--seed", type=int)
        item.set_defaults(func=func)
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
