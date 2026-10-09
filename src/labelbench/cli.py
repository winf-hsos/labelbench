"""Command line interface: labelbench init | new | templates | check | run | compare | report | predict."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from labelbench.checks import SEVERITIES
from labelbench.classifier import ClassifierError
from labelbench.io import read_json
from labelbench.providers import ProviderError
from labelbench.scaffold import PROVIDERS, TEMPLATES, ScaffoldError
from labelbench.task import TaskError, load_task

ICONS = {"error": "x", "warning": "!", "info": "i"}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="labelbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a new project with an example task")
    p.add_argument("directory", nargs="?", default=".")
    p.add_argument("--llm", choices=PROVIDERS, help="also create an LLM classifier for this provider")
    p.add_argument("--model", help="model name for --llm (default depends on the provider)")

    p = sub.add_parser("new", help="create a task or a classifier from a template")
    new = p.add_subparsers(dest="kind", required=True)
    q = new.add_parser("task", help="create an empty task in tasks/NAME")
    q.add_argument("name")
    q = new.add_parser("classifier", help="create a classifier config (and code or prompts)")
    q.add_argument("name")
    q.add_argument("--template", choices=TEMPLATES,
                   help="rules or llm; --provider alone implies llm")
    q.add_argument("--provider", choices=PROVIDERS, help="LLM provider, for --template llm")
    q.add_argument("--model", help="model name, for --template llm")
    q.add_argument("--task", help="existing task whose features and labels pre-fill the template")

    sub.add_parser("templates", help="list classifier templates and LLM providers")

    p = sub.add_parser("check", help="check a task for consistency problems")
    p.add_argument("--task", required=True)
    p.add_argument("--all", action="store_true", help="list every case, not only the first ones")

    p = sub.add_parser("run", help="classify and evaluate a task")
    p.add_argument("--task", required=True)
    p.add_argument("--clf", required=True, help="classifier config (YAML)")
    p.add_argument("--split", help="dev (default if the task has splits) or test")
    p.add_argument("--runs-dir", default="runs")

    p = sub.add_parser("compare", help="compare two runs item by item")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--out", help="directory for the comparison (default: next to run A)")

    p = sub.add_parser("report", help="recompute metrics.json and report.html of a run")
    p.add_argument("run")

    p = sub.add_parser("predict", help="classify unlabelled items")
    p.add_argument("--task", required=True, help="task whose label schema is used")
    p.add_argument("--items", required=True, help="CSV with the id and feature columns")
    p.add_argument("--clf", required=True)
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (TaskError, ClassifierError, ScaffoldError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_init(args: argparse.Namespace) -> int:
    from labelbench.scaffold import init_project

    files = init_project(args.directory, llm=args.llm, model=args.model)
    _print_created(files, Path(args.directory))
    print("\nNext steps:")
    if args.directory != ".":
        print(f"  cd {args.directory}")
    print("  labelbench check --task tasks/example")
    print("  labelbench run --task tasks/example --clf configs/keywords-v1.yaml")
    if args.llm:
        print("  labelbench run --task tasks/example --clf configs/llm-v1.yaml")
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    from labelbench.scaffold import new_classifier, new_task

    if args.kind == "task":
        files = new_task(".", args.name)
        _print_created(files, Path("."))
        print("\nFill labels.csv and gold.csv, adjust task.yaml, then run:")
        print(f"  labelbench check --task tasks/{args.name}")
        return 0
    template = args.template or ("llm" if args.provider else None)
    if template is None:
        raise ScaffoldError("Choose --template rules, or --provider for an LLM classifier.")
    files = new_classifier(".", args.name, template, provider=args.provider,
                           model=args.model, task=args.task)
    _print_created(files, Path("."))
    config = next(f for f in files if f.suffix == ".yaml")
    task = args.task or "tasks/NAME"
    print("\nAdjust the files, then run:")
    print(f"  labelbench run --task {task} --clf {config.as_posix()}")
    return 0


def cmd_templates(args: argparse.Namespace) -> int:
    from labelbench.scaffold import describe_templates

    print(describe_templates())
    return 0


def _print_created(files: list[Path], root: Path) -> None:
    print("Created:")
    for f in files:
        try:
            shown = f.relative_to(root)
        except ValueError:
            shown = f
        print(f"  {shown.as_posix()}")


def cmd_check(args: argparse.Namespace) -> int:
    task = load_task(args.task, strict=False)
    print(f"{task.root}: {len(task.rows)} items, {len(task.labels)} labels")
    if not task.issues:
        print("No issues found.")
        return 0
    for issue in task.issues:
        print(f"\n[{ICONS[issue.severity]}] {issue.severity.upper()} {issue.code} ({issue.count})")
        print(f"    {issue.message}")
        shown = issue.examples if args.all else issue.examples[:5]
        for example in shown:
            print(f"    - {example}")
        if shown and len(shown) < issue.count:
            print(f"    ... {issue.count - len(shown)} more"
                  + ("" if args.all else " (--all shows up to 25)"))
    counts = {s: sum(1 for i in task.issues if i.severity == s) for s in SEVERITIES}
    print(f"\n{counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} note(s).")
    return 1 if counts["error"] else 0


def cmd_run(args: argparse.Namespace) -> int:
    from labelbench.runner import run

    run_dir = run(args.task, args.clf, split=args.split, runs_dir=args.runs_dir)
    _print_metrics(run_dir)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from labelbench.compare import compare

    out = compare(args.run_a, args.run_b, args.out)
    doc = read_json(out / "comparison.json")
    for warning in doc["warnings"]:
        print(f"warning: {warning}")
    for level, s in doc["levels"].items():
        delta = (s["accuracy_b"] - s["accuracy_a"]) * 100
        print(f"{level:<20} A {s['accuracy_a']:.1%}  B {s['accuracy_b']:.1%}  ({delta:+.1f} pp)  "
              f"fixed {s['fixed']}, broken {s['broken']}, McNemar p = {s['mcnemar_p']:.3f}")
    print(f"report: {out / 'report.html'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from labelbench.runner import evaluate_run_dir

    evaluate_run_dir(args.run)
    _print_metrics(Path(args.run))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from labelbench.runner import predict

    out = predict(args.task, args.items, args.clf, args.out)
    print(f"predictions: {out / 'predictions.csv'}")
    return 0


def _print_metrics(run_dir: Path) -> None:
    metrics = read_json(run_dir / "metrics.json")
    split = metrics.get("split") or "all"
    print(f"{metrics['task']} ({split}, {metrics['n_items']} items)")
    for level, m in metrics["levels"].items():
        lo, hi = m["accuracy_ci"]
        print(f"  {level:<20} accuracy {m['accuracy']:.1%} [{lo:.1%}, {hi:.1%}]  "
              f"macro-F1 {m['macro_f1']:.2f}  invalid {m['invalid_share']:.1%}")
    checks = read_json(run_dir / "checks.json") if (run_dir / "checks.json").is_file() else []
    warnings = sum(1 for c in checks if c["severity"] == "warning")
    if warnings:
        print(f"  {warnings} data warning(s), see 'Data checks' in the report")
    print(f"report: {run_dir / 'report.html'}")


COMMANDS = {
    "init": cmd_init,
    "new": cmd_new,
    "templates": cmd_templates,
    "check": cmd_check,
    "run": cmd_run,
    "compare": cmd_compare,
    "report": cmd_report,
    "predict": cmd_predict,
}

if __name__ == "__main__":
    sys.exit(main())
