"""
Command-line entrypoint: `anchor <command>` (or `python -m anchor.cli <command>`)

Commands:
    demo                         Run the bundled toy pipeline (two retriever
                                 variants) against the sample dataset.
    run PIPELINE DATASET -v V    Evaluate your own pipeline. PIPELINE is
                                 "module:function" — a function taking a
                                 query and returning (chunks, answer), or a
                                 zero-arg factory with --factory.
    compare V_A V_B              Compare two pipeline versions in the DB.
    gate V [--min-recall ...]    Exit non-zero if V misses a floor or drops
                                 too far below --baseline. For CI.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .compare import compare_versions
from .gate import GATE_METRICS, run_gate
from .judge import FaithfulnessJudge
from .runner import evaluate_pipeline, summarize
from .tracer import Tracer

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
DEMO_VERSIONS = {"demo-tfidf": "tfidf", "demo-keyword": "keyword"}


def run_demo(db_path: str = "anchor_traces.db", judge: FaithfulnessJudge = None, quiet: bool = False) -> dict:
    """Evaluate both demo pipeline variants. Re-running replaces the previous
    demo runs instead of piling up duplicates. Returns {version: summary}."""
    if not EXAMPLES_DIR.is_dir():
        raise SystemExit("`anchor demo` needs a source checkout (the examples/ folder); clone the repo to run it.")
    sys.path.insert(0, str(EXAMPLES_DIR))
    from dummy_rag_pipeline import make_pipeline  # type: ignore

    dataset = json.loads((EXAMPLES_DIR / "sample_dataset.json").read_text())
    tracer = Tracer(db_path)
    reports = {}
    try:
        for version, retriever in DEMO_VERSIONS.items():
            tracer.delete_version(version)
            results = evaluate_pipeline(
                make_pipeline(retriever=retriever), dataset, pipeline_version=version, k=3, tracer=tracer, judge=judge
            )
            reports[version] = summarize(results)
    finally:
        tracer.close()

    if not quiet:
        print(json.dumps(reports, indent=2))
        print(f"\nRuns written to {db_path}. Next:")
        print(f"  anchor compare demo-tfidf demo-keyword --db {db_path}")
        print("  streamlit run dashboard/app.py")
    return reports


def _load_pipeline(spec: str, factory: bool):
    if ":" not in spec:
        raise SystemExit(f"pipeline must look like 'module:function', got {spec!r}")
    module_name, attr = spec.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    fn = getattr(importlib.import_module(module_name), attr)
    return fn() if factory else fn


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="anchor", description="Deterministic-first RAG evaluation.")
    sub = parser.add_subparsers(dest="command")

    demo_p = sub.add_parser("demo", help="Run the bundled demo pipelines + sample dataset")
    demo_p.add_argument("--db", default="anchor_traces.db")

    run_p = sub.add_parser("run", help="Evaluate your own pipeline on a JSON dataset")
    run_p.add_argument("pipeline", help="module:function")
    run_p.add_argument("dataset", help="path to a JSON list of {query, reference_chunk_ids?}")
    run_p.add_argument("-v", "--version", required=True, help="pipeline_version label to store runs under")
    run_p.add_argument("-k", type=int, default=5, help="cutoff for recall@k / nDCG@k (default 5)")
    run_p.add_argument("--factory", action="store_true", help="call PIPELINE() first to build the pipeline")
    run_p.add_argument("--judge", choices=["auto", "llm", "heuristic"], default="auto")
    run_p.add_argument("--db", default="anchor_traces.db")

    cmp_p = sub.add_parser("compare", help="Compare two pipeline versions")
    cmp_p.add_argument("version_a")
    cmp_p.add_argument("version_b")
    cmp_p.add_argument("--db", default="anchor_traces.db")

    gate_p = sub.add_parser("gate", help="Fail (exit 1) if a version misses its thresholds")
    gate_p.add_argument("version")
    gate_p.add_argument("--db", default="anchor_traces.db")
    for name in GATE_METRICS:
        gate_p.add_argument(f"--min-{name}", type=float, metavar="X", help=f"floor for mean {name}")
    gate_p.add_argument("--baseline", help="version to compare against for --max-*-drop checks")
    for name in GATE_METRICS:
        gate_p.add_argument(f"--max-{name}-drop", type=float, metavar="X", help=f"allowed drop in {name} vs baseline")

    args = parser.parse_args(argv)

    if args.command == "demo":
        run_demo(args.db)
    elif args.command == "run":
        dataset = json.loads(Path(args.dataset).read_text())
        pipeline_fn = _load_pipeline(args.pipeline, args.factory)
        judge = FaithfulnessJudge(backend=args.judge)
        results = evaluate_pipeline(pipeline_fn, dataset, pipeline_version=args.version, k=args.k, judge=judge, db_path=args.db)
        print(json.dumps(summarize(results), indent=2))
    elif args.command == "compare":
        print(json.dumps(compare_versions(args.db, args.version_a, args.version_b), indent=2))
    elif args.command == "gate":
        floors = {n: getattr(args, f"min_{n}") for n in GATE_METRICS if getattr(args, f"min_{n}") is not None}
        drops = {n: getattr(args, f"max_{n}_drop") for n in GATE_METRICS if getattr(args, f"max_{n}_drop") is not None}
        if drops and not args.baseline:
            parser.error("--max-*-drop checks need --baseline")
        if not floors and not drops:
            parser.error("give at least one --min-* or --max-*-drop threshold")
        result = run_gate(args.db, args.version, floors=floors, baseline=args.baseline, max_drop=drops)
        print(result.report())
        return 0 if result.passed else 1
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
