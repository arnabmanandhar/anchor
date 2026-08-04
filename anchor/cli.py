"""
Command-line entrypoint: `python -m anchor.cli <command>`

Commands:
    demo                  Run the bundled toy pipeline against the sample
                           dataset and print a summary.
    compare V_A V_B DB    Compare two pipeline_version labels stored in DB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare import compare_versions
from .runner import evaluate_pipeline, summarize


def _run_demo():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from dummy_rag_pipeline import make_pipeline  # type: ignore

    dataset_path = Path(__file__).resolve().parent.parent / "examples" / "sample_dataset.json"
    dataset = json.loads(dataset_path.read_text())

    pipeline_fn = make_pipeline()
    results = evaluate_pipeline(pipeline_fn, dataset, pipeline_version="demo-v1", db_path="anchor_traces.db")
    report = summarize(results)
    print(json.dumps(report, indent=2))
    print(f"\n{len(results)} runs written to anchor_traces.db")
    print("Run `streamlit run dashboard/app.py` to view them.")


def main():
    parser = argparse.ArgumentParser(prog="anchor")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Run the bundled demo pipeline + sample dataset")

    cmp_parser = sub.add_parser("compare", help="Compare two pipeline versions")
    cmp_parser.add_argument("version_a")
    cmp_parser.add_argument("version_b")
    cmp_parser.add_argument("--db", default="anchor_traces.db")

    args = parser.parse_args()

    if args.command == "demo":
        _run_demo()
    elif args.command == "compare":
        result = compare_versions(args.db, args.version_a, args.version_b)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
