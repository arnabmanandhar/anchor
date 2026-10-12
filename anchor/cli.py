"""
Command-line entrypoint: `anchor <command>` (or `python -m anchor.cli <command>`)

Commands:
    demo                         Run the bundled toy pipeline (two retriever
                                 variants) against the sample dataset.
    run PIPELINE DATASET -v V    Evaluate your own pipeline. PIPELINE is
                                 "module:function" — a function taking a
                                 query and returning (chunks, answer), or a
                                 zero-arg factory with --factory.
    score RECORDS -v V           Score outputs your pipeline already produced
                                 (JSON / JSONL / CSV of query, retrieved
                                 chunks, answer, reference ids).
    compare V_A V_B              Compare two pipeline versions in the DB.
    gate V [--min-recall ...]    Exit non-zero if V misses a floor or drops
                                 too far below --baseline. For CI.
    synth CORPUS -n N -o FILE    Generate a question set from a corpus, with
                                 reference chunk ids known by construction.
    providers [--check]          List supported model providers, which keys
                                 are set, and optionally test them.

Model selection (run, score, synth): --provider and --model, or the
ANCHOR_LLM_PROVIDER / ANCHOR_LLM_MODEL environment variables. Keys are read
only from the environment (e.g. GEMINI_API_KEY, GROQ_API_KEY,
OPENAI_API_KEY), never from command-line flags, which would leak into shell
history and process listings.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from . import llm
from .compare import compare_versions
from .gate import GATE_METRICS, run_gate
from .judge import FaithfulnessJudge
from .records import load_records
from .runner import evaluate_pipeline, evaluate_records, summarize
from .sandbox import parse_corpus
from .synth import DEFAULT_MIX, generate_questions
from .tracer import Tracer

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
DEMO_VERSIONS = {"demo-tfidf": "tfidf", "demo-keyword": "keyword"}


def run_demo(db_path: str = "anchor_traces.db", judge: FaithfulnessJudge = None, quiet: bool = False,
             connection: llm.Connection | None = None) -> dict:
    """Evaluate both demo pipeline variants. Re-running replaces the previous
    demo runs instead of piling up duplicates. Returns {version: summary}."""
    if not EXAMPLES_DIR.is_dir():
        raise SystemExit("`anchor demo` needs a source checkout (the examples/ folder); clone the repo to run it.")
    sys.path.insert(0, str(EXAMPLES_DIR))
    from dummy_rag_pipeline import make_pipeline  # type: ignore

    dataset = json.loads((EXAMPLES_DIR / "sample_dataset.json").read_text())
    tracer = Tracer(db_path)
    if judge is None and connection is not None:
        judge = FaithfulnessJudge(connection, cache=tracer)
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


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=list(llm.PROVIDERS), help="model provider (default: first one with a key set)")
    p.add_argument("--model", help="model id (default: the provider's suggested model)")
    p.add_argument("--base-url", help="endpoint for --provider custom (e.g. http://localhost:11434/v1)")


def _connection(args) -> llm.Connection | None:
    """Connection from flags + environment. Keys only ever come from the environment."""
    if not args.provider:
        conn = llm.connection_from_env()
        if conn and args.model:
            conn = llm.Connection(conn.provider, args.model, conn.key, conn.base_url)
        return conn
    spec = llm.PROVIDERS[args.provider]
    key = next((os.environ[k] for k in spec.env_keys if os.environ.get(k)), None) or os.environ.get("ANCHOR_LLM_API_KEY")
    if not key and args.provider != "custom":
        raise SystemExit(f"--provider {args.provider} needs {' or '.join(spec.env_keys)} set in the environment")
    if args.base_url:
        llm.check_base_url(args.base_url)
    return llm.Connection(args.provider, args.model or spec.models[0], llm.Secret(key) if key else None, args.base_url)


def _providers(check: bool) -> int:
    failed = 0
    for p in llm.PROVIDERS.values():
        key_var = next((k for k in p.env_keys if os.environ.get(k)), None)
        status = f"key set ({key_var})" if key_var else f"no key ({' / '.join(p.env_keys)})"
        print(f"{p.id:12} {p.label:28} {status}")
        print(f"{'':12} default model {p.models[0]}" + (f" · free: {p.free_tier}" if p.free_tier else ""))
        if check and key_var:
            conn = llm.Connection(p.id, p.models[0], llm.Secret(os.environ[key_var]))
            try:
                llm.ping(conn)
                print(f"{'':12} check: OK")
            except llm.LLMError as e:
                failed += 1
                print(f"{'':12} check: FAILED ({e})")
    return 1 if failed else 0


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
    _add_model_args(demo_p)

    run_p = sub.add_parser("run", help="Evaluate your own pipeline on a JSON dataset")
    run_p.add_argument("pipeline", help="module:function")
    run_p.add_argument("dataset", help="path to a JSON list of {query, reference_chunk_ids?}")
    run_p.add_argument("-v", "--version", required=True, help="pipeline_version label to store runs under")
    run_p.add_argument("-k", type=int, default=5, help="cutoff for recall@k / nDCG@k (default 5)")
    run_p.add_argument("--factory", action="store_true", help="call PIPELINE() first to build the pipeline")
    run_p.add_argument("--judge", choices=["auto", "llm", "heuristic"], default="auto")
    run_p.add_argument("--db", default="anchor_traces.db")
    run_p.add_argument("--workers", type=int, default=1, help="cases to evaluate in parallel (pipeline must be thread-safe)")
    _add_model_args(run_p)

    score_p = sub.add_parser("score", help="Score pre-computed pipeline outputs from a file")
    score_p.add_argument("records", help="JSON / JSONL / CSV file of records (see anchor/records.py)")
    score_p.add_argument("-v", "--version", required=True, help="pipeline_version label to store runs under")
    score_p.add_argument("-k", type=int, default=5, help="cutoff for recall@k / nDCG@k (default 5)")
    score_p.add_argument("--judge", choices=["auto", "llm", "heuristic"], default="auto")
    score_p.add_argument("--db", default="anchor_traces.db")
    score_p.add_argument("--workers", type=int, default=4, help="records to judge in parallel (default 4)")
    _add_model_args(score_p)

    syn_p = sub.add_parser("synth", help="Generate questions (with reference chunk ids) from a corpus")
    syn_p.add_argument("corpus", help="corpus file: CSV id,text · JSON · JSONL · or text split by blank lines")
    syn_p.add_argument("-n", type=int, default=20, help="number of questions (default 20)")
    syn_p.add_argument("-o", "--output", help="write the questions here (JSON); default prints to stdout")
    syn_p.add_argument("--generator", choices=["auto", "template", "llm"], default="auto",
                       help="auto uses a model if a provider key is set, else offline templates")
    syn_p.add_argument("--mix", default=",".join(f"{t}={w:g}" for t, w in DEFAULT_MIX.items()),
                       help="type weights, e.g. single_hop=6,multi_hop=2,unanswerable=2")
    syn_p.add_argument("--drop-easy", action="store_true",
                       help="drop questions that plain keyword overlap already retrieves perfectly")
    syn_p.add_argument("-k", type=int, default=3, help="BM25 cutoff used for the 'hard' label (default 3)")
    syn_p.add_argument("--seed", type=int, default=0)
    _add_model_args(syn_p)

    prov_p = sub.add_parser("providers", help="List model providers and which keys are set")
    prov_p.add_argument("--check", action="store_true", help="send one tiny request per configured key")

    cmp_p = sub.add_parser("compare", help="Compare two pipeline versions")
    cmp_p.add_argument("version_a")
    cmp_p.add_argument("version_b")
    cmp_p.add_argument("--db", default="anchor_traces.db")
    cmp_p.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")

    gate_p = sub.add_parser("gate", help="Fail (exit 1) if a version misses its thresholds")
    gate_p.add_argument("version")
    gate_p.add_argument("--db", default="anchor_traces.db")
    for name in GATE_METRICS:
        gate_p.add_argument(f"--min-{name}", type=float, metavar="X", help=f"floor for mean {name}")
    gate_p.add_argument("--baseline", help="version to compare against for --max-*-drop checks")
    for name in GATE_METRICS:
        gate_p.add_argument(f"--max-{name}-drop", type=float, metavar="X", help=f"allowed drop in {name} vs baseline")
    gate_p.add_argument("--require-significant", action="store_true",
                        help="fail a --max-*-drop check only if the drop is statistically significant")
    gate_p.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")

    args = parser.parse_args(argv)

    if args.command == "demo":
        run_demo(args.db, connection=_connection(args))
    elif args.command == "run":
        dataset = json.loads(Path(args.dataset).read_text())
        pipeline_fn = _load_pipeline(args.pipeline, args.factory)
        tracer = Tracer(args.db)
        judge = FaithfulnessJudge(_connection(args) if args.judge != "heuristic" else None, backend=args.judge, cache=tracer)
        results = evaluate_pipeline(pipeline_fn, dataset, pipeline_version=args.version, k=args.k, judge=judge,
                                    tracer=tracer, max_workers=args.workers)
        print(json.dumps(summarize(results), indent=2))
    elif args.command == "score":
        path = Path(args.records)
        records = load_records(path.read_bytes(), path.name)
        tracer = Tracer(args.db)
        judge = FaithfulnessJudge(_connection(args) if args.judge != "heuristic" else None, backend=args.judge, cache=tracer)
        results = evaluate_records(records, pipeline_version=args.version, k=args.k, judge=judge, tracer=tracer,
                                   max_workers=args.workers)
        print(json.dumps(summarize(results), indent=2))
    elif args.command == "synth":
        path = Path(args.corpus)
        chunks = parse_corpus(path.read_bytes(), path.name)
        try:
            mix = {t.strip(): float(w) for t, w in (part.split("=") for part in args.mix.split(",") if part.strip())}
        except ValueError:
            parser.error("--mix looks like single_hop=6,multi_hop=2,unanswerable=2")
        conn = _connection(args) if args.generator != "template" else None
        generator = args.generator if args.generator != "auto" else ("llm" if conn else "template")
        if generator == "llm" and not conn:
            parser.error("--generator llm needs a provider key in the environment (see `anchor providers`)")
        questions = generate_questions(chunks, n=args.n, mix=mix, generator=generator, connection=conn,
                                       seed=args.seed, drop_easy=args.drop_easy, k=args.k)
        text = json.dumps(questions, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        types = Counter(q["type"] for q in questions)
        levels = Counter(q["difficulty"] for q in questions if q["difficulty"] != "n/a")
        print(f"{len(questions)} questions ({generator}): " + ", ".join(f"{v} {k}" for k, v in types.items())
              + (" · difficulty " + ", ".join(f"{v} {k}" for k, v in levels.items()) if levels else ""),
              file=sys.stderr)
    elif args.command == "providers":
        return _providers(args.check)
    elif args.command == "compare":
        print(json.dumps(compare_versions(args.db, args.version_a, args.version_b, alpha=args.alpha), indent=2))
    elif args.command == "gate":
        floors = {n: getattr(args, f"min_{n}") for n in GATE_METRICS if getattr(args, f"min_{n}") is not None}
        drops = {n: getattr(args, f"max_{n}_drop") for n in GATE_METRICS if getattr(args, f"max_{n}_drop") is not None}
        if drops and not args.baseline:
            parser.error("--max-*-drop checks need --baseline")
        if not floors and not drops:
            parser.error("give at least one --min-* or --max-*-drop threshold")
        result = run_gate(args.db, args.version, floors=floors, baseline=args.baseline, max_drop=drops,
                          require_significant=args.require_significant, alpha=args.alpha)
        print(result.report())
        return 0 if result.passed else 1
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
