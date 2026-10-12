"""Command-line entry points for the portable MUSE memory components."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from .io import put, read


def make_llm(gate_log=None):
    from .llm import DirectLLM
    key = os.environ.get("MUSE_API_KEY")
    url = os.environ.get("MUSE_BASE_URL")
    if not key or not url:
        raise ValueError("Set MUSE_API_KEY and MUSE_BASE_URL")
    llm = DirectLLM(key, url, model=os.environ.get("MUSE_MODEL", "glm-5.3-flash"),
                    effort=os.environ.get("MUSE_REASONING_EFFORT", "high"), timeout=300)
    if gate_log is not None:
        from .pure_llm import PureLLMGate
        return PureLLMGate(llm, gate_log)
    return llm


def make_embedder():
    from .retrieval import RawEmbedder
    return RawEmbedder(os.environ.get("MUSE_EMBEDDING_BASE_URL", "http://127.0.0.1:18081/v1"),
                       os.environ.get("MUSE_EMBEDDING_MODEL", "qwen3-embedding-8b"))


def current_context(interaction):
    from .context import RecentContext
    return RecentContext().observe(interaction["assistant"], interaction["hook"],
                                   interaction.get("turn_id", "current"))


def run(args):
    from .construction import consolidate_one, empty, extract_evidence, review_bank
    from .pipeline import build_artifacts, get_initial

    if args.command == "extract":
        evidence = read(args.input)
        if not isinstance(evidence, list):
            evidence = [evidence]
        refs = [e["source_ref"] for e in evidence]
        if len(refs) != len(set(refs)):
            raise ValueError("Duplicate historical source IDs")
        llm = make_llm()
        put(args.output, {e["source_ref"]: extract_evidence(llm, e) for e in evidence})
    elif args.command in {"consolidate", "review"}:
        memories = read(args.memories)
        evidence = read(args.evidence)
        if isinstance(evidence, list):
            evidence = {e["source_ref"]: e for e in evidence}
        elif isinstance(evidence, dict) and "source_ref" in evidence:
            evidence = {evidence["source_ref"]: evidence}
        if set(memories) != set(evidence):
            raise ValueError("Evidence must cover exactly the extracted sources")
        source_order = list(memories)
        llm = make_llm()
        if args.command == "consolidate":
            bank = empty()
            for ref in source_order:
                bank, _audit = consolidate_one(llm, bank, ref, memories, evidence, source_order)
        else:
            bank, _audit = review_bank(llm, read(args.bank), memories, evidence, source_order)
        put(args.output, bank)
    elif args.command == "build-bank":
        summary = build_artifacts(read(args.memories), read(args.targets), args.output,
                                  make_embedder(), read(args.global_bank) if args.global_bank else None,
                                  k=args.k)
        print(json.dumps(summary, indent=2))
    elif args.command == "initial":
        card = get_initial(args.bank, args.task)
        Path(args.output).write_text(card["text"], encoding="utf-8")
    elif args.command == "retrieve":
        from .multichannel import Retriever, candidate_window, queries
        from .io import read as load
        interaction = load(args.interaction)
        ranker = Retriever(args.bank, args.routes, embedder=make_embedder())
        ranking = ranker.rank(queries(current_context(interaction)), load(args.initial_sources))
        rows, candidates = candidate_window(ranking["ranking"], ranker.cases,
                                            set(load(args.seen_sources)) if args.seen_sources else set())
        put(args.output, {**ranking, "rows": rows, "candidates": candidates})
    elif args.command == "gate":
        from .gate import render_messages
        from .review_runtime import complete_review
        candidates = read(args.candidates)["candidates"]
        if not candidates:
            put(args.output, {"inject_ids": [], "review_note": ""})
            return
        history = read(args.history) if args.history else []
        messages = render_messages(args.task_tldr, history + current_context(read(args.interaction)), candidates)
        decision, _audit = complete_review(make_llm(args.log_dir), messages, candidates, Path(args.log_dir))
        put(args.output, decision)
    elif args.command == "runtime":
        from . import MUSERuntime
        card = get_initial(args.bank, args.task)
        llm = make_llm(args.output) if not args.no_gate else None
        runtime = MUSERuntime(args.bank, args.output, {"tldr": args.task_tldr}, llm,
                              {"gate_enabled": not args.no_gate},
                              initial_source_refs=card["source_tasks"], route_bank=args.routes)
        interactions = read(args.interactions)
        if not isinstance(interactions, list):
            interactions = [interactions]
        history = []
        deliveries = []

        async def guide(text):
            deliveries.append(text)
            return {"delivered": True}

        for index, interaction in enumerate(interactions):
            runtime.set_public_history(history)
            asyncio.run(runtime.on_checkpoint(interaction["assistant"], interaction["hook"],
                                               interaction.get("turn_id", str(index)), guide))
            history.extend(current_context(interaction))
        put(Path(args.output) / "delivered.json", deliveries)
        put(Path(args.output) / "metrics.json", runtime.metrics())


def parser():
    p = argparse.ArgumentParser(description="MUSE memory components for scientific coding agents")
    subs = p.add_subparsers(dest="command", required=True)
    extract = subs.add_parser("extract", help="Extract memories from supplied historical evidence")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    for command in ("consolidate", "review"):
        sub = subs.add_parser(command, help="Build or review the shared global memory")
        sub.add_argument("--memories", required=True)
        sub.add_argument("--evidence", required=True)
        sub.add_argument("--output", required=True)
        if command == "review":
            sub.add_argument("--bank", required=True)
    build = subs.add_parser("build-bank", help="Build frozen initial and four-route retrieval artifacts")
    build.add_argument("--memories", required=True)
    build.add_argument("--targets", required=True)
    build.add_argument("--global-bank")
    build.add_argument("--output", required=True)
    build.add_argument("-k", type=int, default=5)
    initial = subs.add_parser("initial", help="Render global and approach-level memory for a target")
    initial.add_argument("--bank", required=True)
    initial.add_argument("--task", required=True)
    initial.add_argument("--output", required=True)
    retrieve = subs.add_parser("retrieve", help="Retrieve four Top1 matches and merge eligible sources")
    for flag in ("bank", "routes", "interaction", "initial-sources", "output"):
        retrieve.add_argument("--" + flag, required=True)
    retrieve.add_argument("--seen-sources")
    gate = subs.add_parser("gate", help="Select one historical source with specific review advice, or skip")
    for flag in ("candidates", "task-tldr", "interaction", "log-dir", "output"):
        gate.add_argument("--" + flag, required=True)
    gate.add_argument("--history")
    runtime = subs.add_parser("runtime", help="Process supplied interactions and save injection messages")
    for flag in ("bank", "routes", "task", "task-tldr", "interactions", "output"):
        runtime.add_argument("--" + flag, required=True)
    runtime.add_argument("--no-gate", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, FileExistsError) as exc:
        p.exit(2, "muse: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
