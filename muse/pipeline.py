"""Portable construction of frozen initial and runtime retrieval artifacts."""

from pathlib import Path

from .bank import build, initial_card
from .construction import bank_text, empty
from .initial import build_initial
from .io import digest, put
from .multichannel import build_route_bank
from .render_v5 import initial as render_initial


def build_artifacts(cases, targets, destination, embedder, global_bank=None, k=5):
    """Build from extracted historical memories and pre-memory target prompts.

    ``cases`` maps source IDs to memories returned by ``extract_evidence``.
    ``targets`` maps disjoint target IDs to their complete first user prompts.
    No target trajectories, submissions, or evaluation outcomes are inputs.
    Output paths must be new so existing frozen banks cannot be overwritten.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a new artifact directory")
    if not cases or not targets or set(cases) & set(targets):
        raise ValueError("Nonempty, disjoint extraction and target IDs are required")
    if not 1 <= k <= len(cases):
        raise ValueError("k must not exceed the number of historical sources")
    for value in targets.values():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Each target must have a nonempty first user prompt")
    destination.mkdir(parents=True)
    source = destination / "source"
    source.mkdir()
    put(source / "cases.json", cases)
    put(source / "index.json", {
        "model": embedder.model, "embedding_url": embedder.url,
    })
    split = {"train": sorted(cases), "test": sorted(targets)}
    put(source / "split.json", split)
    texts = {ref: memory["scientific_problem"] for ref, memory in cases.items()}
    texts.update(targets)
    spec = {
        **split,
        "embedding": {"model": embedder.model, "url": embedder.url},
        "instructions": {
            ref: {
                "source_kind": "first_user_prompt_before_memory",
                "selector": "training_first_user_prompt" if ref in cases
                else "pre_initial_memory_prompt",
                "format": "inline", "text": text, "text_sha256": digest(text),
            }
            for ref, text in texts.items()
        },
    }
    put(source / "initial-input.json", spec)
    global_bank = global_bank if global_bank is not None else empty()
    put(source / "global.json", global_bank)
    global_text = bank_text(global_bank)

    def render(_task, refs):
        return global_text + "\n\n" + "\n\n".join(render_initial(cases[ref]) for ref in refs)

    initial_root = destination / "initial"
    build_initial(source / "initial-input.json", initial_root, embedder,
                  {"muse": render}, top_k=k)
    bank_root = destination / "bank"
    build(source, bank_root, embedder=embedder, initial_artifacts=initial_root)
    build_route_bank(bank_root, destination / "routes", embedder=embedder)
    return {"bank": str(bank_root), "routes": str(destination / "routes"),
            "initial": str(initial_root), "k": k, "sources": len(cases),
            "targets": len(targets)}


def get_initial(bank, task_id):
    """Return full global memory plus the selected approach-level references."""
    return initial_card(bank, task_id, "muse")
