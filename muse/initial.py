"""V8 initial retrieval from complete first-user text.
Training keys use accepted cold first-user content; test queries use the current
consumer's complete pre-history user content. Explicit manifest sources are
hash-bound, train/test are disjoint, and both sides receive no embedding prefix.
RAG and SM share source ranking while their original value callbacks stay intact.

Source manifest example (one row per task in train plus test):
  {"train": ["train_task"], "test": ["test_task"], "instructions": {
     "train_task": {"source_kind": "first_user_prompt_before_memory",
       "selector": "training_first_user_prompt", "format": "messages_json",
       "path": "/absolute/cold/messages.json", "sha256": "<file hash>",
       "text_sha256": "<complete first-user text hash>"},
     "test_task": {"source_kind": "first_user_prompt_before_memory",
       "selector": "pre_initial_memory_prompt", "format": "inline",
       "text": "<actual current consumer user content before adding history>",
       "text_sha256": "<complete text hash>"}},
   "embedding": {"model": "qwen3-embedding-8b",
                 "url": "http://127.0.0.1:18081/v1"}}
See input_manifest_kbx.json / input_manifest_pde.json for hash-bound real sources.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def first_user_text(document):
    messages = document.get("messages") if isinstance(document, dict) else document
    if not isinstance(messages, list):
        raise ValueError("Expected a messages array")
    for index, message in enumerate(messages):
        role = message.get("role", message.get("info", {}).get("role"))
        if role != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content, index
        parts = message.get("parts", content)
        if not isinstance(parts, list):
            raise ValueError("First user has no exact text representation")
        texts = [p.get("text") for p in parts if p.get("type") == "text"]
        if len(texts) != 1 or not isinstance(texts[0], str):
            raise ValueError(
                "First user must have exactly one text part; do not invent joining separators"
            )
        return texts[0], index
    raise ValueError("No user message")


def _extract_file(row, cache=None):
    path = Path(row.get("path", ""))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Explicit source file missing: " + str(path))
    cache = {} if cache is None else cache
    cache_key = str(path.resolve())
    if cache_key not in cache:
        raw = path.read_bytes()
        cache[cache_key] = (digest(raw), json.loads(raw))
    actual, document = cache[cache_key]
    if actual != row.get("sha256"):
        raise ValueError("Source file hash mismatch: " + str(path))
    fmt = row.get("format")
    meta = {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": actual,
        "format": fmt,
    }
    if fmt == "messages_json":
        text, index = first_user_text(document)
        meta["message_index"] = index
    elif fmt == "input_json":
        field = row.get("field")
        if field not in ("prompt", "content"):
            raise ValueError("Input source requires explicit prompt/content field")
        text = document[field]
        meta["field"] = field
    elif fmt == "prompt_map_json":
        key = row.get("task_key")
        text = document[key]
        meta["task_key"] = key
    else:
        raise ValueError("Unsupported first-user source format")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Source did not contain nonempty exact user text")
    return text, meta


def read_sources(manifest_path):
    """Read complete first-user content without stripping, adding or rewriting text."""
    manifest_path = Path(manifest_path).resolve()
    raw_manifest = manifest_path.read_bytes()
    spec = json.loads(raw_manifest)
    train, test = spec.get("train"), spec.get("test")
    if (
        not isinstance(train, list)
        or not isinstance(test, list)
        or not train
        or not test
    ):
        raise ValueError("Explicit nonempty train and test task lists are required")
    if any(not isinstance(t, str) or not t for t in train + test):
        raise ValueError("Task IDs must be nonempty strings")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("Duplicate task IDs in a split")
    if set(train) & set(test):
        raise ValueError("Training/test overlap")
    instructions = spec.get("instructions", {})
    if set(instructions) != set(train + test):
        raise ValueError("Source manifest must cover exactly train plus test")
    texts, sources, cache = {}, {}, {}
    for task in sorted(train + test):
        row = instructions[task]
        if (
            not isinstance(row, dict)
            or row.get("source_kind") != "first_user_prompt_before_memory"
        ):
            raise ValueError(
                task
                + ": exact first-user source required; old instruction-file banks are not accepted"
            )
        purpose = (
            "training_first_user_prompt"
            if task in train
            else "pre_initial_memory_prompt"
        )
        if row.get("selector") != purpose:
            raise ValueError(
                task + ": incorrect source selector for train/test purpose"
            )
        if row.get("format") == "inline":
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(task + ": nonempty exact inline text required")
            meta = {"format": "inline", "sha256": digest(text.encode("utf-8"))}
        else:
            text, meta = _extract_file(row, cache)
        text_hash = digest(text.encode("utf-8"))
        if text_hash != row.get("text_sha256"):
            raise ValueError(task + ": first-user text hash mismatch")
        checks = []
        for verification in row.get("verify_against", []):
            other, evidence = _extract_file(verification, cache)
            if text != other:
                raise ValueError(task + ": first-user source mismatch")
            checks.append(evidence)
        texts[task] = text
        sources[task] = {
            **meta,
            "text_sha256": text_hash,
            "chars": len(text),
            "source_kind": "first_user_prompt_before_memory",
            "selector": purpose,
            "verified_against": checks,
        }
    return (
        spec,
        texts,
        sources,
        {"path": str(manifest_path), "sha256": digest(raw_manifest)},
    )


def _normalized(matrix, size):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != size or matrix.shape[1] == 0:
        raise ValueError("Invalid embedding dimensions")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(norms).all()
        or not (norms > 0).all()
    ):
        raise ValueError("Invalid embedding values")
    return matrix / norms


def build_initial(
    manifest_path, destination, embedder, renderers, top_k=5, renderer_sources=()
):
    """Build isolated cards. renderers[arm](test_task, ordered_train_tasks) returns
    the ORIGINAL rendered value string (or a metadata dict with a text string).
    The callbacks own value formatting; this function never edits their text.
    Embedder.embed receives exact instruction strings and returns (vectors, usage).
    """
    spec, texts, sources, manifest_source = read_sources(manifest_path)
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= len(spec["train"])
    ):
        raise ValueError("Invalid initial top_k")
    if not renderers or not all(
        isinstance(k, str) and callable(v) for k, v in renderers.items()
    ):
        raise ValueError("Explicit original-value renderer callbacks required")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(
            "V8 initial output must be new; existing outputs are immutable"
        )
    renderer_hashes = {}
    for path in renderer_sources:
        path = Path(path).resolve()
        renderer_hashes[str(path)] = digest(path.read_bytes())
    train, test = sorted(spec["train"]), sorted(spec["test"])
    # No instruction/prefix/template is added to either side.
    key_vectors, key_usage = embedder.embed([texts[t] for t in train])
    query_vectors, query_usage = embedder.embed([texts[t] for t in test])
    key_vectors = _normalized(key_vectors, len(train))
    query_vectors = _normalized(query_vectors, len(test))
    if key_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Key/query embedding dimensions differ")
    scores = query_vectors @ key_vectors.T
    selection, initial = {}, {}
    for qi, task in enumerate(test):
        order = sorted(
            range(len(train)), key=lambda i: (-float(scores[qi, i]), train[i])
        )
        ranking = [{"task_id": train[i], "score": float(scores[qi, i])} for i in order]
        selected = [row["task_id"] for row in ranking[:top_k]]
        selection[task] = {
            "source_tasks": selected,
            "ranking": ranking,
            "query_sha256": sources[task]["text_sha256"],
        }
        initial[task] = {}
        for arm, render in renderers.items():
            value = render(task, list(selected))
            card = {"text": value} if isinstance(value, str) else dict(value)
            if not isinstance(card.get("text"), str):
                raise ValueError(f"{arm}/{task}: renderer did not return value text")
            if "source_tasks" in card and card["source_tasks"] != selected:
                raise ValueError(
                    f"{arm}/{task}: renderer used a different source ordering"
                )
            card.update(
                source_tasks=selected,
                sha256=digest(card["text"].encode("utf-8")),
                selection_policy="Shared complete first-user semantic ranking; no prefix",
            )
            initial[task][arm] = card
    # Reject concurrent changes; never publish a mixed-source bank.
    if digest(Path(manifest_source["path"]).read_bytes()) != manifest_source["sha256"]:
        raise ValueError("Instruction manifest changed while building")
    checked_source_paths = {}
    for task, row in sources.items():
        for source in [row, *row.get("verified_against", [])]:
            if source.get("path"):
                path = source["path"]
                if path not in checked_source_paths:
                    checked_source_paths[path] = digest(Path(path).read_bytes())
                if checked_source_paths[path] != source["sha256"]:
                    raise ValueError(f"{task}: source changed while building")
    for path, expected in renderer_hashes.items():
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError("Value renderer source changed while building")
    destination.mkdir(parents=True, exist_ok=False)

    def put(name, value):
        (destination / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    put("instructions.json", texts)
    put("split.json", {"train": train, "test": test})
    put("SOURCES.json", sources)
    put("selection.json", selection)
    put("initial.json", initial)
    np.save(destination / "train_vectors.npy", key_vectors)
    np.save(destination / "test_vectors.npy", query_vectors)
    files = [
        "instructions.json",
        "split.json",
        "SOURCES.json",
        "selection.json",
        "initial.json",
        "train_vectors.npy",
        "test_vectors.npy",
    ]
    result = {
        "version": "V8-initial-first-user-prompt-v1",
        "benchmark": spec.get("benchmark"),
        "train_n": len(train),
        "test_n": len(test),
        "top_k": top_k,
        "embedding": spec.get("embedding", {}),
        "instruction_prefix_used": False,
        "retrieval_policy": "first_user_prompt_v8",
        "source_manifest": manifest_source,
        "renderer_sources": renderer_hashes,
        "value_policy": "Original value renderer output, unchanged",
        "retrieval_uses_values_or_outcomes": False,
        "similarity": "Normalized float32 cosine; descending; task_id tie break",
        "usage": {"keys": key_usage, "queries": query_usage},
        "files": {name: digest((destination / name).read_bytes()) for name in files},
    }
    put("MANIFEST.json", result)
    return result
