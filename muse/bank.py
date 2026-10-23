"""Isolated V8 decision/choice index; historical values stay unchanged.

Step-only banks support offline diagnostics. A task worker must request
``verify_v8_bank(..., require_initial=True)`` before launching a task.
"""

import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from .io import read, put, sha, digest, verify_manifest, PACKAGE_ROOT
from .key import key_text, VERSION
from .initial import INITIAL_VERSION, INITIAL_POLICY, problem_text

INITIAL_SOURCE_KIND = "first_user_prompt_before_memory"
INITIAL_FILES = {
    "instructions.json",
    "split.json",
    "SOURCES.json",
    "selection.json",
    "initial.json",
    "train_vectors.npy",
    "test_vectors.npy",
}
STEP_FILES = {"index.json", "cases.json", "key_vectors.npy"}


def _manifest(root):
    root = Path(root)
    manifest = read(root / "MANIFEST.json")
    for rel in manifest.get("files", {}):
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Manifest contains a path outside its artifact")
    return verify_manifest(root)


def _initial_manifest(root, split, index, verify_source_files=False):
    root = Path(root)
    m = _manifest(root)
    if m.get("version") != INITIAL_VERSION:
        raise ValueError(
            "Initial artifacts must be rebuilt with the current problem retrieval policy"
        )
    if m.get("retrieval_policy") != INITIAL_POLICY:
        raise ValueError("Initial retrieval policy mismatch")
    if (
        m.get("instruction_prefix_used") is not False
        or m.get("retrieval_uses_values_or_outcomes") is not False
    ):
        raise ValueError(
            "Initial retrieval must use problem text without an embedding prefix"
        )
    if not INITIAL_FILES <= set(m["files"]):
        raise ValueError("Incomplete V8 initial artifact")
    source_split = read(root / "split.json")
    for side in ("train", "test"):
        if set(source_split[side]) != set(split[side]):
            raise ValueError("V8 initial/step bank split mismatch: " + side)
    if set(source_split["train"]) & set(source_split["test"]):
        raise ValueError("V8 initial training/test overlap")
    sources = read(root / "SOURCES.json")
    texts = read(
        root / "instructions.json"
    )  # Compatibility filename; contents are full first user prompts.
    tasks = set(source_split["train"]) | set(source_split["test"])
    if set(sources) != tasks or set(texts) != tasks:
        raise ValueError("V8 first-user-prompt sources do not match the frozen split")
    source_documents = {}

    def check_source(row, expected, task):
        fmt = row.get("format")
        if fmt not in {"messages_json", "input_json", "prompt_map_json", "inline"}:
            raise ValueError("Unsupported V8 first-user source format: " + task)
        source_hash = row.get("sha256")
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(c not in "0123456789abcdef" for c in source_hash)
        ):
            raise ValueError("Missing V8 first-user source file/text hash: " + task)
        if fmt == "inline":
            if source_hash != digest(expected):
                raise ValueError("V8 inline first-user hash mismatch: " + task)
            return
        path = Path(row.get("path", ""))
        if not path.is_absolute():
            raise ValueError("V8 first-user source path must be absolute: " + task)
        if fmt == "prompt_map_json" and row.get("task_key") != task:
            raise ValueError("V8 first-user prompt-map task selector mismatch: " + task)
        if fmt == "input_json" and row.get("field") not in {"prompt", "content"}:
            raise ValueError(
                "V8 input source requires an explicit prompt/content selector: " + task
            )
        if not verify_source_files:
            return
        cache_key = (str(path), source_hash)
        if cache_key not in source_documents:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != source_hash:
                raise ValueError("V8 first-user source file hash mismatch: " + task)
            source_documents[cache_key] = json.loads(raw)
        document = source_documents[cache_key]
        if fmt == "messages_json":
            from .initial import first_user_text

            actual, _ = first_user_text(document)
        elif fmt == "input_json":
            actual = document[row["field"]]
        else:
            actual = document[row["task_key"]]
        if actual != expected:
            raise ValueError(
                "V8 first-user source content differs from the frozen text: " + task
            )

    for task in tasks:
        row, text = sources[task], texts[task]
        purpose = (
            "training_first_user_prompt"
            if task in source_split["train"]
            else "pre_initial_memory_prompt"
        )
        if (
            row.get("source_kind") != INITIAL_SOURCE_KIND
            or row.get("selector") != purpose
        ):
            raise ValueError(
                "V8 initial source is not the required first user prompt: " + task
            )
        if (
            not isinstance(text, str)
            or not text.strip()
            or row.get("text_sha256") != digest(text)
        ):
            raise ValueError("V8 first-user-prompt text/hash mismatch: " + task)
        check_source(row, text, task)
        for evidence in row.get("verified_against", []):
            check_source(evidence, text, task)
    embedding = m.get("embedding", {})
    if (
        embedding.get("model") != index["model"]
        or embedding.get("url") != index["embedding_url"]
    ):
        raise ValueError("V8 initial/step embedding model or endpoint mismatch")
    cards = read(root / "initial.json")
    selection = read(root / "selection.json")
    if set(cards) != set(source_split["test"]) or set(selection) != set(cards):
        raise ValueError(
            "V8 initial cards/ranking do not cover the full frozen test split"
        )
    for task, arms in cards.items():
        if selection[task].get("query_sha256") != digest(problem_text(texts[task])):
            raise ValueError("Initial retrieval query hash mismatch: " + task)
        selected = selection[task]["source_tasks"]
        if (
            len(selected) != m["top_k"]
            or len(set(selected)) != len(selected)
            or not set(selected) <= set(source_split["train"])
        ):
            raise ValueError("Invalid V8 initial source selection: " + task)
        if not arms:
            raise ValueError("V8 initial task has no rendered arm: " + task)
        for card in arms.values():
            if card.get("source_tasks") != selected or not isinstance(
                card.get("text"), str
            ):
                raise ValueError("V8 initial value/source ordering mismatch: " + task)
    return m


def verify_v8_bank(bank, require_initial=False):
    """Reject an old bank even if its ordinary file-hash manifest is valid."""
    root = Path(bank)
    m = _manifest(root)
    if m.get("version") != "V8" or m.get("artifact_type") != "step_bank":
        raise ValueError("V8 requires an independently rebuilt V8 step bank")
    if m.get("key_template_version") != VERSION or m.get("key_code_sha256") != sha(
        PACKAGE_ROOT / "key.py"
    ):
        raise ValueError("V8 bank was not built with the current decision/choice key")
    if not STEP_FILES <= set(m["files"]):
        raise ValueError("Incomplete V8 step bank")
    index = read(root / "index.json")
    if (
        index.get("version") != "V8"
        or index.get("key_template_version") != VERSION
        or index.get("instruction_prefix") != ""
    ):
        raise ValueError("V8 index uses an old key or embedding instruction")
    cases = read(root / "cases.json")
    expected = {
        (ref, i) for ref, memory in cases.items() for i in range(len(memory["L2"]))
    }
    actual = set()
    for row in index["keys"]:
        pair = (row["source_ref"], row["decision_index"])
        if pair not in expected or pair in actual:
            raise ValueError(
                "V8 key has unknown, missing or duplicate decision identity"
            )
        ref, i = pair
        if row["unit_id"] != f"{ref}-D{i+1:03d}" or row["text"] != key_text(
            cases[ref], cases[ref]["L2"][i]
        ):
            raise ValueError(
                "V8 key text/identity differs from its unchanged historical value"
            )
        actual.add(pair)
    if (
        actual != expected
        or m["decision_count"] != len(expected)
        or m["source_count"] != len(cases)
        or m["source_refs"] != sorted(cases)
    ):
        raise ValueError("V8 bank source/decision coverage mismatch")
    binding = m.get("initial_artifacts")
    if binding:
        if len(read(root / "split.json")["train"]) != len(cases):
            raise ValueError(
                "V8 initial binding requires every frozen training source case"
            )
        initial_root = root / "initial-v8"
        if sha(initial_root / "MANIFEST.json") != binding["manifest_sha256"]:
            raise ValueError("Bound V8 initial manifest changed")
        _initial_manifest(initial_root, read(root / "split.json"), index)
        for alias, original in (
            ("memory/initial.json", "initial.json"),
            ("prompt-rag-v1/selection.json", "selection.json"),
        ):
            if alias not in m["files"] or sha(root / alias) != sha(
                initial_root / original
            ):
                raise ValueError("V8 initial alias is not the bound rebuilt artifact")
        if not m.get("experiment_ready") or m["scope"] != "full_training_bank":
            raise ValueError("Inconsistent V8 experiment readiness")
    elif m.get("experiment_ready") or require_initial:
        raise ValueError(
            "V8 step-only bank cannot launch a task; bind rebuilt V8 initial artifacts first"
        )
    return m


def _bind_initial(dest, manifest, initial_artifacts):
    if manifest["scope"] != "full_training_bank" or not (dest / "split.json").is_file():
        raise ValueError("Initial artifacts require a full bank with its frozen split")
    source = Path(initial_artifacts).resolve()
    split = read(dest / "split.json")
    if len(split["train"]) != manifest["source_count"]:
        raise ValueError(
            "V8 initial binding requires every frozen training source case"
        )
    im = _initial_manifest(
        source, split, read(dest / "index.json"), verify_source_files=True
    )
    before = sha(source / "MANIFEST.json")
    for rel in [*im["files"], "MANIFEST.json"]:
        dst = dest / "initial-v8" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dst)
        manifest["files"][str(dst.relative_to(dest))] = sha(dst)
    for alias, original in (
        ("memory/initial.json", "initial.json"),
        ("prompt-rag-v1/selection.json", "selection.json"),
    ):
        dst = dest / alias
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest / "initial-v8" / original, dst)
        manifest["files"][alias] = sha(dst)
    _initial_manifest(
        dest / "initial-v8", read(dest / "split.json"), read(dest / "index.json")
    )
    if sha(source / "MANIFEST.json") != before:
        raise ValueError("V8 initial manifest changed during binding")
    manifest.update(
        initial_artifacts={"source": str(source), "manifest_sha256": before},
        initial_policy=INITIAL_POLICY,
        experiment_ready=True,
    )


def build(source_bank, destination, refs=None, embedder=None, initial_artifacts=None):
    source, dest = Path(source_bank).resolve(), Path(destination).resolve()
    if dest == source or source in dest.parents or dest.exists():
        raise ValueError(
            "V8 bank output must be a new directory outside the source bank"
        )
    if refs is not None and initial_artifacts is not None:
        raise ValueError(
            "A diagnostic training subset cannot bind experiment initial artifacts"
        )
    if (source / "MANIFEST.json").exists():
        _manifest(source)
    paths = [source / "cases.json", source / "index.json"]
    paths += [
        source / rel
        for rel in (
            "MANIFEST.json",
            "split.json",
            "SM_SELECTION_FROZEN.json",
            "COLD_SOURCES.json",
            "SM_SOURCES.json",
            "BANKS_READY.json",
            "PLAN.json",
            "PLAN.freeze.json",
        )
        if (source / rel).exists()
    ]
    source_hashes = {str(p): sha(p) for p in paths}
    if (source / "PLAN.freeze.json").exists() and read(source / "PLAN.freeze.json")[
        "sha256"
    ] != sha(source / "PLAN.json"):
        raise ValueError("Source PLAN hash mismatch")
    if (source / "PLAN.json").exists():
        plan = read(source / "PLAN.json")
        for rel in ("cases.json", "index.json"):
            expected = plan.get("files", {}).get(rel)
            if expected and sha(source / rel) != expected:
                raise ValueError("Frozen source bank file mismatch: " + rel)
    cases = read(source / "cases.json")
    if refs is not None:
        if not refs or len(set(refs)) != len(refs) or not set(refs) <= set(cases):
            raise ValueError("Empty, duplicate or unknown training source subset")
        cases = {ref: cases[ref] for ref in refs}
    source_index = read(source / "index.json")
    docs = [
        {
            "unit_id": f"{ref}-D{i+1:03d}",
            "source_ref": ref,
            "decision_index": i,
            "text": key_text(memory, decision),
        }
        for ref, memory in sorted(cases.items())
        for i, decision in enumerate(memory["L2"])
    ]
    if not docs:
        raise ValueError("No historical decisions")
    index = {
        "version": "V8",
        "key_template_version": VERSION,
        "model": source_index["model"],
        "embedding_url": source_index["embedding_url"],
        "instruction_prefix": "",
        "keys": docs,
    }
    if embedder is None:
        from .retrieval import RawEmbedder

        embedder = RawEmbedder(index["embedding_url"], index["model"])
    vectors, usage = embedder.embed([d["text"] for d in docs])
    vectors = np.asarray(vectors, dtype=np.float32)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(docs)
        or not vectors.shape[1]
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Incomplete or invalid V8 embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or not (norms > 0).all():
        raise ValueError("Zero or invalid V8 embedding norm")
    vectors /= norms
    dest.mkdir(parents=True, exist_ok=False)
    np.save(dest / "key_vectors.npy", vectors)
    put(dest / "index.json", index)
    if refs is None:
        shutil.copy2(
            source / "cases.json", dest / "cases.json"
        )  # Byte-identical historical payloads.
    else:
        put(dest / "cases.json", cases)
    generated = sorted(STEP_FILES)
    if refs is None and (source / "split.json").exists():
        shutil.copy2(source / "split.json", dest / "split.json")
        generated.append("split.json")
    for p, h in source_hashes.items():
        if sha(p) != h:
            raise ValueError("Source changed during V8 bank construction: " + p)
    manifest = {
        "version": "V8",
        "artifact_type": "step_bank",
        "key_template_version": VERSION,
        "source_bank": str(source),
        "source_files": source_hashes,
        "source_refs": sorted(cases),
        "source_count": len(cases),
        "decision_count": len(docs),
        "scope": (
            "training_subset_smoke_only" if refs is not None else "full_training_bank"
        ),
        "key_code_sha256": sha(PACKAGE_ROOT / "key.py"),
        "embedding_usage": usage,
        "value_policy": (
            "Byte-identical cases.json"
            if refs is None
            else "Unchanged selected source case objects"
        ),
        "initial_policy": "No initial artifacts: step retrieval diagnostics only",
        "experiment_ready": False,
        "files": {rel: sha(dest / rel) for rel in generated},
    }
    if initial_artifacts is not None:
        _bind_initial(dest, manifest, initial_artifacts)
    put(dest / "MANIFEST.json", manifest)
    verify_v8_bank(dest, require_initial=initial_artifacts is not None)
    return manifest


def bind_initial(bank, initial_artifacts, destination):
    """Publish a NEW experiment bank using existing step vectors; never edit a bank."""
    source, dest = Path(bank).resolve(), Path(destination).resolve()
    m = verify_v8_bank(source)
    if m.get("initial_artifacts"):
        raise ValueError(
            "Source bank already has initial artifacts; build a fresh binding explicitly"
        )
    if dest == source or source in dest.parents or dest.exists():
        raise ValueError("Bound V8 bank output must be new and outside the source bank")
    before = sha(source / "MANIFEST.json")
    dest.mkdir(parents=True, exist_ok=False)
    for rel in m["files"]:
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dst)
    manifest = {
        **m,
        "files": dict(m["files"]),
        "bound_from_step_bank": {"path": str(source), "manifest_sha256": before},
    }
    _bind_initial(dest, manifest, initial_artifacts)
    if sha(source / "MANIFEST.json") != before:
        raise ValueError("V8 step bank manifest changed during binding")
    put(dest / "MANIFEST.json", manifest)
    verify_v8_bank(dest, require_initial=True)
    return manifest


def initial_card(bank, task_id, arm):
    root = Path(bank)
    verify_v8_bank(root, require_initial=True)
    return read(root / "memory/initial.json")[task_id][arm]
