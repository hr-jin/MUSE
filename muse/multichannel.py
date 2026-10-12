"""Four independent unchunked routes; Top1 each, merge by historical source."""

from pathlib import Path
import json
import numpy as np
from .io import read, put, sha, verify_manifest
from .bank import verify_v8_bank
from .context import result_text
from .retrieval import RawEmbedder

VERSION = "muse-four-routes-top1-v1"
ROUTES = (
    ("assistant_to_l2", "assistant", "l2"),
    ("arguments_to_code", "arguments", "code"),
    ("arguments_to_l2", "arguments", "l2"),
    ("result_to_error", "result", "error"),
)


def failure_text(evaluation):
    """Copy textual diagnostics from the frozen public evaluation, never numeric error metrics."""
    value = evaluation.get("error")
    if isinstance(value, str) and value.strip():
        return value
    values = []
    for item in evaluation.get("evaluations", []):
        if item.get("status") == "PASS":
            continue
        metric = item.get("metrics") or {}
        candidates = [
            item.get("diagnostic"),
            item.get("error"),
            metric.get("fail_reason"),
            (metric.get("gate_breakdown") or {}).get("failure_reason"),
        ]
        for text in candidates:
            if isinstance(text, str) and text.strip():
                if text not in values:
                    values.append(text)
                break
    if not values:
        for name in ("diagnostic", "failure_reason", "fail_reason"):
            text = evaluation.get(name)
            if isinstance(text, str) and text.strip() and text not in values:
                values.append(text)
    return "\n\n".join(values)


def documents(cases, index):
    return {
        "l2": [dict(x) for x in index["keys"]],
        "code": [
            {
                "unit_id": ref + "-submission",
                "source_ref": ref,
                "text": m["L1"]["submissioned_solution"],
            }
            for ref, m in sorted(cases.items())
            if m["L1"]["submissioned_solution"].strip()
        ],
        "error": [
            {
                "unit_id": ref + "-error",
                "source_ref": ref,
                "text": failure_text(m["official_evaluation"]),
            }
            for ref, m in sorted(cases.items())
            if failure_text(m["official_evaluation"])
        ],
    }


def queries(context):
    if not context:
        return {}
    r = context[-1]
    tools = r.get("tools", [])
    if not tools:
        return {}
    t = tools[-1]
    name = t["name"]
    if name not in {"Write", "Bash"}:
        return {}
    q = {"assistant": r.get("assistant_text", "")}
    q["arguments"] = t["input"].get("content" if name == "Write" else "command", "")
    if name == "Bash":
        q["result"] = result_text(t.get("output", ""))
    return {k: v for k, v in q.items() if isinstance(v, str) and v.strip()}


def build_route_bank(base_bank, destination, embedder=None):
    base = Path(base_bank).resolve()
    out = Path(destination).resolve()
    verify_v8_bank(base)
    if out.exists() or base == out or base in out.parents:
        raise ValueError("New independent route bank required")
    cases = read(base / "cases.json")
    index = read(base / "index.json")
    docs = documents(cases, index)
    e = embedder or RawEmbedder(
        index["embedding_url"], index["model"], batch_size=2, timeout=120
    )
    inputs = {
        n: sha(base / n)
        for n in ("MANIFEST.json", "cases.json", "index.json", "key_vectors.npy")
    }
    vs = {"l2": np.load(base / "key_vectors.npy")}
    usage = {}
    for kind in ("code", "error"):
        if docs[kind]:
            vs[kind], usage[kind] = e.embed([d["text"] for d in docs[kind]])
        else:
            vs[kind] = np.empty((0, vs["l2"].shape[1]), dtype=np.float32)
    if any(sha(base / n) != h for n, h in inputs.items()):
        raise ValueError("Source changed during build")
    out.mkdir(parents=True)
    put(
        out / "index.json",
        {
            "version": VERSION,
            "model": index["model"],
            "embedding_url": index["embedding_url"],
            "pools": docs,
        },
    )
    files = ["index.json"]
    for kind, v in vs.items():
        v = np.asarray(v, dtype=np.float32)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        if v.ndim != 2 or not np.isfinite(v).all() or (norm <= 0).any():
            raise ValueError("Invalid embedding")
        np.save(out / (kind + ".npy"), v / norm if len(v) else v)
        files.append(kind + ".npy")
    m = {
        "version": VERSION,
        "base_bank": str(base),
        "source_hashes": inputs,
        "files": {n: sha(out / n) for n in files},
        "usage": usage,
        "pool_sizes": {k: len(v) for k, v in docs.items()},
    }
    put(out / "MANIFEST.json", m)
    Retriever(base, out)
    return m


class Retriever:
    def __init__(self, base_bank, route_bank, embedder=None):
        base = Path(base_bank).resolve()
        verify_v8_bank(base)
        root = Path(route_bank)
        m = verify_manifest(root)
        if m["version"] != VERSION or Path(m["base_bank"]).resolve() != base:
            raise ValueError("Route/source mismatch")
        if any(sha(base / n) != h for n, h in m["source_hashes"].items()):
            raise ValueError("Route source hashes changed")
        self.cases = read(base / "cases.json")
        index = read(base / "index.json")
        self.index = read(root / "index.json")
        self.docs = self.index["pools"]
        if self.docs != documents(self.cases, index):
            raise ValueError("Route documents differ from frozen historical values")
        self.vectors = {k: np.load(root / (k + ".npy")) for k in self.docs}
        dim = None
        for k, v in self.vectors.items():
            if v.ndim != 2 or len(v) != len(self.docs[k]) or not np.isfinite(v).all():
                raise ValueError("Invalid route vectors")
            if dim is None:
                dim = v.shape[1]
            if v.shape[1] != dim:
                raise ValueError("Embedding dimension mismatch")
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            if (norms <= 0).any():
                raise ValueError("Zero vector")
            self.vectors[k] = v / norms if len(v) else v
        self.embedder = embedder or RawEmbedder(index["embedding_url"], index["model"])

    def rank(self, qs, initial_sources, top_k=1):
        if top_k != 1:
            raise ValueError("Each route must use Top1")
        excluded = set(initial_sources)
        if not excluded <= set(self.cases):
            raise ValueError("Unknown initially injected source")
        if not qs:
            return {
                "ranking": [],
                "routes": [],
                "queries": {},
                "embedding": {},
                "excluded_initial_sources": sorted(excluded),
            }
        qv, usage = self.embedder.embed(list(qs.values()))
        qvectors = dict(zip(qs, qv))
        routes = []
        merged = {}
        for name, q, kind in ROUTES:
            eligible = [
                j
                for j, d in enumerate(self.docs[kind])
                if d["source_ref"] not in excluded
            ]
            if q not in qvectors or not eligible:
                routes.append(
                    {
                        "route": name,
                        "query_kind": q,
                        "key_kind": kind,
                        "top": [],
                        "status": "empty_query_or_pool",
                    }
                )
                continue
            scores = self.vectors[kind][eligible] @ qvectors[q]
            order = sorted(
                range(len(eligible)),
                key=lambda j: (
                    -float(scores[j]),
                    self.docs[kind][eligible[j]]["unit_id"],
                ),
            )[:1]
            hits = []
            for rank, j in enumerate(order, 1):
                d = self.docs[kind][eligible[j]]
                ref = d["source_ref"]
                hit = {
                    "route": name,
                    "rank": rank,
                    "unit_id": d["unit_id"],
                    "source_ref": ref,
                    "score": float(scores[j]),
                    "key_kind": kind,
                }
                if "decision_index" in d:
                    hit["decision_index"] = d["decision_index"]
                hits.append(hit)
                entry = merged.setdefault(
                    ref,
                    {
                        "unit_id": ref,
                        "source_ref": ref,
                        "decision_indices": [],
                        "route_hits": [],
                    },
                )
                entry["route_hits"].append(hit)
                if (
                    "decision_index" in d
                    and d["decision_index"] not in entry["decision_indices"]
                ):
                    entry["decision_indices"].append(d["decision_index"])
            routes.append(
                {
                    "route": name,
                    "query_kind": q,
                    "key_kind": kind,
                    "eligible_keys": len(eligible),
                    "top": hits,
                    "status": "retrieved",
                }
            )
        # Preserve route/rank order, never compare cross-route cosine scores.
        return {
            "ranking": list(merged.values()),
            "routes": routes,
            "queries": qs,
            "embedding": usage,
            "excluded_initial_sources": sorted(excluded),
        }


def candidate_window(ranking, cases, seen):
    rows = []
    candidates = []
    for r in ranking:
        if r["source_ref"] in seen:
            continue
        row = {**r, "candidate_id": f"candidate_{len(rows)+1}"}
        m = cases[r["source_ref"]]
        rows.append(row)
        candidates.append(
            {
                "candidate_id": row["candidate_id"],
                "scientific_problem": m["scientific_problem"],
                "L2": [m["L2"][j] for j in row["decision_indices"]],
                "submissioned_solution": m["L1"]["submissioned_solution"],
                "official_evaluation": m["official_evaluation"],
            }
        )
    if len(candidates) > 4:
        raise AssertionError("Four routes Top1 exceeded four candidates")
    return rows, candidates
