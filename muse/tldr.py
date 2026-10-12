import fcntl, time
from pathlib import Path
from .io import PACKAGE_ROOT, digest, read, put
from .format_protocol import complete_validated, total_usage, REVISION


def parse_tldr(text):
    # Layout whitespace has no semantic role in a one-sentence summary.
    summary = " ".join(text.split())
    if not summary or len(summary.split()) > 80:
        raise ValueError("TLDR must contain one sentence of at most 80 words")
    return summary


def prepare_tldr(instruction, cache_dir, llm):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Original task instruction required")
    template = (PACKAGE_ROOT / "tldr_prompt.txt").read_text()
    fingerprint = {
        "instruction_sha256": digest(instruction),
        "prompt_sha256": digest(template),
        "model": llm.model,
        "effort": llm.effort,
    }
    key = digest(str(sorted(fingerprint.items())))
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (key + ".json")
    with (root / (key + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            r = read(path)
            if r["fingerprint"] != fingerprint or r["tldr_sha256"] != digest(r["tldr"]):
                raise ValueError("Cached TLDR provenance mismatch")
            return {**r, "cache_hit": True}
        rules, tail = template.split("### Task instruction\n", 1)
        _, output = tail.rsplit("### Output\n", 1)
        messages = [
            {"role": "system", "content": rules + "### Output\n" + output},
            {"role": "user", "content": instruction},
        ]
        raw_dir = root / "raw" / key / str(time.time_ns())
        summary, attempts = complete_validated(
            llm, messages, False, parse_tldr, raw_dir
        )
        row = {
            "fingerprint": fingerprint,
            "tldr": summary,
            "tldr_sha256": digest(summary),
            "usage": total_usage(attempts),
            "cost_cny": sum(a.get("cost_cny", 0) for a in attempts),
            "elapsed_seconds": sum(a["elapsed_seconds"] for a in attempts),
            "protocol_revision": REVISION,
            "raw_attempts_path": str(raw_dir),
            "request_count": len(attempts),
        }
        put(path, row)
        return {**row, "cache_hit": False}
