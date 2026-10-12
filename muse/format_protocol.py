"""Formatting/transport repair only; the decision criteria and evidence stay fixed."""

import json, time
from .io import put
from .llm import cost_cny

REVISION = "v6-format-v2"
MAX_ATTEMPTS = 2


def retryable(exc):
    if isinstance(exc, ValueError):
        return True
    code = getattr(exc, "status_code", None)
    return (
        code in (408, 409, 429)
        or (isinstance(code, int) and code >= 500)
        or type(exc).__name__
        in ("APITimeoutError", "APIConnectionError", "TimeoutError")
    )


def total_usage(attempts):
    result = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    for a in attempts:
        u = a.get("usage", {})
        result["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
        result["completion_tokens"] += u.get("completion_tokens", 0) or 0
        result["prompt_tokens_details"]["cached_tokens"] += (
            u.get("prompt_tokens_details") or {}
        ).get("cached_tokens", 0) or 0
    return result


def complete_validated(
    llm, messages, json_output, parser, out_dir, max_attempts=MAX_ATTEMPTS
):
    """Persist every response before parsing. Retry only this direct completion."""
    attempts = []
    out_dir = __import__("pathlib").Path(out_dir)
    correction = None
    for n in range(1, max_attempts + 1):
        request = [dict(m) for m in messages]
        if correction:
            request[0]["content"] += (
                "\nFormatting repair: "
                + correction
                + " Return the requested output using the same supplied evidence."
            )
        directory = out_dir / f"attempt-{n:02d}"
        put(
            directory / "request.json",
            {
                "protocol_revision": REVISION,
                "model": llm.model,
                "reasoning_effort": llm.effort,
                "messages": request,
                "response_format": (
                    {"type": "json_object"} if json_output else {"type": "text"}
                ),
            },
        )
        started = time.time()
        record = {"attempt": n, "epoch": started}
        try:
            out = llm.complete(request, json_output=json_output)
            record.update(
                content=out.text,
                reasoning_content=out.reasoning,
                finish_reason=out.finish_reason,
                usage=out.usage,
                elapsed_seconds=out.elapsed_seconds,
                cost_cny=cost_cny(out.usage),
            )
            put(directory / "response.json", record)
            if out.finish_reason != "stop":
                raise ValueError("Completion did not finish normally")
            if not isinstance(out.text, str):
                raise ValueError("Completion has no text")
            value = parser(out.text)
            if json_output:
                record["ignored_extra_fields"] = sorted(
                    set(json.loads(out.text)) - {"inject_ids"}
                )
            record["accepted"] = True
            attempts.append(record)
            put(
                directory / "validation.json",
                {
                    "accepted": True,
                    "ignored_extra_fields": record.get("ignored_extra_fields", []),
                },
            )
            put(out_dir / "response.json", record)
            put(out_dir / "attempts.json", attempts)
            return value, attempts
        except Exception as exc:
            record.setdefault("elapsed_seconds", time.time() - started)
            record["accepted"] = False
            # Exception messages from an SDK may contain credentials or request details.
            record["error"] = {
                "type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
            }
            if isinstance(exc, ValueError):
                record["error"]["validation"] = str(exc)
            attempts.append(record)
            put(
                directory / "validation.json",
                {"accepted": False, "error": record["error"]},
            )
            put(out_dir / "attempts.json", attempts)
            if n == max_attempts or not retryable(exc):
                try:
                    exc.format_attempts = attempts
                except Exception:
                    pass
                raise
            correction = str(exc) if isinstance(exc, ValueError) else None
    raise AssertionError("Unreachable")
