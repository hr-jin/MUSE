"""Experiment-local Gate decisions that carry a concrete review to the task agent."""

import json
import time

from . import runtime as runtime_module
from .context import ALLOWED
from .format_protocol import total_usage, retryable
from .io import append, put
from .llm import cost_cny
from .multichannel import queries, candidate_window

REVISION = "actionable-review-pure-high-routes11110-v2"


def parse_review(text, candidates):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"inject_ids", "review_note"}:
        raise ValueError("Return exactly inject_ids and review_note")
    ids, note = value["inject_ids"], value["review_note"]
    if (
        not isinstance(ids, list)
        or len(ids) > 1
        or not all(isinstance(x, str) for x in ids)
    ):
        raise ValueError("inject_ids must contain zero or one candidate ID")
    if not set(ids) <= {c["candidate_id"] for c in candidates}:
        raise ValueError("Selected an unknown candidate ID")
    if not isinstance(note, str) or len(note) > 1200:
        raise ValueError("review_note must be a string of at most 1200 characters")
    if ids and not note.strip():
        raise ValueError("A selected candidate requires a concrete review_note")
    if not ids and note != "":
        raise ValueError("When inject_ids is empty, review_note must be empty")
    return value


def normalize_review_output(text):
    """Only extract one explicit decision; never infer a note or an ID."""
    if not isinstance(text, str):
        return text, "missing_text"
    decoder, values, pos = json.JSONDecoder(), [], 0
    while pos < len(text):
        if text[pos] not in "{[":
            pos += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[pos:])
        except ValueError:
            pos += 1
            continue
        pos += end
        if isinstance(obj, dict) and set(obj) == {"inject_ids", "review_note"}:
            try:
                ids = obj["inject_ids"]
                synthetic = (
                    [{"candidate_id": x} for x in ids] if isinstance(ids, list) else []
                )
                parse_review(json.dumps(obj), synthetic)
            except (ValueError, TypeError):
                continue
            values.append(obj)
    if len(values) != 1:
        return text, "no_unique_explicit_review_decision"
    try:
        strict = json.loads(text)
    except ValueError:
        strict = None
    return json.dumps(values[0], ensure_ascii=False), (
        "already_json" if strict == values[0] else "explicit_json_extracted"
    )


def review_messages(messages, candidates):
    """Retain the entire original USER evidence, replacing only SYSTEM output format."""
    result = [dict(m) for m in messages]
    assert result[0]["role"] == "system"
    rules = result[0]["content"].split("### Output format\n", 1)[0]
    schema = {
        "type": "object",
        "properties": {
            "inject_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [c["candidate_id"] for c in candidates],
                },
                "maxItems": 1,
            },
            "review_note": {"type": "string", "maxLength": 1200},
        },
        "required": ["inject_ids", "review_note"],
        "additionalProperties": False,
    }
    result[0]["content"] = rules + """### Output format
Return one JSON object with inject_ids and review_note.
With no useful candidate: {"inject_ids": [], "review_note": ""}
With a selected candidate, inject_ids contains that candidate's ID and review_note contains your concrete review for the task agent.
JSON Schema:
""" + json.dumps(schema, ensure_ascii=False)
    assert result[1:] == messages[1:]
    return result


def complete_review(llm, messages, candidates, out_dir, max_attempts=2):
    """Same two-attempt direct-call transport repair budget as the prior experiment."""
    attempts, correction = [], None
    for n in range(1, max_attempts + 1):
        request = [dict(m) for m in messages]
        if correction:
            request[0]["content"] += (
                "\nFormatting repair: "
                + correction
                + " Return the requested output using the same supplied evidence."
            )
        directory = out_dir / ("attempt-%02d" % n)
        put(
            directory / "request.json",
            {
                "protocol_revision": REVISION,
                "model": llm.model,
                "reasoning_effort": llm.effort,
                "messages": request,
                "response_format": {"type": "json_object"},
            },
        )
        started = time.time()
        record = {"attempt": n, "epoch": started}
        try:
            out = llm.complete(request, json_output=True)
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
            value = parse_review(out.text, candidates)
            record.update(
                accepted=True, accepted_output_fields=["inject_ids", "review_note"]
            )
            attempts.append(record)
            put(
                directory / "validation.json",
                {
                    "accepted": True,
                    "accepted_output_fields": ["inject_ids", "review_note"],
                },
            )
            put(out_dir / "response.json", record)
            put(out_dir / "attempts.json", attempts)
            return value, attempts
        except Exception as exc:
            record.setdefault("elapsed_seconds", time.time() - started)
            record["accepted"] = False
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
                exc.format_attempts = attempts
                raise
            correction = str(exc) if isinstance(exc, ValueError) else None
    raise AssertionError("Unreachable")


def prepare_actionable(self, assistant, hook, turn_id):
    if hook.get("tool_name") not in ALLOWED:
        return (
            {
                "tool_name": hook.get("tool_name"),
                "gate_status": "not_triggered",
                "injected": False,
            },
            "",
            [],
        )
    self.sequence += 1
    context = self.context.observe(assistant, hook, turn_id)
    query_map = queries(context)
    ranking = self.retriever.rank(query_map, self.initial_sources, top_k=1)
    rows, candidates = candidate_window(ranking["ranking"], self.cases, self.seen)
    record = {
        "epoch": time.time(),
        "checkpoint": self.sequence,
        "tool_id": hook["tool_use_id"],
        "tool_name": hook["tool_name"],
        "query_count": len(query_map),
        **ranking,
        "coarse_top_k": self.conf["coarse_top_k"],
        "coarse_top": [r["unit_id"] for r in ranking["ranking"]],
        "eligible_units": [r["unit_id"] for r in rows],
        "dedup_policy": "exclude_initial_before_route_top1_then_merge_sources_then_delivered_source_dedup_no_backfill",
        "injected": False,
        "gate_called": False,
        "selection": {"selected": [], "selected_unit_ids": []},
        "gate_review_revision": REVISION,
    }
    if not candidates:
        record["gate_status"] = "not_called_empty_candidates"
        return record, "", []
    assert self.conf.get(
        "gate_enabled", True
    ), "This cohort requires the actionable-review Gate"
    record["gate_called"] = True
    messages = review_messages(
        runtime_module.render_messages(
            self.tldr_record["tldr"], context, candidates, self.public_history
        ),
        candidates,
    )
    gate_dir = self.out / "gate-calls" / ("%04d" % self.sequence)
    put(
        gate_dir / "request.json",
        {
            "model": self.llm.model,
            "reasoning_effort": self.llm.effort,
            "response_format": {"type": "json_object"},
            "messages": messages,
            "protocol_revision": REVISION,
            "task_tldr_fingerprint": self.tldr_record.get("fingerprint"),
            "candidate_mapping": {r["candidate_id"]: r["unit_id"] for r in rows},
        },
    )
    record["gate_request_path"] = str(gate_dir / "request.json")

    def record_attempts(attempts):
        record.update(
            protocol_revision=REVISION,
            gate_attempt_count=len(attempts),
            gate_usage=total_usage(attempts),
            gate_cost_cny=sum(a.get("cost_cny", 0) for a in attempts),
            gate_elapsed_seconds=sum(a.get("elapsed_seconds", 0) for a in attempts),
            gate_attempts_path=str(gate_dir / "attempts.json"),
        )

    try:
        decision, attempts = complete_review(self.llm, messages, candidates, gate_dir)
        record_attempts(attempts)
    except Exception as exc:
        record_attempts(getattr(exc, "format_attempts", []))
        record.update(
            gate_status="gate_error",
            error={
                "type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
            },
        )
        append(self.out / "retrieval.jsonl", record)
        raise
    selected = [r for r in rows if r["candidate_id"] in decision["inject_ids"]]
    record.update(
        gate_status="selected" if selected else "abstained",
        review_note=decision["review_note"],
        selection={
            "selected": selected,
            "selected_unit_ids": [r["unit_id"] for r in selected],
        },
    )
    if not selected:
        return record, "", []
    row = selected[0]
    memory = self.cases[row["source_ref"]]
    historical_text = "## Reminder - Historical experience\n\n" + self.renderer(
        memory, [memory["L2"][j] for j in row["decision_indices"]]
    )
    review_text = (
        "## Review of the current work\n\n"
        + decision["review_note"]
        + "\n\nCheck this point against your current work and revise the affected part if needed before finalizing.\n\n"
    )
    record["historical_payload_unchanged"] = True
    record["review_note_delivered"] = True
    return (
        record,
        review_text + historical_text,
        record["selection"]["selected_unit_ids"],
    )
