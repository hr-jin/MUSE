"""Build pre-checkpoint public activity without reasoning or future evidence."""

import copy
import hashlib
import json
from collections import Counter

CURRENT_MARKER = "### Current agent activity\n\n"
CANDIDATE_MARKER = "\n\n### Candidate historical memories\n\n"
HISTORY_MARKER = "### Prior agent activity, in chronological order\n\n"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_paths(value):
    """Keep framework-neutral public activity unchanged."""
    return copy.deepcopy(value)


def extract_prior(messages, current_activity):
    current = current_activity[0]
    target_turn = current["turn_id"]
    target_call = current["tools"][0]["call_id"]
    found = [
        (i, j, message, part)
        for i, message in enumerate(messages)
        for j, part in enumerate(message.get("parts", []))
        if part.get("type") == "tool" and part.get("callID") == target_call
    ]
    assert len(found) == 1, "Checkpoint must have one exact tool match"
    message_index, part_index, target_message, target_part = found[0]
    assert target_message["info"]["id"] == target_turn
    cutoff_ms = target_part["state"]["time"]["start"]
    assert target_part["tool"] == current["tools"][0]["name"]
    assert (
        normalize_paths(target_part["state"]["input"]) == current["tools"][0]["input"]
    )
    original_current_text = "\n".join(
        p.get("text", "")
        for p in target_message["parts"]
        if p.get("type") == "text"
        and p.get("time", {}).get("end", cutoff_ms + 1) <= cutoff_ms
    )
    assert (
        original_current_text == current["assistant_text"]
    ), "Current text differs from frozen request"
    records = []
    tools = Counter()
    excluded = Counter()
    included_tool_ids = []
    for mi, message in enumerate(messages[: message_index + 1]):
        info = message.get("info", {})
        if info.get("role") != "assistant":
            # Original task is represented by the unchanged TLDR. Historical
            # memory messages are not repeated inside current activity.
            excluded["user_messages"] += 1
            continue
        if info.get("semantics", {}).get("providerVisibility") == "hidden":
            excluded["hidden_messages"] += 1
            continue
        record = {"turn_id": info["id"], "assistant_text": "", "tools": []}
        texts = []
        for pi, part in enumerate(message.get("parts", [])):
            kind = part.get("type")
            if kind == "reasoning":
                excluded["reasoning_parts"] += 1
                excluded["reasoning_characters"] += len(part.get("text", ""))
                continue
            if kind == "text":
                if mi == message_index:
                    # Already present verbatim in unchanged current activity.
                    continue
                assert part.get("time", {}).get("end", cutoff_ms + 1) <= cutoff_ms
                texts.append(part.get("text", ""))
            elif kind == "tool":
                if mi == message_index and pi >= part_index:
                    break
                state = part["state"]
                if state["status"] in {"completed", "error"}:
                    assert (
                        state.get("time", {}).get("end", cutoff_ms + 1) <= cutoff_ms
                    ), "Terminal tool after checkpoint cutoff"
                else:
                    assert state["status"] in {
                        "running",
                        "pending",
                    }, "Unknown prior tool state"
                    assert (
                        state.get("time", {}).get("start", cutoff_ms + 1) <= cutoff_ms
                    ), "Prior tool started in the future"
                    assert (
                        "output" not in state and "error" not in state
                    ), "Unobserved result in prior history"
                item = {
                    "call_id": part["callID"],
                    "name": part["tool"],
                    "input": copy.deepcopy(state.get("input", {})),
                    "status": state["status"],
                }
                for field in ["output", "error"]:
                    if field in state:
                        item[field] = copy.deepcopy(state[field])
                record["tools"].append(item)
                tools[part["tool"]] += 1
                included_tool_ids.append(part["callID"])
        record["assistant_text"] = "\n".join(texts)
        if texts or record["tools"]:
            records.append(record)
    assert target_call not in included_tool_ids
    expected_prior_calls = [
        p["callID"]
        for mi, message in enumerate(messages[: message_index + 1])
        if message.get("info", {}).get("role") == "assistant"
        and message.get("info", {}).get("semantics", {}).get("providerVisibility")
        != "hidden"
        for pi, p in enumerate(message.get("parts", []))
        if p.get("type") == "tool" and (mi < message_index or pi < part_index)
    ]
    assert included_tool_ids == expected_prior_calls, "Prior tool coverage mismatch"
    audit = {
        "target_turn_id": target_turn,
        "target_call_id": target_call,
        "target_message_index": message_index,
        "target_part_index": part_index,
        "checkpoint_tool_start_ms": cutoff_ms,
        "prior_assistant_records": len(records),
        "prior_tools": dict(tools),
        "prior_tool_ids": included_tool_ids,
        "excluded": dict(excluded),
        "current_activity_duplicated": False,
        "reasoning_included": False,
        "future_messages_included": False,
        "truncation_added": False,
        "source_outputs_preserved": True,
    }
    audit["project_path_alias"] = "none"
    return normalize_paths(records), audit


def expand_request(baseline, history):
    request = copy.deepcopy(baseline)
    user = request["messages"][1]["content"]
    assert user.count(CURRENT_MARKER) == 1
    rendered = json.dumps(history, ensure_ascii=False, indent=2)
    request["messages"][1]["content"] = user.replace(
        CURRENT_MARKER, HISTORY_MARKER + rendered + "\n\n" + CURRENT_MARKER, 1
    )
    assert request["messages"][0] == baseline["messages"][0]
    assert request["candidate_mapping"] == baseline["candidate_mapping"]
    assert (
        request["messages"][1]["content"].split(CURRENT_MARKER, 1)[1]
        == user.split(CURRENT_MARKER, 1)[1]
    )
    return request


def verify_request(baseline, expanded, history):
    assert expanded == expand_request(
        baseline, history
    ), "Expanded request changed beyond prior activity"
    user = expanded["messages"][1]["content"]
    candidates = json.loads(user.split(CANDIDATE_MARKER, 1)[1])
    return candidates


def reconcile_tool_receipts(messages, current_activity, receipts):
    """Use already observed PostToolUse receipts when session state lags.
    The cutoff and source-call checks prevent later outputs entering history.
    """
    from datetime import datetime

    snapshot = copy.deepcopy(messages)
    target = current_activity[0]["tools"][0]["call_id"]
    found = [
        (mi, pi, p)
        for mi, m in enumerate(snapshot)
        for pi, p in enumerate(m.get("parts", []))
        if p.get("type") == "tool" and p.get("callID") == target
    ]
    assert len(found) == 1, "Receipt reconciliation: target call not unique"
    target_mi, target_pi, target_part = found[0]
    cutoff = target_part["state"]["time"]["start"]
    indexed = {}
    for hook in receipts:
        cid = hook.get("tool_use_id", hook.get("toolCallId"))
        ts = hook.get("timestamp")
        if not cid or not isinstance(ts, str):
            continue
        end = round(
            datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000
        )
        if cid == target or end > cutoff:
            continue
        indexed[cid] = (hook, end)
    repaired = []
    pending = []
    for mi, message in enumerate(snapshot[: target_mi + 1]):
        if message.get("info", {}).get("role") != "assistant":
            continue
        for pi, part in enumerate(message.get("parts", [])):
            if mi == target_mi and pi >= target_pi:
                break
            if part.get("type") != "tool":
                continue
            state = part["state"]
            times = state.get("time", {})
            if (
                state.get("status") in {"completed", "error"}
                and times.get("end", cutoff + 1) <= cutoff
            ):
                continue
            cid = part["callID"]
            if cid in indexed:
                hook, end = indexed[cid]
                assert part["tool"] == hook.get(
                    "tool_name", hook.get("toolName")
                ), "Receipt tool mismatch"
                assert normalize_paths(state.get("input", {})) == normalize_paths(
                    hook.get("tool_input", hook.get("toolInput", {}))
                ), "Receipt input mismatch"
                assert (
                    times.get("start", 0) <= end <= cutoff
                ), "Receipt outside checkpoint boundary"
                response = copy.deepcopy(
                    hook.get(
                        "tool_response", hook.get("toolResponse", hook.get("error", ""))
                    )
                )
                state["status"] = (
                    "error"
                    if hook.get("hookEventName") == "PostToolUseFailure"
                    else "completed"
                )
                state["output"] = response
                state.pop("error", None)
                state.setdefault("time", {})["end"] = end
                repaired.append(
                    {
                        "call_id": cid,
                        "receipt_time_ms": end,
                        "snapshot_status": part.get("state", {}).get("status"),
                    }
                )
            else:
                # The operation was launched, but no result was yet observed.
                # Preserve its input, without importing any later final output.
                assert (
                    times.get("start", cutoff + 1) <= cutoff
                ), "Future tool in prior history"
                state["status"] = "running"
                state.pop("output", None)
                state.pop("error", None)
                state.setdefault("time", {}).pop("end", None)
                pending.append(cid)
    return snapshot, {
        "revision": "post-tool-receipt-reconciliation-v1",
        "reconciled_calls": repaired,
        "pending_input_only_calls": pending,
        "cutoff_ms": cutoff,
        "future_receipts_excluded": True,
    }
