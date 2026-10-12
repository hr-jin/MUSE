"""V8 multi-route: current Write/Bash checkpoint; retrieval query construction lives in multichannel.py."""

import copy, json, re

ALLOWED = {"Write", "Bash"}


def arguments(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {"content": value}
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def result_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(x for x in (result_text(v) for v in value) if x.strip())
    if isinstance(value, dict):
        # Do not turn exit codes, byte counts, filenames or IDs into queries.
        return "\n".join(
            result_text(value[k])
            for k in ("stdout", "stderr", "content", "text", "message", "error")
            if k in value
        )
    return ""


class RecentContext:
    """One checkpoint only; never carry earlier actions, even in the same turn."""

    def __init__(self, turns=1):
        if turns != 1:
            raise ValueError("V8 requires current checkpoint only")
        self.turns = 1
        self.records = []

    def observe(self, assistant, hook, turn_id):
        if hook.get("tool_name") not in ALLOWED:
            self.records = []
            return self.snapshot()
        text = assistant.get("content") or ""
        if not isinstance(text, str):
            text = result_text(text)
        tools = []
        if hook.get("tool_name") in ALLOWED:
            tools = [
                {
                    "call_id": hook["tool_use_id"],
                    "name": hook["tool_name"],
                    "input": arguments(hook.get("tool_input", {})),
                    "output": copy.deepcopy(
                        hook.get("tool_response", hook.get("error", ""))
                    ),
                }
            ]
        self.records = (
            [{"turn_id": turn_id, "assistant_text": text, "tools": tools}]
            if text.strip() or tools
            else []
        )
        return self.snapshot()

    def snapshot(self):
        return copy.deepcopy(self.records)


def current_assistant(client, event):
    stream = [
        e
        for e in client.events
        if e.get("type") == "model.streaming"
        and e.get("payload", {}).get("assistantMessageId")
    ]
    if not stream:
        raise ValueError("No native assistant streaming record")
    ident = stream[-1]["payload"]["assistantMessageId"]
    parts = [
        e["payload"] for e in stream if e["payload"]["assistantMessageId"] == ident
    ]
    return {
        "content": "".join(
            p.get("delta", "") for p in parts if p.get("kind") == "text_delta"
        )
    }, ident


def replay_context(messages, message_index, call_id, conf):
    recent = RecentContext(conf["recent_assistant_turns"])
    found = False
    for mi, m in enumerate(messages):
        if mi > message_index:
            break
        if m.get("info", {}).get("role") != "assistant":
            continue
        text = ""
        for p in m.get("parts", []):
            if p.get("type") == "text":
                text += p.get("text", "")
            if p.get("type") != "tool":
                continue
            state = p.get("state", {})
            h = {
                "tool_name": p["tool"],
                "tool_use_id": p["callID"],
                "tool_input": state.get("input", {}),
                "tool_response": state.get("output", state.get("error", "")),
            }
            recent.observe({"content": text}, h, str(mi))
            if mi == message_index and p["callID"] == call_id:
                found = True
                break
        if found:
            break
    if not found:
        raise ValueError("Requested replay checkpoint not found")
    return recent
