import json, re
from .io import PACKAGE_ROOT


def render_prompt(tldr, context, candidates):
    template = (PACKAGE_ROOT / "gate_prompt.txt").read_text()
    values = {
        "task_tldr": tldr,
        "recent_agent_activity": json.dumps(context, ensure_ascii=False, indent=2),
        "candidate_memories": json.dumps(candidates, ensure_ascii=False, indent=2),
    }
    return re.sub(
        r"\{\{(task_tldr|recent_agent_activity|candidate_memories)\}\}",
        lambda m: values[m.group(1)],
        template,
    )


def output_schema(candidates):
    return {
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


def render_messages(tldr, context, candidates, prior_activity=None):
    rendered = render_prompt(tldr, context, candidates)
    rules, data = rendered.split("### Current task summary\n", 1)
    data, output = data.rsplit("### Output format\n", 1)
    if prior_activity:
        marker = "### Current agent activity\n\n"
        history = json.dumps(prior_activity, ensure_ascii=False, indent=2)
        data = data.replace(
            marker,
            "### Prior agent activity, in chronological order\n\n"
            + history
            + "\n\n"
            + marker,
            1,
        )
    system = (
        rules
        + "### Output format\n"
        + output
        + "\nJSON Schema:\n"
        + json.dumps(output_schema(candidates), ensure_ascii=False)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "### Current task summary\n" + data},
    ]


def parse_decision(text, candidates):
    """Return selected IDs after validating the full actionable-review decision."""
    from .review_runtime import parse_review

    return parse_review(text, candidates)["inject_ids"]
