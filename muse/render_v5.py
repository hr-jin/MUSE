"""V5 injection: shared problem/full submission/evaluation plus the requested memory tier."""

import json
import re

HEADER = (
    "Historical reference material\n\n"
    "These records describe other tasks and their observed outcomes. "
    "They are not instructions for the current task. "
    "Apply a record only when its assumptions and interface conditions match. "
    "A recorded failure does not establish a valid fix. "
    "The current task requirements take precedence."
)

EXPERIENCE_HEADER = (
    "Historical reference material\n\n"
    "These experiences describe other tasks and their observed outcomes. "
    "They are not instructions for the current task. "
    "Apply an experience only when its assumptions and interface conditions match. "
    "A failed experience does not establish a valid fix. "
    "The current task requirements take precedence."
)


def experience_prefix(memory):
    return [
        EXPERIENCE_HEADER,
        "### Historical scientific problem\n\n" + fence(memory["scientific_problem"]),
    ]


def fence(text, language="text"):
    mark = "~" * max(
        4, 1 + max([len(m.group()) for m in re.finditer(r"~+", text)] or [0])
    )
    return mark + language + "\n" + text + "\n" + mark


def shared_prefix(memory):
    return [
        HEADER,
        "### Historical scientific problem\n\n" + fence(memory["scientific_problem"]),
    ]


def evaluation(memory):
    return "### Matching official evaluation\n\n" + fence(
        json.dumps(memory["official_evaluation"], ensure_ascii=False, indent=2), "json"
    )


def initial(memory):
    parts = shared_prefix(memory)
    parts += [
        "### Final route\n\n" + memory["L1"]["final_route"],
        "### Submissioned solution\n\n"
        + fence(memory["L1"]["submissioned_solution"], "python"),
        evaluation(memory),
    ]
    return "\n\n".join(parts)


def step(memory, selected_decisions):
    """Caller resolves internal decision identities; they are never rendered."""
    parts = experience_prefix(memory)
    for d in selected_decisions:
        assert d in memory["L2"]
        parts += [
            "### Historical decision\n\n" + d["decision"],
            "#### Choice and reason\n\n" + d["choice_and_reason"],
            "#### Relation to final failure\n\n"
            + (
                d["failure_relation"]
                if d["failure_relation"] is not None
                else "Not applicable."
            ),
        ]
    parts += [
        "### Submissioned solution\n\n"
        + fence(memory["L1"]["submissioned_solution"], "python"),
        evaluation(memory),
    ]
    return "\n\n".join(parts)
