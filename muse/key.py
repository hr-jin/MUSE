VERSION = "raw-decision-choice-v8"


def key_text(memory, decision):
    parts = [decision["decision"], decision["choice_and_reason"]]
    if not all(isinstance(p, str) and p.strip() for p in parts):
        raise ValueError("Missing V8 decision field")
    return "\n\n".join(parts)
