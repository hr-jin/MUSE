"""Portable extraction and GlobalMemBank updates for scientific coding memory.

The bank update rules adapt the original incremental construction pipeline.
Historical inputs are copied rather than modified, and model responses must
pass both the released JSON schemas and source-order checks before application.
"""

import copy
import json
from pathlib import Path

import jsonschema


EVIDENCE_STATUSES = {"observed", "derived_from_contract", "proposed_unverified"}
CORPUS_AUDIT = "CORPUS_AUDIT"


def _resource(name):
    package = Path(__file__).resolve().parent
    for directory in (package / "prompts", package.parent / "prompts"):
        path = directory / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError("Missing MUSE prompt resource: " + name)


def _schema(name):
    return json.loads(_resource(name))


def _validate_schema(value, name):
    try:
        jsonschema.validate(value, _schema(name))
    except jsonschema.ValidationError as exc:
        raise ValueError("Invalid " + name + ": " + exc.message) from exc


def _order(source_order):
    if not isinstance(source_order, (list, tuple)):
        raise ValueError("source_order must be an ordered list of source IDs")
    refs = list(source_order)
    if any(not isinstance(ref, str) or not ref.strip() or ref == CORPUS_AUDIT for ref in refs):
        raise ValueError("Invalid source ID in source_order")
    if len(refs) != len(set(refs)):
        raise ValueError("Duplicate source IDs in source_order")
    return refs


def _validate_evidence(evidence, ref=None):
    required = {"source_ref", "scientific_problem", "submission", "official_evaluation", "trajectory"}
    if not isinstance(evidence, dict) or not required <= set(evidence):
        raise ValueError("Evidence requires " + ", ".join(sorted(required)))
    source = evidence["source_ref"]
    if not isinstance(source, str) or not source.strip() or source == CORPUS_AUDIT:
        raise ValueError("Evidence requires a nonempty source_ref")
    if ref is not None and source != ref:
        raise ValueError("Evidence source_ref does not match its source ID")
    if not isinstance(evidence["scientific_problem"], str) or not evidence["scientific_problem"].strip():
        raise ValueError("scientific_problem must be nonempty text")
    if not isinstance(evidence["submission"], str):
        raise ValueError("submission must contain the historical submission text")
    if not isinstance(evidence["official_evaluation"], dict):
        raise ValueError("official_evaluation must be an object")
    if not isinstance(evidence["trajectory"], (str, list, dict)):
        raise ValueError("trajectory must contain the historical public trajectory")
    return source


def validate_draft(draft, ref=None, source_order=None):
    """Validate extracted lessons without altering historical source material."""
    if source_order is not None and ref not in _order(source_order):
        raise ValueError("Unknown extraction source")
    _validate_schema(draft, "extraction_schema.json")
    if not draft["final_route"].strip():
        raise ValueError("The approach-level lesson must contain text")
    for decision in draft["decisions"]:
        if not decision["decision"].strip() or not decision["choice_and_reason"].strip():
            raise ValueError("Implementation descriptions and rationale must contain text")
        relation = decision["failure_relation"]
        if relation is not None and not relation.strip():
            raise ValueError("failure_relation must contain text or be null")


def assemble(ref, draft, evidence):
    """Assemble the source memory while copying its problem, code, and outcome."""
    _validate_evidence(evidence, ref)
    validate_draft(draft)
    return {
        "source_ref": ref,
        "scientific_problem": copy.deepcopy(evidence["scientific_problem"]),
        "official_evaluation": copy.deepcopy(evidence["official_evaluation"]),
        "L1": {
            "final_route": draft["final_route"],
            "submissioned_solution": copy.deepcopy(evidence["submission"]),
        },
        "L2": copy.deepcopy(draft["decisions"]),
    }


def empty():
    return {"entries": [], "next_id": 1, "processed_sources": []}


def bank_text(bank):
    """Render global memory, keeping unverified proposals visibly labeled."""
    return "## GlobalMemBank\n\n" + "\n\n---\n\n".join(
        ("Proposed, unverified: " if entry["evidence_status"] == "proposed_unverified"
         and not entry["text"].lower().startswith("proposed, unverified") else "")
        + entry["text"].strip()
        for entry in bank["entries"]
    ) + "\n"


def validate_bank(bank, source_order):
    refs = _order(source_order)
    if not isinstance(bank, dict) or set(bank) != {"entries", "next_id", "processed_sources"}:
        raise ValueError("Invalid GlobalMemBank fields")
    seen = bank["processed_sources"]
    if not isinstance(seen, list) or seen != refs[:len(seen)]:
        raise ValueError("Processed sources must be a prefix of source_order")
    if type(bank["next_id"]) is not int or bank["next_id"] < 1:
        raise ValueError("next_id must be a positive integer")
    if not isinstance(bank["entries"], list):
        raise ValueError("GlobalMemBank entries must be a list")
    ids = []
    for entry in bank["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"id", "text", "source_refs", "evidence_status"}:
            raise ValueError("Invalid global entry fields")
        eid = entry["id"]
        if (not isinstance(eid, str) or not eid.startswith("G") or not eid[1:].isdigit()
                or not 0 < int(eid[1:]) < bank["next_id"]):
            raise ValueError("Global entry ID is outside its assigned ID range")
        if not isinstance(entry["text"], str) or not entry["text"].strip():
            raise ValueError("Global entry text must be nonempty")
        sources = entry["source_refs"]
        if (not isinstance(sources, list) or not sources
                or any(not isinstance(source, str) for source in sources)
                or len(sources) != len(set(sources)) or not set(sources) <= set(seen)):
            raise ValueError("Global entry references unknown or unprocessed sources")
        if not isinstance(entry["evidence_status"], str) or entry["evidence_status"] not in EVIDENCE_STATUSES:
            raise ValueError("Unknown global entry evidence status")
        ids.append(eid)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate global entry IDs")


def apply(bank, response, ref, allowed, source_order):
    """Apply validated incremental operations, preserving unchanged entries."""
    refs = _order(source_order)
    validate_bank(bank, refs)
    allowed = list(allowed)
    if ref == CORPUS_AUDIT:
        if bank["processed_sources"] != refs or allowed != refs:
            raise ValueError("Bank-wide review requires the complete processed source order")
    else:
        expected = refs[:len(bank["processed_sources"]) + 1]
        if (len(expected) != len(bank["processed_sources"]) + 1
                or allowed != expected or ref != expected[-1]):
            raise ValueError("Consolidation must process exactly the next source")
    _validate_schema(response, "operations_schema.json")
    if response["source_ref"] != ref:
        raise ValueError("Operations source_ref does not match the current source")
    out = copy.deepcopy(bank)
    entries = {entry["id"]: entry for entry in out["entries"]}
    ids = list(entries)
    touched, audit = set(), []
    for operation in response["operations"]:
        op = copy.deepcopy(operation)
        if not set(op["source_refs"]) <= set(allowed):
            raise ValueError("Unknown or future source in a global memory operation")
        if not op["basis"].strip():
            raise ValueError("Each operation requires an evidence basis")
        action, eid = op["action"], op["entry_id"]
        if action == "add":
            if eid is not None or not op["text"] or op["evidence_status"] is None:
                raise ValueError("Add operations require null entry_id, text, and evidence_status")
            eid = "G%04d" % out["next_id"]
            out["next_id"] += 1
            ids.append(eid)
        elif eid not in entries or eid in touched:
            raise ValueError("Unknown entry or duplicate operation on the same entry")
        old = copy.deepcopy(entries.get(eid))
        if action == "retire":
            if op["text"] is not None or op["evidence_status"] is not None:
                raise ValueError("Retire operations require null text and evidence_status")
            del entries[eid]
            ids.remove(eid)
        else:
            if not isinstance(op["text"], str) or not op["text"].strip() or op["evidence_status"] is None:
                raise ValueError("Global memory updates require nonempty text and evidence_status")
            entries[eid] = {key: copy.deepcopy(op[key]) for key in ("text", "source_refs", "evidence_status")}
            entries[eid]["id"] = eid
        touched.add(eid)
        audit.append({"entry_id": eid, "operation": op, "previous": old,
                      "current": copy.deepcopy(entries.get(eid))})
    out["entries"] = [entries[eid] for eid in ids]
    if ref != CORPUS_AUDIT:
        out["processed_sources"].append(ref)
    validate_bank(out, refs)
    return out, audit


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


def _completion_json(llm, system, context):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, indent=2)}]
    completion = llm.complete(messages, json_output=True)
    if getattr(completion, "finish_reason", "stop") != "stop":
        raise ValueError("Memory construction completion did not finish normally")
    try:
        return json.loads(completion.text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise ValueError("Memory construction requires a JSON completion") from exc


def _inline_system(name, description):
    original = _resource(name)
    _, separator, body = original.partition("\n\n")
    if not separator:
        raise ValueError("Prompt resource has no instruction body: " + name)
    return (description + "\n\n"
            "The next message contains the request, historical inputs, and output schema as inline JSON. "
            "These fields replace the /workspace/ file locations in the original prompt. "
            "Return the schema-conforming JSON object directly. No filesystem or tool access is available.\n\n"
            + body)


def extract_evidence(llm, evidence):
    """Extract and assemble a memory from one frozen historical source."""
    ref = _validate_evidence(evidence)
    system = _inline_system("extraction.txt", "Construct Scientific Memory from ONE frozen historical scientific coding attempt.")
    draft = _completion_json(llm, system, {"evidence": copy.deepcopy(evidence),
                                         "extraction_schema": _schema("extraction_schema.json")})
    validate_draft(draft)
    return assemble(ref, draft, evidence)


def _source_inputs(memories, evidence, refs, allowed):
    if not isinstance(memories, dict) or not isinstance(evidence, dict):
        raise ValueError("memories and evidence must be maps keyed by source ID")
    if not set(memories) <= set(refs) or not set(evidence) <= set(refs):
        raise ValueError("Unknown source in memories or evidence")
    if not set(allowed) <= set(memories) or not set(allowed) <= set(evidence):
        raise ValueError("Historical memory and evidence are required for every available source")
    for ref in allowed:
        _validate_evidence(evidence[ref], ref)
        memory = memories[ref]
        if (not isinstance(memory, dict) or not {"scientific_problem", "official_evaluation", "L1", "L2"} <= set(memory)
                or not isinstance(memory["L1"], dict)):
            raise ValueError("Incomplete historical memory for source " + ref)
        if memory.get("source_ref", ref) != ref:
            raise ValueError("Historical memory source ID does not match its map key")
        if (memory["scientific_problem"] != evidence[ref]["scientific_problem"]
                or memory["official_evaluation"] != evidence[ref]["official_evaluation"]
                or memory["L1"].get("submissioned_solution") != evidence[ref]["submission"]):
            raise ValueError("Historical memory differs from its original source material")
        validate_draft({"final_route": memory["L1"].get("final_route"), "decisions": memory["L2"]})
    return {"memories": {ref: copy.deepcopy(memories[ref]) for ref in allowed},
            "evidence": {ref: copy.deepcopy(evidence[ref]) for ref in allowed}}


def consolidate_one(llm, bank, ref, memories, evidence, source_order):
    """Consolidate exactly the next source, with only the processed prefix visible."""
    refs = _order(source_order)
    validate_bank(bank, refs)
    allowed = refs[:len(bank["processed_sources"]) + 1]
    if len(allowed) != len(bank["processed_sources"]) + 1 or ref != allowed[-1]:
        raise ValueError("Consolidation must process exactly the next source")
    inputs = _source_inputs(memories, evidence, refs, allowed)
    context = {"request": {"source_ref": ref, "source_ordinal": len(bank["processed_sources"]),
                           "available_sources": allowed}, "bank": copy.deepcopy(bank),
               **inputs, "operations_schema": _schema("operations_schema.json")}
    system = _inline_system("consolidation.txt", "Build GlobalMemBank incrementally from frozen Scientific Memory and historical evidence. Process exactly the source identified in request.source_ref.")
    response = _completion_json(llm, system, context)
    return apply(bank, response, ref, allowed, refs)


def review_bank(llm, bank, memories, evidence, source_order):
    """Review the fully consolidated bank against the complete frozen source set."""
    refs = _order(source_order)
    validate_bank(bank, refs)
    if bank["processed_sources"] != refs:
        raise ValueError("Bank-wide review requires every source to be processed")
    inputs = _source_inputs(memories, evidence, refs, refs)
    context = {"request": {"source_ref": CORPUS_AUDIT, "available_sources": refs},
               "bank": copy.deepcopy(bank), **inputs, "operations_schema": _schema("operations_schema.json")}
    system = _inline_system("global-review.txt", 'Audit the incrementally constructed GlobalMemBank using only the frozen historical scientific coding corpus. Return operations with source_ref="CORPUS_AUDIT".')
    response = _completion_json(llm, system, context)
    return apply(bank, response, CORPUS_AUDIT, refs, refs)
