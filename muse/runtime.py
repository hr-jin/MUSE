import asyncio, json, time
from pathlib import Path
from .io import append, put, read, configuration, PACKAGE_ROOT
from .context import RecentContext, ALLOWED
from .multichannel import queries, Retriever, candidate_window
from .gate import render_messages, parse_decision
from .format_protocol import complete_validated, total_usage, REVISION
from .llm import cost_cny
from . import render_v5


class V8Runtime:
    def __init__(
        self,
        bank,
        out,
        tldr_record,
        llm,
        conf=None,
        renderer=None,
        retriever=None,
        initial_source_refs=None,
        route_bank=None,
        public_history=None,
    ):
        self.conf = configuration(conf)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.tldr_record = tldr_record
        self.llm = llm
        self.cases = read(Path(bank) / "cases.json")
        if initial_source_refs is None:
            raise ValueError("Supply actually injected initial source_refs explicitly")
        self.initial_sources = set(initial_source_refs)
        if not self.initial_sources <= set(self.cases):
            raise ValueError("Unknown initial source")
        if retriever is None and route_bank is None:
            raise ValueError("Rebuilt multichannel route_bank required")
        self.retriever = retriever or Retriever(bank, route_bank)
        self.renderer = renderer or render_v5.step
        self.context = RecentContext(self.conf["recent_assistant_turns"])
        self.public_history = list(public_history or [])
        self.public_message_snapshot = None
        self.public_tool_receipts = None
        self.seen = set()
        self.completed_checkpoints = set()
        self.sequence = 0
        # Task resumptions need an explicit context replay; do not silently reset.
        if (self.out / "retrieval.jsonl").exists():
            raise ValueError("V8 output already exists; use a fresh isolated attempt")
        put(self.out / "V8_CONFIG.json", self.conf)
        put(self.out / "TLDR_USED.json", tldr_record)

    def set_public_history(self, records):
        """Supply earlier public activity, excluding private reasoning and current/future actions."""
        import copy

        self.public_history = copy.deepcopy(list(records))

    def prepare(self, assistant, hook, turn_id):
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
        if self.conf.get("gate_enabled", True):
            if self.public_message_snapshot is not None:
                from .history_context import extract_prior, reconcile_tool_receipts
                from .context import RecentContext

                current = RecentContext(1).observe(assistant, hook, turn_id)
                messages = self.public_message_snapshot
                if self.public_tool_receipts is not None:
                    messages, _ = reconcile_tool_receipts(
                        messages, current, self.public_tool_receipts
                    )
                history, _ = extract_prior(messages, current)
                self.set_public_history(history)
            from .review_runtime import prepare_actionable

            return prepare_actionable(self, assistant, hook, turn_id)
        return self.prepare_direct(assistant, hook, turn_id)

    def prepare_direct(self, assistant, hook, turn_id):
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
        ranking = self.retriever.rank(queries(context), self.initial_sources, top_k=1)
        rows, _ = candidate_window(ranking["ranking"], self.cases, self.seen)
        record = {
            "epoch": time.time(),
            "checkpoint": self.sequence,
            "tool_id": hook["tool_use_id"],
            "tool_name": hook["tool_name"],
            **ranking,
            "gate_called": False,
            "injected": False,
            "dedup_policy": "exclude_initial_before_top1_then_merge_sources_then_delivered_source_dedup_no_backfill",
            "selection": {"selected": [], "selected_unit_ids": []},
        }
        if not rows:
            record["gate_status"] = "not_called_empty_candidates"
            return record, "", []
        # Similarities come from the same normalized embedding space.
        # Select exactly one source with the highest retained route-hit score.
        row = max(rows, key=lambda r: max(h["score"] for h in r["route_hits"]))
        ids = [row["unit_id"]]
        record.update(
            gate_status="direct_no_gate",
            selection={"selected": [row], "selected_unit_ids": ids},
        )
        memory = self.cases[row["source_ref"]]
        text = "## Reminder - Historical experience\n\n" + self.renderer(
            memory, [memory["L2"][j] for j in row["decision_indices"]]
        )
        return record, text, ids

    async def on_checkpoint(self, assistant, hook, turn_id, guide):
        # Guard here as well as in the native adapter: unsupported tools do no work.
        if hook.get("tool_name") not in ALLOWED:
            return
        if hook["tool_use_id"] in self.completed_checkpoints:
            return

        async def run():
            record, text, ids = await asyncio.to_thread(
                self.prepare, assistant, hook, turn_id
            )
            if ids:
                try:
                    receipt = await guide(text)
                except Exception:
                    record["gate_status"] = "delivery_error"
                    append(self.out / "retrieval.jsonl", record)
                    raise
                # Mark only messages accepted by the native guide transport.
                self.seen.update(ids)
                record.update(injected=True, text=text, receipt=receipt)
            append(self.out / "retrieval.jsonl", record)
            self.completed_checkpoints.add(hook["tool_use_id"])
            put(
                self.out / "STATE.json",
                {
                    "seen_sources": sorted(self.seen),
                    "processed_checkpoints": len(self.completed_checkpoints),
                    "initial_sources_excluded": sorted(self.initial_sources),
                },
            )

        try:
            return await asyncio.wait_for(
                run(), self.conf["checkpoint_timeout_seconds"]
            )
        except asyncio.TimeoutError:
            append(
                self.out / "errors.jsonl",
                {
                    "tool_id": hook["tool_use_id"],
                    "error": "checkpoint_timeout",
                    "budget_seconds": self.conf["checkpoint_timeout_seconds"],
                },
            )
            raise

    def metrics(self):
        result = {
            "protocol_revision": REVISION,
            "coarse_top_k": self.conf["coarse_top_k"],
            "checkpoints": 0,
            "gate_calls": 0,
            "gate_abstentions": 0,
            "empty_candidate_skips": 0,
            "injected_memories": 0,
            "gate_errors": 0,
            "gate_prompt_tokens": 0,
            "gate_completion_tokens": 0,
            "gate_cached_tokens": 0,
            "gate_cost_cny": 0.0,
            "tldr_cache_hit": self.tldr_record.get("cache_hit"),
            "tldr_creation_cost_cny": (
                self.tldr_record.get("cost_cny", 0)
                if self.tldr_record.get("cache_hit") is False
                else 0
            ),
        }
        path = self.out / "retrieval.jsonl"
        if not path.exists():
            return result
        for line in path.read_text().splitlines():
            r = json.loads(line)
            u = r.get("gate_usage", {})
            result["checkpoints"] += 1
            result["gate_calls"] += r.get(
                "gate_attempt_count", int(r.get("gate_called", False))
            )
            result["gate_abstentions"] += int(r.get("gate_status") == "abstained")
            result["empty_candidate_skips"] += int(
                r.get("gate_status") == "not_called_empty_candidates"
            )
            result["injected_memories"] += int(r.get("injected", False))
            result["gate_errors"] += int(r.get("gate_status") == "gate_error")
            result["gate_prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            result["gate_completion_tokens"] += u.get("completion_tokens", 0) or 0
            result["gate_cached_tokens"] += (u.get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            ) or 0
            result["gate_cost_cny"] += r.get("gate_cost_cny", 0)
        return result
