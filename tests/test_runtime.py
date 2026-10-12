import asyncio, json, tempfile, unittest
from pathlib import Path
import numpy as np
from muse.context import RecentContext
from muse.gate import render_messages
from muse.io import configuration
from muse.key import key_text
from muse.llm import Completion
from muse.multichannel import ROUTES, Retriever, candidate_window, queries
from muse.review_runtime import normalize_review_output, parse_review
from muse.runtime import V8Runtime


def memory(ref):
    return {
        "scientific_problem": "problem " + ref,
        "L1": {
            "final_route": "approach " + ref,
            "submissioned_solution": "code " + ref,
        },
        "L2": [
            {
                "decision": "implementation " + ref,
                "choice_and_reason": "choice " + ref,
                "failure_relation": None,
            }
        ],
        "official_evaluation": {"correct": False, "error": "historical error " + ref},
    }


class Embedder:
    def embed(self, texts):
        v = {
            "assistant": [1.0, 0.0],
            "arguments": [0.0, 1.0],
            "result": [2**-0.5, 2**-0.5],
        }
        return np.asarray([v[t] for t in texts]), {}


def retriever():
    r = Retriever.__new__(Retriever)
    r.cases = {ref: memory(ref) for ref in "ABCD"}
    r.embedder = Embedder()
    r.docs = {
        kind: [
            {
                "unit_id": ref + "-" + kind,
                "source_ref": ref,
                **({"decision_index": 0} if kind == "l2" else {}),
            }
            for ref in "ABCD"
        ]
        for kind in ("l2", "code", "error")
    }
    r.vectors = {
        "l2": np.asarray([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.0, 1.0]]),
        "code": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, 0.8]]),
        "error": np.asarray([[1.0, 0.0], [0.8, 0.6], [2**-0.5, 2**-0.5], [0.0, 1.0]]),
    }
    return r


class FakeLLM:
    model = "glm-5.3-flash"
    effort = "high"

    def __init__(self, response=None):
        self.calls = []
        self.response = response or {
            "inject_ids": ["candidate_1"],
            "review_note": "Check the current array layout against the historical API requirement.",
        }

    def complete(self, messages, json_output=False):
        self.calls.append(messages)
        return Completion(json.dumps(self.response), {}, finish_reason="stop")


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, directory, llm=None, initial=(), gate=True, ranker=None):
        r = ranker or retriever()
        b = Path(directory) / "bank"
        b.mkdir()
        (b / "cases.json").write_text(json.dumps(r.cases))
        return V8Runtime(
            b,
            Path(directory) / "out",
            {"tldr": "Solve the current scientific task"},
            llm or FakeLLM(),
            {"gate_enabled": gate},
            retriever=r,
            initial_source_refs=list(initial),
        )

    def hook(self, ident="tool-1", name="Bash"):
        return {
            "tool_name": name,
            "tool_use_id": ident,
            "tool_input": {"command": "arguments", "content": "arguments"},
            "tool_response": "result",
        }

    def test_config_is_four_top1_and_supports_no_gate(self):
        c = configuration({"gate_enabled": False})
        self.assertEqual(c["coarse_top_k"], 1)
        self.assertEqual(c["max_merged_candidates"], 4)
        self.assertFalse(c["evaluation_in_key"])
        self.assertEqual(len(ROUTES), 4)

    def test_key_contains_only_implementation_and_choice(self):
        m = memory("A")
        self.assertEqual(key_text(m, m["L2"][0]), "implementation A\n\nchoice A")

    def test_queries_only_current_interaction(self):
        c = RecentContext(1)
        c.observe({"content": "OLD"}, self.hook("old"), "old")
        q = queries(
            c.observe(
                {"content": "assistant", "reasoning": "PRIVATE"}, self.hook(), "turn"
            )
        )
        self.assertEqual(
            q, {"assistant": "assistant", "arguments": "arguments", "result": "result"}
        )
        q = queries(
            c.observe({"content": "assistant"}, self.hook(name="Write"), "turn")
        )
        self.assertEqual(q, {"assistant": "assistant", "arguments": "arguments"})

    def test_four_top1_routes_merge_by_source(self):
        out = retriever().rank(
            {"assistant": "assistant", "arguments": "arguments", "result": "result"},
            set(),
        )
        self.assertEqual(
            [r["source_ref"] for r in out["ranking"]], ["A", "B", "D", "C"]
        )
        self.assertEqual(len(out["routes"]), 4)
        self.assertTrue(all(len(r["top"]) == 1 for r in out["routes"]))

    def test_initial_exclusion_precedes_ranking(self):
        out = retriever().rank({"assistant": "assistant"}, {"A"})
        self.assertEqual(out["ranking"][0]["source_ref"], "B")
        self.assertNotIn("A", [r["source_ref"] for r in out["ranking"]])

    def test_delivered_source_dedup_has_no_backfill(self):
        r = retriever()
        out = r.rank({"assistant": "assistant"}, set())
        self.assertEqual(candidate_window(out["ranking"], r.cases, {"A"}), ([], []))

    def test_gate_sees_history_and_delivers_review_plus_original(self):
        llm = FakeLLM()
        with tempfile.TemporaryDirectory() as t:
            r = self.make_runtime(t, llm)
            r.set_public_history(
                [
                    {
                        "turn_id": "prior",
                        "assistant_text": "previous visible action",
                        "tools": [],
                    }
                ]
            )
            record, text, ids = r.prepare({"content": "assistant"}, self.hook(), "turn")
            self.assertEqual(ids, ["A"])
            self.assertIn("Check the current array layout", text)
            self.assertIn("code A", text)
            self.assertIn("historical error A", text)
            self.assertIn("previous visible action", llm.calls[0][1]["content"])
            self.assertIn("arguments", llm.calls[0][1]["content"])
            self.assertTrue(record["review_note_delivered"])
            self.assertEqual(r.seen, set())

    def test_no_gate_highest_similarity_single_source(self):
        ranker = retriever()
        ranker.vectors["l2"][0] = [0.9, 0.1]
        ranker.vectors["error"][2] = [1.0, 0.0]
        with tempfile.TemporaryDirectory() as t:
            r = self.make_runtime(t, gate=False, ranker=ranker)
            record, text, ids = r.prepare({"content": "assistant"}, self.hook(), "turn")
            self.assertEqual(ids, ["B"])
            self.assertIn("code B", text)
            self.assertFalse(record["gate_called"])
            self.assertNotIn("Review of the current work", text)

    def test_abstention_skips_injection(self):
        with tempfile.TemporaryDirectory() as t:
            r = self.make_runtime(t, FakeLLM({"inject_ids": [], "review_note": ""}))
            record, text, ids = r.prepare({"content": "assistant"}, self.hook(), "turn")
            self.assertEqual((text, ids), ("", []))
            self.assertEqual(record["gate_status"], "abstained")

    def test_seen_only_after_acknowledged_delivery(self):
        with tempfile.TemporaryDirectory() as t:
            r = self.make_runtime(t)

            async def guide(text):
                self.assertNotIn("A", r.seen)
                return {"delivered": True}

            asyncio.run(
                r.on_checkpoint({"content": "assistant"}, self.hook(), "turn", guide)
            )
            self.assertEqual(r.seen, {"A"})

    def test_failed_delivery_keeps_source_eligible(self):
        with tempfile.TemporaryDirectory() as t:
            r = self.make_runtime(t)

            async def guide(text):
                raise RuntimeError("delivery failed")

            with self.assertRaises(RuntimeError):
                asyncio.run(
                    r.on_checkpoint(
                        {"content": "assistant"}, self.hook(), "turn", guide
                    )
                )
            self.assertEqual(r.seen, set())

    def test_native_snapshot_excludes_private_reasoning_current_and_future_tools(self):
        llm = FakeLLM()
        previous = {
            "info": {"role": "assistant", "id": "prior"},
            "parts": [
                {"type": "text", "text": "PUBLIC PRIOR", "time": {"end": 20}},
                {"type": "reasoning", "text": "PRIVATE THOUGHT"},
                {
                    "type": "tool",
                    "callID": "old-tool",
                    "tool": "Bash",
                    "state": {
                        "status": "completed",
                        "input": {"command": "old"},
                        "output": "PUBLIC OUTPUT",
                        "time": {"start": 10, "end": 20},
                    },
                },
            ],
        }
        current = {
            "info": {"role": "assistant", "id": "turn"},
            "parts": [
                {"type": "text", "text": "assistant", "time": {"end": 90}},
                {
                    "type": "tool",
                    "callID": "tool-1",
                    "tool": "Bash",
                    "state": {
                        "status": "completed",
                        "input": self.hook()["tool_input"],
                        "output": "result",
                        "time": {"start": 100, "end": 110},
                    },
                },
            ],
        }
        future = {
            "info": {"role": "assistant", "id": "future"},
            "parts": [
                {
                    "type": "text",
                    "text": "FUTURE HIDDEN FROM GATE",
                    "time": {"end": 200},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as t:
            r = self.make_runtime(t, llm)
            r.public_message_snapshot = [previous, current, future]
            r.prepare({"content": "assistant"}, self.hook(), "turn")
            user = llm.calls[0][1]["content"]
            self.assertIn("PUBLIC PRIOR", user)
            self.assertIn("PUBLIC OUTPUT", user)
            self.assertNotIn("PRIVATE THOUGHT", user)
            self.assertNotIn("FUTURE HIDDEN FROM GATE", user)
            self.assertEqual(user.count('"call_id": "tool-1"'), 1)

    def test_dynamic_gate_schema_and_strict_review_validation(self):
        candidates = [{"candidate_id": "a"}, {"candidate_id": "b"}]
        m = render_messages("summary", [], candidates)
        self.assertIn('"enum": ["a", "b"]', m[0]["content"])
        for invalid in [
            {"inject_ids": ["a", "b"], "review_note": "check"},
            {"inject_ids": ["a"], "review_note": ""},
            {"inject_ids": [], "review_note": "unselected advice"},
            {"inject_ids": ["unknown"], "review_note": "check"},
        ]:
            with self.assertRaises(ValueError):
                parse_review(json.dumps(invalid), candidates)
        obj = {"inject_ids": ["a"], "review_note": "check this operation"}
        normalized, reason = normalize_review_output(
            "prefix " + json.dumps(obj) + " suffix"
        )
        self.assertEqual(json.loads(normalized), obj)
        self.assertEqual(reason, "explicit_json_extracted")


if __name__ == "__main__":
    unittest.main()
