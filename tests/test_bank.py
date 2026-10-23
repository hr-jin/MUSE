import hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
from muse.bank import build, bind_initial, initial_card, verify_v8_bank
from muse.initial import build_initial, INITIAL_VERSION, INITIAL_POLICY
from muse.io import put, sha, digest
from muse.multichannel import build_route_bank, Retriever
from test_runtime import memory


class DeterministicEmbedding:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = np.asarray(
            [
                [float(b + 1) for b in hashlib.sha256(text.encode()).digest()[:8]]
                for text in texts
            ],
            dtype=np.float32,
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors, {}


class BankTests(unittest.TestCase):
    def test_initial_binding_route_bank_and_hash_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            train = list("ABCDE")
            cases = {ref: memory(ref) for ref in train}
            embedding = DeterministicEmbedding()
            put(source / "cases.json", cases)
            put(source / "split.json", {"train": train, "test": ["T"]})
            put(
                source / "index.json",
                {
                    "model": "synthetic-embedding",
                    "embedding_url": "http://localhost:1/v1",
                },
            )
            put(
                source / "MANIFEST.json",
                {
                    "files": {
                        name: sha(source / name)
                        for name in ("cases.json", "split.json", "index.json")
                    }
                },
            )
            instructions = {}
            for task in train + ["T"]:
                text = ("Synthetic scientific task " + task
                        + "\n\n## Workspace protocol\nUse the assigned workspace.")
                instructions[task] = {
                    "source_kind": "first_user_prompt_before_memory",
                    "selector": (
                        "training_first_user_prompt"
                        if task in train
                        else "pre_initial_memory_prompt"
                    ),
                    "format": "inline",
                    "text": text,
                    "text_sha256": digest(text),
                }
            manifest = root / "input.json"
            put(
                manifest,
                {
                    "train": train,
                    "test": ["T"],
                    "instructions": instructions,
                    "embedding": {
                        "model": "synthetic-embedding",
                        "url": "http://localhost:1/v1",
                    },
                },
            )
            initial = root / "initial"
            build_initial(
                manifest,
                initial,
                embedding,
                {"muse": lambda task, refs: {"text": "initial " + ",".join(refs)}},
                top_k=5,
            )
            self.assertEqual(embedding.calls, [
                ["Synthetic scientific task " + task for task in train],
                ["Synthetic scientific task T"],
            ])
            saved = json.loads((initial / "instructions.json").read_text())
            self.assertEqual(saved, {task: row["text"] for task, row in instructions.items()})
            sources = json.loads((initial / "SOURCES.json").read_text())
            self.assertEqual(sources["T"]["text_sha256"], digest(instructions["T"]["text"]))
            selection = json.loads((initial / "selection.json").read_text())
            self.assertEqual(selection["T"]["query_sha256"], digest("Synthetic scientific task T"))
            step = root / "step"
            build(source, step, embedder=embedding)
            with self.assertRaises(ValueError):
                verify_v8_bank(step, require_initial=True)
            bound = root / "bound"
            initial_manifest = json.loads((initial / "MANIFEST.json").read_text())
            initial_manifest["version"] = "V8-initial-first-user-prompt-v1"
            initial_manifest["retrieval_policy"] = "first_user_prompt_v8"
            put(initial / "MANIFEST.json", initial_manifest)
            with self.assertRaisesRegex(ValueError, "must be rebuilt"):
                bind_initial(step, initial, root / "old-policy")
            initial_manifest["version"] = INITIAL_VERSION
            initial_manifest["retrieval_policy"] = INITIAL_POLICY
            put(initial / "MANIFEST.json", initial_manifest)
            bind_initial(step, initial, bound)
            verify_v8_bank(bound, require_initial=True)
            self.assertEqual(
                (bound / "cases.json").read_bytes(),
                (source / "cases.json").read_bytes(),
            )
            self.assertEqual(len(initial_card(bound, "T", "muse")["source_tasks"]), 5)
            card = initial_card(bound, "T", "muse")
            self.assertEqual(card["text"], "initial " + ",".join(card["source_tasks"]))
            routes = root / "routes"
            build_route_bank(bound, routes, embedding)
            ranked = Retriever(bound, routes, embedding).rank(
                {"assistant": "current implementation"}, set()
            )
            self.assertLessEqual(len(ranked["ranking"]), 4)
            (bound / "cases.json").write_text("{}")
            with self.assertRaises(ValueError):
                verify_v8_bank(bound)


if __name__ == "__main__":
    unittest.main()
