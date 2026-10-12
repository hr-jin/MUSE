"""Offline integration checks for package construction and CLI entry points."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muse.bank import verify_v8_bank
from muse.construction import empty
from muse.multichannel import Retriever
from muse.pipeline import build_artifacts, get_initial


class Embedder:
    model = "test-embedding"
    url = "http://localhost:18081/v1"

    def embed(self, texts):
        rows = []
        for text in texts:
            values = np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8)
            values = values.astype(np.float32) + 1
            rows.append(values / np.linalg.norm(values))
        return np.asarray(rows, dtype=np.float32), {}


def memory(label):
    return {
        "scientific_problem": "Synthetic scientific task " + label,
        "L1": {"final_route": "Use a suitable numerical scheme.",
               "submissioned_solution": "synthetic implementation " + label},
        "L2": [{"decision": "Prepare coordinate arrays.",
                "choice_and_reason": "Use the array layout required by the interface.",
                "failure_relation": None}],
        "official_evaluation": {"error": "Synthetic shape diagnostic " + label},
    }


class PipelineTests(unittest.TestCase):
    def test_frozen_artifacts_initial_and_four_route_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            cases = {"historical-a": memory("a"), "historical-b": memory("b")}
            artifacts = build_artifacts(cases, {"target": "Solve this scientific task"},
                                        output, Embedder(), empty(), k=1)
            verify_v8_bank(artifacts["bank"], require_initial=True)
            card = get_initial(artifacts["bank"], "target")
            self.assertEqual(len(card["source_tasks"]), 1)
            self.assertIn("GlobalMemBank", card["text"])
            ranker = Retriever(artifacts["bank"], artifacts["routes"], Embedder())
            ranked = ranker.rank({"assistant": "Prepare coordinate arrays."}, card["source_tasks"])
            self.assertTrue(ranked["ranking"])
            self.assertTrue(all(row["source_ref"] not in card["source_tasks"]
                                for row in ranked["ranking"]))
            with self.assertRaises(FileExistsError):
                build_artifacts(cases, {"target": "task"}, output, Embedder(), k=1)

    def test_target_overlap_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            with self.assertRaises(ValueError):
                build_artifacts({"same": memory("a")}, {"same": "task"}, output, Embedder(), k=1)
            self.assertFalse(output.exists())

    def test_cli_help_does_not_require_provider_credentials(self):
        result = subprocess.run([sys.executable, "-m", "muse.cli", "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("build-bank", result.stdout)
        self.assertIn("runtime", result.stdout)


if __name__ == "__main__":
    unittest.main()
