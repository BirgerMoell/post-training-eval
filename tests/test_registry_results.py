import json
import os
import tempfile
import unittest
from pathlib import Path

from post_training_eval.checkpoints import parse_checkpoint
from post_training_eval.planner import build_plan
from post_training_eval.registry import benchmark_index, validate_registry
from post_training_eval.results import build_site_data, ingest_lm_eval
from post_training_eval.gates import compare_runs


class RegistryResultTests(unittest.TestCase):
    def test_registry_is_coherent(self):
        self.assertEqual(validate_registry(), [])
        self.assertIn("ifeval", benchmark_index())

    def test_plan_preserves_external_gaps(self):
        plan = build_plan(parse_checkpoint("birgermoell/oellm-9b-256k-sft"), "smoke", 3)
        self.assertTrue(any(step["runnable"] for step in plan["steps"]))
        safety = next(step for step in plan["steps"] if step["id"] == "safety-smoke")
        self.assertFalse(safety["runnable"])

    def test_megatron_plan_blocks_eval_until_prepared(self):
        plan = build_plan(parse_checkpoint("megatron:///scratch/model"), "smoke")
        blocked = [step for step in plan["steps"] if step["driver"] != "inspect"]
        self.assertTrue(all(not step["runnable"] for step in blocked))

    def test_lm_eval_ingestion(self):
        raw = {"results": {"ifeval": {"prompt_level_strict_acc,none": 0.75}}, "n-samples": {"ifeval": {"effective": 100}}, "versions": {"lm_eval": "0.4.11"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(raw))
            run = ingest_lm_eval(path, "owner/model", "test-run", "abc", "lm_eval ...")
        self.assertEqual(run["metrics"][0]["value"], 75.0)
        self.assertEqual(run["metrics"][0]["capability"], "instruction-chat")

    def test_site_has_models(self):
        data = build_site_data()
        self.assertGreaterEqual(len(data["models"]), 2)
        self.assertGreater(len(data["runs"]), 0)

    def test_gate_detects_regression(self):
        baseline = {"run_id": "base", "metrics": [{"capability": "instruction-chat", "benchmark": "ifeval", "metric": "prompt_level_strict_accuracy", "value": 50, "scale": "percentage", "direction": "higher"}]}
        candidate = {"run_id": "candidate", "metrics": [{"capability": "instruction-chat", "benchmark": "ifeval", "metric": "prompt_level_strict_accuracy", "value": 45, "scale": "percentage", "direction": "higher"}]}
        report = compare_runs(candidate, baseline, 2)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["regressions"][0]["improvement"], -5)


if __name__ == "__main__":
    unittest.main()
