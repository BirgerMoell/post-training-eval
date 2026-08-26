import json
import os
import tempfile
import unittest
from pathlib import Path

from post_training_eval.checkpoints import parse_checkpoint
from post_training_eval.planner import build_plan
from post_training_eval.registry import benchmark_index, validate_registry
from post_training_eval.results import ResultError, _task_to_benchmark, build_site_data, ingest_lm_eval, ingest_lm_eval_directory, ingest_oellm_csv, publish_run
from post_training_eval.gates import compare_runs
from post_training_eval.holdouts import _sample_by_bucket


class RegistryResultTests(unittest.TestCase):
    def test_registry_is_coherent(self):
        self.assertEqual(validate_registry(), [])
        self.assertIn("ifeval", benchmark_index())

    def test_plan_preserves_external_gaps(self):
        plan = build_plan(parse_checkpoint("birgermoell/oellm-9b-256k-sft"), "smoke", 3)
        self.assertTrue(any(step["runnable"] for step in plan["steps"]))
        safety = next(step for step in plan["steps"] if step["id"] == "safety-smoke")
        self.assertFalse(safety["runnable"])

    def test_quick_plan_samples_every_registered_oellm_task(self):
        plan = build_plan(parse_checkpoint("owner/model"), "quick")
        self.assertTrue(plan["profile"]["diagnostic"])
        survey = next(step for step in plan["steps"] if step["id"] == "all-oellm-tasks-quick")
        self.assertEqual(survey["task_groups"], ["all"])
        self.assertEqual(survey["command"][-2:], ["--limit", "8"])
        self.assertTrue(survey["runnable"])

    def test_quick_limit_can_be_overridden(self):
        plan = build_plan(parse_checkpoint("owner/model"), "quick", 3)
        survey = next(step for step in plan["steps"] if step["id"] == "all-oellm-tasks-quick")
        self.assertEqual(survey["command"][-2:], ["--limit", "3"])

    def test_quick_endpoint_plan_stratifies_holdouts(self):
        plan = build_plan(parse_checkpoint("openai://http://localhost:8000/v1#model"), "quick")
        survey = next(step for step in plan["steps"] if step["id"] == "all-oellm-tasks-quick")
        holdouts = next(step for step in plan["steps"] if step["id"] == "capability-holdouts-quick")
        self.assertFalse(survey["runnable"])
        self.assertTrue(holdouts["runnable"])
        self.assertEqual(holdouts["command"][-2:], ["--samples-per-bucket", "2"])

    def test_megatron_plan_blocks_eval_until_prepared(self):
        plan = build_plan(parse_checkpoint("megatron:///scratch/model"), "smoke")
        blocked = [step for step in plan["steps"] if step["driver"] != "inspect"]
        self.assertTrue(all(not step["runnable"] for step in blocked))

    def test_lm_eval_ingestion(self):
        raw = {"date": 1787730830.0, "results": {"ifeval": {"prompt_level_strict_acc,none": 0.75}}, "n-samples": {"ifeval": {"effective": 100}}, "versions": {"lm_eval": "0.4.11"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(raw))
            run = ingest_lm_eval(path, "owner/model", "test-run", "abc", "lm_eval ...")
        self.assertEqual(run["metrics"][0]["value"], 75.0)
        self.assertEqual(run["metrics"][0]["capability"], "instruction-chat")
        self.assertEqual(run["started_at"], "2026-08-26T07:53:50+00:00")

    def test_lm_eval_limited_run_is_automatically_diagnostic(self):
        raw = {"config": {"limit": 8.0}, "results": {"ifeval": {"prompt_level_strict_acc,none": 0.5}}, "n-samples": {"ifeval": {"original": 541, "effective": 8}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(raw))
            run = ingest_lm_eval(path, "owner/model", "quick-run", "abc", "lm_eval --limit 8")
        self.assertTrue(run["diagnostic"])
        self.assertIn("directional", run["limitations"][0])

    def test_multilingual_task_mapping_preserves_language(self):
        raw = {"results": {"sib200_swe_Latn": {"acc_norm,none": 0.8}}, "n-samples": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(raw))
            run = ingest_lm_eval(path, "owner/model", "sib-run", "abc", "lm_eval ...")
        self.assertEqual(run["metrics"][0]["benchmark"], "sib-200")
        self.assertEqual(run["metrics"][0]["language"], "swe_Latn")

    def test_all_group_families_have_stable_benchmark_ids(self):
        cases = {
            "xcsqa_deu_Latn": "xcsqa",
            "pawsx_fra_Latn": "paws-x",
            "xnli_bul_Cyrl": "xnli",
            "opensubtitles_multi40_en_to_sv": "opensubtitles-multi40",
            "arc_challenge_mt_sv": "arc-challenge-mt",
            "global_piqa_prompted_swe_latn": "global-piqa",
            "arc_easy": "open-sci",
            "bigbench_operators_generate_until": "dclm-core",
        }
        self.assertEqual({task: _task_to_benchmark(task) for task in cases}, cases)

    def test_lm_eval_directory_ingestion_merges_batches(self):
        first = {"config": {"limit": 8}, "results": {"arc_easy": {"acc_norm,none": 0.5}}, "n-samples": {"arc_easy": {"original": 100, "effective": 8}}}
        second = {"config": {"limit": 8}, "results": {"xnli_bul_Cyrl": {"acc,none": 0.75}}, "n-samples": {"xnli_bul_Cyrl": {"original": 100, "effective": 8}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "results_a.json").write_text(json.dumps(first))
            (root / "b" / "results_b.json").write_text(json.dumps(second))
            run = ingest_lm_eval_directory(root, "owner/model", "quick-merged", "abc", "pteval local-quick --limit 8")
        self.assertEqual({metric["benchmark"] for metric in run["metrics"]}, {"open-sci", "xnli"})
        self.assertTrue(run["diagnostic"])

    def test_lm_eval_translation_metrics_keep_score_scale(self):
        raw = {"results": {"opensubtitles_multi40_en_to_sv": {"bleu,none": 31.5}}, "n-samples": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(raw))
            run = ingest_lm_eval(path, "owner/model", "translation-run", "abc", "lm_eval")
        self.assertEqual(run["metrics"][0]["value"], 31.5)
        self.assertEqual(run["metrics"][0]["scale"], "score")

    def test_oellm_collector_csv_ingestion(self):
        content = "model_name,task,n_shot,performance,metric_name\n/path/model,flores200:swe_Latn-eng_Latn,0,42.5,chrf++\n/path/model,global_mgsm_de,0,0.296,exact_match\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collected.csv"
            path.write_text(content)
            run = ingest_oellm_csv(path, "owner/model", "mixed-run", "abc", "oellm-eval collect")
        self.assertEqual(run["metrics"][0]["scale"], "score")
        self.assertEqual(run["metrics"][1]["value"], 29.6)
        self.assertEqual(run["metrics"][1]["language"], "de")

    def test_oellm_quick_ingestion_is_explicitly_diagnostic(self):
        content = "model_name,task,n_shot,performance,metric_name\n/path/model,global_mgsm_de,0,0.296,exact_match\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collected.csv"
            path.write_text(content)
            run = ingest_oellm_csv(path, "owner/model", "quick-csv", "abc", "oellm-eval --limit 8", diagnostic=True)
        self.assertTrue(run["diagnostic"])

    def test_oellm_limit_command_is_automatically_diagnostic(self):
        content = "model_name,task,n_shot,performance,metric_name\n/path/model,global_mgsm_de,0,0.296,exact_match\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collected.csv"
            path.write_text(content)
            run = ingest_oellm_csv(path, "owner/model", "quick-auto", "abc", "oellm-eval schedule --limit=8")
        self.assertTrue(run["diagnostic"])

    def test_endpoint_quick_sample_covers_every_bucket(self):
        rows = [
            {"id": f"{bucket}-{language}", "bucket": bucket, "language": language}
            for bucket in ("safety", "tools", "grounding")
            for language in ("de", "fr", "sv")
        ]
        sample = _sample_by_bucket(rows, 2)
        self.assertEqual(len(sample), 6)
        self.assertEqual({row["bucket"] for row in sample}, {"safety", "tools", "grounding"})
        self.assertTrue(all(sum(row["bucket"] == bucket for row in sample) == 2 for bucket in ("safety", "tools", "grounding")))

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

    def test_published_run_ids_are_immutable(self):
        run = {"schema_version": 1, "run_id": "immutable-run", "status": "completed", "model": {"id": "owner/model", "revision": "abc", "format": "hf"}, "provenance": {"kind": "fresh-reproduced", "source": "test"}, "metrics": [{"capability": "instruction-chat", "benchmark": "ifeval", "metric": "prompt_level_strict_accuracy", "value": 50, "scale": "percentage"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            first.write_text(json.dumps(run))
            publish_run(first, root)
            changed = dict(run)
            changed["metrics"] = [{**run["metrics"][0], "value": 49}]
            second = root / "second.json"
            second.write_text(json.dumps(changed))
            with self.assertRaises(ResultError):
                publish_run(second, root)


if __name__ == "__main__":
    unittest.main()
