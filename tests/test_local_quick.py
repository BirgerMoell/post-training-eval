import unittest
from pathlib import Path

from post_training_eval.evalchemy_quick_adapter import capped_cpu_count, configure_benchmark, retry_filelock_fork
from post_training_eval.local_quick import (
    SWEEP_TASKS,
    TaskSpec,
    batch_tasks,
    build_evalchemy_command,
    build_lm_eval_command,
    evalchemy_max_tokens,
    select_tasks,
)


class LocalQuickTests(unittest.TestCase):
    def test_batches_never_mix_suite_or_fewshot_protocol(self):
        tasks = [
            TaskSpec("b", 0, "lm-eval-harness"),
            TaskSpec("a", 0, "lm-eval-harness"),
            TaskSpec("c", 5, "lm-eval-harness"),
            TaskSpec("d", 0, "lighteval"),
        ]
        batches = batch_tasks(tasks, 2)
        self.assertEqual([[task.task for task in batch] for batch in batches], [["a", "b"], ["c"], ["d"]])
        self.assertTrue(all(len({(task.suite, task.n_shot) for task in batch}) == 1 for batch in batches))

    def test_evalchemy_batches_are_atomic_per_task(self):
        tasks = [
            TaskSpec("AIME24", 0, "evalchemy"),
            TaskSpec("MATH500", 0, "evalchemy"),
            TaskSpec("GPQADiamond", 0, "evalchemy"),
            TaskSpec("HumanEval", 0, "evalchemy"),
        ]
        batches = batch_tasks(tasks)
        self.assertEqual(
            [[task.task for task in batch] for batch in batches],
            [["GPQADiamond"], ["HumanEval"], ["AIME24"], ["MATH500"]],
        )

    def test_lm_eval_command_is_bounded_deterministic_and_chat_templated(self):
        command = build_lm_eval_command(
            "/venv/bin/python",
            "/models/checkpoint",
            [TaskSpec("arc_easy", 0, "lm-eval-harness")],
            Path("/runs/chunk"),
            Path("/oellm/tasks"),
            8,
        )
        self.assertIn("--apply_chat_template", command)
        self.assertEqual(command[command.index("--batch_size") + 1], "1")
        self.assertEqual(command[command.index("--limit") + 1], "8")
        self.assertIn("dtype=bfloat16", command[command.index("--model_args") + 1])

    def test_evalchemy_command_uses_same_chat_and_precision_protocol(self):
        command = build_evalchemy_command(
            "/evalchemy/bin/python",
            "/models/checkpoint",
            [TaskSpec("MATH500", 0, "evalchemy")],
            Path("/runs/chunk"),
            8,
            "/tokenizers/checkpoint",
        )
        self.assertIn("--apply_chat_template", command)
        self.assertNotIn("--log_samples", command)
        self.assertNotIn("-m", command)
        self.assertTrue(any(value.endswith("evalchemy_quick_adapter.py") for value in command))
        self.assertEqual(command[command.index("--max_tokens") + 1], "2048")
        model_args = command[command.index("--model_args") + 1]
        self.assertIn("dtype=bfloat16", model_args)
        self.assertIn("tokenizer=/tokenizers/checkpoint", model_args)

    def test_fast_profile_uses_short_generation_caps(self):
        math = TaskSpec("MATH500", 0, "evalchemy")
        gpqa = TaskSpec("GPQADiamond", 0, "evalchemy")
        self.assertEqual(evalchemy_max_tokens(math, "fast"), 384)
        self.assertEqual(evalchemy_max_tokens(gpqa, "fast"), 128)
        command = build_evalchemy_command(
            "/evalchemy/bin/python",
            "/models/checkpoint",
            [math],
            Path("/runs/chunk"),
            2,
            profile="fast",
        )
        self.assertEqual(command[command.index("--max_tokens") + 1], "384")

    def test_sweep_selects_exact_curated_registry_tasks(self):
        registry = [TaskSpec(name, 0, "evalchemy" if name[0].isupper() else "lm-eval-harness") for name in SWEEP_TASKS]
        registry.append(TaskSpec("AIME24", 0, "evalchemy"))
        selected = select_tasks(registry, SWEEP_TASKS)
        self.assertEqual([task.task for task in selected], list(SWEEP_TASKS))
        self.assertEqual(evalchemy_max_tokens(TaskSpec("GPQADiamond", 0, "evalchemy"), "sweep"), 128)

    def test_sweep_rejects_unknown_task(self):
        with self.assertRaisesRegex(ValueError, "Unknown oellm-eval task"):
            select_tasks([TaskSpec("ifeval", 0, "lm-eval-harness")], ["missing-task"])

    def test_evalchemy_adapter_caps_questions_and_repetitions(self):
        class Benchmark:
            n_repeat = 10

            def load_questions(self):
                return list(range(20))

        benchmark = Benchmark()
        policy = configure_benchmark(benchmark, question_limit=8, repeat_limit=1)
        self.assertEqual(benchmark.load_questions(), list(range(8)))
        self.assertEqual(benchmark.n_repeat, 1)
        self.assertEqual(policy["sampling"], "load_questions slice")
        self.assertEqual(policy["data_preprocessing_workers"], 4)

    def test_cpu_count_is_temporarily_capped(self):
        import os

        original = os.cpu_count
        with capped_cpu_count(3):
            self.assertLessEqual(os.cpu_count() or 0, 3)
        self.assertIs(os.cpu_count, original)

    def test_evalchemy_adapter_uses_debug_for_inline_loaders(self):
        class InlineBenchmark:
            debug = False

        benchmark = InlineBenchmark()
        policy = configure_benchmark(benchmark, question_limit=8, repeat_limit=1)
        self.assertTrue(benchmark.debug)
        self.assertEqual(policy["sampling"], "upstream debug subset")

    def test_filelock_fork_race_is_retried(self):
        calls = []

        def flaky():
            calls.append(True)
            if len(calls) < 3:
                raise RuntimeError("os.fork is unsafe while filelock is changing descriptor ownership")
            return "saved"

        self.assertEqual(retry_filelock_fork(flaky, attempts=3, delay_seconds=0), "saved")
        self.assertEqual(len(calls), 3)

    def test_unrelated_runtime_error_is_not_retried(self):
        calls = []

        def broken():
            calls.append(True)
            raise RuntimeError("different failure")

        with self.assertRaisesRegex(RuntimeError, "different failure"):
            retry_filelock_fork(broken, attempts=3, delay_seconds=0)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
