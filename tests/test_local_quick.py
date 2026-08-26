import unittest
from pathlib import Path

from post_training_eval.evalchemy_quick_adapter import configure_benchmark
from post_training_eval.local_quick import TaskSpec, batch_tasks, build_evalchemy_command, build_lm_eval_command


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

    def test_evalchemy_batches_never_mix_generation_caps(self):
        tasks = [
            TaskSpec("AIME24", 0, "evalchemy"),
            TaskSpec("MATH500", 0, "evalchemy"),
            TaskSpec("GPQADiamond", 0, "evalchemy"),
            TaskSpec("HumanEval", 0, "evalchemy"),
        ]
        batches = batch_tasks(tasks)
        self.assertEqual(
            [[task.task for task in batch] for batch in batches],
            [["HumanEval"], ["GPQADiamond"], ["AIME24", "MATH500"]],
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
        self.assertIn("--log_samples", command)
        self.assertNotIn("-m", command)
        self.assertTrue(any(value.endswith("evalchemy_quick_adapter.py") for value in command))
        self.assertEqual(command[command.index("--max_tokens") + 1], "2048")
        model_args = command[command.index("--model_args") + 1]
        self.assertIn("dtype=bfloat16", model_args)
        self.assertIn("tokenizer=/tokenizers/checkpoint", model_args)

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

    def test_evalchemy_adapter_uses_debug_for_inline_loaders(self):
        class InlineBenchmark:
            debug = False

        benchmark = InlineBenchmark()
        policy = configure_benchmark(benchmark, question_limit=8, repeat_limit=1)
        self.assertTrue(benchmark.debug)
        self.assertEqual(policy["sampling"], "upstream debug subset")


if __name__ == "__main__":
    unittest.main()
