import unittest
from pathlib import Path

from post_training_eval.local_quick import TaskSpec, batch_tasks, build_lm_eval_command


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


if __name__ == "__main__":
    unittest.main()
