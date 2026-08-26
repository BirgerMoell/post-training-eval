from __future__ import annotations

import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable


LIMIT_ENV = "PTEVAL_EVALCHEMY_QUESTION_LIMIT"
REPEAT_LIMIT_ENV = "PTEVAL_EVALCHEMY_REPEAT_LIMIT"


def _bounded_questions(loader: Callable[..., Any], limit: int) -> Callable[..., Any]:
    @wraps(loader)
    def bounded(*args: Any, **kwargs: Any) -> Any:
        questions = loader(*args, **kwargs)
        effective = min(limit, len(questions))
        if hasattr(questions, "select"):
            return questions.select(range(effective))
        return questions[:effective]

    return bounded


def configure_benchmark(instance: Any, question_limit: int, repeat_limit: int) -> dict[str, Any]:
    """Bound an Evalchemy chat benchmark without changing the upstream checkout."""
    if question_limit < 1 or repeat_limit < 1:
        raise ValueError("Evalchemy quick limits must be positive")

    policy: dict[str, Any] = {
        "question_limit": question_limit,
        "repeat_limit": None,
        "sampling": None,
    }
    loader = getattr(instance, "load_questions", None)
    if callable(loader):
        instance.load_questions = _bounded_questions(loader, question_limit)
        policy["sampling"] = "load_questions slice"
    elif hasattr(instance, "debug"):
        # A few Evalchemy tasks (notably HumanEval) load examples inline. Their
        # native debug mode is the only supported bounded path and uses two
        # examples per task-defined slice (for HumanEval, per language).
        instance.debug = True
        policy["sampling"] = "upstream debug subset"
    else:
        raise RuntimeError(
            f"Cannot enforce a quick sample limit for Evalchemy task {type(instance).__name__}"
        )

    if hasattr(instance, "n_repeat"):
        instance.n_repeat = min(int(instance.n_repeat), repeat_limit)
        policy["repeat_limit"] = instance.n_repeat

    logger = getattr(instance, "logger", None)
    if logger:
        logger.info(
            "pteval diagnostic adapter: sampling=%s, question_limit=%s, repeat_limit=%s",
            policy["sampling"],
            question_limit,
            policy["repeat_limit"],
        )
    return policy


def patch_task_manager(task_manager_class: type[Any], question_limit: int, repeat_limit: int) -> None:
    original_register = task_manager_class._register_benchmark

    @wraps(original_register)
    def register(self: Any, name: str, benchmark_class: type[Any]) -> None:
        original_register(self, name, benchmark_class)
        instance = self.benchmark_instances.get(name)
        if instance is not None:
            configure_benchmark(instance, question_limit, repeat_limit)

    task_manager_class._register_benchmark = register


def _positive_env(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} must be set by pteval local-quick")
    value = int(raw)
    if value < 1:
        raise RuntimeError(f"{name} must be positive")
    return value


def main() -> None:
    question_limit = _positive_env(LIMIT_ENV)
    repeat_limit = _positive_env(REPEAT_LIMIT_ENV)

    # Accelerate executes this adapter by absolute path, so explicitly expose
    # the Evalchemy checkout supplied as the subprocess working directory.
    sys.path.insert(0, str(Path.cwd()))

    # Evalchemy is imported only inside its dedicated environment. Patching the
    # manager before eval.eval constructs it keeps the upstream source pristine.
    from eval.task import TaskManager

    patch_task_manager(TaskManager, question_limit, repeat_limit)

    from eval.eval import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
