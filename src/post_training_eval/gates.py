from __future__ import annotations

from typing import Any

from .registry import benchmark_index


def metric_key(metric: dict[str, Any]) -> tuple[Any, ...]:
    return (metric.get("capability"), metric.get("benchmark"), metric.get("metric"), metric.get("language"), metric.get("slice"))


def compare_runs(candidate: dict[str, Any], baseline: dict[str, Any], max_regression: float = 2.0) -> dict[str, Any]:
    baseline_metrics = {metric_key(metric): metric for metric in baseline.get("metrics", [])}
    benchmarks = benchmark_index()
    comparisons = []
    regressions = []
    targets_missed = []
    for metric in candidate.get("metrics", []):
        key = metric_key(metric)
        base = baseline_metrics.get(key)
        if base and metric.get("scale") == base.get("scale"):
            direction = metric.get("direction") or benchmarks.get(metric["benchmark"], {}).get("direction", "higher")
            raw_delta = float(metric["value"]) - float(base["value"])
            improvement = -raw_delta if direction == "lower" else raw_delta
            comparison = {"key": list(key), "baseline": base["value"], "candidate": metric["value"], "improvement": round(improvement, 6), "direction": direction, "passed": improvement >= -max_regression}
            comparisons.append(comparison)
            if not comparison["passed"]:
                regressions.append(comparison)
        benchmark = benchmarks.get(metric.get("benchmark"), {})
        target = benchmark.get("target")
        if target is not None and metric.get("metric") == benchmark.get("metric"):
            passed = metric["value"] <= target if benchmark.get("direction") == "lower" else metric["value"] >= target
            if not passed:
                targets_missed.append({"key": list(key), "value": metric["value"], "target": target, "direction": benchmark.get("direction")})
    return {
        "schema_version": 1,
        "candidate_run": candidate.get("run_id"),
        "baseline_run": baseline.get("run_id"),
        "max_regression_points": max_regression,
        "status": "failed" if regressions else "passed",
        "comparison_count": len(comparisons),
        "regressions": regressions,
        "targets_missed": targets_missed,
        "comparisons": comparisons,
        "note": "Target misses are reported separately; a development regression gate fails only on comparable metrics beyond tolerance.",
    }

