# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep aggregation for parameter sweeping."""

from enum import Enum
from typing import NamedTuple


class OptimizationDirection(Enum):
    """Direction of optimization for a metric.

    Note: This is defined here for the parameter sweeping feature. Ideally, this
    would be a property of BaseMetric itself (e.g., BaseMetric.optimization_direction).
    See "Future Extensions" section for discussion of adding this to the metrics system.
    """

    MAXIMIZE = "maximize"  # Higher is better (e.g., throughput)
    MINIMIZE = "minimize"  # Lower is better (e.g., latency)


class Objective(NamedTuple):
    """Definition of an optimization objective.

    Args:
        metric_key: The metric tag (e.g., "request_throughput_avg", "ttft_p99_ms")
        direction: Whether to maximize or minimize this metric

    Note: In the future, direction could be derived from BaseMetric.optimization_direction
    if that property is added to the metrics system.
    """

    metric_key: str
    direction: OptimizationDirection


# Default objectives for common use case
DEFAULT_PARETO_OBJECTIVES = [
    Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
    Objective("ttft_p99_ms", OptimizationDirection.MINIMIZE),
]


def identify_pareto_optimal(
    per_value_stats: dict[int, dict],
    objectives: list[Objective] | None = None,
) -> list[int]:
    """Identify Pareto optimal configurations across N objectives.

    A configuration is Pareto optimal if no other configuration is strictly better
    on ALL objectives simultaneously.

    Args:
        per_value_stats: Statistics for each sweep value
        objectives: List of objectives to optimize. If None, uses DEFAULT_PARETO_OBJECTIVES
            (throughput vs latency).

            Example: [
                Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
                Objective("ttft_p99_ms", OptimizationDirection.MINIMIZE),
            ]

    Returns:
        List of sweep values that are Pareto optimal

    Example:
        # 2D Pareto frontier (throughput vs latency) - uses defaults
        pareto = identify_pareto_optimal(stats)

        # 3D Pareto frontier (throughput, latency, cost)
        objectives = [
            Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
            Objective("ttft_p99_ms", OptimizationDirection.MINIMIZE),
            Objective("cost_per_request", OptimizationDirection.MINIMIZE),
        ]
        pareto = identify_pareto_optimal(stats, objectives)
    """
    if objectives is None:
        objectives = DEFAULT_PARETO_OBJECTIVES

    pareto_optimal = []

    for value1, stats1 in per_value_stats.items():
        # Extract objective values for this configuration
        values1 = [stats1[obj.metric_key]["mean"] for obj in objectives]

        is_dominated = False
        for value2, stats2 in per_value_stats.items():
            if value1 == value2:
                continue

            # Extract objective values for comparison configuration
            values2 = [stats2[obj.metric_key]["mean"] for obj in objectives]

            # Check if value2 dominates value1
            # Domination means: better or equal on all objectives, AND strictly better on at least one
            better_or_equal_count = 0
            strictly_better_count = 0

            for i, obj in enumerate(objectives):
                if obj.direction == OptimizationDirection.MAXIMIZE:
                    if values2[i] > values1[i]:
                        strictly_better_count += 1
                        better_or_equal_count += 1
                    elif values2[i] == values1[i]:
                        better_or_equal_count += 1
                    # else: values2[i] < values1[i], worse on this objective
                else:  # MINIMIZE
                    if values2[i] < values1[i]:
                        strictly_better_count += 1
                        better_or_equal_count += 1
                    elif values2[i] == values1[i]:
                        better_or_equal_count += 1
                    # else: values2[i] > values1[i], worse on this objective

            # value2 dominates value1 if it's better or equal on all AND strictly better on at least one
            if better_or_equal_count == len(objectives) and strictly_better_count > 0:
                is_dominated = True
                break

        if not is_dominated:
            pareto_optimal.append(value1)

    return sorted(pareto_optimal)


def analyze_trends(
    per_value_stats: dict[int, dict],
    sweep_values: list[int],
    metric_key: str,
) -> dict:
    """Analyze how a metric changes across sweep values.

    Args:
        per_value_stats: Statistics for each sweep value
        sweep_values: List of sweep values in order
        metric_key: The metric to analyze (e.g., "request_throughput_avg")

    Returns:
        Dictionary with:
            - inflection_points: List of sweep values where trend changes significantly
            - rate_of_change: List of changes between consecutive values

    Note: Pattern (increasing/decreasing/plateau/mixed) is derivable from rate_of_change:
        - All positive → increasing
        - All negative → decreasing
        - All near zero → plateau
        - Mixed signs → mixed

    Example:
        >>> per_value_stats = {
        ...     10: {"request_throughput_avg": {"mean": 100}},
        ...     20: {"request_throughput_avg": {"mean": 180}},
        ...     30: {"request_throughput_avg": {"mean": 270}},
        ...     40: {"request_throughput_avg": {"mean": 285}},
        ... }
        >>> result = analyze_trends(per_value_stats, [10, 20, 30, 40], "request_throughput_avg")
        >>> result["rate_of_change"]
        [80.0, 90.0, 15.0]
        >>> result["inflection_points"]
        [40]
    """
    # Extract metric values in sweep order
    values = [per_value_stats[v][metric_key]["mean"] for v in sweep_values]

    # Compute rate of change between consecutive values
    rate_of_change = []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        rate_of_change.append(delta)

    # Identify inflection points (where rate of change changes significantly)
    inflection_points = []
    for i in range(1, len(rate_of_change)):
        prev_rate = rate_of_change[i - 1]
        curr_rate = rate_of_change[i]

        # Significant change in rate: sign flip or magnitude change > 50%
        # Sign flip: product is negative
        has_sign_flip = prev_rate * curr_rate < 0

        # Magnitude change > 50% (avoid division by zero)
        has_magnitude_change = False
        if prev_rate != 0:
            has_magnitude_change = abs(curr_rate - prev_rate) > 0.5 * abs(prev_rate)

        if has_sign_flip or has_magnitude_change:
            # Inflection point is at sweep_values[i+1] (the value where the new rate starts)
            inflection_points.append(sweep_values[i + 1])

    return {
        "inflection_points": inflection_points,
        "rate_of_change": rate_of_change,
    }


class SweepAggregation:
    """Compute sweep-level statistics and analysis."""

    @staticmethod
    def compute(
        per_value_stats: dict[int, dict],
        sweep_values: list[int],
        parameter_name: str = "concurrency",
    ) -> dict:
        """Compute sweep-level aggregate statistics.

        Args:
            per_value_stats: Statistics for each sweep value (value -> confidence stats)
            sweep_values: List of sweep values in order
            parameter_name: Name of the parameter being swept (default: "concurrency")

        Returns:
            Dictionary with:
                - metadata: Parameter name, values, and counts
                - per_value_metrics: Metrics for each sweep value
                - best_configurations: Best values for key metrics
                - pareto_optimal: List of Pareto optimal sweep values
                - trends: Trend analysis for key metrics

        Example:
            >>> per_value_stats = {
            ...     10: {"request_throughput_avg": {"mean": 100, "std": 5, ...}},
            ...     20: {"request_throughput_avg": {"mean": 180, "std": 8, ...}},
            ... }
            >>> result = SweepAggregation.compute(per_value_stats, [10, 20], "concurrency")
            >>> result["metadata"]["num_values"]
            2
            >>> result["best_configurations"]["best_throughput"]["value"]
            20
        """
        # Build metadata section
        metadata = {
            "parameter_name": parameter_name,
            "parameter_values": sweep_values,
            "num_values": len(sweep_values),
        }

        # Per-value metrics section (convert int keys to strings for JSON compatibility)
        per_value_metrics = {
            str(value): stats for value, stats in per_value_stats.items()
        }

        # Identify best configurations
        best_configurations = {}
        if per_value_stats:
            # Best throughput: highest request_throughput_avg
            if all(
                "request_throughput_avg" in stats for stats in per_value_stats.values()
            ):
                best_throughput_value = max(
                    per_value_stats.items(),
                    key=lambda item: item[1]["request_throughput_avg"]["mean"],
                )
                best_configurations["best_throughput"] = {
                    "value": best_throughput_value[0],
                    "metric": best_throughput_value[1]["request_throughput_avg"][
                        "mean"
                    ],
                    "unit": best_throughput_value[1]["request_throughput_avg"].get(
                        "unit", "requests/sec"
                    ),
                }

            # Best latency: lowest ttft_p99_ms (or request_latency_p99 as fallback)
            latency_metric = None
            if all("ttft_p99_ms" in stats for stats in per_value_stats.values()):
                latency_metric = "ttft_p99_ms"
            elif all(
                "request_latency_p99" in stats for stats in per_value_stats.values()
            ):
                latency_metric = "request_latency_p99"

            if latency_metric:
                best_latency_value = min(
                    per_value_stats.items(),
                    key=lambda item: item[1][latency_metric]["mean"],
                )
                best_configurations["best_latency_p99"] = {
                    "value": best_latency_value[0],
                    "metric": best_latency_value[1][latency_metric]["mean"],
                    "unit": best_latency_value[1][latency_metric].get("unit", "ms"),
                }

        # Identify Pareto optimal configurations
        # Only compute if we have the required metrics for default objectives in ALL sweep values
        pareto_optimal = []
        if per_value_stats:
            # Check if all required metrics are present in ALL sweep values
            has_required_metrics = all(
                all(obj.metric_key in stats for obj in DEFAULT_PARETO_OBJECTIVES)
                for stats in per_value_stats.values()
            )
            if has_required_metrics:
                pareto_optimal = identify_pareto_optimal(per_value_stats)
            else:
                # Try fallback objectives (request_latency_p99 instead of ttft_p99_ms)
                fallback_objectives = [
                    Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
                    Objective("request_latency_p99", OptimizationDirection.MINIMIZE),
                ]
                has_fallback_metrics = all(
                    all(obj.metric_key in stats for obj in fallback_objectives)
                    for stats in per_value_stats.values()
                )
                if has_fallback_metrics:
                    pareto_optimal = identify_pareto_optimal(
                        per_value_stats, fallback_objectives
                    )

        # Analyze trends for key metrics
        trends = {}
        if per_value_stats and len(sweep_values) > 1:
            # Get the first stats dict to determine available metrics
            first_stats = next(iter(per_value_stats.values()))

            # Key metrics to analyze for trends (with fallbacks)
            key_metrics = ["request_throughput_avg"]

            # Add latency metric (prefer ttft_p99_ms, fallback to request_latency_p99)
            if "ttft_p99_ms" in first_stats:
                key_metrics.append("ttft_p99_ms")
            elif "request_latency_p99" in first_stats:
                key_metrics.append("request_latency_p99")

            for metric_key in key_metrics:
                # Check if this metric exists in all sweep values
                if all(metric_key in stats for stats in per_value_stats.values()):
                    trends[metric_key] = analyze_trends(
                        per_value_stats, sweep_values, metric_key
                    )

        return {
            "metadata": metadata,
            "per_value_metrics": per_value_metrics,
            "best_configurations": best_configurations,
            "pareto_optimal": pareto_optimal,
            "trends": trends,
        }
