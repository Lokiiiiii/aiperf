# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Confidence aggregation strategy for multi-run results."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from aiperf.orchestrator.aggregation.base import AggregateResult, AggregationStrategy
from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConfidenceMetric:
    """Statistics for a single metric across runs.

    Attributes:
        mean: Sample mean
        std: Sample standard deviation (ddof=1)
        min: Minimum value
        max: Maximum value
        cv: Coefficient of variation (std/mean)
        se: Standard error (std/sqrt(n))
        ci_low: Lower bound of confidence interval
        ci_high: Upper bound of confidence interval
        t_critical: t-distribution critical value used for CI
        unit: Unit of measurement (e.g., "ms", "requests/sec")
    """

    mean: float
    std: float
    min: float
    max: float
    cv: float
    se: float
    ci_low: float
    ci_high: float
    t_critical: float
    unit: str

    def to_json_result(self):
        """Convert to JsonMetricResult for export.

        Maps confidence statistics to JSON export format:
        - mean → avg (mean of run-level averages)
        - std → std (std of run-level averages)
        - min/max → min/max (across runs)

        Confidence-specific fields (cv, se, ci_low, ci_high, t_critical)
        are added as extra fields via JsonExportData's extra="allow" setting.

        Returns:
            JsonMetricResult compatible with existing exporters
        """
        from aiperf.common.models.export_models import JsonMetricResult

        return JsonMetricResult(
            unit=self.unit,
            avg=self.mean,
            std=self.std,
            min=self.min,
            max=self.max,
        )


class ConfidenceAggregation(AggregationStrategy):
    """Aggregation strategy for confidence reporting.

    Computes mean, std, CV, and confidence intervals for each metric.

    Attributes:
        confidence_level: Confidence level for intervals (default: 0.95)
    """

    def __init__(self, confidence_level: float = 0.95):
        """Initialize ConfidenceAggregation.

        Args:
            confidence_level: Confidence level for intervals (0 < level < 1)

        Raises:
            ValueError: If confidence_level is not between 0 and 1
        """
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"Invalid confidence level: {confidence_level}. "
                "Confidence level must be between 0 and 1 (exclusive). "
                "Common values: 0.90 (90%), 0.95 (95%), 0.99 (99%)."
            )
        self.confidence_level = confidence_level

    def get_aggregation_type(self) -> str:
        """Return aggregation type identifier."""
        return "confidence"

    def aggregate(self, results: list[RunResult]) -> AggregateResult:
        """Aggregate results for confidence reporting.

        Args:
            results: List of RunResult from orchestrator

        Returns:
            AggregateResult with confidence statistics

        Raises:
            ValueError: If fewer than 2 successful runs
        """
        # Separate successful and failed runs
        successful = [r for r in results if r.success]
        failed = [
            {"label": r.label, "error": r.error} for r in results if not r.success
        ]

        if len(successful) < 2:
            if len(successful) == 0:
                raise ValueError(
                    "All runs failed - cannot compute confidence statistics. "
                    f"Total runs: {len(results)}, Failed runs: {len(failed)}. "
                    "Please check the error messages in the logs and ensure your "
                    "benchmark configuration is correct."
                )
            else:
                raise ValueError(
                    f"Insufficient successful runs for confidence intervals. "
                    f"Got {len(successful)} successful run(s), but need at least 2. "
                    f"Total runs: {len(results)}, Failed runs: {len(failed)}. "
                    "Consider increasing --num-profile-runs or investigating why runs are failing."
                )

        # Aggregate each metric
        metrics = self._aggregate_metrics(successful)

        return AggregateResult(
            aggregation_type="confidence",
            num_runs=len(results),
            num_successful_runs=len(successful),
            failed_runs=failed,
            metrics=metrics,
            metadata={
                "confidence_level": self.confidence_level,
                "run_labels": [r.label for r in successful],
            },
        )

    def _aggregate_metrics(
        self, results: list[RunResult]
    ) -> dict[str, ConfidenceMetric]:
        """Aggregate each metric across runs.

        Args:
            results: List of successful RunResult

        Returns:
            Dict mapping metric name to ConfidenceMetric
        """
        # Get all metric names from first result
        if not results or not results[0].summary_metrics:
            return {}

        metric_names = results[0].summary_metrics.keys()

        aggregated = {}
        for metric_name in metric_names:
            # Extract values for this metric across all runs
            values = [
                r.summary_metrics[metric_name]
                for r in results
                if metric_name in r.summary_metrics
            ]

            if not values:
                logger.warning(f"Metric {metric_name} not found in any run")
                continue

            # Compute statistics
            aggregated[metric_name] = self._compute_confidence_stats(
                values, metric_name
            )

        return aggregated

    def _compute_confidence_stats(
        self, values: list[float], metric_name: str
    ) -> ConfidenceMetric:
        """Compute confidence statistics for a single metric.

        Args:
            values: List of metric values across runs
            metric_name: Name of the metric (e.g., "time_to_first_token_avg")

        Returns:
            ConfidenceMetric with computed statistics
        """
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))  # Sample std (N-1)

        # Coefficient of variation (handle division by zero)
        # CV is expressed as a ratio (not percentage), so no *100
        cv = std / mean if mean != 0 else float("inf")

        # Standard error
        se = std / np.sqrt(n)

        # Confidence interval using t-distribution
        alpha = 1 - self.confidence_level
        df = n - 1
        t_critical = float(stats.t.ppf(1 - alpha / 2, df))

        margin = t_critical * se
        ci_low = mean - margin
        ci_high = mean + margin

        # Get unit from metric definition
        unit = self._get_metric_unit(metric_name)

        return ConfidenceMetric(
            mean=mean,
            std=std,
            min=float(min(values)),
            max=float(max(values)),
            cv=cv,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            t_critical=t_critical,
            unit=unit,
        )

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit from metric definition in MetricRegistry.

        Extracts the metric tag from the full metric name (e.g., "time_to_first_token_avg"
        → "time_to_first_token") and looks up the display unit from the MetricRegistry.

        Args:
            metric_name: Full metric name with statistical suffix (e.g., "time_to_first_token_avg")

        Returns:
            Unit string from metric definition (e.g., "ms", "requests/sec")
            Returns empty string if metric not found in registry.
        """
        from aiperf.metrics.metric_registry import MetricRegistry

        # Extract metric tag by removing statistical suffix (_avg, _p50, _p90, etc.)
        # Common suffixes: _avg, _min, _max, _std, _p50, _p90, _p95, _p99, _count
        stat_suffixes = [
            "_avg",
            "_min",
            "_max",
            "_std",
            "_p50",
            "_p90",
            "_p95",
            "_p99",
            "_count",
        ]

        metric_tag = metric_name
        for suffix in stat_suffixes:
            if metric_name.endswith(suffix):
                metric_tag = metric_name[: -len(suffix)]
                break

        try:
            # Get metric instance from registry
            metric = MetricRegistry.get_instance(metric_tag)

            # Use display_unit if available, otherwise use base unit
            unit = metric.display_unit if metric.display_unit else metric.unit

            # Convert unit enum to string if needed
            return str(unit) if unit else ""

        except Exception:
            # Metric not found in registry - log warning and return empty string
            logger.warning(
                f"Metric '{metric_tag}' not found in MetricRegistry. "
                f"Cannot determine unit for '{metric_name}'. Using empty string."
            )
            return ""
