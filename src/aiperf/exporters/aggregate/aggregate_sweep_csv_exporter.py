# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSV exporter for sweep aggregate results."""

import csv
import io

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateSweepCsvExporter(AggregateBaseExporter):
    """Exports sweep aggregate results to CSV format.

    Creates a CSV with multiple sections:
    - Per-value metrics table (one row per sweep value)
    - Blank line separator
    - Best configurations section
    - Pareto optimal points section
    - Trends section
    - Metadata section

    Uses similar formatting approach as AggregateConfidenceCsvExporter for consistency.
    """

    def get_file_name(self) -> str:
        """Return CSV file name.

        Returns:
            str: "profile_export_aiperf_sweep.csv"
        """
        return "profile_export_aiperf_sweep.csv"

    def _generate_content(self) -> str:
        """Generate CSV content from sweep aggregate result.

        The result contains:
        - result.metadata: Contains sweep metadata + best_configurations, pareto_optimal, trends
        - result.metrics: Contains per_value_metrics (the actual metrics dict)

        Format (long format for easy analysis):
        Each row represents one metric for one sweep value:
        concurrency,metric,mean,std,min,max,cv,se,ci_low,ci_high,unit

        Returns:
            str: CSV content string
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Get parameter name from metadata (default to "concurrency")
        param_name = self._result.metadata.get("parameter_name", "concurrency")

        # Write header
        header = [
            param_name,
            "metric",
            "mean",
            "std",
            "min",
            "max",
            "cv",
            "se",
            "ci_low",
            "ci_high",
            "unit",
        ]
        writer.writerow(header)

        # Per-value metrics are in result.metrics
        per_value_metrics = self._result.metrics

        # Write data rows (one row per metric per sweep value)
        for value in sorted(per_value_metrics.keys(), key=int):
            metrics = per_value_metrics[value]

            for metric_name in sorted(metrics.keys()):
                metric_data = metrics[metric_name]

                if isinstance(metric_data, dict):
                    row = [
                        value,
                        metric_name,
                        self._format_number(metric_data.get("mean")),
                        self._format_number(metric_data.get("std")),
                        self._format_number(metric_data.get("min")),
                        self._format_number(metric_data.get("max")),
                        self._format_number(metric_data.get("cv"), decimals=4),
                        self._format_number(metric_data.get("se")),
                        self._format_number(metric_data.get("ci_low")),
                        self._format_number(metric_data.get("ci_high")),
                        metric_data.get("unit", ""),
                    ]
                    writer.writerow(row)

        return buf.getvalue()

    def _format_number(self, value, decimals: int = 2) -> str:
        """Format a number for CSV output.

        Args:
            value: Number to format
            decimals: Number of decimal places

        Returns:
            str: Formatted number or empty string if None
        """
        if value is None:
            return ""
        if isinstance(value, float):
            if value == float("inf"):
                return "inf"
            if value == float("-inf"):
                return "-inf"
            return f"{value:.{decimals}f}"
        return str(value)
