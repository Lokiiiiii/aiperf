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

        Format: Multiple sections separated by blank lines:
        1. Per-value metrics table (WIDE format - one row per sweep value, metrics as columns)
        2. Best configurations section
        3. Pareto optimal points section
        4. Trends section
        5. Metadata section

        Returns:
            str: CSV content string
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Get parameter name from metadata (default to "concurrency")
        param_name = self._result.metadata.get("parameter_name", "concurrency")

        # Section 1: Per-value metrics table (WIDE format)
        per_value_metrics = self._result.metrics

        if per_value_metrics:
            # Build header: parameter_value, metric1_mean, metric1_std, ..., metric2_mean, metric2_std, ...
            header = ["parameter_value"]

            # Get all metric names from first value
            first_value = sorted(per_value_metrics.keys(), key=int)[0]
            metric_names = sorted(per_value_metrics[first_value].keys())

            # Add columns for each metric's statistics
            for metric_name in metric_names:
                header.extend(
                    [
                        f"{metric_name}_mean",
                        f"{metric_name}_std",
                        f"{metric_name}_min",
                        f"{metric_name}_max",
                        f"{metric_name}_cv",
                    ]
                )

            writer.writerow(header)

            # Write data rows (one row per sweep value)
            for value_str in sorted(per_value_metrics.keys(), key=int):
                row = [value_str]
                metrics = per_value_metrics[value_str]

                for metric_name in metric_names:
                    metric_data = metrics.get(metric_name, {})
                    if isinstance(metric_data, dict):
                        row.extend(
                            [
                                self._format_number(metric_data.get("mean")),
                                self._format_number(metric_data.get("std")),
                                self._format_number(metric_data.get("min")),
                                self._format_number(metric_data.get("max")),
                                self._format_number(metric_data.get("cv"), decimals=4),
                            ]
                        )
                    else:
                        # If not a dict, fill with empty values
                        row.extend(["", "", "", "", ""])

                writer.writerow(row)

        # Section 2: Best Configurations
        writer.writerow([])  # Blank line
        writer.writerow(["Best Configurations"])
        best_configs = self._result.metadata.get("best_configurations", {})
        if best_configs:
            writer.writerow(["Configuration", "Value", "Metric", "Unit"])
            for config_name, config_data in best_configs.items():
                # Format config name: best_throughput -> Best Throughput
                formatted_name = config_name.replace("_", " ").title()
                writer.writerow(
                    [
                        formatted_name,
                        config_data.get("value", ""),
                        self._format_number(config_data.get("metric")),
                        config_data.get("unit", ""),
                    ]
                )

        # Section 3: Pareto Optimal Points
        writer.writerow([])  # Blank line
        writer.writerow(["Pareto Optimal Points"])
        pareto_optimal = self._result.metadata.get("pareto_optimal", [])
        if pareto_optimal:
            writer.writerow([param_name])
            for value in pareto_optimal:
                writer.writerow([value])
        else:
            writer.writerow(["None"])

        # Section 4: Trends
        writer.writerow([])  # Blank line
        writer.writerow(["Trends"])
        trends = self._result.metadata.get("trends", {})
        if trends:
            for metric_name, trend_data in trends.items():
                writer.writerow([f"Metric: {metric_name}"])

                # Inflection Points
                inflection_points = trend_data.get("inflection_points", [])
                writer.writerow(["Inflection Points"])
                if inflection_points:
                    writer.writerow([param_name])
                    for point in inflection_points:
                        writer.writerow([point])
                else:
                    writer.writerow(["None"])

                # Rate of Change
                rate_of_change = trend_data.get("rate_of_change", [])
                writer.writerow(["Rate of Change"])
                if rate_of_change:
                    writer.writerow(["Interval", "Rate"])
                    for i, rate in enumerate(rate_of_change):
                        writer.writerow(
                            [f"Interval {i + 1}", self._format_number(rate)]
                        )
                else:
                    writer.writerow(["None"])

                writer.writerow([])  # Blank line between metrics

        # Section 5: Metadata
        writer.writerow([])  # Blank line
        writer.writerow(["Metadata"])
        writer.writerow(["Field", "Value"])
        writer.writerow(["Aggregation Type", self._result.aggregation_type])
        writer.writerow(["Parameter Name", param_name])
        writer.writerow(
            ["Parameter Values", str(self._result.metadata.get("parameter_values", []))]
        )
        writer.writerow(
            ["Number of Values", self._result.metadata.get("num_values", 0)]
        )
        writer.writerow(["Number of Profile Runs", self._result.num_runs])
        writer.writerow(["Number of Successful Runs", self._result.num_successful_runs])

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
