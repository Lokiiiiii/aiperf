# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep aggregate exporters."""

import csv
import json

import pytest

from aiperf.exporters.aggregate import (
    AggregateExporterConfig,
    AggregateSweepCsvExporter,
    AggregateSweepJsonExporter,
)
from aiperf.orchestrator.aggregation.base import AggregateResult


@pytest.fixture
def sample_sweep_aggregate():
    """Create a sample sweep aggregate result for testing."""
    # Simulate the output from SweepAggregation.compute()
    sweep_data = {
        "metadata": {
            "parameter_name": "concurrency",
            "parameter_values": [10, 20, 30],
            "num_values": 3,
        },
        "per_value_metrics": {
            "10": {
                "request_throughput_avg": {
                    "mean": 100.5,
                    "std": 5.2,
                    "min": 95.0,
                    "max": 108.0,
                    "cv": 0.052,
                    "unit": "requests/sec",
                },
                "ttft_p99_ms": {
                    "mean": 120.5,
                    "std": 8.1,
                    "min": 110.0,
                    "max": 130.0,
                    "cv": 0.067,
                    "unit": "ms",
                },
            },
            "20": {
                "request_throughput_avg": {
                    "mean": 180.2,
                    "std": 9.5,
                    "min": 170.0,
                    "max": 195.0,
                    "cv": 0.053,
                    "unit": "requests/sec",
                },
                "ttft_p99_ms": {
                    "mean": 135.8,
                    "std": 10.2,
                    "min": 125.0,
                    "max": 150.0,
                    "cv": 0.075,
                    "unit": "ms",
                },
            },
            "30": {
                "request_throughput_avg": {
                    "mean": 250.7,
                    "std": 12.3,
                    "min": 235.0,
                    "max": 270.0,
                    "cv": 0.049,
                    "unit": "requests/sec",
                },
                "ttft_p99_ms": {
                    "mean": 155.3,
                    "std": 15.5,
                    "min": 140.0,
                    "max": 175.0,
                    "cv": 0.100,
                    "unit": "ms",
                },
            },
        },
        "best_configurations": {
            "best_throughput": {
                "value": 30,
                "metric": 250.7,
                "unit": "requests/sec",
            },
            "best_latency_p99": {
                "value": 10,
                "metric": 120.5,
                "unit": "ms",
            },
        },
        "pareto_optimal": [10, 30],
        "trends": {
            "request_throughput_avg": {
                "inflection_points": [],
                "rate_of_change": [79.7, 70.5],
            },
            "ttft_p99_ms": {
                "inflection_points": [],
                "rate_of_change": [15.3, 19.5],
            },
        },
    }

    # Create AggregateResult matching the structure from cli_runner
    result = AggregateResult(
        aggregation_type="sweep",
        num_runs=15,  # 3 values × 5 trials
        num_successful_runs=15,
        failed_runs=[],
        metadata=sweep_data["metadata"].copy(),
        metrics=sweep_data["per_value_metrics"],
    )

    # Store additional sweep-specific data in metadata (like cli_runner does)
    result.metadata["best_configurations"] = sweep_data["best_configurations"]
    result.metadata["pareto_optimal"] = sweep_data["pareto_optimal"]
    result.metadata["trends"] = sweep_data["trends"]

    return result


class TestAggregateSweepJsonExporter:
    """Tests for AggregateSweepJsonExporter."""

    @pytest.mark.asyncio
    async def test_json_export_creates_file(self, tmp_path, sample_sweep_aggregate):
        """Test that JSON export creates the expected file."""
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepJsonExporter(config)

        json_path = await exporter.export()

        # Verify file exists
        assert json_path.exists()
        assert json_path.name == "profile_export_aiperf_sweep.json"
        assert json_path.parent == output_dir

    @pytest.mark.asyncio
    async def test_json_export_schema_compliance(
        self, tmp_path, sample_sweep_aggregate
    ):
        """Test that JSON export conforms to expected schema.

        Validates: Requirements 11.2, 11.3, 11.4
        Tasks: 9.1, 9.4, 9.5, 9.6, 9.7, 9.8
        """
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepJsonExporter(config)

        json_path = await exporter.export()

        # Load and verify JSON structure
        with open(json_path) as f:
            data = json.load(f)

        # Verify top-level fields
        assert data["aggregation_type"] == "sweep"
        assert data["num_profile_runs"] == 15
        assert data["num_successful_runs"] == 15

        # Verify metadata section (Task 9.4)
        assert "metadata" in data
        metadata = data["metadata"]
        assert metadata["parameter_name"] == "concurrency"
        assert metadata["parameter_values"] == [10, 20, 30]
        assert metadata["num_values"] == 3

        # Verify per-value metrics section (Task 9.5)
        assert "per_value_metrics" in data
        per_value = data["per_value_metrics"]
        assert "10" in per_value
        assert "20" in per_value
        assert "30" in per_value

        # Check metric structure for one value
        value_10_metrics = per_value["10"]
        assert "request_throughput_avg" in value_10_metrics
        assert "ttft_p99_ms" in value_10_metrics

        throughput = value_10_metrics["request_throughput_avg"]
        assert throughput["mean"] == 100.5
        assert throughput["std"] == 5.2
        assert throughput["unit"] == "requests/sec"

        # Verify best configurations section (Task 9.6)
        assert "best_configurations" in data
        best = data["best_configurations"]
        assert "best_throughput" in best
        assert best["best_throughput"]["value"] == 30
        assert best["best_throughput"]["metric"] == 250.7
        assert "best_latency_p99" in best
        assert best["best_latency_p99"]["value"] == 10

        # Verify Pareto optimal section (Task 9.7)
        assert "pareto_optimal" in data
        assert data["pareto_optimal"] == [10, 30]

        # Verify trends section (Task 9.8)
        assert "trends" in data
        trends = data["trends"]
        assert "request_throughput_avg" in trends
        assert "ttft_p99_ms" in trends

        throughput_trend = trends["request_throughput_avg"]
        assert "inflection_points" in throughput_trend
        assert "rate_of_change" in throughput_trend
        assert throughput_trend["rate_of_change"] == [79.7, 70.5]

    @pytest.mark.asyncio
    async def test_json_export_with_failed_runs(self, tmp_path):
        """Test JSON export includes failed runs information."""
        sweep_data = {
            "metadata": {
                "parameter_name": "concurrency",
                "parameter_values": [10, 20],
                "num_values": 2,
            },
            "per_value_metrics": {
                "10": {
                    "request_throughput_avg": {
                        "mean": 100.0,
                        "std": 5.0,
                        "min": 95.0,
                        "max": 105.0,
                        "cv": 0.05,
                    },
                },
            },
            "best_configurations": {},
            "pareto_optimal": [],
            "trends": {},
        }

        aggregate = AggregateResult(
            aggregation_type="sweep",
            num_runs=10,
            num_successful_runs=5,
            failed_runs=[
                {"run_index": 5, "error": "Connection timeout"},
                {"run_index": 6, "error": "Connection timeout"},
            ],
            metadata={},
            metrics=sweep_data,
        )

        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(result=aggregate, output_dir=output_dir)
        exporter = AggregateSweepJsonExporter(config)

        json_path = await exporter.export()

        with open(json_path) as f:
            data = json.load(f)

        assert data["num_successful_runs"] == 5
        assert "failed_runs" in data
        assert len(data["failed_runs"]) == 2

    @pytest.mark.asyncio
    async def test_json_export_creates_directory(
        self, tmp_path, sample_sweep_aggregate
    ):
        """Test that export creates output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "sweep_aggregate"
        assert not output_dir.exists()

        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepJsonExporter(config)

        json_path = await exporter.export()

        # Verify directory was created
        assert output_dir.exists()
        assert json_path.exists()


class TestAggregateSweepCsvExporter:
    """Tests for AggregateSweepCsvExporter."""

    @pytest.mark.asyncio
    async def test_csv_export_creates_file(self, tmp_path, sample_sweep_aggregate):
        """Test that CSV export creates the expected file."""
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepCsvExporter(config)

        csv_path = await exporter.export()

        # Verify file exists
        assert csv_path.exists()
        assert csv_path.name == "profile_export_aiperf_sweep.csv"
        assert csv_path.parent == output_dir

    @pytest.mark.asyncio
    async def test_csv_export_format(self, tmp_path, sample_sweep_aggregate):
        """Test that CSV export has correct format and sections.

        Validates: Requirements 11.5
        Tasks: 9.3, 9.4, 9.5, 9.6, 9.7, 9.8
        """
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepCsvExporter(config)

        csv_path = await exporter.export()

        # Read CSV content
        csv_content = csv_path.read_text()

        # Verify sections are present
        assert "parameter_value" in csv_content  # Per-value metrics table header
        assert "Best Configurations" in csv_content
        assert "Pareto Optimal Points" in csv_content
        assert "Trends" in csv_content
        assert "Metadata" in csv_content

        # Parse CSV and verify structure
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # First row should be per-value metrics header
        header = rows[0]
        assert header[0] == "parameter_value"
        assert "request_throughput_avg_mean" in header
        assert "ttft_p99_ms_mean" in header

        # Should have data rows for each sweep value
        assert rows[1][0] == "10"
        assert rows[2][0] == "20"
        assert rows[3][0] == "30"

    @pytest.mark.asyncio
    async def test_csv_per_value_metrics_table(self, tmp_path, sample_sweep_aggregate):
        """Test that per-value metrics table is correctly formatted."""
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepCsvExporter(config)

        csv_path = await exporter.export()

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check header
        header = rows[0]
        assert header[0] == "parameter_value"

        # Check data for value 10
        row_10 = rows[1]
        assert row_10[0] == "10"

        # Find throughput mean column
        throughput_mean_idx = header.index("request_throughput_avg_mean")
        assert float(row_10[throughput_mean_idx]) == 100.5

    @pytest.mark.asyncio
    async def test_csv_best_configurations_section(
        self, tmp_path, sample_sweep_aggregate
    ):
        """Test that best configurations section is present and correct."""
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepCsvExporter(config)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        # Verify best configurations section
        assert "Best Configurations" in csv_content
        assert "Best Throughput" in csv_content
        assert "Best Latency P99" in csv_content

    @pytest.mark.asyncio
    async def test_csv_pareto_optimal_section(self, tmp_path, sample_sweep_aggregate):
        """Test that Pareto optimal section is present and correct."""
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepCsvExporter(config)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        # Verify Pareto optimal section
        assert "Pareto Optimal Points" in csv_content

    @pytest.mark.asyncio
    async def test_csv_trends_section(self, tmp_path, sample_sweep_aggregate):
        """Test that trends section is present and correct."""
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepCsvExporter(config)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        # Verify trends section
        assert "Trends" in csv_content
        assert "Inflection Points" in csv_content
        assert "Rate of Change" in csv_content

    @pytest.mark.asyncio
    async def test_csv_metadata_section(self, tmp_path, sample_sweep_aggregate):
        """Test that metadata section is present and correct."""
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepCsvExporter(config)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        # Verify metadata section
        assert "Metadata" in csv_content
        assert "Aggregation Type" in csv_content
        assert "Parameter Name" in csv_content
        assert "Parameter Values" in csv_content

    @pytest.mark.asyncio
    async def test_csv_number_formatting(self, tmp_path, sample_sweep_aggregate):
        """Test that numbers are formatted correctly in CSV."""
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )
        exporter = AggregateSweepCsvExporter(config)

        csv_path = await exporter.export()

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check that numbers are formatted with appropriate precision
        header = rows[0]
        throughput_mean_idx = header.index("request_throughput_avg_mean")

        # Value should be formatted to 2 decimal places
        value = rows[1][throughput_mean_idx]
        assert value == "100.50"

    @pytest.mark.asyncio
    async def test_csv_empty_pareto_optimal(self, tmp_path):
        """Test CSV export when no Pareto optimal points exist."""
        sweep_data = {
            "metadata": {
                "parameter_name": "concurrency",
                "parameter_values": [10],
                "num_values": 1,
            },
            "per_value_metrics": {
                "10": {
                    "request_throughput_avg": {
                        "mean": 100.0,
                        "std": 5.0,
                        "min": 95.0,
                        "max": 105.0,
                        "cv": 0.05,
                    },
                },
            },
            "best_configurations": {},
            "pareto_optimal": [],  # Empty
            "trends": {},
        }

        # Create AggregateResult matching the structure from cli_runner
        aggregate = AggregateResult(
            aggregation_type="sweep",
            num_runs=5,
            num_successful_runs=5,
            failed_runs=[],
            metadata=sweep_data["metadata"].copy(),
            metrics=sweep_data["per_value_metrics"],
        )

        # Store additional sweep-specific data in metadata (like cli_runner does)
        aggregate.metadata["best_configurations"] = sweep_data["best_configurations"]
        aggregate.metadata["pareto_optimal"] = sweep_data["pareto_optimal"]
        aggregate.metadata["trends"] = sweep_data["trends"]

        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(result=aggregate, output_dir=output_dir)
        exporter = AggregateSweepCsvExporter(config)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        # Should handle empty Pareto optimal gracefully
        assert "Pareto Optimal Points" in csv_content
        assert "None" in csv_content


class TestSweepExportersIntegration:
    """Integration tests for both JSON and CSV sweep exporters."""

    @pytest.mark.asyncio
    async def test_both_exporters_produce_consistent_data(
        self, tmp_path, sample_sweep_aggregate
    ):
        """Test that JSON and CSV exporters produce consistent data."""
        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(
            result=sample_sweep_aggregate, output_dir=output_dir
        )

        json_exporter = AggregateSweepJsonExporter(config)
        csv_exporter = AggregateSweepCsvExporter(config)

        json_path = await json_exporter.export()
        csv_path = await csv_exporter.export()

        # Load JSON data
        with open(json_path) as f:
            json_data = json.load(f)

        # Load CSV data
        csv_content = csv_path.read_text()

        # Verify key data points are consistent
        assert str(json_data["metadata"]["num_values"]) in csv_content
        assert json_data["metadata"]["parameter_name"] in csv_content

        # Verify Pareto optimal values appear in both
        for value in json_data["pareto_optimal"]:
            assert str(value) in csv_content

    @pytest.mark.asyncio
    async def test_exporters_handle_minimal_data(self, tmp_path):
        """Test that exporters handle minimal sweep data gracefully."""
        minimal_sweep_data = {
            "metadata": {
                "parameter_name": "concurrency",
                "parameter_values": [10],
                "num_values": 1,
            },
            "per_value_metrics": {},
            "best_configurations": {},
            "pareto_optimal": [],
            "trends": {},
        }

        # Create AggregateResult matching the structure from cli_runner
        aggregate = AggregateResult(
            aggregation_type="sweep",
            num_runs=1,
            num_successful_runs=1,
            failed_runs=[],
            metadata=minimal_sweep_data["metadata"].copy(),
            metrics=minimal_sweep_data["per_value_metrics"],
        )

        # Store additional sweep-specific data in metadata (like cli_runner does)
        aggregate.metadata["best_configurations"] = minimal_sweep_data[
            "best_configurations"
        ]
        aggregate.metadata["pareto_optimal"] = minimal_sweep_data["pareto_optimal"]
        aggregate.metadata["trends"] = minimal_sweep_data["trends"]

        output_dir = tmp_path / "sweep_aggregate"
        config = AggregateExporterConfig(result=aggregate, output_dir=output_dir)

        json_exporter = AggregateSweepJsonExporter(config)
        csv_exporter = AggregateSweepCsvExporter(config)

        # Should not raise exceptions
        json_path = await json_exporter.export()
        csv_path = await csv_exporter.export()

        assert json_path.exists()
        assert csv_path.exists()
