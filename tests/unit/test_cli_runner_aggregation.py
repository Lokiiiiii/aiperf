# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-value aggregation in cli_runner.py"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.cli_runner import _aggregate_per_sweep_value
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunResult


class TestAggregatePerSweepValue:
    """Test per-value aggregation for sweep + confidence mode."""

    @pytest.fixture
    def mock_results_repeated_mode(self) -> list[RunResult]:
        """Create mock results for repeated mode sweep + confidence.

        Structure: 2 trials × 3 concurrency values = 6 runs
        Trial 1: [10, 20, 30]
        Trial 2: [10, 20, 30]
        """
        results = []

        # Trial 1
        for value_index, concurrency in enumerate([10, 20, 30]):
            result = RunResult(
                label=f"trial_0001_concurrency_{concurrency}",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec",
                        avg=100.0 + concurrency,
                        std=5.0,
                        min=95.0 + concurrency,
                        max=105.0 + concurrency,
                    ),
                    "ttft": JsonMetricResult(
                        unit="ms",
                        avg=50.0 + concurrency,
                        p99=100.0 + concurrency,
                    ),
                },
                metadata={
                    "trial_index": 0,
                    "value_index": value_index,
                    "concurrency": concurrency,
                    "sweep_mode": "repeated",
                },
            )
            results.append(result)

        # Trial 2
        for value_index, concurrency in enumerate([10, 20, 30]):
            result = RunResult(
                label=f"trial_0002_concurrency_{concurrency}",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec",
                        avg=102.0 + concurrency,
                        std=5.0,
                        min=97.0 + concurrency,
                        max=107.0 + concurrency,
                    ),
                    "ttft": JsonMetricResult(
                        unit="ms",
                        avg=52.0 + concurrency,
                        p99=102.0 + concurrency,
                    ),
                },
                metadata={
                    "trial_index": 1,
                    "value_index": value_index,
                    "concurrency": concurrency,
                    "sweep_mode": "repeated",
                },
            )
            results.append(result)

        return results

    @pytest.fixture
    def mock_results_independent_mode(self) -> list[RunResult]:
        """Create mock results for independent mode sweep + confidence.

        Structure: 3 concurrency values × 2 trials = 6 runs
        Concurrency 10: [trial1, trial2]
        Concurrency 20: [trial1, trial2]
        Concurrency 30: [trial1, trial2]
        """
        results = []

        for concurrency in [10, 20, 30]:
            for trial_index in range(2):
                result = RunResult(
                    label=f"concurrency_{concurrency}_trial_{trial_index + 1:04d}",
                    success=True,
                    summary_metrics={
                        "request_throughput": JsonMetricResult(
                            unit="requests/sec",
                            avg=100.0 + concurrency + trial_index,
                            std=5.0,
                            min=95.0 + concurrency,
                            max=105.0 + concurrency,
                        ),
                        "ttft": JsonMetricResult(
                            unit="ms",
                            avg=50.0 + concurrency + trial_index,
                            p99=100.0 + concurrency,
                        ),
                    },
                    metadata={
                        "trial_index": trial_index,
                        "value_index": [10, 20, 30].index(concurrency),
                        "concurrency": concurrency,
                        "sweep_mode": "independent",
                    },
                )
                results.append(result)

        return results

    def test_groups_results_by_concurrency_value(
        self, mock_results_repeated_mode: list[RunResult], tmp_path: Path
    ):
        """Test that results are correctly grouped by concurrency value."""
        with patch(
            "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
        ) as mock_agg_class:
            mock_agg = Mock()
            mock_agg_class.return_value = mock_agg

            # Mock aggregate to return a simple result
            mock_agg.aggregate.return_value = Mock(
                metadata={},
                metrics={},
            )

            with patch("asyncio.run") as mock_run:
                # Mock asyncio.run to return tuple of paths
                mock_run.return_value = (tmp_path / "test.json", tmp_path / "test.csv")

                _aggregate_per_sweep_value(
                    mock_results_repeated_mode,
                    confidence_level=0.95,
                    base_dir=tmp_path,
                    sweep_mode="repeated",
                )

            # Should call aggregate 3 times (once per concurrency value)
            assert mock_agg.aggregate.call_count == 3

            # Verify each call has 2 results (2 trials per value)
            for call in mock_agg.aggregate.call_args_list:
                results = call[0][0]
                assert len(results) == 2

    def test_writes_to_correct_directory_repeated_mode(
        self, mock_results_repeated_mode: list[RunResult], tmp_path: Path
    ):
        """Test that aggregate files are written to correct directories in repeated mode."""
        with patch(
            "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
        ) as mock_agg_class:
            mock_agg = Mock()
            mock_agg_class.return_value = mock_agg
            mock_agg.aggregate.return_value = Mock(metadata={}, metrics={})

            with (
                patch(
                    "aiperf.exporters.aggregate.AggregateConfidenceJsonExporter"
                ) as mock_json,
                patch(
                    "aiperf.exporters.aggregate.AggregateConfidenceCsvExporter"
                ) as mock_csv,
            ):
                # Mock exporters
                mock_json_instance = Mock()
                mock_json_instance.export = AsyncMock(
                    return_value=tmp_path / "test.json"
                )
                mock_json.return_value = mock_json_instance

                mock_csv_instance = Mock()
                mock_csv_instance.export = AsyncMock(return_value=tmp_path / "test.csv")
                mock_csv.return_value = mock_csv_instance

                _aggregate_per_sweep_value(
                    mock_results_repeated_mode,
                    confidence_level=0.95,
                    base_dir=tmp_path,
                    sweep_mode="repeated",
                )

            # Verify directories: base_dir/aggregate/concurrency_XX/
            expected_dirs = [
                tmp_path / "aggregate" / "concurrency_10",
                tmp_path / "aggregate" / "concurrency_20",
                tmp_path / "aggregate" / "concurrency_30",
            ]

            for expected_dir in expected_dirs:
                assert expected_dir.exists()

    def test_writes_to_correct_directory_independent_mode(
        self, mock_results_independent_mode: list[RunResult], tmp_path: Path
    ):
        """Test that aggregate files are written to correct directories in independent mode."""
        with patch(
            "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
        ) as mock_agg_class:
            mock_agg = Mock()
            mock_agg_class.return_value = mock_agg
            mock_agg.aggregate.return_value = Mock(metadata={}, metrics={})

            with (
                patch(
                    "aiperf.exporters.aggregate.AggregateConfidenceJsonExporter"
                ) as mock_json,
                patch(
                    "aiperf.exporters.aggregate.AggregateConfidenceCsvExporter"
                ) as mock_csv,
            ):
                # Mock exporters
                mock_json_instance = Mock()
                mock_json_instance.export = AsyncMock(
                    return_value=tmp_path / "test.json"
                )
                mock_json.return_value = mock_json_instance

                mock_csv_instance = Mock()
                mock_csv_instance.export = AsyncMock(return_value=tmp_path / "test.csv")
                mock_csv.return_value = mock_csv_instance

                _aggregate_per_sweep_value(
                    mock_results_independent_mode,
                    confidence_level=0.95,
                    base_dir=tmp_path,
                    sweep_mode="independent",
                )

            # Verify directories: base_dir/concurrency_XX/aggregate/
            expected_dirs = [
                tmp_path / "concurrency_10" / "aggregate",
                tmp_path / "concurrency_20" / "aggregate",
                tmp_path / "concurrency_30" / "aggregate",
            ]

            for expected_dir in expected_dirs:
                assert expected_dir.exists()

    def test_skips_values_with_insufficient_successful_runs(
        self, mock_results_repeated_mode: list[RunResult], tmp_path: Path
    ):
        """Test that values with < 2 successful runs are skipped."""
        # Mark one run as failed for concurrency=20
        for result in mock_results_repeated_mode:
            if (
                result.metadata["concurrency"] == 20
                and result.metadata["trial_index"] == 1
            ):
                result.success = False
                result.error = "Test failure"

        with patch(
            "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
        ) as mock_agg_class:
            mock_agg = Mock()
            mock_agg_class.return_value = mock_agg
            mock_agg.aggregate.return_value = Mock(metadata={}, metrics={})

            with patch("asyncio.run") as mock_run:
                # Mock asyncio.run to return tuple of paths
                mock_run.return_value = (tmp_path / "test.json", tmp_path / "test.csv")

                _aggregate_per_sweep_value(
                    mock_results_repeated_mode,
                    confidence_level=0.95,
                    base_dir=tmp_path,
                    sweep_mode="repeated",
                )

            # Should only call aggregate 2 times (concurrency 10 and 30)
            # Concurrency 20 has only 1 successful run
            assert mock_agg.aggregate.call_count == 2

    def test_adds_sweep_metadata_to_aggregate_result(
        self, mock_results_repeated_mode: list[RunResult], tmp_path: Path
    ):
        """Test that sweep-specific metadata is added to aggregate results."""
        with patch(
            "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
        ) as mock_agg_class:
            mock_agg = Mock()
            mock_agg_class.return_value = mock_agg

            # Create a mock aggregate result that we can inspect
            mock_aggregate_result = Mock(metadata={}, metrics={})
            mock_agg.aggregate.return_value = mock_aggregate_result

            with patch("asyncio.run") as mock_run:
                # Mock asyncio.run to return tuple of paths
                mock_run.return_value = (tmp_path / "test.json", tmp_path / "test.csv")

                _aggregate_per_sweep_value(
                    mock_results_repeated_mode,
                    confidence_level=0.95,
                    base_dir=tmp_path,
                    sweep_mode="repeated",
                )

            # Verify metadata was added for each concurrency value
            # Check that concurrency and sweep_mode were set
            assert mock_aggregate_result.metadata["sweep_mode"] == "repeated"
            # The last call should have concurrency=30
            assert mock_aggregate_result.metadata["concurrency"] == 30

    def test_handles_empty_results_gracefully(self, tmp_path: Path):
        """Test that function handles empty results list gracefully."""
        with patch(
            "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
        ) as mock_agg_class:
            mock_agg = Mock()
            mock_agg_class.return_value = mock_agg

            # Should not raise an error
            _aggregate_per_sweep_value(
                [],
                confidence_level=0.95,
                base_dir=tmp_path,
                sweep_mode="repeated",
            )

            # Should not call aggregate
            mock_agg.aggregate.assert_not_called()

    def test_handles_results_without_concurrency_metadata(self, tmp_path: Path):
        """Test that function handles results without concurrency metadata gracefully."""
        results = [
            RunResult(
                label="test_run",
                success=True,
                summary_metrics={},
                metadata={},  # No concurrency metadata
            )
        ]

        with patch(
            "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
        ) as mock_agg_class:
            mock_agg = Mock()
            mock_agg_class.return_value = mock_agg

            # Should not raise an error
            _aggregate_per_sweep_value(
                results,
                confidence_level=0.95,
                base_dir=tmp_path,
                sweep_mode="repeated",
            )

            # Should not call aggregate
            mock_agg.aggregate.assert_not_called()

    def test_uses_correct_confidence_level(
        self, mock_results_repeated_mode: list[RunResult], tmp_path: Path
    ):
        """Test that the correct confidence level is passed to ConfidenceAggregation."""
        with patch(
            "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
        ) as mock_agg_class:
            mock_agg = Mock()
            mock_agg_class.return_value = mock_agg
            mock_agg.aggregate.return_value = Mock(metadata={}, metrics={})

            with patch("asyncio.run") as mock_run:
                # Mock asyncio.run to return tuple of paths
                mock_run.return_value = (tmp_path / "test.json", tmp_path / "test.csv")

                _aggregate_per_sweep_value(
                    mock_results_repeated_mode,
                    confidence_level=0.99,
                    base_dir=tmp_path,
                    sweep_mode="repeated",
                )

            # Verify ConfidenceAggregation was created with correct confidence level
            mock_agg_class.assert_called_once_with(confidence_level=0.99)
