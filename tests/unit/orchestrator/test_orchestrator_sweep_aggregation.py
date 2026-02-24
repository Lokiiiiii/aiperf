# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator sweep aggregation computation methods."""

from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.orchestrator.aggregation.base import AggregateResult
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator


@dataclass
class MockConfidenceMetric:
    """Mock confidence metric for testing."""

    avg: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    std: float
    cv: float


class TestComputeSweepAggregates:
    """Tests for _compute_sweep_aggregates method."""

    @pytest.fixture
    def mock_service_config(self):
        """Create a mock service config."""
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def sample_run_results(self, tmp_path):
        """Create sample run results with sweep metadata."""
        return [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "concurrency_10" / "run_0001",
                metadata={"concurrency": 10, "sweep_mode": "repeated"},
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "concurrency_10" / "run_0002",
                metadata={"concurrency": 10, "sweep_mode": "repeated"},
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "concurrency_20" / "run_0001",
                metadata={"concurrency": 20, "sweep_mode": "repeated"},
            ),
            RunResult(
                label="run_0004",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "concurrency_20" / "run_0002",
                metadata={"concurrency": 20, "sweep_mode": "repeated"},
            ),
        ]

    @pytest.fixture
    def sample_per_value_aggregates(self):
        """Create sample per-value aggregates."""
        return {
            10: AggregateResult(
                aggregation_type="confidence",
                num_runs=2,
                num_successful_runs=2,
                failed_runs=[],
                metadata={"concurrency": 10},
                metrics={
                    "throughput": MockConfidenceMetric(
                        avg=100.0,
                        min=95.0,
                        max=105.0,
                        p50=100.0,
                        p90=103.0,
                        p95=104.0,
                        p99=105.0,
                        std=5.0,
                        cv=0.05,
                    ),
                    "ttft_p99": MockConfidenceMetric(
                        avg=150.0,
                        min=145.0,
                        max=155.0,
                        p50=150.0,
                        p90=153.0,
                        p95=154.0,
                        p99=155.0,
                        std=5.0,
                        cv=0.03,
                    ),
                },
            ),
            20: AggregateResult(
                aggregation_type="confidence",
                num_runs=2,
                num_successful_runs=2,
                failed_runs=[],
                metadata={"concurrency": 20},
                metrics={
                    "throughput": MockConfidenceMetric(
                        avg=180.0,
                        min=175.0,
                        max=185.0,
                        p50=180.0,
                        p90=183.0,
                        p95=184.0,
                        p99=185.0,
                        std=5.0,
                        cv=0.03,
                    ),
                    "ttft_p99": MockConfidenceMetric(
                        avg=200.0,
                        min=195.0,
                        max=205.0,
                        p50=200.0,
                        p90=203.0,
                        p95=204.0,
                        p99=205.0,
                        std=5.0,
                        cv=0.03,
                    ),
                },
            ),
        }

    def test_compute_sweep_aggregates_returns_aggregate_result(
        self,
        mock_service_config,
        sample_run_results,
        sample_per_value_aggregates,
        tmp_path,
    ):
        """Test _compute_sweep_aggregates returns AggregateResult."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Mock SweepAggregation.compute
        mock_sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}]
            },
            "per_combination_metrics": [],
            "best_configurations": {},
            "pareto_optimal": [],
        }

        with patch(
            "aiperf.orchestrator.aggregation.sweep.SweepAggregation.compute",
            return_value=mock_sweep_dict,
        ):
            result = orchestrator._compute_sweep_aggregates(
                sample_run_results, sample_per_value_aggregates, confidence_level=0.95
            )

            assert isinstance(result, AggregateResult)
            assert result.aggregation_type == "sweep"
            assert result.num_runs == 4
            assert result.num_successful_runs == 4

    def test_compute_sweep_aggregates_detects_parameter_name(
        self,
        mock_service_config,
        sample_run_results,
        sample_per_value_aggregates,
        tmp_path,
    ):
        """Test _compute_sweep_aggregates detects parameter name from results."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}]
            },
            "per_combination_metrics": [],
            "best_configurations": {},
            "pareto_optimal": [],
        }

        with patch(
            "aiperf.orchestrator.aggregation.sweep.SweepAggregation.compute",
            return_value=mock_sweep_dict,
        ) as mock_compute:
            orchestrator._compute_sweep_aggregates(
                sample_run_results, sample_per_value_aggregates, confidence_level=0.95
            )

            # Verify SweepAggregation.compute was called
            mock_compute.assert_called_once()
            # Check that sweep_parameters includes concurrency
            call_args = mock_compute.call_args
            sweep_params = call_args[0][1]  # Second positional argument
            assert sweep_params == [{"name": "concurrency", "values": [10, 20]}]

    def test_compute_sweep_aggregates_converts_metrics_to_dicts(
        self,
        mock_service_config,
        sample_run_results,
        sample_per_value_aggregates,
        tmp_path,
    ):
        """Test _compute_sweep_aggregates converts ConfidenceMetric to dicts."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}]
            },
            "per_combination_metrics": [],
            "best_configurations": {},
            "pareto_optimal": [],
        }

        with patch(
            "aiperf.orchestrator.aggregation.sweep.SweepAggregation.compute",
            return_value=mock_sweep_dict,
        ) as mock_compute:
            orchestrator._compute_sweep_aggregates(
                sample_run_results, sample_per_value_aggregates, confidence_level=0.95
            )

            # Verify metrics were converted to dicts
            call_args = mock_compute.call_args
            per_combination_stats = call_args[0][0]  # First positional argument

            # Check that metrics are dicts, not dataclass instances
            for _coord, metrics in per_combination_stats.items():
                assert isinstance(metrics, dict)
                for _metric_name, metric_value in metrics.items():
                    assert isinstance(metric_value, dict)
                    assert "avg" in metric_value
                    assert "p99" in metric_value

    def test_compute_sweep_aggregates_adds_metadata(
        self,
        mock_service_config,
        sample_run_results,
        sample_per_value_aggregates,
        tmp_path,
    ):
        """Test _compute_sweep_aggregates adds required metadata."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}]
            },
            "per_combination_metrics": [],
            "best_configurations": {
                "best_throughput": {"parameters": {"concurrency": 20}}
            },
            "pareto_optimal": [{"concurrency": 10}, {"concurrency": 20}],
        }

        with patch(
            "aiperf.orchestrator.aggregation.sweep.SweepAggregation.compute",
            return_value=mock_sweep_dict,
        ):
            result = orchestrator._compute_sweep_aggregates(
                sample_run_results, sample_per_value_aggregates, confidence_level=0.95
            )

            # Verify metadata
            assert result.metadata["sweep_mode"] == "repeated"
            assert result.metadata["confidence_level"] == 0.95
            assert result.metadata["aggregation_type"] == "sweep"
            assert result.metadata["num_trials_per_value"] == 2
            assert "best_configurations" in result.metadata
            assert "pareto_optimal" in result.metadata

    def test_compute_sweep_aggregates_counts_failed_runs(
        self, mock_service_config, sample_per_value_aggregates, tmp_path
    ):
        """Test _compute_sweep_aggregates correctly counts failed runs."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Create results with one failure
        results_with_failure = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0001",
                metadata={"concurrency": 10, "sweep_mode": "repeated"},
            ),
            RunResult(
                label="run_0002",
                success=False,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0002",
                metadata={"concurrency": 10, "sweep_mode": "repeated"},
                error="Connection timeout",
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0003",
                metadata={"concurrency": 20, "sweep_mode": "repeated"},
            ),
        ]

        mock_sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}]
            },
            "per_combination_metrics": [],
            "best_configurations": {},
            "pareto_optimal": [],
        }

        with patch(
            "aiperf.orchestrator.aggregation.sweep.SweepAggregation.compute",
            return_value=mock_sweep_dict,
        ):
            result = orchestrator._compute_sweep_aggregates(
                results_with_failure, sample_per_value_aggregates, confidence_level=0.95
            )

            assert result.num_runs == 3
            assert result.num_successful_runs == 2
            assert len(result.failed_runs) == 1
            assert result.failed_runs[0]["label"] == "run_0002"
            assert "timeout" in result.failed_runs[0]["error"].lower()

    def test_compute_sweep_aggregates_handles_missing_parameter_name(
        self, mock_service_config, sample_per_value_aggregates, tmp_path
    ):
        """Test _compute_sweep_aggregates handles missing parameter name gracefully."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Create results without sweep metadata
        results_no_param = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0001",
                metadata={},  # No parameter metadata
            ),
        ]

        result = orchestrator._compute_sweep_aggregates(
            results_no_param, sample_per_value_aggregates, confidence_level=0.95
        )

        # Should return None when parameter name can't be determined
        assert result is None

    def test_compute_sweep_aggregates_stores_best_configurations(
        self,
        mock_service_config,
        sample_run_results,
        sample_per_value_aggregates,
        tmp_path,
    ):
        """Test _compute_sweep_aggregates stores best configurations in metadata."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_best_configs = {
            "best_throughput": {
                "parameters": {"concurrency": 20},
                "metric": 180.0,
                "unit": "req/s",
            },
            "best_latency_p99": {
                "parameters": {"concurrency": 10},
                "metric": 150.0,
                "unit": "ms",
            },
        }

        mock_sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}]
            },
            "per_combination_metrics": [],
            "best_configurations": mock_best_configs,
            "pareto_optimal": [],
        }

        with patch(
            "aiperf.orchestrator.aggregation.sweep.SweepAggregation.compute",
            return_value=mock_sweep_dict,
        ):
            result = orchestrator._compute_sweep_aggregates(
                sample_run_results, sample_per_value_aggregates, confidence_level=0.95
            )

            assert result.metadata["best_configurations"] == mock_best_configs
            assert "best_throughput" in result.metadata["best_configurations"]
            assert "best_latency_p99" in result.metadata["best_configurations"]

    def test_compute_sweep_aggregates_stores_pareto_optimal(
        self,
        mock_service_config,
        sample_run_results,
        sample_per_value_aggregates,
        tmp_path,
    ):
        """Test _compute_sweep_aggregates stores Pareto optimal points in metadata."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_pareto = [{"concurrency": 10}, {"concurrency": 20}]

        mock_sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}]
            },
            "per_combination_metrics": [],
            "best_configurations": {},
            "pareto_optimal": mock_pareto,
        }

        with patch(
            "aiperf.orchestrator.aggregation.sweep.SweepAggregation.compute",
            return_value=mock_sweep_dict,
        ):
            result = orchestrator._compute_sweep_aggregates(
                sample_run_results, sample_per_value_aggregates, confidence_level=0.95
            )

            assert result.metadata["pareto_optimal"] == mock_pareto
            assert len(result.metadata["pareto_optimal"]) == 2

    def test_compute_sweep_aggregates_calculates_trials_per_value(
        self,
        mock_service_config,
        sample_run_results,
        sample_per_value_aggregates,
        tmp_path,
    ):
        """Test _compute_sweep_aggregates calculates num_trials_per_value correctly."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}]
            },
            "per_combination_metrics": [],
            "best_configurations": {},
            "pareto_optimal": [],
        }

        with patch(
            "aiperf.orchestrator.aggregation.sweep.SweepAggregation.compute",
            return_value=mock_sweep_dict,
        ):
            result = orchestrator._compute_sweep_aggregates(
                sample_run_results, sample_per_value_aggregates, confidence_level=0.95
            )

            # We have 2 runs for concurrency=10 and 2 runs for concurrency=20
            assert result.metadata["num_trials_per_value"] == 2
