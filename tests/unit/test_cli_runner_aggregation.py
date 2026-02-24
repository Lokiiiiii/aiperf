# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aggregation in MultiRunOrchestrator"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator


class TestOrchestratorAggregation:
    """Test aggregation methods in MultiRunOrchestrator."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> MultiRunOrchestrator:
        """Create orchestrator instance."""
        service_config = ServiceConfig()
        return MultiRunOrchestrator(base_dir=tmp_path, service_config=service_config)

    @pytest.fixture
    def mock_config(self) -> UserConfig:
        """Create mock user config."""
        config = Mock(spec=UserConfig)
        config.loadgen = Mock()
        config.loadgen.confidence_level = 0.95
        config.loadgen.profile_run_cooldown_seconds = 10
        config.loadgen.num_profile_runs = 2
        return config

    @pytest.fixture
    def mock_results_confidence_only(self) -> list[RunResult]:
        """Create mock results for confidence-only mode (no sweep)."""
        results = []
        for i in range(3):
            result = RunResult(
                label=f"trial_{i + 1:04d}",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec",
                        avg=100.0 + i,
                        std=5.0,
                        min=95.0,
                        max=105.0,
                    ),
                },
                metadata={"trial_index": i},
            )
            results.append(result)
        return results

    @pytest.fixture
    def mock_results_sweep_only(self) -> list[RunResult]:
        """Create mock results for sweep-only mode (no confidence trials)."""
        results = []
        for concurrency in [10, 20, 30]:
            result = RunResult(
                label=f"concurrency_{concurrency}",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec",
                        avg=100.0 + concurrency,
                        std=5.0,
                    ),
                },
                metadata={"concurrency": concurrency},
            )
            results.append(result)
        return results

    @pytest.fixture
    def mock_results_sweep_and_confidence(self) -> list[RunResult]:
        """Create mock results for sweep + confidence mode."""
        results = []
        for concurrency in [10, 20, 30]:
            for trial in range(2):
                result = RunResult(
                    label=f"concurrency_{concurrency}_trial_{trial + 1:04d}",
                    success=True,
                    summary_metrics={
                        "request_throughput": JsonMetricResult(
                            unit="requests/sec",
                            avg=100.0 + concurrency + trial,
                            std=5.0,
                        ),
                    },
                    metadata={
                        "concurrency": concurrency,
                        "trial_index": trial,
                        "sweep_mode": "repeated",
                    },
                )
                results.append(result)
        return results

    def test_aggregate_results_confidence_only(
        self,
        orchestrator: MultiRunOrchestrator,
        mock_config: UserConfig,
        mock_results_confidence_only: list[RunResult],
    ):
        """Test aggregation for confidence-only mode."""
        aggregates = orchestrator.aggregate_results(
            mock_results_confidence_only, mock_config
        )

        assert "aggregate" in aggregates
        assert aggregates["aggregate"].aggregation_type == "confidence"
        assert aggregates["aggregate"].num_runs == 3
        assert aggregates["aggregate"].num_successful_runs == 3

    def test_aggregate_results_sweep_only(
        self,
        orchestrator: MultiRunOrchestrator,
        mock_config: UserConfig,
        mock_results_sweep_only: list[RunResult],
    ):
        """Test aggregation for sweep-only mode (should return empty dict)."""
        aggregates = orchestrator.aggregate_results(
            mock_results_sweep_only, mock_config
        )

        # Sweep-only mode doesn't need aggregation
        assert aggregates == {}

    def test_aggregate_results_sweep_and_confidence(
        self,
        orchestrator: MultiRunOrchestrator,
        mock_config: UserConfig,
        mock_results_sweep_and_confidence: list[RunResult],
    ):
        """Test aggregation for sweep + confidence mode."""
        aggregates = orchestrator.aggregate_results(
            mock_results_sweep_and_confidence, mock_config
        )

        assert "per_value_aggregates" in aggregates
        assert "sweep_aggregate" in aggregates

        # Should have 3 per-value aggregates (one for each concurrency)
        assert len(aggregates["per_value_aggregates"]) == 3
        assert 10 in aggregates["per_value_aggregates"]
        assert 20 in aggregates["per_value_aggregates"]
        assert 30 in aggregates["per_value_aggregates"]

        # Sweep aggregate should exist
        assert aggregates["sweep_aggregate"].aggregation_type == "sweep"

    def test_has_sweep_metadata(
        self,
        orchestrator: MultiRunOrchestrator,
        mock_results_sweep_only: list[RunResult],
        mock_results_confidence_only: list[RunResult],
    ):
        """Test detection of sweep metadata."""
        assert orchestrator._has_sweep_metadata(mock_results_sweep_only) is True
        assert orchestrator._has_sweep_metadata(mock_results_confidence_only) is False

    def test_has_multiple_trials_per_value(
        self,
        orchestrator: MultiRunOrchestrator,
        mock_results_sweep_and_confidence: list[RunResult],
        mock_results_sweep_only: list[RunResult],
    ):
        """Test detection of multiple trials per value."""
        assert (
            orchestrator._has_multiple_trials_per_value(
                mock_results_sweep_and_confidence
            )
            is True
        )
        assert (
            orchestrator._has_multiple_trials_per_value(mock_results_sweep_only)
            is False
        )

    def test_aggregate_per_sweep_value(
        self,
        orchestrator: MultiRunOrchestrator,
        mock_results_sweep_and_confidence: list[RunResult],
    ):
        """Test per-value aggregation."""
        per_value_aggregates = orchestrator._aggregate_per_sweep_value(
            mock_results_sweep_and_confidence, confidence_level=0.95
        )

        # Should have 3 aggregates
        assert len(per_value_aggregates) == 3

        # Each should have correct metadata
        for concurrency, aggregate in per_value_aggregates.items():
            assert aggregate.metadata["concurrency"] == concurrency
            assert aggregate.metadata["sweep_mode"] == "repeated"
            assert aggregate.aggregation_type == "confidence"

    def test_aggregate_per_sweep_value_skips_insufficient_runs(
        self,
        orchestrator: MultiRunOrchestrator,
        mock_results_sweep_and_confidence: list[RunResult],
    ):
        """Test that values with < 2 successful runs are skipped."""
        # Mark one run as failed for concurrency=20
        for result in mock_results_sweep_and_confidence:
            if (
                result.metadata["concurrency"] == 20
                and result.metadata["trial_index"] == 1
            ):
                result.success = False
                result.error = "Test failure"

        per_value_aggregates = orchestrator._aggregate_per_sweep_value(
            mock_results_sweep_and_confidence, confidence_level=0.95
        )

        # Should only have 2 aggregates (10 and 30, not 20)
        assert len(per_value_aggregates) == 2
        assert 10 in per_value_aggregates
        assert 30 in per_value_aggregates
        assert 20 not in per_value_aggregates

    def test_compute_sweep_aggregates(
        self,
        orchestrator: MultiRunOrchestrator,
        mock_results_sweep_and_confidence: list[RunResult],
    ):
        """Test sweep-level aggregation."""
        # First compute per-value aggregates
        per_value_aggregates = orchestrator._aggregate_per_sweep_value(
            mock_results_sweep_and_confidence, confidence_level=0.95
        )

        # Then compute sweep aggregates
        sweep_aggregate = orchestrator._compute_sweep_aggregates(
            mock_results_sweep_and_confidence,
            per_value_aggregates,
            confidence_level=0.95,
        )

        assert sweep_aggregate is not None
        assert sweep_aggregate.aggregation_type == "sweep"
        assert "best_configurations" in sweep_aggregate.metadata
        assert "pareto_optimal" in sweep_aggregate.metadata
