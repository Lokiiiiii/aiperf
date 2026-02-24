# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator aggregation and export methods."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.orchestrator.aggregation.base import AggregateResult
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator


class TestOrchestratorAggregation:
    """Tests for orchestrator aggregation methods."""

    @pytest.fixture
    def mock_service_config(self):
        """Create a mock service config."""
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {
            "workers_max": 4,
            "workers_min": 1,
        }
        return mock_config

    @pytest.fixture
    def mock_user_config(self):
        """Create a mock user config."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.warmup_request_count = 10
        return config

    @pytest.fixture
    def sample_run_results(self, tmp_path):
        """Create sample run results for testing."""
        return [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0001",
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0002",
            ),
        ]

    @pytest.fixture
    def sample_aggregate_result(self):
        """Create a sample aggregate result."""
        return AggregateResult(
            aggregation_type="confidence",
            num_runs=5,
            num_successful_runs=5,
            failed_runs=[],
            metadata={},
            metrics={},
        )

    def test_aggregate_and_export_no_aggregates_returns_early(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test _aggregate_and_export returns early when no aggregates."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Mock aggregate_results to return empty dict
        with patch.object(orchestrator, "aggregate_results", return_value={}):
            # Should not raise any errors
            orchestrator._aggregate_and_export([], mock_user_config)

    def test_aggregate_and_export_confidence_only_mode(
        self,
        mock_service_config,
        mock_user_config,
        sample_run_results,
        sample_aggregate_result,
        tmp_path,
    ):
        """Test _aggregate_and_export with confidence-only aggregation."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        aggregates = {"aggregate": sample_aggregate_result}

        with (
            patch.object(orchestrator, "aggregate_results", return_value=aggregates),
            patch.object(orchestrator, "_export_confidence_aggregate") as mock_export,
        ):
            orchestrator._aggregate_and_export(sample_run_results, mock_user_config)

            # Verify export was called with correct arguments
            mock_export.assert_called_once_with(
                sample_aggregate_result, mock_user_config
            )

    def test_aggregate_and_export_sweep_mode(
        self,
        mock_service_config,
        mock_user_config,
        sample_run_results,
        sample_aggregate_result,
        tmp_path,
    ):
        """Test _aggregate_and_export with sweep aggregation."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        per_value_agg = {10: sample_aggregate_result}
        sweep_agg = sample_aggregate_result

        aggregates = {
            "per_value_aggregates": per_value_agg,
            "sweep_aggregate": sweep_agg,
        }

        with (
            patch.object(orchestrator, "aggregate_results", return_value=aggregates),
            patch.object(orchestrator, "_export_sweep_aggregates") as mock_export,
        ):
            orchestrator._aggregate_and_export(sample_run_results, mock_user_config)

            # Verify export was called with correct arguments
            mock_export.assert_called_once_with(
                per_value_agg, sweep_agg, mock_user_config
            )


class TestConfidenceAggregateExport:
    """Tests for confidence aggregate export."""

    @pytest.fixture
    def mock_service_config(self):
        """Create a mock service config."""
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def mock_user_config(self):
        """Create a mock user config."""
        return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

    @pytest.fixture
    def sample_aggregate_result(self):
        """Create a sample aggregate result."""
        return AggregateResult(
            aggregation_type="confidence",
            num_runs=5,
            num_successful_runs=5,
            failed_runs=[],
            metadata={},
            metrics={},
        )

    def test_export_confidence_aggregate_creates_directory(
        self, mock_service_config, mock_user_config, sample_aggregate_result, tmp_path
    ):
        """Test that export creates aggregate directory."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Mock the exporters at the module level where they're imported
        mock_json_path = tmp_path / "aggregate" / "profile_export_aiperf.json"
        mock_csv_path = tmp_path / "aggregate" / "profile_export_aiperf.csv"

        with (
            patch(
                "aiperf.exporters.aggregate.AggregateConfidenceJsonExporter"
            ) as mock_json_exporter,
            patch(
                "aiperf.exporters.aggregate.AggregateConfidenceCsvExporter"
            ) as mock_csv_exporter,
        ):
            # Setup mock exporters
            mock_json_instance = Mock()
            mock_json_instance.export = AsyncMock(return_value=mock_json_path)
            mock_json_exporter.return_value = mock_json_instance

            mock_csv_instance = Mock()
            mock_csv_instance.export = AsyncMock(return_value=mock_csv_path)
            mock_csv_exporter.return_value = mock_csv_instance

            # Execute
            orchestrator._export_confidence_aggregate(
                sample_aggregate_result, mock_user_config
            )

            # Verify directory was created
            aggregate_dir = tmp_path / "aggregate"
            assert aggregate_dir.exists()
            assert aggregate_dir.is_dir()

    def test_export_confidence_aggregate_calls_exporters(
        self, mock_service_config, mock_user_config, sample_aggregate_result, tmp_path
    ):
        """Test that export calls both JSON and CSV exporters."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_json_path = tmp_path / "aggregate" / "profile_export_aiperf.json"
        mock_csv_path = tmp_path / "aggregate" / "profile_export_aiperf.csv"

        with (
            patch(
                "aiperf.exporters.aggregate.AggregateConfidenceJsonExporter"
            ) as mock_json_exporter,
            patch(
                "aiperf.exporters.aggregate.AggregateConfidenceCsvExporter"
            ) as mock_csv_exporter,
        ):
            # Setup mock exporters
            mock_json_instance = Mock()
            mock_json_instance.export = AsyncMock(return_value=mock_json_path)
            mock_json_exporter.return_value = mock_json_instance

            mock_csv_instance = Mock()
            mock_csv_instance.export = AsyncMock(return_value=mock_csv_path)
            mock_csv_exporter.return_value = mock_csv_instance

            # Execute
            orchestrator._export_confidence_aggregate(
                sample_aggregate_result, mock_user_config
            )

            # Verify both exporters were called
            mock_json_instance.export.assert_called_once()
            mock_csv_instance.export.assert_called_once()


class TestSweepAggregateExport:
    """Tests for sweep aggregate export."""

    @pytest.fixture
    def mock_service_config(self):
        """Create a mock service config."""
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def mock_user_config(self):
        """Create a mock user config."""
        return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

    @pytest.fixture
    def sample_per_value_aggregates(self):
        """Create sample per-value aggregates."""
        return {
            10: AggregateResult(
                aggregation_type="confidence",
                num_runs=3,
                num_successful_runs=3,
                failed_runs=[],
                metadata={"concurrency": 10},
                metrics={},
            ),
            20: AggregateResult(
                aggregation_type="confidence",
                num_runs=3,
                num_successful_runs=3,
                failed_runs=[],
                metadata={"concurrency": 20},
                metrics={},
            ),
        }

    @pytest.fixture
    def sample_sweep_aggregate(self):
        """Create a sample sweep aggregate."""
        return AggregateResult(
            aggregation_type="sweep",
            num_runs=6,
            num_successful_runs=6,
            failed_runs=[],
            metadata={
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}],
                "best_configurations": {},
                "pareto_optimal": [],
            },
            metrics=[],
        )

    def test_export_sweep_aggregates_exports_per_value_aggregates(
        self,
        mock_service_config,
        mock_user_config,
        sample_per_value_aggregates,
        sample_sweep_aggregate,
        tmp_path,
    ):
        """Test that sweep export creates per-value aggregate directories."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        with (
            patch(
                "aiperf.exporters.aggregate.AggregateConfidenceJsonExporter"
            ) as mock_conf_json,
            patch(
                "aiperf.exporters.aggregate.AggregateConfidenceCsvExporter"
            ) as mock_conf_csv,
            patch(
                "aiperf.exporters.aggregate.AggregateSweepJsonExporter"
            ) as mock_sweep_json,
            patch(
                "aiperf.exporters.aggregate.AggregateSweepCsvExporter"
            ) as mock_sweep_csv,
        ):
            # Setup mocks for confidence exporters
            mock_conf_json_instance = Mock()
            mock_conf_json_instance.export = AsyncMock(
                return_value=tmp_path / "conf.json"
            )
            mock_conf_json.return_value = mock_conf_json_instance

            mock_conf_csv_instance = Mock()
            mock_conf_csv_instance.export = AsyncMock(
                return_value=tmp_path / "conf.csv"
            )
            mock_conf_csv.return_value = mock_conf_csv_instance

            # Setup mocks for sweep exporters
            mock_sweep_json_instance = Mock()
            mock_sweep_json_instance.export = AsyncMock(
                return_value=tmp_path / "sweep.json"
            )
            mock_sweep_json.return_value = mock_sweep_json_instance

            mock_sweep_csv_instance = Mock()
            mock_sweep_csv_instance.export = AsyncMock(
                return_value=tmp_path / "sweep.csv"
            )
            mock_sweep_csv.return_value = mock_sweep_csv_instance

            # Execute
            orchestrator._export_sweep_aggregates(
                sample_per_value_aggregates,
                sample_sweep_aggregate,
                mock_user_config,
            )

            # Verify confidence exporters were called for each value
            assert mock_conf_json_instance.export.call_count == 2
            assert mock_conf_csv_instance.export.call_count == 2

    def test_export_sweep_aggregates_calls_sweep_exporters(
        self,
        mock_service_config,
        mock_user_config,
        sample_per_value_aggregates,
        sample_sweep_aggregate,
        tmp_path,
    ):
        """Test that sweep export calls JSON and CSV exporters."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        with (
            patch(
                "aiperf.exporters.aggregate.AggregateConfidenceJsonExporter"
            ) as mock_conf_json,
            patch(
                "aiperf.exporters.aggregate.AggregateConfidenceCsvExporter"
            ) as mock_conf_csv,
            patch(
                "aiperf.exporters.aggregate.AggregateSweepJsonExporter"
            ) as mock_sweep_json,
            patch(
                "aiperf.exporters.aggregate.AggregateSweepCsvExporter"
            ) as mock_sweep_csv,
        ):
            # Setup mocks for confidence exporters
            mock_conf_json_instance = Mock()
            mock_conf_json_instance.export = AsyncMock(
                return_value=tmp_path / "conf.json"
            )
            mock_conf_json.return_value = mock_conf_json_instance

            mock_conf_csv_instance = Mock()
            mock_conf_csv_instance.export = AsyncMock(
                return_value=tmp_path / "conf.csv"
            )
            mock_conf_csv.return_value = mock_conf_csv_instance

            # Setup mocks for sweep exporters
            mock_sweep_json_instance = Mock()
            mock_sweep_json_instance.export = AsyncMock(
                return_value=tmp_path / "sweep.json"
            )
            mock_sweep_json.return_value = mock_sweep_json_instance

            mock_sweep_csv_instance = Mock()
            mock_sweep_csv_instance.export = AsyncMock(
                return_value=tmp_path / "sweep.csv"
            )
            mock_sweep_csv.return_value = mock_sweep_csv_instance

            # Execute
            orchestrator._export_sweep_aggregates(
                sample_per_value_aggregates,
                sample_sweep_aggregate,
                mock_user_config,
            )

            # Verify sweep exporters were called
            mock_sweep_json_instance.export.assert_called_once()
            mock_sweep_csv_instance.export.assert_called_once()


class TestAggregateResults:
    """Tests for aggregate_results method."""

    @pytest.fixture
    def mock_service_config(self):
        """Create a mock service config."""
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def mock_user_config(self):
        """Create a mock user config."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.num_profile_runs = 5
        return config

    @pytest.fixture
    def sample_run_results(self, tmp_path):
        """Create sample run results."""
        return [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0001",
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0002",
            ),
        ]

    def test_aggregate_results_confidence_only_mode(
        self, mock_service_config, mock_user_config, sample_run_results, tmp_path
    ):
        """Test aggregate_results in confidence-only mode."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Mock the aggregation computation with mutable metadata
        mock_aggregate = AggregateResult(
            aggregation_type="confidence",
            num_runs=2,
            num_successful_runs=2,
            failed_runs=[],
            metadata={},  # This will be a real dict, not a Mock
            metrics={},
        )

        with patch(
            "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
        ) as mock_agg_class:
            mock_agg_instance = Mock()
            mock_agg_instance.aggregate.return_value = mock_aggregate
            mock_agg_class.return_value = mock_agg_instance

            result = orchestrator.aggregate_results(
                sample_run_results, mock_user_config
            )

            # Should return confidence aggregate
            assert "aggregate" in result
            assert result["aggregate"] == mock_aggregate
            # Verify cooldown was added to metadata
            assert "cooldown_seconds" in result["aggregate"].metadata

    def test_aggregate_results_sweep_mode(self, mock_service_config, tmp_path):
        """Test aggregate_results in sweep mode with multiple trials per value."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Create config with sweep parameters (concurrency as list)
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        # Set concurrency as a list to trigger sweep mode
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 3

        # Create run results with sweep metadata - multiple runs per value
        run_results = [
            # First value (10) - 2 runs
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "concurrency_10" / "run_0001",
                metadata={"concurrency": 10},
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "concurrency_10" / "run_0002",
                metadata={"concurrency": 10},
            ),
            # Second value (20) - 2 runs
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "concurrency_20" / "run_0001",
                metadata={"concurrency": 20},
            ),
            RunResult(
                label="run_0004",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "concurrency_20" / "run_0002",
                metadata={"concurrency": 20},
            ),
        ]

        # Mock the aggregation computation with mutable metadata
        mock_per_value_agg = AggregateResult(
            aggregation_type="confidence",
            num_runs=2,
            num_successful_runs=2,
            failed_runs=[],
            metadata={"concurrency": 10},
            metrics={},
        )

        mock_sweep_agg = AggregateResult(
            aggregation_type="sweep",
            num_runs=4,
            num_successful_runs=4,
            failed_runs=[],
            metadata={},
            metrics=[],
        )

        with (
            patch(
                "aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation"
            ) as mock_conf_agg,
            patch(
                "aiperf.orchestrator.aggregation.sweep.SweepAggregation"
            ) as mock_sweep_agg_class,
        ):
            mock_conf_instance = Mock()
            mock_conf_instance.aggregate.return_value = mock_per_value_agg
            mock_conf_agg.return_value = mock_conf_instance

            mock_sweep_instance = Mock()
            mock_sweep_instance.compute.return_value = mock_sweep_agg
            mock_sweep_agg_class.return_value = mock_sweep_instance

            result = orchestrator.aggregate_results(run_results, config)

            # Should return both per-value and sweep aggregates
            assert "per_value_aggregates" in result
            assert "sweep_aggregate" in result

    def test_aggregate_results_empty_results_returns_empty(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test aggregate_results with empty results returns empty dict."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        result = orchestrator.aggregate_results([], mock_user_config)

        # Should return empty dict for empty results
        assert result == {}


class TestCollectFailedSweepValues:
    """Tests for _collect_failed_sweep_values method."""

    @pytest.fixture
    def mock_service_config(self):
        """Create a mock service config."""
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    def test_collect_failed_sweep_values_no_failures_returns_empty(
        self, mock_service_config, tmp_path
    ):
        """Test _collect_failed_sweep_values with no failures."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0001",
                metadata={"concurrency": 10},
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0002",
                metadata={"concurrency": 20},
            ),
        ]

        failed_values = orchestrator._collect_failed_sweep_values(results)

        assert failed_values == []

    def test_collect_failed_sweep_values_with_failures_collects_values(
        self, mock_service_config, tmp_path
    ):
        """Test _collect_failed_sweep_values collects failed values."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0001",
                metadata={"concurrency": 10},
            ),
            RunResult(
                label="run_0002",
                success=False,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0002",
                metadata={"concurrency": 20},
                error="Connection timeout",
            ),
        ]

        failed_values = orchestrator._collect_failed_sweep_values(results)

        assert len(failed_values) == 1
        assert failed_values[0]["value"] == 20
        assert "timestamp" in failed_values[0]

    def test_collect_failed_sweep_values_deduplicates_same_value(
        self, mock_service_config, tmp_path
    ):
        """Test _collect_failed_sweep_values deduplicates multiple failures at same value."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        results = [
            RunResult(
                label="run_0001",
                success=False,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0001",
                metadata={"concurrency": 20},
                error="Error 1",
            ),
            RunResult(
                label="run_0002",
                success=False,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0002",
                metadata={"concurrency": 20},
                error="Error 2",
            ),
            RunResult(
                label="run_0003",
                success=False,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0003",
                metadata={"concurrency": 20},
                error="Error 3",
            ),
        ]

        failed_values = orchestrator._collect_failed_sweep_values(results)

        # Should only report the value once
        assert len(failed_values) == 1
        assert failed_values[0]["value"] == 20

    def test_collect_failed_sweep_values_ignores_non_sweep_failures(
        self, mock_service_config, tmp_path
    ):
        """Test _collect_failed_sweep_values ignores failures without sweep metadata."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        results = [
            RunResult(
                label="run_0001",
                success=False,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0001",
                metadata={"concurrency": 20},
                error="Sweep error",
            ),
            RunResult(
                label="run_0002",
                success=False,
                summary_metrics={},
                artifacts_path=tmp_path / "run_0002",
                metadata={},  # No sweep metadata
                error="Non-sweep error",
            ),
        ]

        failed_values = orchestrator._collect_failed_sweep_values(results)

        # Should only collect the sweep failure
        assert len(failed_values) == 1
        assert failed_values[0]["value"] == 20
