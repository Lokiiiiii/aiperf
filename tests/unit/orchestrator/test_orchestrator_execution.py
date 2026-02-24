# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator execution methods."""

from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import FixedTrialsStrategy, ParameterSweepStrategy


class TestOrchestratorExecution:
    """Tests for orchestrator execution logic."""

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
        config.loadgen.concurrency = 10
        config.loadgen.num_profile_runs = 1
        config.input.random_seed = 42
        return config

    @pytest.fixture
    def sweep_config(self):
        """Create a config with sweep parameters."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20, 30]
        config.loadgen.num_profile_runs = 1
        config.loadgen.parameter_sweep_mode = "repeated"
        return config

    @pytest.fixture
    def confidence_config(self):
        """Create a config with confidence parameters."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = 10
        config.loadgen.num_profile_runs = 5
        return config

    @pytest.fixture
    def sweep_and_confidence_config(self):
        """Create a config with both sweep and confidence parameters."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 3
        config.loadgen.parameter_sweep_mode = "repeated"
        return config

    def test_execute_auto_detects_single_run(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that execute auto-detects single run mode."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_result = RunResult(
            label="run_0001",
            success=True,
            summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
            artifacts_path=tmp_path,
        )

        with patch.object(
            orchestrator, "_execute_single_run", return_value=mock_result
        ):
            results = orchestrator.execute(mock_user_config)

        assert len(results) == 1
        assert results[0].label == "run_0001"

    def test_execute_auto_detects_sweep_only(
        self, mock_service_config, sweep_config, tmp_path
    ):
        """Test that execute auto-detects sweep-only mode."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_results = [
            RunResult(
                label=f"concurrency_{val}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"concurrency_{val}",
            )
            for val in [10, 20, 30]
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator.execute(sweep_config)

        assert len(results) == 3

    def test_execute_auto_detects_confidence_only(
        self, mock_service_config, confidence_config, tmp_path
    ):
        """Test that execute auto-detects confidence-only mode."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        mock_results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"run_{i:04d}",
            )
            for i in range(1, 6)
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator.execute(confidence_config)

        assert len(results) == 5

    def test_execute_auto_detects_sweep_and_confidence(
        self, mock_service_config, sweep_and_confidence_config, tmp_path
    ):
        """Test that execute auto-detects sweep + confidence mode."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # 2 sweep values × 3 trials = 6 runs
        mock_results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"run_{i:04d}",
            )
            for i in range(1, 7)
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator.execute(sweep_and_confidence_config)

        assert len(results) == 6

    def test_execute_composed_repeated_mode(
        self, mock_service_config, sweep_and_confidence_config, tmp_path
    ):
        """Test _execute_composed with repeated mode."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Set to repeated mode
        sweep_and_confidence_config.loadgen.parameter_sweep_mode = "repeated"

        mock_results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"run_{i:04d}",
                metadata={"trial_index": i // 2, "value_index": i % 2},
            )
            for i in range(6)
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator._execute_composed(sweep_and_confidence_config)

        assert len(results) == 6
        # Verify metadata is set
        assert all("trial_index" in r.metadata for r in results)
        assert all("value_index" in r.metadata for r in results)

    def test_execute_composed_independent_mode(
        self, mock_service_config, sweep_and_confidence_config, tmp_path
    ):
        """Test _execute_composed with independent mode."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Set to independent mode
        sweep_and_confidence_config.loadgen.parameter_sweep_mode = "independent"

        mock_results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"run_{i:04d}",
                metadata={"trial_index": i % 3, "value_index": i // 3},
            )
            for i in range(6)
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator._execute_composed(sweep_and_confidence_config)

        assert len(results) == 6
        # Verify metadata is set
        assert all("trial_index" in r.metadata for r in results)
        assert all("value_index" in r.metadata for r in results)

    def test_create_sweep_strategy(self, mock_service_config, sweep_config, tmp_path):
        """Test _create_sweep_strategy creates correct strategy."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = orchestrator._create_sweep_strategy(sweep_config)

        assert isinstance(strategy, ParameterSweepStrategy)
        assert strategy.parameter_name == "concurrency"
        assert strategy.parameter_values == [10, 20, 30]

    def test_create_sweep_strategy_raises_without_sweep_param(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test _create_sweep_strategy raises error without sweep parameter."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        with pytest.raises(ValueError, match="No sweep parameter detected"):
            orchestrator._create_sweep_strategy(mock_user_config)

    def test_create_confidence_strategy(
        self, mock_service_config, confidence_config, tmp_path
    ):
        """Test _create_confidence_strategy creates correct strategy."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = orchestrator._create_confidence_strategy(confidence_config)

        assert isinstance(strategy, FixedTrialsStrategy)
        assert strategy.num_trials == 5

    def test_execute_strategy_logs_failed_sweep_values(
        self, mock_service_config, tmp_path
    ):
        """Test _execute_strategy logs failed sweep values."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20, 30]

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 30],
        )

        mock_results = [
            RunResult(
                label="concurrency_10",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "concurrency_10",
                metadata={"concurrency": 10},
            ),
            RunResult(
                label="concurrency_20",
                success=False,
                error="Connection timeout",
                artifacts_path=tmp_path / "concurrency_20",
                metadata={"concurrency": 20},
            ),
            RunResult(
                label="concurrency_30",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "concurrency_30",
                metadata={"concurrency": 30},
            ),
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch("aiperf.orchestrator.orchestrator.logger") as mock_logger,
        ):
            results = orchestrator._execute_strategy(config, strategy)

        # Verify warning was logged for failed sweep value
        assert mock_logger.warning.called
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("Some sweep values failed" in str(call) for call in warning_calls)
        assert len(results) == 3

    def test_execute_strategy_no_warning_without_sweep_metadata(
        self, mock_service_config, tmp_path
    ):
        """Test _execute_strategy doesn't log sweep warnings for non-sweep runs."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

        strategy = FixedTrialsStrategy(num_trials=2)

        mock_results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "run_0001",
                metadata={},  # No concurrency metadata
            ),
            RunResult(
                label="run_0002",
                success=False,
                error="Some error",
                artifacts_path=tmp_path / "run_0002",
                metadata={},  # No concurrency metadata
            ),
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch("aiperf.orchestrator.orchestrator.logger") as mock_logger,
        ):
            results = orchestrator._execute_strategy(config, strategy)

        # Verify no sweep-related warnings were logged
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert not any("sweep values failed" in str(call) for call in warning_calls)
        assert len(results) == 2

    def test_compose_trials_then_sweep_execution_order(
        self, mock_service_config, tmp_path
    ):
        """Test _compose_trials_then_sweep executes in correct order."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2

        sweep_strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20],
        )
        trial_strategy = FixedTrialsStrategy(num_trials=2)

        executed_configs = []

        def capture_config(config, label, path):
            executed_configs.append(config.loadgen.concurrency)
            return RunResult(
                label=label,
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=path,
                metadata={"concurrency": config.loadgen.concurrency},
            )

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=capture_config
        ):
            results = orchestrator._compose_trials_then_sweep(
                config, trial_strategy, sweep_strategy
            )

        # Verify execution order: trial 1 [10, 20], trial 2 [10, 20]
        assert executed_configs == [10, 20, 10, 20]
        assert len(results) == 4

    def test_compose_sweep_then_trials_execution_order(
        self, mock_service_config, tmp_path
    ):
        """Test _compose_sweep_then_trials executes in correct order."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2

        sweep_strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20],
        )
        trial_strategy = FixedTrialsStrategy(num_trials=2)

        executed_configs = []

        def capture_config(config, label, path):
            executed_configs.append(config.loadgen.concurrency)
            return RunResult(
                label=label,
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=path,
                metadata={"concurrency": config.loadgen.concurrency},
            )

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=capture_config
        ):
            results = orchestrator._compose_sweep_then_trials(
                config, sweep_strategy, trial_strategy
            )

        # Verify execution order: value 10 [trial1, trial2], value 20 [trial1, trial2]
        assert executed_configs == [10, 10, 20, 20]
        assert len(results) == 4

    def test_compose_trials_then_sweep_applies_cooldowns(
        self, mock_service_config, tmp_path
    ):
        """Test _compose_trials_then_sweep applies cooldowns correctly."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2
        config.loadgen.parameter_sweep_cooldown_seconds = 0.1
        config.loadgen.profile_run_cooldown_seconds = 0.2

        sweep_strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20],
            cooldown_seconds=0.1,
        )
        trial_strategy = FixedTrialsStrategy(num_trials=2, cooldown_seconds=0.2)

        mock_results = [
            RunResult(
                label=f"run_{i}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"run_{i}",
                metadata={},
            )
            for i in range(4)
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch("aiperf.orchestrator.orchestrator.time.sleep") as mock_sleep,
        ):
            orchestrator._compose_trials_then_sweep(
                config, trial_strategy, sweep_strategy
            )

        # Should have sweep cooldown (between values) and trial cooldown (between trials)
        # Trial 1: value 10, [sweep cooldown], value 20
        # [trial cooldown]
        # Trial 2: value 10, [sweep cooldown], value 20
        assert mock_sleep.call_count == 3  # 1 sweep + 1 trial + 1 sweep

    def test_compose_sweep_then_trials_applies_cooldowns(
        self, mock_service_config, tmp_path
    ):
        """Test _compose_sweep_then_trials applies cooldowns correctly."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2
        config.loadgen.parameter_sweep_cooldown_seconds = 0.1
        config.loadgen.profile_run_cooldown_seconds = 0.2

        sweep_strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20],
            cooldown_seconds=0.1,
        )
        trial_strategy = FixedTrialsStrategy(num_trials=2, cooldown_seconds=0.2)

        mock_results = [
            RunResult(
                label=f"run_{i}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"run_{i}",
                metadata={},
            )
            for i in range(4)
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch("aiperf.orchestrator.orchestrator.time.sleep") as mock_sleep,
        ):
            orchestrator._compose_sweep_then_trials(
                config, sweep_strategy, trial_strategy
            )

        # Should have trial cooldown (between trials) and sweep cooldown (between values)
        # Value 10: trial 1, [trial cooldown], trial 2
        # [sweep cooldown]
        # Value 20: trial 1, [trial cooldown], trial 2
        assert mock_sleep.call_count == 3  # 1 trial + 1 sweep + 1 trial
