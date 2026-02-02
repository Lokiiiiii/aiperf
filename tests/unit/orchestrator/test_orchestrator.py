# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator."""

import copy
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import FixedTrialsStrategy


class TestMultiRunOrchestrator:
    """Tests for MultiRunOrchestrator."""

    @pytest.fixture
    def mock_service_config(self):
        """Create a mock service config."""
        return Mock(spec=ServiceConfig)

    @pytest.fixture
    def mock_user_config(self):
        """Create a mock user config."""
        from aiperf.common.config import EndpointConfig, UserConfig
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.warmup_request_count = 10
        config.loadgen.warmup_num_sessions = None
        config.loadgen.warmup_duration = None
        config.loadgen.warmup_concurrency = None
        config.loadgen.warmup_request_rate = None
        config.loadgen.warmup_prefill_concurrency = None
        config.input.random_seed = 42
        return config

    def test_execute_with_fixed_trials_strategy(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test execute with FixedTrialsStrategy."""
        orchestrator = MultiRunOrchestrator(
            tmp_path, mock_service_config
        )

        strategy = FixedTrialsStrategy(num_trials=3, cooldown_seconds=0.0)

        # Mock the _execute_single_run method
        mock_results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft_avg": 100.0},
                artifacts_path=tmp_path / "run_0001",
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={"ttft_avg": 105.0},
                artifacts_path=tmp_path / "run_0002",
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={"ttft_avg": 102.0},
                artifacts_path=tmp_path / "run_0003",
            ),
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator.execute(mock_user_config, strategy)

        # Verify we got 3 results
        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].label == "run_0001"
        assert results[1].label == "run_0002"
        assert results[2].label == "run_0003"

    def test_execute_handles_failures(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that execute handles run failures gracefully."""
        orchestrator = MultiRunOrchestrator(
            tmp_path, mock_service_config
        )

        strategy = FixedTrialsStrategy(num_trials=3, cooldown_seconds=0.0)

        # Mock results with one failure
        mock_results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft_avg": 100.0},
                artifacts_path=tmp_path / "run_0001",
            ),
            RunResult(
                label="run_0002",
                success=False,
                error="Connection timeout",
                artifacts_path=tmp_path / "run_0002",
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={"ttft_avg": 102.0},
                artifacts_path=tmp_path / "run_0003",
            ),
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator.execute(mock_user_config, strategy)

        # Verify we got 3 results
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "Connection timeout"
        assert results[2].success is True

    def test_extract_summary_metrics(
        self, mock_service_config, tmp_path
    ):
        """Test extracting summary metrics from JSON file."""
        orchestrator = MultiRunOrchestrator(
            tmp_path, mock_service_config
        )

        # Create a mock profile_export_aiperf.json file with the correct structure
        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)

        json_content = {
            "time_to_first_token": {
                "unit": "ms",
                "avg": 150.5,
                "min": 100.0,
                "max": 200.0,
                "p50": 145.0,
                "p99": 195.0,
            },
            "request_throughput": {
                "unit": "requests/sec",
                "avg": 25.4,
                "min": 20.0,
                "max": 30.0,
            },
        }

        import json

        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            json.dump(json_content, f)

        # Extract metrics
        metrics = orchestrator._extract_summary_metrics(artifacts_path)

        # Verify metrics were extracted
        assert "time_to_first_token_avg" in metrics
        assert metrics["time_to_first_token_avg"] == 150.5
        assert "time_to_first_token_p99" in metrics
        assert metrics["time_to_first_token_p99"] == 195.0
        assert "request_throughput_avg" in metrics
        assert metrics["request_throughput_avg"] == 25.4

    def test_extract_summary_metrics_missing_file(
        self, mock_service_config, tmp_path
    ):
        """Test extracting metrics when file doesn't exist."""
        orchestrator = MultiRunOrchestrator(
            tmp_path, mock_service_config
        )

        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)

        # Extract metrics (file doesn't exist)
        metrics = orchestrator._extract_summary_metrics(artifacts_path)

        # Should return empty dict
        assert metrics == {}

    def test_warmup_disabled_after_first_run(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that warmup is disabled after the first run."""
        orchestrator = MultiRunOrchestrator(
            tmp_path, mock_service_config
        )

        strategy = FixedTrialsStrategy(num_trials=2, cooldown_seconds=0.0)

        configs_used = []

        def mock_execute(config, strategy, run_index):
            # Capture the config used
            configs_used.append(copy.deepcopy(config))
            return RunResult(
                label=strategy.get_run_label(run_index),
                success=True,
                summary_metrics={"ttft_avg": 100.0},
                artifacts_path=tmp_path / strategy.get_run_label(run_index),
            )

        with patch.object(orchestrator, "_execute_single_run", side_effect=mock_execute):
            with patch("aiperf.orchestrator.orchestrator.copy.deepcopy", side_effect=copy.deepcopy):
                orchestrator.execute(mock_user_config, strategy)

        # Verify first run has warmup, second run doesn't
        assert len(configs_used) == 2
        # First run should have warmup (original config)
        assert configs_used[0].loadgen.warmup_request_count == 10
        # Second run should have warmup disabled
        assert configs_used[1].loadgen.warmup_request_count is None
