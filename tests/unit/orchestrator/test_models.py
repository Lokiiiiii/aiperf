# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for orchestrator data models."""

from pathlib import Path

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.orchestrator.models import RunConfig, RunResult


class TestRunConfig:
    """Tests for RunConfig data model."""

    def test_create_run_config(self):
        """Test creating a RunConfig."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        run_config = RunConfig(
            config=config,
            label="run_0001",
            metadata={"run_index": 0, "seed": 42}
        )

        assert run_config.config == config
        assert run_config.label == "run_0001"
        assert run_config.metadata == {"run_index": 0, "seed": 42}

    def test_run_config_with_empty_metadata(self):
        """Test creating a RunConfig with empty metadata."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        run_config = RunConfig(
            config=config,
            label="run_0001",
            metadata={}
        )

        assert run_config.metadata == {}


class TestRunResult:
    """Tests for RunResult data model."""

    def test_create_successful_run_result(self):
        """Test creating a successful RunResult."""
        result = RunResult(
            label="run_0001",
            success=True,
            summary_metrics={"ttft_avg": 100.0, "tpot_avg": 10.0},
            artifacts_path=Path("/tmp/run_0001")
        )

        assert result.label == "run_0001"
        assert result.success is True
        assert result.summary_metrics == {"ttft_avg": 100.0, "tpot_avg": 10.0}
        assert result.artifacts_path == Path("/tmp/run_0001")
        assert result.error is None

    def test_create_failed_run_result(self):
        """Test creating a failed RunResult."""
        result = RunResult(
            label="run_0002",
            success=False,
            error="Connection timeout",
            artifacts_path=Path("/tmp/run_0002")
        )

        assert result.label == "run_0002"
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.summary_metrics == {}  # Defaults to empty dict
        assert result.artifacts_path == Path("/tmp/run_0002")

    def test_run_result_with_none_metrics(self):
        """Test RunResult with None summary_metrics."""
        result = RunResult(
            label="run_0003",
            success=True,
            summary_metrics=None,
            artifacts_path=Path("/tmp/run_0003")
        )

        assert result.summary_metrics is None
