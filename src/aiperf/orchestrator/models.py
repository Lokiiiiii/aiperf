# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for multi-run orchestration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiperf.common.config import UserConfig


@dataclass
class RunConfig:
    """Configuration for a single benchmark run.

    Attributes:
        config: The benchmark configuration to execute
        label: Human-readable label for this run (e.g., "run_0001")
        metadata: Additional metadata about this run (e.g., trial number, parameter values)
    """

    config: UserConfig
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result from executing a single benchmark run.

    Attributes:
        label: Label identifying this run
        success: Whether the run completed successfully
        summary_metrics: Run-level summary statistics (e.g., {"ttft_p99_ms": 152.7})
        error: Error message if run failed
        artifacts_path: Path to run artifacts directory
    """

    label: str
    success: bool
    summary_metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    artifacts_path: Path | None = None
