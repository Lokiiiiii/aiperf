# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for aggregate exporters."""

from dataclasses import dataclass
from pathlib import Path

from aiperf.orchestrator.aggregation.base import AggregateResult


@dataclass(slots=True)
class AggregateExporterConfig:
    """Configuration for aggregate exporters.
    
    Simpler than ExporterConfig because aggregate exports don't need:
    - ProfileResults (single-run data)
    - TelemetryExportData (per-run telemetry)
    - ServerMetricsResults (per-run server metrics)
    - Full UserConfig (just need output directory)
    
    Attributes:
        result: AggregateResult to export
        output_dir: Directory where export file will be written
    """
    
    result: AggregateResult
    output_dir: Path
