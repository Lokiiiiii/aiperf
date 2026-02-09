# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-run orchestrator for AIPerf benchmarks."""

import logging
import subprocess
import sys
import time
from pathlib import Path

import orjson

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import ExecutionStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "MultiRunOrchestrator",
]


class MultiRunOrchestrator:
    """Orchestrates execution of multiple benchmark runs using a strategy.

    The strategy decides:
    - What to run next (which config)
    - When to stop (based on results so far)
    - How to label runs
    - Cooldown duration between runs
    - Artifact path structure

    This orchestrator sits above the SystemController and coordinates multiple
    single-run executions.
    """

    def __init__(
        self,
        base_dir: Path,
        service_config: ServiceConfig,
    ):
        """Initialize MultiRunOrchestrator.

        Args:
            base_dir: Base directory for all artifacts
            service_config: Service configuration for SystemController
        """
        self.base_dir = Path(base_dir)
        self.service_config = service_config

    def execute(
        self, base_config: UserConfig, strategy: ExecutionStrategy
    ) -> list[RunResult]:
        """Execute runs based on strategy.

        Args:
            base_config: Base benchmark configuration
            strategy: Execution strategy that decides what to run

        Returns:
            List of RunResult, one per run executed
        """
        results = []
        run_index = 0

        logger.info(
            f"Starting multi-run benchmark with strategy: {strategy.__class__.__name__}"
        )

        # Let strategy validate config before starting
        strategy.validate_config(base_config)

        should_continue = strategy.should_continue(results)

        while should_continue:
            # Strategy decides next config (including warmup handling)
            config = strategy.get_next_config(base_config, results)

            # Strategy provides label
            label = strategy.get_run_label(run_index)

            logger.info(f"[{run_index + 1}] Executing {label}...")

            # Execute run - strategy determines artifact path
            result = self._execute_single_run(config, strategy, run_index)
            results.append(result)

            if result.success:
                logger.info(f"[{run_index + 1}] {label} completed successfully")
            else:
                logger.error(f"[{run_index + 1}] {label} failed: {result.error}")

            run_index += 1

            # Check if there will be another run (single call to should_continue)
            should_continue = strategy.should_continue(results)

            # Apply cooldown only if there's another run coming
            if should_continue:
                cooldown = strategy.get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Applying cooldown: {cooldown}s")
                    time.sleep(cooldown)

        successful = sum(1 for r in results if r.success)
        logger.info(f"All runs complete: {successful}/{len(results)} successful")

        return results

    def _execute_single_run(
        self, config: UserConfig, strategy: ExecutionStrategy, run_index: int
    ) -> RunResult:
        """Execute a single benchmark run in a subprocess.

        Each run is executed in a separate subprocess to ensure complete isolation.
        This allows the SystemController to call os._exit() without affecting the orchestrator.

        Args:
            config: Benchmark configuration
            strategy: Execution strategy (determines artifact path and label)
            run_index: Zero-based run index

        Returns:
            RunResult with success status and metrics or error
        """
        try:
            # Strategy determines artifact path and label
            artifacts_path = strategy.get_run_path(self.base_dir, run_index)
            artifacts_path.mkdir(parents=True, exist_ok=True)
            label = strategy.get_run_label(run_index)

            config = config.model_copy(deep=True)
            config.output.artifact_directory = artifacts_path

            # Serialize configs to JSON
            # Use exclude_defaults=True to avoid serializing fields that weren't explicitly set
            # This prevents validation errors on deserialization for fields with conditional validators
            config_data = {
                "user_config": config.model_dump(
                    mode="json", exclude_defaults=True, exclude_none=True
                ),
                "service_config": self.service_config.model_dump(
                    mode="json", exclude_defaults=True, exclude_none=True
                ),
            }

            # Write config to artifact directory for debugging and reproducibility
            # This allows users to see exactly what config was used for each run
            config_file = artifacts_path / "run_config.json"
            with open(config_file, "wb") as f:
                f.write(orjson.dumps(config_data, option=orjson.OPT_INDENT_2))

            # Run the benchmark in a subprocess using the dedicated runner module
            # The runner loads the config and calls _run_single_benchmark()
            # No timeout is set - SystemController handles benchmark duration and grace period internally
            # stdin/stdout are passed through to terminal so Textual can detect TTY and render live dashboard
            # stderr is captured for error reporting
            # -u flag forces unbuffered output so live dashboard updates are visible immediately
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",  # Unbuffered output - critical for live dashboard rendering
                    "-m",
                    "aiperf.orchestrator.subprocess_runner",
                    str(config_file),
                ],
                stdin=sys.stdin,  # Pass through stdin so Textual can detect interactive TTY
                stdout=sys.stdout,  # Pass through stdout for live dashboard rendering
                stderr=subprocess.PIPE,  # Capture for error reporting
                text=True,
            )

            if result.returncode != 0:
                error_msg = f"Benchmark failed with exit code {result.returncode}"
                if result.stderr:
                    # Get last 2000 chars of stderr for debugging
                    error_msg += f"\nStderr: {result.stderr[-2000:]}"
                logger.error(error_msg)
                return RunResult(
                    label=label,
                    success=False,
                    error=error_msg,
                    artifacts_path=artifacts_path,
                )

            # Extract summary metrics from the artifacts
            # The SystemController writes results to files, so we read them back
            summary_metrics = self._extract_summary_metrics(artifacts_path)

            # Check if the run produced any meaningful results
            # If no metrics were extracted or request_count is 0, treat as failure
            if not summary_metrics:
                error_msg = (
                    "No metrics found in artifacts - run may have failed to complete"
                )
                logger.error(error_msg)
                return RunResult(
                    label=label,
                    success=False,
                    error=error_msg,
                    artifacts_path=artifacts_path,
                )

            # Check if any requests completed successfully
            request_count = summary_metrics.get("request_count_avg", 0)
            if request_count == 0:
                error_msg = "No successful requests completed"
                logger.error(error_msg)
                return RunResult(
                    label=label,
                    success=False,
                    error=error_msg,
                    artifacts_path=artifacts_path,
                )

            return RunResult(
                label=label,
                success=True,
                summary_metrics=summary_metrics,
                artifacts_path=artifacts_path,
            )
        except Exception as e:
            logger.exception(f"Error executing run {label}")
            return RunResult(
                label=label,
                success=False,
                error=str(e),
                artifacts_path=artifacts_path if "artifacts_path" in locals() else None,
            )

    def _extract_summary_metrics(self, artifacts_path: Path) -> dict[str, float]:
        """Extract run-level summary statistics from artifacts.

        Reads the profile_export_aiperf.json file written by the SystemController
        and extracts the summary metrics using Pydantic deserialization.

        This method dynamically discovers all metrics in the export file without
        hardcoding metric names or statistical keys, making it robust to future
        additions of new metrics.

        Args:
            artifacts_path: Path to run artifacts directory

        Returns:
            Dict mapping metric name to value (e.g., {"time_to_first_token_p99": 152.7})
        """
        # Read the profile export JSON file
        json_file = artifacts_path / "profile_export_aiperf.json"

        if not json_file.exists():
            logger.warning(f"Profile export file not found: {json_file}")
            return {}

        try:
            # Load JSON as dict directly for efficient iteration
            with open(json_file, "rb") as f:
                data = orjson.loads(f.read())

            # Extract metrics dynamically by iterating over all fields
            # The export data uses extra="allow", so it can have additional metrics
            # beyond what's explicitly defined in the model
            metrics = {}

            # Iterate over all fields in the data
            for field_name, field_value in data.items():
                # Check if this field is a metric (has the metric structure with "unit")
                if isinstance(field_value, dict) and "unit" in field_value:
                    # This is a metric - extract all statistical values
                    for stat_key, stat_value in field_value.items():
                        # Skip non-numeric fields (like "unit")
                        if stat_key != "unit" and stat_value is not None:
                            # Create metric name like "time_to_first_token_p99"
                            full_metric_name = f"{field_name}_{stat_key}"
                            try:
                                metrics[full_metric_name] = float(stat_value)
                            except (ValueError, TypeError):
                                # Skip non-numeric values (e.g., strings, objects)
                                logger.debug(
                                    f"Skipping non-numeric field {full_metric_name}: {stat_value}"
                                )
                                continue

            return metrics

        except Exception:
            logger.exception(f"Error extracting metrics from {json_file}")
            return {}
