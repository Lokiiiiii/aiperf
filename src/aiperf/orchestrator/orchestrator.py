# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-run orchestrator for AIPerf benchmarks."""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import (
    ExecutionStrategy,
    FixedTrialsStrategy,
    ParameterSweepStrategy,
)

if TYPE_CHECKING:
    from aiperf.common.models.export_models import JsonMetricResult

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
        self, base_config: UserConfig, strategy: ExecutionStrategy | None = None
    ) -> list[RunResult]:
        """Execute benchmark with potential strategy composition.

        This method detects what mode to run based on configuration:
        - Sweep only: concurrency is list, num_profile_runs = 1
        - Confidence only: concurrency is int, num_profile_runs > 1
        - Both: concurrency is list, num_profile_runs > 1
        - Single run: concurrency is int, num_profile_runs = 1

        Args:
            base_config: Base benchmark configuration
            strategy: Optional execution strategy. If None, strategy is auto-detected
                     from config. If provided, uses the given strategy directly.

        Returns:
            List of RunResult, one per run executed
        """
        # If strategy provided explicitly, use it directly
        if strategy is not None:
            return self._execute_strategy(base_config, strategy)

        # Auto-detect mode from configuration
        has_sweep = isinstance(base_config.loadgen.concurrency, list)
        has_confidence = base_config.loadgen.num_profile_runs > 1

        if has_sweep and has_confidence:
            # Both: Compose strategies
            logger.info(
                f"Executing parameter sweep with confidence trials "
                f"(mode: {base_config.loadgen.parameter_sweep_mode})"
            )
            return self._execute_composed(base_config)
        elif has_sweep:
            # Just sweep
            logger.info("Executing parameter sweep (no confidence trials)")
            sweep_strategy = self._create_sweep_strategy(base_config)
            return self._execute_strategy(base_config, sweep_strategy)
        elif has_confidence:
            # Just confidence
            logger.info(
                f"Executing confidence trials (n={base_config.loadgen.num_profile_runs})"
            )
            confidence_strategy = self._create_confidence_strategy(base_config)
            return self._execute_strategy(base_config, confidence_strategy)
        else:
            # Single run
            logger.info("Executing single benchmark run")
            result = self._execute_single_run(
                base_config, label="run_0001", artifact_path=self.base_dir
            )
            return [result]

    def _create_sweep_strategy(self, config: UserConfig) -> "ParameterSweepStrategy":
        """Create parameter sweep strategy from config.

        Args:
            config: User configuration with sweep parameters

        Returns:
            ParameterSweepStrategy configured from config
        """
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        return ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=config.loadgen.concurrency,
            cooldown_seconds=config.loadgen.parameter_sweep_cooldown_seconds,
            same_seed=config.loadgen.parameter_sweep_same_seed,
            auto_set_seed=True,
        )

    def _create_confidence_strategy(self, config: UserConfig) -> "FixedTrialsStrategy":
        """Create confidence/fixed trials strategy from config.

        Args:
            config: User configuration with confidence parameters

        Returns:
            FixedTrialsStrategy configured from config
        """
        from aiperf.orchestrator.strategies import FixedTrialsStrategy

        return FixedTrialsStrategy(
            num_trials=config.loadgen.num_profile_runs,
            cooldown_seconds=config.loadgen.profile_run_cooldown_seconds,
            auto_set_seed=True,
            disable_warmup_after_first=True,
        )

    def _execute_composed(self, config: UserConfig) -> list[RunResult]:
        """Execute with both sweep and confidence strategies composed.

        Determines composition order based on parameter_sweep_mode:
        - repeated: for trial in trials: for value in values
        - independent: for value in values: for trial in trials

        Args:
            config: User configuration

        Returns:
            List of all run results
        """
        sweep_strategy = self._create_sweep_strategy(config)
        confidence_strategy = self._create_confidence_strategy(config)

        # Determine composition order based on mode
        if config.loadgen.parameter_sweep_mode == "repeated":
            # Repeated: for trial in trials: for value in values
            return self._compose_trials_then_sweep(
                config, confidence_strategy, sweep_strategy
            )
        else:  # independent
            # Independent: for value in values: for trial in trials
            return self._compose_sweep_then_trials(
                config, sweep_strategy, confidence_strategy
            )

    def _execute_strategy(
        self, config: UserConfig, strategy: ExecutionStrategy
    ) -> list[RunResult]:
        """Execute runs using a single strategy (no composition).

        Args:
            config: Base benchmark configuration
            strategy: Execution strategy to use

        Returns:
            List of run results
        """
        results = []
        run_index = 0

        logger.info(
            f"Starting multi-run benchmark with strategy: {strategy.__class__.__name__}"
        )

        # Let strategy validate config before starting
        strategy.validate_config(config)

        should_continue = strategy.should_continue(results)

        while should_continue:
            # Strategy decides next config (including warmup handling)
            run_config = strategy.get_next_config(config, results)

            # Strategy provides label
            label = strategy.get_run_label(run_index)

            # Strategy determines artifact path
            artifact_path = strategy.get_run_path(self.base_dir, run_index)

            logger.info(f"[{run_index + 1}] Executing {label}...")

            # Execute run
            result = self._execute_single_run(run_config, label, artifact_path)
            results.append(result)

            if result.success:
                logger.info(f"[{run_index + 1}] {label} completed successfully")
            else:
                logger.error(f"[{run_index + 1}] {label} failed: {result.error}")

            run_index += 1

            # Check if there will be another run
            should_continue = strategy.should_continue(results)

            # Apply cooldown only if there's another run coming
            if should_continue:
                cooldown = strategy.get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Applying cooldown: {cooldown}s")
                    time.sleep(cooldown)

        successful = sum(1 for r in results if r.success)
        logger.info(f"All runs complete: {successful}/{len(results)} successful")

        # Log failed sweep values if any (for sweep-only mode)
        if any("concurrency" in r.metadata for r in results):
            failed_values = self._collect_failed_sweep_values(results)
            if failed_values:
                logger.warning(
                    f"Some sweep values failed: {[fv['value'] for fv in failed_values]}"
                )
                for fv in failed_values:
                    logger.warning(
                        f"  {fv['parameter_name']}={fv['value']}: {fv['error']}"
                    )

        return results

    def _compose_trials_then_sweep(
        self,
        config: UserConfig,
        trial_strategy: "FixedTrialsStrategy",
        sweep_strategy: "ParameterSweepStrategy",
    ) -> list[RunResult]:
        """Repeated mode: for trial in trials: for value in values

        Example with 3 values, 5 trials:
        Trial 1: [10, 20, 30]
        Trial 2: [10, 20, 30]
        Trial 3: [10, 20, 30]
        Trial 4: [10, 20, 30]
        Trial 5: [10, 20, 30]

        Path nesting: trial_strategy.get_run_path() returns base for sweep_strategy
        Result: base_dir/profile_runs/run_0001/concurrency_10/

        Args:
            config: Base user configuration
            trial_strategy: Strategy for confidence trials
            sweep_strategy: Strategy for parameter sweep

        Returns:
            List of all run results
        """

        all_results = []
        trial_results = []

        logger.info(
            f"Starting repeated mode: {trial_strategy.num_trials} trials × "
            f"{len(sweep_strategy.parameter_values)} sweep values"
        )

        # Validate config before starting
        trial_strategy.validate_config(config)
        sweep_strategy.validate_config(config)

        # Outer loop: trials
        while trial_strategy.should_continue(trial_results):
            trial_index = len(trial_results)

            # Get trial-specific config (handles warmup, seed, etc.)
            trial_config = trial_strategy.get_next_config(config, trial_results)

            # Get trial's base directory
            trial_dir = trial_strategy.get_run_path(self.base_dir, trial_index)

            trial_label = trial_strategy.get_run_label(trial_index)
            logger.info(
                f"[Trial {trial_index + 1}/{trial_strategy.num_trials}] Starting {trial_label}"
            )

            # Inner loop: sweep values
            sweep_results = []
            while sweep_strategy.should_continue(sweep_results):
                value_index = len(sweep_results)

                # Get sweep-specific config (sets concurrency, derives seed)
                run_config = sweep_strategy.get_next_config(trial_config, sweep_results)

                # Get sweep path nested under trial directory
                run_path = sweep_strategy.get_run_path(trial_dir, value_index)

                # Build label
                sweep_label = sweep_strategy.get_run_label(value_index)
                label = f"{trial_label}_{sweep_label}"

                logger.info(
                    f"  [{value_index + 1}/{len(sweep_strategy.parameter_values)}] "
                    f"Executing {sweep_label}..."
                )

                # Execute run
                result = self._execute_single_run(run_config, label, run_path)
                result.metadata.update(
                    {
                        "trial_index": trial_index,
                        "value_index": value_index,
                        "concurrency": run_config.loadgen.concurrency,
                        "sweep_mode": "repeated",
                    }
                )
                sweep_results.append(result)
                all_results.append(result)

                if result.success:
                    logger.info(f"  [{value_index + 1}] {sweep_label} completed")
                else:
                    logger.error(
                        f"  [{value_index + 1}] {sweep_label} failed: {result.error}"
                    )

                # Apply sweep cooldown (between values within trial)
                if sweep_strategy.should_continue(sweep_results):
                    cooldown = sweep_strategy.get_cooldown_seconds()
                    if cooldown > 0:
                        logger.info(f"  Applying sweep cooldown: {cooldown}s")
                        time.sleep(cooldown)

            trial_results.append(sweep_results[-1])  # Track trial completion

            # Apply trial cooldown (between trials)
            if trial_strategy.should_continue(trial_results):
                cooldown = trial_strategy.get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Applying trial cooldown: {cooldown}s")
                    time.sleep(cooldown)

        successful = sum(1 for r in all_results if r.success)
        logger.info(
            f"Repeated mode complete: {successful}/{len(all_results)} runs successful"
        )

        # Log failed sweep values if any
        failed_values = self._collect_failed_sweep_values(all_results)
        if failed_values:
            logger.warning(
                f"Some sweep values failed: {[fv['value'] for fv in failed_values]}"
            )
            for fv in failed_values:
                logger.warning(f"  {fv['parameter_name']}={fv['value']}: {fv['error']}")

        return all_results

    def _compose_sweep_then_trials(
        self,
        config: UserConfig,
        sweep_strategy: "ParameterSweepStrategy",
        trial_strategy: "FixedTrialsStrategy",
    ) -> list[RunResult]:
        """Independent mode: for value in values: for trial in trials

        Example with 3 values, 5 trials:
        Value 10: [trial1, trial2, trial3, trial4, trial5]
        Value 20: [trial1, trial2, trial3, trial4, trial5]
        Value 30: [trial1, trial2, trial3, trial4, trial5]

        Path nesting: sweep_strategy.get_run_path() returns base for trial_strategy
        Result: base_dir/concurrency_10/profile_runs/run_0001/

        Args:
            config: Base user configuration
            sweep_strategy: Strategy for parameter sweep
            trial_strategy: Strategy for confidence trials

        Returns:
            List of all run results
        """

        all_results = []
        sweep_results = []

        logger.info(
            f"Starting independent mode: {len(sweep_strategy.parameter_values)} sweep values × "
            f"{trial_strategy.num_trials} trials"
        )

        # Validate config before starting
        sweep_strategy.validate_config(config)
        trial_strategy.validate_config(config)

        # Outer loop: sweep values
        while sweep_strategy.should_continue(sweep_results):
            value_index = len(sweep_results)

            # Get sweep-specific config (sets concurrency, derives seed)
            value_config = sweep_strategy.get_next_config(config, sweep_results)

            # Get sweep's base directory
            sweep_dir = sweep_strategy.get_run_path(self.base_dir, value_index)

            sweep_label = sweep_strategy.get_run_label(value_index)
            logger.info(
                f"[Value {value_index + 1}/{len(sweep_strategy.parameter_values)}] "
                f"Starting {sweep_label}"
            )

            # Inner loop: trials
            trial_results = []
            while trial_strategy.should_continue(trial_results):
                trial_index = len(trial_results)

                # Get trial-specific config (handles warmup, seed, etc.)
                run_config = trial_strategy.get_next_config(value_config, trial_results)

                # Get trial path nested under sweep directory
                run_path = trial_strategy.get_run_path(sweep_dir, trial_index)

                # Build label
                trial_label = trial_strategy.get_run_label(trial_index)
                label = f"{sweep_label}_{trial_label}"

                logger.info(
                    f"  [{trial_index + 1}/{trial_strategy.num_trials}] "
                    f"Executing {trial_label}..."
                )

                # Execute run
                result = self._execute_single_run(run_config, label, run_path)
                result.metadata.update(
                    {
                        "trial_index": trial_index,
                        "value_index": value_index,
                        "concurrency": run_config.loadgen.concurrency,
                        "sweep_mode": "independent",
                    }
                )
                trial_results.append(result)
                all_results.append(result)

                if result.success:
                    logger.info(f"  [{trial_index + 1}] {trial_label} completed")
                else:
                    logger.error(
                        f"  [{trial_index + 1}] {trial_label} failed: {result.error}"
                    )

                # Apply trial cooldown (between trials within value)
                if trial_strategy.should_continue(trial_results):
                    cooldown = trial_strategy.get_cooldown_seconds()
                    if cooldown > 0:
                        logger.info(f"  Applying trial cooldown: {cooldown}s")
                        time.sleep(cooldown)

            sweep_results.append(trial_results[-1])  # Track sweep completion

            # Apply sweep cooldown (between values)
            if sweep_strategy.should_continue(sweep_results):
                cooldown = sweep_strategy.get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Applying sweep cooldown: {cooldown}s")
                    time.sleep(cooldown)

        successful = sum(1 for r in all_results if r.success)
        logger.info(
            f"Independent mode complete: {successful}/{len(all_results)} runs successful"
        )

        # Log failed sweep values if any
        failed_values = self._collect_failed_sweep_values(all_results)
        if failed_values:
            logger.warning(
                f"Some sweep values failed: {[fv['value'] for fv in failed_values]}"
            )
            for fv in failed_values:
                logger.warning(f"  {fv['parameter_name']}={fv['value']}: {fv['error']}")

        return all_results

    def _execute_single_run(
        self, config: UserConfig, label: str, artifact_path: Path
    ) -> RunResult:
        """Execute a single benchmark run in a subprocess.

        Each run is executed in a separate subprocess to ensure complete isolation.
        This allows the SystemController to call os._exit() without affecting the orchestrator.

        Args:
            config: Benchmark configuration
            label: Label for this run (e.g., "run_0001", "concurrency_10")
            artifact_path: Path where artifacts should be stored

        Returns:
            RunResult with success status and metrics or error
        """
        try:
            # Ensure artifact directory exists
            artifact_path = Path(artifact_path)
            artifact_path.mkdir(parents=True, exist_ok=True)

            config = config.model_copy(deep=True)
            config.output.artifact_directory = artifact_path

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
            config_file = artifact_path / "run_config.json"
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
                    artifacts_path=artifact_path,
                )

            # Extract summary metrics from the artifacts
            # The SystemController writes results to files, so we read them back
            summary_metrics = self._extract_summary_metrics(artifact_path)

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
                    artifacts_path=artifact_path,
                )

            # Check if any requests completed successfully
            # request_count only counts valid (successful) requests
            # error_request_count counts failed requests
            # If error_request_count > 0 but request_count is missing or 0, all requests failed
            request_count_metric = summary_metrics.get("request_count")
            error_request_count_metric = summary_metrics.get("error_request_count")

            # If no request_count metric exists or it's 0, check if there were any errors
            if not request_count_metric or request_count_metric.avg == 0:
                # If there were error requests, all requests failed
                if error_request_count_metric and error_request_count_metric.avg > 0:
                    error_msg = (
                        f"All {int(error_request_count_metric.avg)} requests failed"
                    )
                    logger.error(error_msg)
                    return RunResult(
                        label=label,
                        success=False,
                        error=error_msg,
                        artifacts_path=artifact_path,
                    )
                # If no errors either, no requests were made at all
                error_msg = "No requests completed"
                logger.error(error_msg)
                return RunResult(
                    label=label,
                    success=False,
                    error=error_msg,
                    artifacts_path=artifact_path,
                )

            return RunResult(
                label=label,
                success=True,
                summary_metrics=summary_metrics,
                artifacts_path=artifact_path,
            )
        except Exception as e:
            logger.exception(f"Error executing run {label}")
            return RunResult(
                label=label,
                success=False,
                error=str(e),
                artifacts_path=artifact_path,
            )

    def _extract_summary_metrics(
        self, artifacts_path: Path
    ) -> dict[str, "JsonMetricResult"]:
        """Extract run-level summary statistics from artifacts.

        Reads the profile_export_aiperf.json file written by the SystemController
        and extracts the summary metrics, preserving the full structure with units.

        Args:
            artifacts_path: Path to run artifacts directory

        Returns:
            Dict mapping metric name to JsonMetricResult (e.g., {"time_to_first_token": JsonMetricResult(unit="ms", avg=150, p99=195)})
        """
        from aiperf.common.models.export_models import JsonMetricResult

        # Read the profile export JSON file
        json_file = artifacts_path / "profile_export_aiperf.json"

        if not json_file.exists():
            logger.warning(f"Profile export file not found: {json_file}")
            return {}

        try:
            # Load JSON as dict directly
            with open(json_file, "rb") as f:
                data = orjson.loads(f.read())

            # Extract metrics - keep the structure intact, don't flatten
            metrics = {}

            for field_name, field_value in data.items():
                # Check if this field is a metric (has the metric structure with "unit")
                if isinstance(field_value, dict) and "unit" in field_value:
                    try:
                        # Parse as JsonMetricResult to preserve full structure
                        metrics[field_name] = JsonMetricResult(**field_value)
                    except Exception as e:
                        logger.debug(f"Skipping field {field_name}: {e}")
                        continue

            return metrics

        except Exception:
            logger.exception(f"Error extracting metrics from {json_file}")
            return {}

    def _collect_failed_sweep_values(
        self, results: list[RunResult]
    ) -> list[dict[str, Any]]:
        """Collect information about failed sweep values.

        Args:
            results: List of all run results

        Returns:
            List of failed value information with structure:
            [
                {
                    "value": 30,
                    "parameter_name": "concurrency",
                    "error": "Connection timeout after 60s",
                    "timestamp": "2025-01-15T10:30:45Z"
                }
            ]
        """
        from datetime import datetime, timezone

        failed_values = []
        seen_failures = set()  # Track (parameter_name, value) to avoid duplicates

        for result in results:
            if not result.success and "concurrency" in result.metadata:
                # Extract parameter info from metadata
                param_value = result.metadata.get("concurrency")
                param_name = "concurrency"  # Currently only concurrency is supported

                # Create unique key to avoid duplicate entries
                failure_key = (param_name, param_value)

                if failure_key not in seen_failures:
                    seen_failures.add(failure_key)
                    failed_values.append(
                        {
                            "value": param_value,
                            "parameter_name": param_name,
                            "error": result.error or "Unknown error",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

        return failed_values
