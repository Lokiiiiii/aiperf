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

    def execute_and_export(
        self, base_config: UserConfig, strategy: ExecutionStrategy | None = None
    ) -> list[RunResult]:
        """Execute benchmark, aggregate results, and export aggregates.

        This is the main entry point that handles the complete workflow:
        1. Execute runs
        2. Aggregate results (if applicable)
        3. Export aggregates (if applicable)

        Follows the same pattern as SystemController which handles execution + export.

        Args:
            base_config: Base benchmark configuration
            strategy: Optional execution strategy. If None, strategy is auto-detected

        Returns:
            List of RunResult, one per run executed
        """
        results = self.execute(base_config, strategy)
        self._aggregate_and_export(results, base_config)
        return results

    def _aggregate_and_export(
        self, results: list[RunResult], config: UserConfig
    ) -> None:
        """Aggregate results and export aggregate files.

        Args:
            results: List of run results from execute()
            config: User configuration used for execution
        """
        # Aggregate results
        aggregates = self.aggregate_results(results, config)

        if not aggregates:
            # No aggregation needed (e.g., sweep-only mode)
            return

        # Export aggregates
        if "aggregate" in aggregates:
            # Confidence-only mode
            self._export_confidence_aggregate(aggregates["aggregate"], config)
        elif "per_value_aggregates" in aggregates and "sweep_aggregate" in aggregates:
            # Sweep + confidence mode
            self._export_sweep_aggregates(
                aggregates["per_value_aggregates"],
                aggregates["sweep_aggregate"],
                config,
            )

    def _export_confidence_aggregate(
        self, aggregate_result: Any, _config: UserConfig
    ) -> None:
        """Export confidence-only aggregate results.

        Args:
            aggregate_result: Aggregate result to export
            _config: User configuration (unused)
        """
        import asyncio

        from aiperf.exporters.aggregate import (
            AggregateConfidenceCsvExporter,
            AggregateConfidenceJsonExporter,
            AggregateExporterConfig,
        )

        # Determine export directory
        aggregate_dir = self.base_dir / "aggregate"

        exporter_config = AggregateExporterConfig(
            result=aggregate_result,
            output_dir=aggregate_dir,
        )

        async def export_artifacts() -> tuple[Path, Path]:
            """Export aggregate artifacts asynchronously."""
            await asyncio.to_thread(aggregate_dir.mkdir, parents=True, exist_ok=True)

            json_exporter = AggregateConfidenceJsonExporter(exporter_config)
            csv_exporter = AggregateConfidenceCsvExporter(exporter_config)

            json_path, csv_path = await asyncio.gather(
                json_exporter.export(),
                csv_exporter.export(),
            )

            return json_path, csv_path

        json_path, csv_path = asyncio.run(export_artifacts())

        logger.info(f"Aggregate JSON written to: {json_path}")
        logger.info(f"Aggregate CSV written to: {csv_path}")

    def _export_sweep_aggregates(
        self,
        per_value_aggregates: dict[int, Any],
        sweep_aggregate: Any,
        config: UserConfig,
    ) -> None:
        """Export sweep + confidence aggregate results.

        Args:
            per_value_aggregates: Per-value aggregate results
            sweep_aggregate: Sweep-level aggregate result
            config: User configuration
        """
        import asyncio

        from aiperf.exporters.aggregate import (
            AggregateConfidenceCsvExporter,
            AggregateConfidenceJsonExporter,
            AggregateExporterConfig,
            AggregateSweepCsvExporter,
            AggregateSweepJsonExporter,
        )

        sweep_mode = sweep_aggregate.metadata.get("sweep_mode", "repeated")

        # Detect parameter name from first aggregate
        param_name = None
        for aggregate in per_value_aggregates.values():
            for key in ["concurrency", "request_rate"]:
                if key in aggregate.metadata:
                    param_name = key
                    break
            if param_name:
                break

        if not param_name:
            logger.warning("Could not determine parameter name for export")
            return

        logger.info("Exporting per-value aggregates...")

        # Export per-value aggregates
        async def export_per_value(
            param_value: int, aggregate: Any
        ) -> tuple[Path, Path]:
            """Export single per-value aggregate."""
            # Determine directory based on sweep mode
            if sweep_mode == "repeated":
                agg_dir = self.base_dir / "aggregate" / f"{param_name}_{param_value}"
            else:  # independent
                agg_dir = self.base_dir / f"{param_name}_{param_value}" / "aggregate"

            await asyncio.to_thread(agg_dir.mkdir, parents=True, exist_ok=True)

            exporter_config = AggregateExporterConfig(
                result=aggregate,
                output_dir=agg_dir,
            )

            json_exporter = AggregateConfidenceJsonExporter(exporter_config)
            csv_exporter = AggregateConfidenceCsvExporter(exporter_config)

            json_path, csv_path = await asyncio.gather(
                json_exporter.export(),
                csv_exporter.export(),
            )

            return json_path, csv_path

        # Export all per-value aggregates concurrently
        async def export_all_per_value() -> list[tuple[Path, Path]]:
            """Export all per-value aggregates."""
            tasks = [
                export_per_value(param_value, aggregate)
                for param_value, aggregate in per_value_aggregates.items()
            ]
            return await asyncio.gather(*tasks)

        asyncio.run(export_all_per_value())

        logger.info(
            f"Exported {len(per_value_aggregates)} per-value aggregates to {self.base_dir}"
        )

        # Export sweep-level aggregate
        logger.info("Exporting sweep-level aggregate...")

        # Determine sweep aggregate directory
        if sweep_mode == "repeated":
            sweep_dir = self.base_dir / "aggregate" / "sweep_aggregate"
        else:  # independent
            sweep_dir = self.base_dir / "sweep_aggregate"

        exporter_config = AggregateExporterConfig(
            result=sweep_aggregate,
            output_dir=sweep_dir,
        )

        async def export_sweep() -> tuple[Path, Path]:
            """Export sweep aggregate."""
            await asyncio.to_thread(sweep_dir.mkdir, parents=True, exist_ok=True)

            json_exporter = AggregateSweepJsonExporter(exporter_config)
            csv_exporter = AggregateSweepCsvExporter(exporter_config)

            json_path, csv_path = await asyncio.gather(
                json_exporter.export(),
                csv_exporter.export(),
            )

            return json_path, csv_path

        json_path, csv_path = asyncio.run(export_sweep())

        logger.info(f"  Sweep JSON: {json_path}")
        logger.info(f"  Sweep CSV: {csv_path}")

        # Log best configurations
        best_configs = sweep_aggregate.metadata.get("best_configurations", {})
        if best_configs:
            logger.info("")
            logger.info("Best Configurations:")
            if "best_throughput" in best_configs:
                best_throughput = best_configs["best_throughput"]
                params_str = ", ".join(
                    f"{k}={v}" for k, v in best_throughput["parameters"].items()
                )
                logger.info(
                    f"  Best throughput: {params_str} "
                    f"({best_throughput['metric']:.2f} {best_throughput['unit']})"
                )
            if "best_latency_p99" in best_configs:
                best_latency = best_configs["best_latency_p99"]
                params_str = ", ".join(
                    f"{k}={v}" for k, v in best_latency["parameters"].items()
                )
                logger.info(
                    f"  Best latency (p99): {params_str} "
                    f"({best_latency['metric']:.2f} {best_latency['unit']})"
                )

        # Log Pareto optimal points
        pareto_optimal = sweep_aggregate.metadata.get("pareto_optimal", [])
        if pareto_optimal:
            logger.info(f"  Pareto optimal points: {pareto_optimal}")

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
        has_sweep = base_config.loadgen.get_sweep_parameter() is not None
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

        Raises:
            ValueError: If no sweep parameter is detected in config
        """
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        sweep_info = config.loadgen.get_sweep_parameter()
        if not sweep_info:
            raise ValueError(
                "No sweep parameter detected in configuration. "
                "To enable parameter sweep, provide a parameter as a comma-separated list. "
                "Example: --concurrency 10,20,30"
            )

        param_name, param_values = sweep_info

        return ParameterSweepStrategy(
            parameter_name=param_name,
            parameter_values=param_values,
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
            auto_set_seed=config.loadgen.set_consistent_seed,
            disable_warmup_after_first=config.loadgen.profile_run_disable_warmup_after_first,
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

    def aggregate_results(
        self, results: list[RunResult], config: UserConfig
    ) -> dict[str, Any]:
        """Aggregate results based on execution mode.

        Determines aggregation strategy based on results metadata:
        - Sweep-only: No aggregation (returns empty dict)
        - Confidence-only: Single aggregate using ConfidenceAggregation
        - Sweep + confidence: Per-value aggregates + sweep-level aggregates

        Args:
            results: List of run results from execute()
            config: User configuration used for execution

        Returns:
            Dict with aggregation results:
            - For confidence-only: {"aggregate": AggregateResult}
            - For sweep+confidence: {
                "per_value_aggregates": {value: AggregateResult},
                "sweep_aggregate": AggregateResult
              }
            - For sweep-only: {} (empty dict, no aggregation needed)
        """
        from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation

        # Detect mode from results metadata
        has_sweep_metadata = self._has_sweep_metadata(results)
        has_multiple_trials = self._has_multiple_trials_per_value(results)

        # Sweep-only mode: no aggregation needed
        if has_sweep_metadata and not has_multiple_trials:
            logger.info("Sweep-only mode: no aggregation needed")
            return {}

        # Confidence-only mode: single aggregate
        if not has_sweep_metadata and has_multiple_trials:
            logger.info("Computing confidence aggregate...")
            aggregation = ConfidenceAggregation(
                confidence_level=config.loadgen.confidence_level
            )
            aggregate_result = aggregation.aggregate(results)
            aggregate_result.metadata["cooldown_seconds"] = (
                config.loadgen.profile_run_cooldown_seconds
            )
            return {"aggregate": aggregate_result}

        # Sweep + confidence mode: per-value and sweep-level aggregates
        if has_sweep_metadata and has_multiple_trials:
            logger.info("Computing per-value and sweep-level aggregates...")
            per_value_aggregates = self._aggregate_per_sweep_value(
                results, config.loadgen.confidence_level
            )
            sweep_aggregate = self._compute_sweep_aggregates(
                results, per_value_aggregates, config.loadgen.confidence_level
            )
            return {
                "per_value_aggregates": per_value_aggregates,
                "sweep_aggregate": sweep_aggregate,
            }

        # Single run or insufficient data
        logger.warning("No aggregation performed: insufficient data or single run")
        return {}

    def _has_sweep_metadata(self, results: list[RunResult]) -> bool:
        """Check if results contain sweep parameter metadata.

        Args:
            results: List of run results

        Returns:
            True if multiple distinct parameter values found
        """
        sweep_param_values = set()
        for r in results:
            for key in ["concurrency", "request_rate"]:
                if key in r.metadata:
                    sweep_param_values.add(r.metadata[key])
                    break
        return len(sweep_param_values) > 1

    def _has_multiple_trials_per_value(self, results: list[RunResult]) -> bool:
        """Check if results contain multiple trials per parameter value.

        For confidence-only mode (no sweep), checks if there are multiple results total.
        For sweep mode, checks if any parameter value has multiple trials.

        Args:
            results: List of run results

        Returns:
            True if multiple trials detected
        """
        from collections import defaultdict

        results_by_value: dict[int, list[RunResult]] = defaultdict(list)
        has_sweep_param = False

        for result in results:
            for key in ["concurrency", "request_rate"]:
                if key in result.metadata:
                    has_sweep_param = True
                    param_value = result.metadata[key]
                    results_by_value[param_value].append(result)
                    break

        # If no sweep parameters found, this is confidence-only mode
        # Check if we have multiple results total
        if not has_sweep_param:
            return len(results) > 1

        # For sweep mode, check if any value has multiple results
        return any(len(trials) > 1 for trials in results_by_value.values())

    def _aggregate_per_sweep_value(
        self, results: list[RunResult], confidence_level: float
    ) -> dict[int, Any]:
        """Aggregate results per sweep value for sweep + confidence mode.

        Groups results by parameter value and applies ConfidenceAggregation
        to each group.

        Args:
            results: List of all run results
            confidence_level: Confidence level for intervals

        Returns:
            Dict mapping parameter value to AggregateResult
        """
        from collections import defaultdict

        from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation

        # Group results by parameter value
        results_by_value: dict[int, list[RunResult]] = defaultdict(list)
        param_name = None

        for result in results:
            for key in ["concurrency", "request_rate"]:
                if key in result.metadata:
                    param_name = key
                    param_value = result.metadata[key]
                    results_by_value[param_value].append(result)
                    break

        if not results_by_value:
            logger.warning("No sweep results found with parameter metadata")
            return {}

        logger.info(
            f"Computing per-value aggregates for {len(results_by_value)} {param_name} values..."
        )

        # Aggregate each parameter value
        aggregation = ConfidenceAggregation(confidence_level=confidence_level)
        per_value_aggregates = {}

        for param_value in sorted(results_by_value.keys()):
            value_results = results_by_value[param_value]

            # Check if we have enough successful runs
            successful = [r for r in value_results if r.success]
            if len(successful) < 2:
                logger.warning(
                    f"Skipping aggregate for {param_name}={param_value}: "
                    f"only {len(successful)} successful run(s), need at least 2"
                )
                continue

            logger.info(
                f"  Aggregating {param_name}={param_value} "
                f"({len(successful)}/{len(value_results)} successful)"
            )

            # Compute aggregate statistics
            aggregate_result = aggregation.aggregate(value_results)

            # Add sweep-specific metadata
            aggregate_result.metadata[param_name] = param_value
            aggregate_result.metadata["sweep_mode"] = value_results[0].metadata.get(
                "sweep_mode", "repeated"
            )

            per_value_aggregates[param_value] = aggregate_result

        return per_value_aggregates

    def _compute_sweep_aggregates(
        self,
        results: list[RunResult],
        per_value_aggregates: dict[int, Any],
        confidence_level: float,
    ) -> Any:
        """Compute sweep-level aggregates (Pareto optimal, best configs).

        Args:
            results: List of all run results
            per_value_aggregates: Per-value aggregate results
            confidence_level: Confidence level for intervals

        Returns:
            AggregateResult with sweep-level statistics
        """
        from aiperf.orchestrator.aggregation.base import AggregateResult
        from aiperf.orchestrator.aggregation.sweep import (
            ParameterCombination,
            SweepAggregation,
        )

        logger.info("Computing sweep-level aggregates...")

        # Detect parameter name from results
        param_name = None
        for result in results:
            for key in ["concurrency", "request_rate"]:
                if key in result.metadata:
                    param_name = key
                    break
            if param_name:
                break

        if not param_name:
            logger.warning("Could not determine sweep parameter name")
            return None

        # Convert per-value aggregates to coordinate-based format
        # Need to convert ConfidenceMetric objects to dicts for SweepAggregation
        per_combination_stats = {}
        for param_value, aggregate in per_value_aggregates.items():
            coord = ParameterCombination({param_name: param_value})

            # Convert metrics from ConfidenceMetric objects to dicts
            metrics_dict = {}
            for metric_name, metric_value in aggregate.metrics.items():
                # ConfidenceMetric is a dataclass, use asdict
                from dataclasses import asdict, is_dataclass

                if is_dataclass(metric_value):
                    metrics_dict[metric_name] = asdict(metric_value)
                elif isinstance(metric_value, dict):
                    metrics_dict[metric_name] = metric_value
                else:
                    # Fallback: try to convert to dict
                    metrics_dict[metric_name] = (
                        dict(metric_value)
                        if hasattr(metric_value, "__iter__")
                        else metric_value
                    )

            per_combination_stats[coord] = metrics_dict

        sweep_parameters = [
            {"name": param_name, "values": sorted(per_value_aggregates.keys())}
        ]

        # Compute sweep aggregation
        sweep_dict = SweepAggregation.compute(per_combination_stats, sweep_parameters)

        # Add metadata
        sweep_mode = results[0].metadata.get("sweep_mode", "repeated")
        sweep_dict["metadata"]["sweep_mode"] = sweep_mode
        sweep_dict["metadata"]["confidence_level"] = confidence_level
        sweep_dict["metadata"]["aggregation_type"] = "sweep"

        # Determine number of trials per value
        from collections import defaultdict

        results_by_value: dict[int, list[RunResult]] = defaultdict(list)
        for result in results:
            if param_name in result.metadata:
                param_value = result.metadata[param_name]
                results_by_value[param_value].append(result)

        if results_by_value:
            first_value = sorted(results_by_value.keys())[0]
            num_trials = len(results_by_value[first_value])
            sweep_dict["metadata"]["num_trials_per_value"] = num_trials

        # Count total and successful runs
        total_runs = len(results)
        successful_runs = len([r for r in results if r.success])
        failed_run_details = [
            {
                "label": r.label,
                "error": str(r.error) if hasattr(r, "error") else "Unknown error",
            }
            for r in results
            if not r.success
        ]

        # Convert to AggregateResult
        sweep_result = AggregateResult(
            aggregation_type="sweep",
            num_runs=total_runs,
            num_successful_runs=successful_runs,
            failed_runs=failed_run_details,
            metadata=sweep_dict["metadata"],
            metrics=sweep_dict["per_combination_metrics"],
        )

        # Store additional sweep-specific data in metadata
        sweep_result.metadata["best_configurations"] = sweep_dict["best_configurations"]
        sweep_result.metadata["pareto_optimal"] = sweep_dict["pareto_optimal"]

        return sweep_result
