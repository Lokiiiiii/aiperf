# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader
from aiperf.plugin.enums import ServiceType, UIType

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.orchestrator.aggregation.base import AggregateResult


def _validate_ui_compatibility(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Validate UI type compatibility with parameter sweeps.

    Raises:
        ValueError: If dashboard UI is explicitly set with parameter sweeps.
    """
    # Check if parameter sweep is enabled
    is_sweep = user_config.loadgen.get_sweep_parameter() is not None

    # Check if dashboard UI was explicitly set by user
    if (
        is_sweep
        and "ui_type" in service_config.model_fields_set
        and service_config.ui_type == UIType.DASHBOARD
    ):
        raise ValueError(
            "Dashboard UI (--ui dashboard) is not supported with parameter sweeps "
            "due to terminal control limitations when running multiple sequential benchmarks. "
            "Use --ui simple (recommended, shows progress bars) or --ui none (no UI output). "
            "Example: aiperf --concurrency 10,20,30 --ui simple ..."
        )


def run_system_controller(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run the system controller with the given configuration.

    If num_profile_runs > 1 OR parameter sweep is detected, runs multi-run orchestration.
    Otherwise, runs a single benchmark (backward compatibility).
    """
    # Validate dashboard UI is not used with parameter sweeps
    _validate_ui_compatibility(user_config, service_config)

    # Check if multi-run mode or parameter sweep is enabled
    is_sweep = user_config.loadgen.get_sweep_parameter() is not None
    is_multi_run = user_config.loadgen.num_profile_runs > 1

    if is_multi_run or is_sweep:
        _run_multi_benchmark(user_config, service_config)
    else:
        _run_single_benchmark(user_config, service_config)


def _run_single_benchmark(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run a single benchmark (original behavior)."""

    # NOTE: On macOS, when using the Textual UI with multiprocessing, terminal corruption
    # (ASCII garbage, freezing) can occur when mouse events interfere with child processes.
    # We apply multiple layers of protection:
    # 1. Set spawn method early (before any multiprocessing operations)
    # 2. Create log_queue before any UI initialization
    # 3. Set FD_CLOEXEC on terminal file descriptors
    # 4. Close terminal FDs in child processes (done in bootstrap.py)

    import multiprocessing
    import platform

    is_macos = platform.system() == "Darwin"
    using_dashboard = service_config.ui_type == UIType.DASHBOARD

    # Force spawn method on macOS to prevent fork-related issues.
    # This should already be the default, but we'll set it explicitly just in case.
    if is_macos and using_dashboard:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("spawn", force=True)

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.common.tokenizer_validator import validate_tokenizer_early

    logger = AIPerfLogger(__name__)

    # Create log_queue before UI initialization to minimize FD inheritance issues.
    log_queue = None
    if using_dashboard:
        from aiperf.common.logging import get_global_log_queue

        log_queue = get_global_log_queue()

        # Set FD_CLOEXEC on terminal file descriptors on macOS.
        # This ensures terminal FDs are closed when child processes spawn.
        if is_macos:
            import fcntl
            import sys

            try:
                for fd in [
                    sys.stdin.fileno(),
                    sys.stdout.fileno(),
                    sys.stderr.fileno(),
                ]:
                    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
                    fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
                logger.debug("Set FD_CLOEXEC on terminal file descriptors for macOS")
            except (OSError, ValueError, AttributeError) as e:
                # Non-fatal if this fails, other layers will protect
                logger.debug(f"Could not set FD_CLOEXEC on terminal descriptors: {e}")
    else:
        from aiperf.common.logging import setup_rich_logging

        setup_rich_logging(user_config, service_config)

    # Create and start the system controller
    logger.info("Starting AIPerf System")

    # Validate tokenizer early (before spawning services) to fail fast.
    user_config.tokenizer.resolved_names = validate_tokenizer_early(user_config, logger)

    # Validate custom GPU metrics CSV file
    if user_config.gpu_telemetry_metrics_file:
        try:
            csv_path = user_config.gpu_telemetry_metrics_file
            logger.info(f"Custom GPU metrics file configured: {csv_path}")

            loader = MetricsConfigLoader()
            custom_metrics, _ = loader.build_custom_metrics_from_csv(csv_path)
            logger.info(
                f"Validated {len(custom_metrics)} custom metrics from {csv_path}"
            )
        except Exception as e:
            logger.exception("Error validating custom GPU metrics file")
            raise_startup_error_and_exit(
                f"Invalid custom GPU metrics file: {e}",
                title="GPU Metrics Configuration Error",
            )

    try:
        bootstrap_and_run_service(
            service_type=ServiceType.SYSTEM_CONTROLLER,
            service_config=service_config,
            user_config=user_config,
            log_queue=log_queue,
        )
    except Exception:
        logger.exception("Error running AIPerf System")
        raise
    finally:
        logger.debug("AIPerf System exited")


def _run_multi_benchmark(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run multiple benchmarks for confidence reporting or parameter sweeps.

    Executes benchmarks according to the detected mode:
    - Parameter sweep only: Tests different parameter values
    - Confidence trials only: Runs same config multiple times
    - Both: Combines sweep and confidence (nested execution)
    """
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.logging import setup_rich_logging
    from aiperf.orchestrator.orchestrator import MultiRunOrchestrator

    # Validate and adjust UI type for multi-run mode
    if (
        "ui_type" in service_config.model_fields_set
        and service_config.ui_type == UIType.DASHBOARD
    ):
        raise ValueError(
            "Dashboard UI (--ui dashboard) is not supported with multi-run mode (--num-profile-runs > 1) "
            "or parameter sweeps due to terminal control limitations when running multiple sequential benchmarks. "
            "Use --ui simple (recommended, shows progress bars) or --ui none (no UI output). "
            "Example: aiperf --concurrency 10,20,30 --ui simple ..."
        )

    # Set default to simple if ui_type wasn't explicitly set
    if "ui_type" not in service_config.model_fields_set:
        service_config.ui_type = UIType.SIMPLE

    # Set up logging so output is visible
    setup_rich_logging(user_config, service_config)

    logger = AIPerfLogger(__name__)

    # Inform user about UI mode (now that logging is set up)
    if "ui_type" not in service_config.model_fields_set:
        logger.info(
            "Multi-run mode: UI automatically set to 'simple' "
            "(use '--ui none' to disable UI output)"
        )

    # Detect mode
    sweep_info = user_config.loadgen.get_sweep_parameter()
    is_sweep = sweep_info is not None
    is_confidence = user_config.loadgen.num_profile_runs > 1

    # Print banner based on mode
    logger.info("=" * 80)
    if is_sweep and is_confidence:
        param_name, param_values = sweep_info
        logger.info("Starting Parameter Sweep with Confidence Trials")
        logger.info(f"  Parameter: {param_name} = {param_values}")
        logger.info(f"  Sweep mode: {user_config.loadgen.parameter_sweep_mode}")
        logger.info(f"  Trials per value: {user_config.loadgen.num_profile_runs}")
        logger.info(f"  Confidence level: {user_config.loadgen.confidence_level:.0%}")
    elif is_sweep:
        param_name, param_values = sweep_info
        logger.info("Starting Parameter Sweep")
        logger.info(f"  Parameter: {param_name} = {param_values}")
        logger.info(
            f"  Sweep cooldown: {user_config.loadgen.parameter_sweep_cooldown_seconds}s"
        )
    elif is_confidence:
        logger.info("Starting Multi-Run Confidence Reporting")
        logger.info(f"  Number of runs: {user_config.loadgen.num_profile_runs}")
        logger.info(f"  Confidence level: {user_config.loadgen.confidence_level:.0%}")
        logger.info(
            f"  Cooldown between runs: {user_config.loadgen.profile_run_cooldown_seconds}s"
        )
    logger.info("=" * 80)

    # Create orchestrator
    orchestrator = MultiRunOrchestrator(
        base_dir=user_config.output.artifact_directory, service_config=service_config
    )

    # Execute runs
    try:
        results = orchestrator.execute(user_config, strategy=None)
    except Exception:
        logger.exception("Error executing multi-run benchmark")
        raise

    # Count successful runs
    successful_runs = [r for r in results if r.success]
    failed_runs = [r for r in results if not r.success]

    logger.info("=" * 80)
    logger.info(f"All runs complete: {len(successful_runs)}/{len(results)} successful")
    if failed_runs:
        logger.warning(f"Failed runs: {', '.join(r.label for r in failed_runs)}")
    logger.info("=" * 80)

    # For sweep-only mode, we're done (no aggregation needed)
    if is_sweep and not is_confidence:
        if len(successful_runs) < len(results):
            logger.warning("Some sweep values failed. Check logs for details.")
            sys.exit(1)
        else:
            logger.info("Parameter sweep completed successfully!")
        return

    # Check if we have enough successful runs for aggregation
    if len(successful_runs) < 2:
        if len(successful_runs) == 1:
            logger.warning(
                "Only 1 successful run - cannot compute confidence statistics. "
                "At least 2 successful runs are required."
            )
        else:
            logger.error(
                "All runs failed - cannot compute aggregate statistics. "
                "Please check the error messages above."
            )
        sys.exit(1)

    # Aggregate results using orchestrator
    logger.info("Computing aggregate statistics...")
    try:
        aggregates = orchestrator.aggregate_results(results, user_config)
    except Exception:
        logger.exception("Error computing aggregate statistics")
        raise

    # Export and display aggregates based on mode
    if "aggregate" in aggregates:
        # Confidence-only mode
        _export_confidence_aggregate(
            aggregates["aggregate"], user_config.output.artifact_directory, logger
        )
    elif "per_value_aggregates" in aggregates and "sweep_aggregate" in aggregates:
        # Sweep + confidence mode
        _export_sweep_aggregates(
            aggregates["per_value_aggregates"],
            aggregates["sweep_aggregate"],
            user_config.output.artifact_directory,
            logger,
        )
    else:
        logger.warning("No aggregates to export")


def _export_confidence_aggregate(
    aggregate_result: "AggregateResult", base_dir: Path, logger: "AIPerfLogger"
) -> None:
    """Export confidence-only aggregate results.

    Args:
        aggregate_result: Aggregate result to export
        base_dir: Base artifact directory
        logger: Logger instance
    """
    import asyncio

    from aiperf.exporters.aggregate import (
        AggregateConfidenceCsvExporter,
        AggregateConfidenceJsonExporter,
        AggregateExporterConfig,
    )

    aggregate_dir = Path(base_dir) / "profile_runs" / "aggregate"

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

    # Print summary
    _print_aggregate_summary(aggregate_result, logger)


def _export_sweep_aggregates(
    per_value_aggregates: dict[int, "AggregateResult"],
    sweep_aggregate: "AggregateResult",
    base_dir: Path,
    logger: "AIPerfLogger",
) -> None:
    """Export sweep + confidence aggregate results.

    Args:
        per_value_aggregates: Per-value aggregate results
        sweep_aggregate: Sweep-level aggregate result
        base_dir: Base artifact directory
        logger: Logger instance
    """
    import asyncio

    from aiperf.exporters.aggregate import (
        AggregateConfidenceCsvExporter,
        AggregateConfidenceJsonExporter,
        AggregateExporterConfig,
        AggregateSweepCsvExporter,
        AggregateSweepJsonExporter,
    )

    base_path = Path(base_dir)
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
        param_value: int, aggregate: "AggregateResult"
    ) -> tuple[Path, Path]:
        """Export single per-value aggregate."""
        if sweep_mode == "repeated":
            agg_dir = base_path / "aggregate" / f"{param_name}_{param_value}"
        else:  # independent
            agg_dir = base_path / f"{param_name}_{param_value}" / "aggregate"

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
    async def export_all_per_value():
        """Export all per-value aggregates."""
        tasks = [
            export_per_value(param_value, aggregate)
            for param_value, aggregate in per_value_aggregates.items()
        ]
        return await asyncio.gather(*tasks)

    asyncio.run(export_all_per_value())

    logger.info(
        f"Exported {len(per_value_aggregates)} per-value aggregates to {base_path}"
    )

    # Export sweep-level aggregate
    logger.info("Exporting sweep-level aggregate...")

    if sweep_mode == "repeated":
        sweep_dir = base_path / "aggregate" / "sweep_aggregate"
    else:  # independent
        sweep_dir = base_path / "sweep_aggregate"

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


def _print_aggregate_summary(
    aggregate_result: "AggregateResult", logger: "AIPerfLogger"
) -> None:
    """Print a comprehensive summary of aggregate statistics to console.

    Args:
        aggregate_result: AggregateResult with computed statistics
        logger: Logger instance for output
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info("AGGREGATE STATISTICS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Aggregation Type: {aggregate_result.aggregation_type}")
    logger.info(f"Total Runs: {aggregate_result.num_runs}")
    logger.info(f"Successful Runs: {aggregate_result.num_successful_runs}")

    if aggregate_result.failed_runs:
        logger.warning(f"Failed Runs ({len(aggregate_result.failed_runs)}):")
        for failed in aggregate_result.failed_runs:
            logger.warning(f"  - {failed['label']}: {failed['error']}")

    # Get confidence level from metadata
    confidence_level = aggregate_result.metadata.get("confidence_level", 0.95)
    logger.info(f"Confidence Level: {confidence_level:.0%}")

    logger.info("")
    logger.info("Key Metrics:")
    logger.info("-" * 80)

    # Define priority metrics to display (in order of preference)
    # We'll look for these base metric names with _avg, _p99, _max suffixes
    priority_metrics = [
        "request_throughput",
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
    ]

    # Build list of metrics to display by finding available stat variants
    metrics_to_display = []
    for base_metric in priority_metrics:
        # Look for _avg first (most common), then _p99, then _max
        for suffix in ["_avg", "_p99", "_max", "_p50"]:
            metric_key = f"{base_metric}{suffix}"
            if metric_key in aggregate_result.metrics:
                # Create display name (e.g., "Request Throughput (Avg)")
                display_name = base_metric.replace("_", " ").title()
                stat_name = suffix[1:].upper()  # Remove leading underscore
                if stat_name == "AVG":
                    stat_name = "Avg"
                elif stat_name.startswith("P"):
                    stat_name = f"P{stat_name[1:]}"  # P99, P50, etc.
                else:
                    stat_name = stat_name.capitalize()

                metrics_to_display.append((metric_key, f"{display_name} ({stat_name})"))
                break  # Only show one stat variant per base metric

    metrics_found = 0
    for metric_key, display_name in metrics_to_display:
        metric = aggregate_result.metrics[metric_key]
        logger.info(f"\n{display_name}:")
        logger.info(f"  Mean:    {metric.mean:>12.4f} {metric.unit}")
        logger.info(f"  Std Dev: {metric.std:>12.4f} {metric.unit}")
        logger.info(f"  Min:     {metric.min:>12.4f} {metric.unit}")
        logger.info(f"  Max:     {metric.max:>12.4f} {metric.unit}")
        logger.info(f"  CV:      {metric.cv:>12.2%}")
        logger.info(
            f"  {confidence_level:.0%} CI: [{metric.ci_low:.4f}, {metric.ci_high:.4f}] {metric.unit}"
        )
        metrics_found += 1

    if metrics_found == 0:
        logger.warning("No key metrics found in aggregate results")

    logger.info("")
    logger.info("-" * 80)
    logger.info("Coefficient of Variation (CV) Interpretation Guide:")
    logger.info("  CV < 5%:   Excellent repeatability (low variance)")
    logger.info("  CV 5-10%:  Good repeatability (moderate variance)")
    logger.info("  CV 10-20%: Fair repeatability (consider more runs)")
    logger.info("  CV > 20%:  High variance (investigate or increase runs)")
    logger.info("")
    logger.info("Confidence Interval (CI) Interpretation:")
    logger.info(
        f"  The {confidence_level:.0%} CI indicates the range where the true mean"
    )
    logger.info(f"  is likely to fall with {confidence_level:.0%} confidence.")
    logger.info("  Narrower intervals indicate more precise estimates.")
    logger.info("=" * 80)
