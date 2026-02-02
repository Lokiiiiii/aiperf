# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader
from aiperf.plugin.enums import ServiceType, UIType


def run_system_controller(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run the system controller with the given configuration.

    If num_profile_runs > 1, runs multi-run orchestration for confidence reporting.
    Otherwise, runs a single benchmark (backward compatibility).
    """
    # Check if multi-run mode is enabled
    if user_config.loadgen.num_profile_runs > 1:
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
    """Run multiple benchmarks for confidence reporting.

    Executes num_profile_runs benchmarks with the same configuration,
    then aggregates results and computes confidence statistics.
    """
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.exporters.aggregate import (
        AggregateConfidenceCsvExporter,
        AggregateConfidenceJsonExporter,
        AggregateExporterConfig,
    )
    from aiperf.module_loader import ensure_modules_loaded
    from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation
    from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
    from aiperf.orchestrator.strategies import FixedTrialsStrategy

    logger = AIPerfLogger(__name__)

    # Ensure modules are loaded
    try:
        ensure_modules_loaded()
    except Exception as e:
        raise_startup_error_and_exit(
            f"Error loading modules: {e}",
            title="Error Loading Modules",
        )

    # Print multi-run banner
    num_runs = user_config.loadgen.num_profile_runs
    confidence_level = user_config.loadgen.confidence_level
    cooldown = user_config.loadgen.profile_run_cooldown_seconds

    logger.info("=" * 80)
    logger.info("Starting Multi-Run Confidence Reporting")
    logger.info(f"  Number of runs: {num_runs}")
    logger.info(f"  Confidence level: {confidence_level:.0%}")
    logger.info(f"  Cooldown between runs: {cooldown}s")
    logger.info("=" * 80)

    # Create strategy
    strategy = FixedTrialsStrategy(
        num_trials=num_runs,
        cooldown_seconds=cooldown,
        auto_set_seed=True,
        disable_warmup_after_first=user_config.loadgen.profile_run_disable_warmup_after_first,
    )

    # Create orchestrator
    orchestrator = MultiRunOrchestrator(
        base_dir=user_config.output.artifact_directory, service_config=service_config
    )

    # Execute runs
    try:
        results = orchestrator.execute(user_config, strategy)
    except Exception:
        logger.exception("Error executing multi-run benchmark")
        raise

    # Count successful runs
    successful_runs = [r for r in results if r.success]
    failed_runs = [r for r in results if not r.success]

    logger.info("=" * 80)
    logger.info(f"All runs complete: {len(successful_runs)}/{num_runs} successful")
    if failed_runs:
        logger.warning(f"Failed runs: {', '.join(r.label for r in failed_runs)}")
    logger.info("=" * 80)

    # Aggregate results if we have at least 2 successful runs
    if len(successful_runs) >= 2:
        logger.info("Computing aggregate statistics...")

        aggregation = ConfidenceAggregation(confidence_level=confidence_level)
        aggregate_result = aggregation.aggregate(results)

        # Add cooldown to metadata
        aggregate_result.metadata["cooldown_seconds"] = cooldown

        # Write aggregate artifacts using exporters directly
        aggregate_dir = strategy.get_aggregate_path(
            user_config.output.artifact_directory
        )
        aggregate_dir.mkdir(parents=True, exist_ok=True)

        # Create exporter config
        exporter_config = AggregateExporterConfig(
            result=aggregate_result,
            output_dir=aggregate_dir,
        )

        # Export JSON
        json_exporter = AggregateConfidenceJsonExporter(exporter_config)
        json_path = json_exporter.export_sync()

        # Export CSV
        csv_exporter = AggregateConfidenceCsvExporter(exporter_config)
        csv_path = csv_exporter.export_sync()

        logger.info(f"Aggregate JSON written to: {json_path}")
        logger.info(f"Aggregate CSV written to: {csv_path}")

        # Print summary
        _print_aggregate_summary(aggregate_result, logger)
    elif len(successful_runs) == 1:
        logger.warning(
            "Only 1 successful run - cannot compute confidence statistics. "
            "At least 2 successful runs are required."
        )
        import sys

        sys.exit(1)
    else:
        logger.error(
            "All runs failed - cannot compute aggregate statistics. "
            "Please check the error messages above."
        )
        import sys

        sys.exit(1)


def _print_aggregate_summary(aggregate_result, logger) -> None:
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

    # Define key metrics to display with their display names
    key_metrics = [
        ("request_throughput", "Request Throughput"),
        ("time_to_first_token_avg", "Time to First Token (Avg)"),
        ("time_to_first_token_p99", "Time to First Token (P99)"),
        ("inter_token_latency_avg", "Inter-Token Latency (Avg)"),
        ("inter_token_latency_p99", "Inter-Token Latency (P99)"),
        ("request_latency_avg", "Request Latency (Avg)"),
        ("request_latency_p99", "Request Latency (P99)"),
    ]

    metrics_found = 0
    for metric_name, display_name in key_metrics:
        if metric_name in aggregate_result.metrics:
            metric = aggregate_result.metrics[metric_name]
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
