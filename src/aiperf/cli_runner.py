# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import sys
from typing import TYPE_CHECKING

from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader
from aiperf.plugin.enums import ServiceType, UIType

if TYPE_CHECKING:
    pass


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

    Follows the same pattern as _run_single_benchmark where the orchestrator
    handles execution, aggregation, and export internally.
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

    # Detect mode for banner
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

    # Execute runs, aggregate, and export (orchestrator handles everything)
    try:
        results = orchestrator.execute_and_export(user_config, strategy=None)
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

    # Check for failures and exit appropriately
    if is_sweep and not is_confidence:
        # Sweep-only mode
        if len(successful_runs) < len(results):
            logger.warning("Some sweep values failed. Check logs for details.")
            sys.exit(1)
        else:
            logger.info("Parameter sweep completed successfully!")
    elif len(successful_runs) < 2:
        # Confidence or sweep+confidence mode needs at least 2 successful runs
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
