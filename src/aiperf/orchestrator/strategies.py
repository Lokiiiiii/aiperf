# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution strategies for multi-run orchestration."""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from aiperf.common.config import UserConfig
from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionStrategy",
    "FixedTrialsStrategy",
    "ParameterSweepStrategy",
]


class ExecutionStrategy(ABC):
    """Base class for execution strategies.

    Strategies decide:
    1. What config to run next (based on results so far)
    2. Whether to continue or stop
    3. How to label runs for artifact organization
    4. Where to store artifacts (path structure)
    5. Cooldown duration between runs
    """

    def validate_config(self, config: UserConfig) -> None:  # noqa: B027
        """Validate that config is suitable for this strategy.

        Override this method to add strategy-specific validation.
        Called by orchestrator before starting execution.

        Args:
            config: User configuration to validate
        """
        # Default implementation: no validation required
        # Subclasses can override to add strategy-specific validation
        pass

    @abstractmethod
    def should_continue(self, results: list[RunResult]) -> bool:
        """Decide whether to run another trial.

        Args:
            results: Results from runs executed so far

        Returns:
            True if should run another trial, False to stop
        """
        pass

    @abstractmethod
    def get_next_config(
        self, base_config: UserConfig, results: list[RunResult]
    ) -> UserConfig:
        """Generate config for next run.

        Args:
            base_config: Base benchmark configuration
            results: Results from runs executed so far

        Returns:
            Configuration for next run
        """
        pass

    @abstractmethod
    def get_run_label(self, run_index: int) -> str:
        """Generate label for run at given index.

        Args:
            run_index: Zero-based index of run

        Returns:
            Label for run (e.g., "run_0001")
        """
        pass

    @abstractmethod
    def get_cooldown_seconds(self) -> float:
        """Return cooldown duration between runs."""
        pass

    @abstractmethod
    def get_run_path(self, base_dir: Path, run_index: int) -> Path:
        """Build path for a run's artifacts.

        Strategy decides the directory structure based on its execution model.
        This allows different strategies to organize artifacts differently:
        - Fixed trials: flat structure (profile_runs/run_0001/)
        - Parameter sweep: hierarchical (concurrency_10/run_0001/)

        Args:
            base_dir: Base artifact directory (Path)
            run_index: Zero-based run index

        Returns:
            Path where this run's artifacts should be stored
        """
        pass

    @abstractmethod
    def get_aggregate_path(self, base_dir: Path) -> Path:
        """Build path for aggregate artifacts.

        Strategy decides where aggregate statistics should be stored.
        This allows different strategies to organize aggregates differently:
        - Fixed trials: single aggregate (aggregate/)
        - Parameter sweep: per-parameter aggregates (concurrency_10/aggregate/)

        Args:
            base_dir: Base artifact directory (Path)

        Returns:
            Path where aggregate artifacts should be stored
        """
        pass


class FixedTrialsStrategy(ExecutionStrategy):
    """Strategy for fixed number of trials with identical config.

    Used for confidence reporting: run same benchmark N times to quantify variance.

    Attributes:
        num_trials: Number of trials to run
        cooldown_seconds: Sleep duration between trials
        auto_set_seed: Auto-set random seed if not specified
    """

    DEFAULT_SEED = 42

    def __init__(
        self,
        num_trials: int,
        cooldown_seconds: float = 0.0,
        auto_set_seed: bool = True,
        disable_warmup_after_first: bool = True,
    ) -> None:
        """Initialize FixedTrialsStrategy.

        Args:
            num_trials: Number of trials to run (must be between 1 and 10)
            cooldown_seconds: Sleep duration between trials (must be >= 0)
            auto_set_seed: Auto-set random seed if not specified
            disable_warmup_after_first: Disable warmup for runs after the first.
                If True (default), only the first run includes warmup, subsequent
                runs measure steady-state performance. If False, all runs include
                warmup (useful for long cooldown periods or cold-start scenarios).

        Raises:
            ValueError: If cooldown_seconds < 0

        Note:
            num_trials validation is handled by Pydantic at the config level
            (LoadGeneratorConfig.num_profile_runs with ge=1, le=10 constraints).
        """
        if cooldown_seconds < 0:
            raise ValueError(
                f"Invalid cooldown duration: {cooldown_seconds} seconds. "
                f"Cooldown must be non-negative (0 or greater). "
                f"Use 0 for no cooldown, or a positive value like 10 for a 10-second pause between runs."
            )

        self.num_trials = num_trials
        self.cooldown_seconds = cooldown_seconds
        self.auto_set_seed = auto_set_seed
        self.disable_warmup_after_first = disable_warmup_after_first

    def validate_config(self, config: UserConfig) -> None:
        """Validate that config is suitable for this strategy.

        For FixedTrialsStrategy with multiple trials, warns if random seed
        is not set and auto_set_seed is disabled, as this may result in
        different workloads across runs.

        Args:
            config: User configuration to validate
        """
        if (
            self.num_trials > 1
            and config.input.random_seed is None
            and not self.auto_set_seed
        ):
            logger.warning(
                "No random seed specified and auto_set_seed is disabled. "
                "This may result in different workloads across runs, "
                "making confidence statistics less meaningful. "
                "Consider setting --random-seed explicitly."
            )

    def should_continue(self, results: list[RunResult]) -> bool:
        """Continue until we've run num_trials."""
        return len(results) < self.num_trials

    def get_next_config(
        self, base_config: UserConfig, results: list[RunResult]
    ) -> UserConfig:
        """Return config for next trial.

        For first trial: ensure random seed is set (for workload consistency).
        For subsequent trials: optionally disable warmup based on strategy settings.

        Warmup behavior is controlled by disable_warmup_after_first:
        - True (default): Only first run has warmup, subsequent runs measure steady-state
        - False: All runs include warmup (useful for long cooldowns or cold-start testing)
        """
        config = base_config

        # Ensure seed is set for all trials when auto_set_seed is enabled
        # This ensures the seed persists across all runs
        if self.auto_set_seed:
            config = self._ensure_random_seed(config)

        # Subsequent trials: optionally disable warmup
        if len(results) > 0 and self.disable_warmup_after_first:
            config = self._disable_warmup(config)

        return config

    def get_run_label(self, run_index: int) -> str:
        """Generate zero-padded label: trial_0001, trial_0002, etc.

        Args:
            run_index: Zero-based index of run

        Returns:
            Sanitized label safe for filesystem paths
        """
        label = f"trial_{run_index + 1:04d}"
        # Sanitize label to prevent path traversal
        return self._sanitize_label(label)

    def _sanitize_label(self, label: str) -> str:
        """Sanitize label to prevent path traversal attacks.

        Args:
            label: Raw label string

        Returns:
            Sanitized label safe for filesystem paths
        """
        # Remove any path separators and parent directory references
        sanitized = re.sub(r"[/\\]|\.\.", "", label)
        # Remove any other potentially dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', "", sanitized)
        return sanitized

    def get_cooldown_seconds(self) -> float:
        """Return configured cooldown duration."""
        return self.cooldown_seconds

    def get_run_path(self, base_dir: Path, run_index: int) -> Path:
        """Build path for a run's artifacts.

        For fixed trials, uses flat structure: base_dir/profile_runs/trial_NNNN/

        Directory Structure Example:
        When base_dir is the auto-generated artifact directory:
        artifacts/llama-3-8b-openai-chat-concurrency_10/profile_runs/trial_0001/
        artifacts/llama-3-8b-openai-chat-concurrency_10/profile_runs/trial_0002/
        artifacts/llama-3-8b-openai-chat-concurrency_10/aggregate/

        The base_dir includes the auto-generated name with model, service, and stimulus:
        - Model name (e.g., "llama-3-8b")
        - Service kind and endpoint type (e.g., "openai-chat")
        - Stimulus (e.g., "concurrency_10", "request_rate_100")

        Args:
            base_dir: Base artifact directory (Path)
            run_index: Zero-based run index

        Returns:
            Path where this run's artifacts should be stored
        """
        base_dir = Path(base_dir)
        label = self.get_run_label(run_index)
        return base_dir / "profile_runs" / label

    def get_aggregate_path(self, base_dir: Path) -> Path:
        """Build path for aggregate artifacts.

        For fixed trials, uses single aggregate directory: base_dir/aggregate/

        Directory Structure Example:
        artifacts/llama-3-8b-openai-chat-concurrency_10/aggregate/

        This directory contains the aggregated results across all runs:
        - profile_export_aiperf_aggregate.json
        - profile_export_aiperf_aggregate.csv

        Args:
            base_dir: Base artifact directory (Path)

        Returns:
            Path where aggregate artifacts should be stored
        """
        base_dir = Path(base_dir)
        return base_dir / "aggregate"

    def _ensure_random_seed(self, config: UserConfig) -> UserConfig:
        """Ensure config has random seed set.

        Auto-sets seed if not specified for multi-run consistency.

        Args:
            config: Base configuration

        Returns:
            Configuration with random seed set
        """
        if config.input.random_seed is None:
            logger.info(
                f"No --random-seed specified. Using default seed {self.DEFAULT_SEED} "
                f"for multi-run consistency. All runs will use identical workloads."
            )
            config = config.model_copy(deep=True)
            config.input.random_seed = self.DEFAULT_SEED

        return config

    def _disable_warmup(self, config: UserConfig) -> UserConfig:
        """Create config copy with warmup disabled.

        This is called for subsequent trials when disable_warmup_after_first=True.
        Disabling warmup after the first trial provides more accurate aggregate
        statistics by measuring steady-state performance without warmup overhead.

        However, users may want warmup on all runs (disable_warmup_after_first=False)
        for scenarios like:
        - Long cooldown periods where system returns to cold state
        - Testing cold-start performance explicitly
        - Ensuring consistent conditions across all runs

        Delegates to LoadGeneratorConfig.disable_warmup() which maintains
        the authoritative list of warmup fields.

        Args:
            config: Original configuration

        Returns:
            Configuration with all warmup parameters set to None
        """
        config = config.model_copy(deep=True)
        config.loadgen.disable_warmup()
        return config


class ParameterSweepStrategy(ExecutionStrategy):
    """Strategy for sweeping a single parameter across multiple values.

    This strategy is COMPLETELY INDEPENDENT - it knows nothing about confidence
    trials or FixedTrialsStrategy. It ONLY handles varying a parameter.

    The orchestrator composes this with FixedTrialsStrategy when both are needed,
    using each strategy's path generation to build nested directory structures.

    Attributes:
        parameter_name: Name of parameter to sweep (e.g., "concurrency")
        parameter_values: List of values to test
        cooldown_seconds: Cooldown between parameter values
        same_seed: Use same seed for all values (default: derive different seeds)
        auto_set_seed: Auto-set base seed if not specified
        base_seed: Base seed for derivation (set on first get_next_config call)
    """

    DEFAULT_SEED = 42

    def __init__(
        self,
        parameter_name: str,
        parameter_values: list[int],
        cooldown_seconds: float = 0.0,
        same_seed: bool = False,
        auto_set_seed: bool = True,
    ) -> None:
        """Initialize parameter sweep strategy.

        Args:
            parameter_name: Name of parameter to sweep (e.g., "concurrency")
            parameter_values: List of values to test
            cooldown_seconds: Cooldown between parameter values (must be >= 0)
            same_seed: Use same seed for all values (default: derive different seeds)
            auto_set_seed: Auto-set base seed if not specified

        Raises:
            ValueError: If cooldown_seconds < 0 or parameter_values is empty
        """
        if cooldown_seconds < 0:
            raise ValueError(
                f"Invalid cooldown duration: {cooldown_seconds} seconds. "
                f"Cooldown must be non-negative (0 or greater). "
                f"Use 0 for no cooldown, or a positive value like 10 for a 10-second pause between parameter values."
            )
        if not parameter_values:
            raise ValueError(
                "Parameter sweep requires at least one value to test. "
                "Provide a comma-separated list of values: --concurrency 10,20,30. "
                "For a single value, use: --concurrency 10 (no comma)."
            )

        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.cooldown_seconds = cooldown_seconds
        self.same_seed = same_seed
        self.auto_set_seed = auto_set_seed
        self.base_seed: int | None = None

    def should_continue(self, results: list[RunResult]) -> bool:
        """Continue until all parameter values are tested.

        Args:
            results: Results from runs executed so far

        Returns:
            True if more parameter values remain, False otherwise
        """
        return len(results) < len(self.parameter_values)

    def get_next_config(
        self, base_config: UserConfig, results: list[RunResult]
    ) -> UserConfig:
        """Generate config for next parameter value.

        Sets the parameter value and derives appropriate random seed.

        Seed derivation logic:
        - Base seed is determined on first call (user-specified or auto-set)
        - If same_seed=True: all values use base_seed
        - If same_seed=False: each value uses base_seed + value_index

        Args:
            base_config: Base benchmark configuration
            results: Results from runs executed so far

        Returns:
            Configuration for next parameter value
        """
        value_index = len(results)
        value = self.parameter_values[value_index]

        # Create deep copy to avoid modifying base config
        config = base_config.model_copy(deep=True)

        # Set parameter value (now a single value, not a list)
        setattr(config.loadgen, self.parameter_name, value)

        # Clear parameter_sweep_mode since we're no longer sweeping in the subprocess
        # The orchestrator handles the sweep at a higher level
        config.loadgen.parameter_sweep_mode = None

        # Initialize base seed on first call
        if self.base_seed is None:
            if config.input.random_seed is not None:
                self.base_seed = config.input.random_seed
            elif self.auto_set_seed:
                self.base_seed = self.DEFAULT_SEED
                logger.info(
                    f"No --random-seed specified. Using default seed {self.DEFAULT_SEED} "
                    f"for parameter sweep consistency."
                )

        # Derive seed for this value
        if self.base_seed is not None:
            if self.same_seed:
                seed = self.base_seed
                if value_index == 0:
                    logger.info(
                        f"Using same seed ({seed}) across all sweep values "
                        f"(--parameter-sweep-same-seed enabled)."
                    )
            else:
                seed = self.base_seed + value_index
                if value_index == 0:
                    logger.info(
                        f"Deriving different seeds per sweep value from base seed {self.base_seed}."
                    )
            config.input.random_seed = seed

        return config

    def get_run_label(self, run_index: int) -> str:
        """Generate label: concurrency_10, concurrency_20, etc.

        Args:
            run_index: Zero-based index of run

        Returns:
            Sanitized label safe for filesystem paths
        """
        value = self.parameter_values[run_index]
        label = f"{self.parameter_name}_{value}"
        # Sanitize label to prevent path traversal
        return self._sanitize_label(label)

    def _sanitize_label(self, label: str) -> str:
        """Sanitize label to prevent path traversal attacks.

        Args:
            label: Raw label string

        Returns:
            Sanitized label safe for filesystem paths
        """
        # Remove any path separators and parent directory references
        sanitized = re.sub(r"[/\\]|\.\.", "", label)
        # Remove any other potentially dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', "", sanitized)
        return sanitized

    def get_cooldown_seconds(self) -> float:
        """Return cooldown between parameter values."""
        return self.cooldown_seconds

    def get_run_path(self, base_dir: Path, run_index: int) -> Path:
        """Build path relative to base_dir.

        Returns: base_dir/concurrency_10/

        When composed, orchestrator uses this as base_dir for inner strategy.

        Directory Structure Examples:

        Single sweep (no composition):
        base_dir/concurrency_10/
        base_dir/concurrency_20/

        Composed with FixedTrialsStrategy (repeated mode):
        base_dir/profile_runs/run_0001/concurrency_10/
        base_dir/profile_runs/run_0001/concurrency_20/

        Composed with FixedTrialsStrategy (independent mode):
        base_dir/concurrency_10/profile_runs/run_0001/
        base_dir/concurrency_20/profile_runs/run_0001/

        Args:
            base_dir: Base artifact directory (Path)
            run_index: Zero-based run index

        Returns:
            Path where this parameter value's artifacts should be stored
        """
        base_dir = Path(base_dir)
        label = self.get_run_label(run_index)
        return base_dir / label

    def get_aggregate_path(self, base_dir: Path) -> Path:
        """Build aggregate path relative to base_dir.

        Returns: base_dir/sweep_aggregate/

        This directory contains sweep-level aggregate statistics comparing
        performance across all parameter values.

        Args:
            base_dir: Base artifact directory (Path)

        Returns:
            Path where sweep aggregate artifacts should be stored
        """
        base_dir = Path(base_dir)
        return base_dir / "sweep_aggregate"
