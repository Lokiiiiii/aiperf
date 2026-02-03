# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution strategies for multi-run orchestration."""

import copy
import logging
from abc import ABC, abstractmethod

from aiperf.common.config import UserConfig
from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionStrategy",
    "FixedTrialsStrategy",
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

    def validate_config(self, config: UserConfig) -> None:
        """Validate that config is suitable for this strategy.

        Override this method to add strategy-specific validation.
        Called by orchestrator before starting execution.

        Args:
            config: User configuration to validate
        """
        # Default implementation: no validation required
        # Subclasses can override to add strategy-specific validation
        _ = config  # Mark as intentionally unused

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
    def get_run_path(self, base_dir, run_index: int):
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
    def get_aggregate_path(self, base_dir):
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
    ):
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
            ValueError: If num_trials < 1 or cooldown_seconds < 0
        """
        if num_trials < 1:
            raise ValueError(f"Invalid num_trials: {num_trials}. Must be at least 1.")

        if cooldown_seconds < 0:
            raise ValueError(
                f"Invalid cooldown_seconds: {cooldown_seconds}. Must be non-negative."
            )

        if num_trials > 100:
            logger.warning(
                f"Large number of trials ({num_trials}) may result in very long execution times. "
                f"Consider reducing num_trials or using a more efficient strategy."
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
        """Generate zero-padded label: run_0001, run_0002, etc.

        Args:
            run_index: Zero-based index of run

        Returns:
            Sanitized label safe for filesystem paths
        """
        label = f"run_{run_index + 1:04d}"
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
        import re

        sanitized = re.sub(r"[/\\]|\.\.", "", label)
        # Remove any other potentially dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', "", sanitized)
        return sanitized

    def get_cooldown_seconds(self) -> float:
        """Return configured cooldown duration."""
        return self.cooldown_seconds

    def get_run_path(self, base_dir, run_index: int):
        """Build path for a run's artifacts.

        For fixed trials, uses flat structure: base_dir/profile_runs/run_NNNN/

        Args:
            base_dir: Base artifact directory (Path)
            run_index: Zero-based run index

        Returns:
            Path where this run's artifacts should be stored
        """
        from pathlib import Path

        base_dir = Path(base_dir)
        label = self.get_run_label(run_index)
        return base_dir / "profile_runs" / label

    def get_aggregate_path(self, base_dir):
        """Build path for aggregate artifacts.

        For fixed trials, uses single aggregate directory: base_dir/aggregate/

        Args:
            base_dir: Base artifact directory (Path)

        Returns:
            Path where aggregate artifacts should be stored
        """
        from pathlib import Path

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
            config = copy.deepcopy(config)
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
        config = copy.deepcopy(config)
        config.loadgen.disable_warmup()
        return config
