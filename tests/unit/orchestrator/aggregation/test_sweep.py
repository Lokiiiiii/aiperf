# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep aggregation components."""

import pytest

from aiperf.orchestrator.aggregation import (
    DEFAULT_PARETO_OBJECTIVES,
    Objective,
    OptimizationDirection,
)


class TestOptimizationDirection:
    """Tests for OptimizationDirection enum."""

    def test_maximize_value(self):
        """Test MAXIMIZE enum value."""
        assert OptimizationDirection.MAXIMIZE.value == "maximize"

    def test_minimize_value(self):
        """Test MINIMIZE enum value."""
        assert OptimizationDirection.MINIMIZE.value == "minimize"

    def test_enum_members(self):
        """Test that enum has exactly two members."""
        assert len(OptimizationDirection) == 2
        assert set(OptimizationDirection) == {
            OptimizationDirection.MAXIMIZE,
            OptimizationDirection.MINIMIZE,
        }


class TestObjective:
    """Tests for Objective named tuple."""

    def test_create_maximize_objective(self):
        """Test creating an objective with MAXIMIZE direction."""
        obj = Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        assert obj.metric_key == "request_throughput_avg"
        assert obj.direction == OptimizationDirection.MAXIMIZE

    def test_create_minimize_objective(self):
        """Test creating an objective with MINIMIZE direction."""
        obj = Objective("ttft_p99_ms", OptimizationDirection.MINIMIZE)
        assert obj.metric_key == "ttft_p99_ms"
        assert obj.direction == OptimizationDirection.MINIMIZE

    def test_objective_is_immutable(self):
        """Test that Objective is immutable (NamedTuple property)."""
        obj = Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        with pytest.raises(AttributeError):
            obj.metric_key = "new_metric"

    def test_objective_equality(self):
        """Test that objectives with same values are equal."""
        obj1 = Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        obj2 = Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        assert obj1 == obj2

    def test_objective_inequality(self):
        """Test that objectives with different values are not equal."""
        obj1 = Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        obj2 = Objective("ttft_p99_ms", OptimizationDirection.MINIMIZE)
        assert obj1 != obj2

    def test_objective_tuple_unpacking(self):
        """Test that Objective can be unpacked like a tuple."""
        obj = Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        metric_key, direction = obj
        assert metric_key == "request_throughput_avg"
        assert direction == OptimizationDirection.MAXIMIZE

    def test_objective_indexing(self):
        """Test that Objective supports indexing."""
        obj = Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        assert obj[0] == "request_throughput_avg"
        assert obj[1] == OptimizationDirection.MAXIMIZE

    def test_objective_repr(self):
        """Test that Objective has a useful repr."""
        obj = Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        repr_str = repr(obj)
        assert "Objective" in repr_str
        assert "request_throughput_avg" in repr_str
        assert "MAXIMIZE" in repr_str


class TestDefaultParetoObjectives:
    """Tests for DEFAULT_PARETO_OBJECTIVES constant."""

    def test_default_objectives_is_list(self):
        """Test that DEFAULT_PARETO_OBJECTIVES is a list."""
        assert isinstance(DEFAULT_PARETO_OBJECTIVES, list)

    def test_default_objectives_has_two_objectives(self):
        """Test that DEFAULT_PARETO_OBJECTIVES contains exactly two objectives."""
        assert len(DEFAULT_PARETO_OBJECTIVES) == 2

    def test_default_objectives_contains_objective_instances(self):
        """Test that all items in DEFAULT_PARETO_OBJECTIVES are Objective instances."""
        for obj in DEFAULT_PARETO_OBJECTIVES:
            assert isinstance(obj, Objective)

    def test_default_objectives_first_is_throughput_maximize(self):
        """Test that first objective is to maximize request_throughput_avg."""
        obj = DEFAULT_PARETO_OBJECTIVES[0]
        assert obj.metric_key == "request_throughput_avg"
        assert obj.direction == OptimizationDirection.MAXIMIZE

    def test_default_objectives_second_is_latency_minimize(self):
        """Test that second objective is to minimize ttft_p99_ms."""
        obj = DEFAULT_PARETO_OBJECTIVES[1]
        assert obj.metric_key == "ttft_p99_ms"
        assert obj.direction == OptimizationDirection.MINIMIZE

    def test_default_objectives_immutable(self):
        """Test that DEFAULT_PARETO_OBJECTIVES objectives are immutable."""
        obj = DEFAULT_PARETO_OBJECTIVES[0]
        with pytest.raises(AttributeError):
            obj.metric_key = "new_metric"


class TestIdentifyParetoOptimal:
    """Tests for identify_pareto_optimal() function."""

    def test_single_configuration_is_pareto_optimal(self):
        """Test that a single configuration is always Pareto optimal."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            }
        }

        result = identify_pareto_optimal(per_value_stats)
        assert result == [10]

    def test_all_configurations_pareto_optimal_when_none_dominates(self):
        """Test that all configurations are Pareto optimal when none dominates another."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        # Config 1: high throughput, high latency
        # Config 2: low throughput, low latency
        # Neither dominates the other
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 100.0},
            },
            20: {
                "request_throughput_avg": {"mean": 50.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats)
        assert sorted(result) == [10, 20]

    def test_dominated_configuration_excluded(self):
        """Test that a dominated configuration is excluded from Pareto optimal set."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        # Config 1: 100 throughput, 50ms latency
        # Config 2: 150 throughput, 40ms latency (dominates config 1)
        # Config 3: 80 throughput, 60ms latency (dominated by config 1)
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 150.0},
                "ttft_p99_ms": {"mean": 40.0},
            },
            30: {
                "request_throughput_avg": {"mean": 80.0},
                "ttft_p99_ms": {"mean": 60.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats)
        assert result == [20]  # Only config 2 is Pareto optimal

    def test_pareto_frontier_with_tradeoffs(self):
        """Test Pareto frontier with multiple optimal points showing tradeoffs."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        # Classic Pareto frontier: as throughput increases, latency increases
        # All three are Pareto optimal (different tradeoff points)
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 50.0},
                "ttft_p99_ms": {"mean": 30.0},
            },
            20: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
            30: {
                "request_throughput_avg": {"mean": 150.0},
                "ttft_p99_ms": {"mean": 80.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats)
        assert sorted(result) == [10, 20, 30]

    def test_uses_default_objectives_when_none_provided(self):
        """Test that function uses DEFAULT_PARETO_OBJECTIVES when objectives=None."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            }
        }

        # Should not raise error and should use default objectives
        result = identify_pareto_optimal(per_value_stats, objectives=None)
        assert result == [10]

    def test_custom_objectives_single_maximize(self):
        """Test with custom objective that only maximizes one metric."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        objectives = [
            Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        ]

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 150.0}},
            30: {"request_throughput_avg": {"mean": 120.0}},
        }

        result = identify_pareto_optimal(per_value_stats, objectives)
        assert result == [20]  # Highest throughput

    def test_custom_objectives_single_minimize(self):
        """Test with custom objective that only minimizes one metric."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        objectives = [Objective("ttft_p99_ms", OptimizationDirection.MINIMIZE)]

        per_value_stats = {
            10: {"ttft_p99_ms": {"mean": 50.0}},
            20: {"ttft_p99_ms": {"mean": 30.0}},
            30: {"ttft_p99_ms": {"mean": 40.0}},
        }

        result = identify_pareto_optimal(per_value_stats, objectives)
        assert result == [20]  # Lowest latency

    def test_custom_objectives_three_dimensions(self):
        """Test with three objectives (N-dimensional Pareto analysis)."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        objectives = [
            Objective("throughput", OptimizationDirection.MAXIMIZE),
            Objective("latency", OptimizationDirection.MINIMIZE),
            Objective("cost", OptimizationDirection.MINIMIZE),
        ]

        # Config 1: high throughput, high latency, low cost
        # Config 2: medium throughput, low latency, medium cost
        # Config 3: low throughput, medium latency, high cost (dominated)
        per_value_stats = {
            10: {
                "throughput": {"mean": 150.0},
                "latency": {"mean": 80.0},
                "cost": {"mean": 10.0},
            },
            20: {
                "throughput": {"mean": 100.0},
                "latency": {"mean": 40.0},
                "cost": {"mean": 20.0},
            },
            30: {
                "throughput": {"mean": 50.0},
                "latency": {"mean": 60.0},
                "cost": {"mean": 30.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats, objectives)
        # Config 3 is dominated by both 1 and 2
        assert sorted(result) == [10, 20]

    def test_equal_values_not_dominated(self):
        """Test that configurations with equal objective values are not dominated."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        # Config 1 and 2 have identical metrics
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats)
        # Both should be Pareto optimal (neither strictly dominates)
        assert sorted(result) == [10, 20]

    def test_strictly_better_on_all_objectives_required(self):
        """Test that domination requires being better or equal on all, strictly better on at least one."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        # Config 2 is better on throughput and equal on latency
        # Config 2 DOES dominate config 1 (better or equal on all, strictly better on one)
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 150.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats)
        # Only config 2 is Pareto optimal (dominates config 1)
        assert result == [20]

    def test_result_is_sorted(self):
        """Test that result is sorted by sweep value."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        per_value_stats = {
            30: {
                "request_throughput_avg": {"mean": 150.0},
                "ttft_p99_ms": {"mean": 80.0},
            },
            10: {
                "request_throughput_avg": {"mean": 50.0},
                "ttft_p99_ms": {"mean": 30.0},
            },
            20: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats)
        assert result == [10, 20, 30]  # Sorted order

    def test_empty_stats_returns_empty_list(self):
        """Test that empty per_value_stats returns empty list."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        result = identify_pareto_optimal({})
        assert result == []

    def test_complex_pareto_frontier(self):
        """Test complex scenario with multiple dominated and non-dominated configs."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 50.0},
                "ttft_p99_ms": {"mean": 100.0},
            },  # Dominated by 20, 30, 50
            20: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 80.0},
            },  # Dominated by 30, 50
            30: {
                "request_throughput_avg": {"mean": 150.0},
                "ttft_p99_ms": {"mean": 60.0},
            },  # Dominated by 50
            40: {
                "request_throughput_avg": {"mean": 120.0},
                "ttft_p99_ms": {"mean": 70.0},
            },  # Dominated by 30, 50
            50: {
                "request_throughput_avg": {"mean": 200.0},
                "ttft_p99_ms": {"mean": 50.0},
            },  # Pareto optimal (best on both)
        }

        result = identify_pareto_optimal(per_value_stats)
        # Only 50 is Pareto optimal (dominates all others)
        assert result == [50]

    def test_true_pareto_frontier_with_multiple_optimal(self):
        """Test a true Pareto frontier where multiple configs are optimal."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        # Create a realistic Pareto frontier where there are tradeoffs
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 50.0},
                "ttft_p99_ms": {"mean": 30.0},
            },  # Low throughput, low latency - Pareto optimal
            20: {
                "request_throughput_avg": {"mean": 80.0},
                "ttft_p99_ms": {"mean": 45.0},
            },  # Medium throughput, medium latency - Pareto optimal
            30: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },  # Good throughput, medium latency - Pareto optimal
            40: {
                "request_throughput_avg": {"mean": 120.0},
                "ttft_p99_ms": {"mean": 70.0},
            },  # Higher throughput, higher latency - Pareto optimal
            50: {
                "request_throughput_avg": {"mean": 110.0},
                "ttft_p99_ms": {"mean": 80.0},
            },  # Dominated by 40 (40 has higher throughput and lower latency)
        }

        result = identify_pareto_optimal(per_value_stats)
        # Configs 10, 20, 30, 40 form the Pareto frontier
        assert result == [10, 20, 30, 40]

    def test_missing_metric_key_raises_error(self):
        """Test that missing metric key in objectives raises KeyError."""
        from aiperf.orchestrator.aggregation import (
            Objective,
            OptimizationDirection,
            identify_pareto_optimal,
        )

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
        }

        # Objective references metric that doesn't exist
        objectives = [Objective("nonexistent_metric", OptimizationDirection.MAXIMIZE)]

        with pytest.raises(KeyError):
            identify_pareto_optimal(per_value_stats, objectives)

    def test_very_close_floating_point_values(self):
        """Test Pareto identification with very close floating point values."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        # Values differ by tiny amounts (floating point precision edge case)
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0000001},
                "ttft_p99_ms": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 100.0000002},  # Slightly higher
                "ttft_p99_ms": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats)
        # Config 20 dominates 10 (strictly better throughput, equal latency)
        assert result == [20]

    def test_large_number_of_configurations(self):
        """Test Pareto identification with many configurations (performance test)."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        # Create 100 configurations where throughput increases and latency decreases
        # This means higher configs dominate lower ones
        per_value_stats = {}
        for i in range(1, 101):
            per_value_stats[i] = {
                "request_throughput_avg": {"mean": float(i * 10)},  # Increases
                "ttft_p99_ms": {"mean": float(101 - i)},  # Decreases
            }

        result = identify_pareto_optimal(per_value_stats)

        # Only the last config (100) is Pareto optimal - it dominates all others
        # (highest throughput AND lowest latency)
        assert result == [100]

    def test_all_dominated_by_one_configuration(self):
        """Test when one configuration dominates all others."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 100.0},
            },
            20: {
                "request_throughput_avg": {"mean": 150.0},
                "ttft_p99_ms": {"mean": 80.0},
            },
            30: {
                "request_throughput_avg": {"mean": 200.0},
                "ttft_p99_ms": {"mean": 50.0},
            },  # Dominates all
            40: {
                "request_throughput_avg": {"mean": 120.0},
                "ttft_p99_ms": {"mean": 90.0},
            },
            50: {
                "request_throughput_avg": {"mean": 180.0},
                "ttft_p99_ms": {"mean": 70.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats)
        # Only config 30 is Pareto optimal
        assert result == [30]

    def test_four_dimensional_pareto_analysis(self):
        """Test Pareto analysis with 4 objectives (high-dimensional)."""
        from aiperf.orchestrator.aggregation import (
            Objective,
            OptimizationDirection,
            identify_pareto_optimal,
        )

        objectives = [
            Objective("throughput", OptimizationDirection.MAXIMIZE),
            Objective("latency", OptimizationDirection.MINIMIZE),
            Objective("cost", OptimizationDirection.MINIMIZE),
            Objective("memory", OptimizationDirection.MINIMIZE),
        ]

        per_value_stats = {
            10: {
                "throughput": {"mean": 100.0},
                "latency": {"mean": 50.0},
                "cost": {"mean": 10.0},
                "memory": {"mean": 1000.0},
            },  # Pareto optimal (best cost and memory)
            20: {
                "throughput": {"mean": 150.0},
                "latency": {"mean": 60.0},
                "cost": {"mean": 15.0},
                "memory": {"mean": 1200.0},
            },  # Pareto optimal (best throughput)
            30: {
                "throughput": {"mean": 120.0},
                "latency": {"mean": 55.0},
                "cost": {"mean": 12.0},
                "memory": {"mean": 1100.0},
            },  # Pareto optimal (balanced tradeoff)
        }

        result = identify_pareto_optimal(per_value_stats, objectives)
        # All three are Pareto optimal - none dominates another on all 4 dimensions
        assert sorted(result) == [10, 20, 30]

    def test_negative_metric_values_in_pareto_analysis(self):
        """Test Pareto analysis works correctly with negative metric values."""
        from aiperf.orchestrator.aggregation import (
            Objective,
            OptimizationDirection,
            identify_pareto_optimal,
        )

        objectives = [
            Objective("profit", OptimizationDirection.MAXIMIZE),  # Can be negative
            Objective("cost", OptimizationDirection.MINIMIZE),  # Can be negative
        ]

        per_value_stats = {
            10: {
                "profit": {"mean": -50.0},  # Worst profit
                "cost": {"mean": 10.0},  # Best cost
            },  # Pareto optimal (best cost)
            20: {
                "profit": {"mean": -30.0},  # Best profit (least negative)
                "cost": {"mean": 20.0},  # Worst cost
            },  # Pareto optimal (best profit)
            30: {
                "profit": {"mean": -40.0},  # Middle profit
                "cost": {"mean": 15.0},  # Middle cost
            },  # Dominated by neither 10 nor 20 - Pareto optimal
        }

        result = identify_pareto_optimal(per_value_stats, objectives)
        # All three form a Pareto frontier with different tradeoffs
        assert sorted(result) == [10, 20, 30]

    def test_zero_values_in_metrics(self):
        """Test Pareto analysis with zero values in metrics."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 0.0},  # Zero throughput
                "ttft_p99_ms": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 0.0},  # Zero latency
            },
            30: {
                "request_throughput_avg": {"mean": 50.0},
                "ttft_p99_ms": {"mean": 25.0},
            },
        }

        result = identify_pareto_optimal(per_value_stats)
        # Config 20 dominates 10 (higher throughput, lower latency)
        # Config 20 dominates 30 (higher throughput, lower latency)
        assert result == [20]

    def test_mixed_domination_patterns(self):
        """Test complex domination patterns with multiple Pareto optimal points."""
        from aiperf.orchestrator.aggregation import identify_pareto_optimal

        # Create a scenario with multiple clusters of Pareto optimal points
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 50.0},
                "ttft_p99_ms": {"mean": 20.0},
            },  # Pareto optimal (best latency)
            20: {
                "request_throughput_avg": {"mean": 45.0},
                "ttft_p99_ms": {"mean": 25.0},
            },  # Dominated by 10
            30: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 40.0},
            },  # Pareto optimal
            40: {
                "request_throughput_avg": {"mean": 95.0},
                "ttft_p99_ms": {"mean": 45.0},
            },  # Dominated by 30
            50: {
                "request_throughput_avg": {"mean": 150.0},
                "ttft_p99_ms": {"mean": 60.0},
            },  # Pareto optimal
            60: {
                "request_throughput_avg": {"mean": 200.0},
                "ttft_p99_ms": {"mean": 80.0},
            },  # Pareto optimal (best throughput)
        }

        result = identify_pareto_optimal(per_value_stats)
        # Configs 10, 30, 50, 60 form the Pareto frontier
        assert result == [10, 30, 50, 60]


class TestAnalyzeTrends:
    """Tests for analyze_trends() function."""

    def test_increasing_trend_no_inflection(self):
        """Test steadily increasing trend with no inflection points."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
            30: {"request_throughput_avg": {"mean": 260.0}},
            40: {"request_throughput_avg": {"mean": 340.0}},
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        # All positive rate of change, similar magnitude
        assert result["rate_of_change"] == [80.0, 80.0, 80.0]
        assert result["inflection_points"] == []

    def test_decreasing_trend_no_inflection(self):
        """Test steadily decreasing trend with no inflection points."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"ttft_p99_ms": {"mean": 100.0}},
            20: {"ttft_p99_ms": {"mean": 90.0}},
            30: {"ttft_p99_ms": {"mean": 80.0}},
            40: {"ttft_p99_ms": {"mean": 70.0}},
        }

        result = analyze_trends(per_value_stats, [10, 20, 30, 40], "ttft_p99_ms")

        # All negative rate of change, similar magnitude
        assert result["rate_of_change"] == [-10.0, -10.0, -10.0]
        assert result["inflection_points"] == []

    def test_plateau_trend(self):
        """Test plateau trend with minimal changes."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 101.0}},
            30: {"request_throughput_avg": {"mean": 100.5}},
            40: {"request_throughput_avg": {"mean": 100.2}},
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        # All near-zero rate of change
        assert result["rate_of_change"] == pytest.approx([1.0, -0.5, -0.3])
        # Inflection at 30 (sign flip from +1.0 to -0.5)
        assert result["inflection_points"] == [30]

    def test_sign_flip_inflection_point(self):
        """Test inflection point detection when rate of change flips sign."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},  # +80
            30: {"request_throughput_avg": {"mean": 270.0}},  # +90
            40: {"request_throughput_avg": {"mean": 285.0}},  # +15 (inflection!)
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [80.0, 90.0, 15.0]
        # Inflection at 40 (rate dropped significantly)
        assert result["inflection_points"] == [40]

    def test_magnitude_change_inflection_point(self):
        """Test inflection point detection when rate magnitude changes > 50%."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 200.0}},  # +100
            30: {"request_throughput_avg": {"mean": 240.0}},  # +40 (60% drop)
            40: {"request_throughput_avg": {"mean": 280.0}},  # +40 (no change)
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [100.0, 40.0, 40.0]
        # Inflection at 30 (rate dropped by 60%)
        assert result["inflection_points"] == [30]

    def test_multiple_inflection_points(self):
        """Test detection of multiple inflection points."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 200.0}},  # +100
            30: {"request_throughput_avg": {"mean": 220.0}},  # +20 (inflection!)
            40: {"request_throughput_avg": {"mean": 320.0}},  # +100 (inflection!)
            50: {"request_throughput_avg": {"mean": 340.0}},  # +20 (inflection!)
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40, 50], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [100.0, 20.0, 100.0, 20.0]
        # Inflections at 30, 40, and 50
        assert result["inflection_points"] == [30, 40, 50]

    def test_negative_to_positive_sign_flip(self):
        """Test inflection point when trend changes from decreasing to increasing."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"ttft_p99_ms": {"mean": 100.0}},
            20: {"ttft_p99_ms": {"mean": 80.0}},  # -20
            30: {"ttft_p99_ms": {"mean": 70.0}},  # -10
            40: {"ttft_p99_ms": {"mean": 90.0}},  # +20 (sign flip!)
        }

        result = analyze_trends(per_value_stats, [10, 20, 30, 40], "ttft_p99_ms")

        assert result["rate_of_change"] == [-20.0, -10.0, 20.0]
        # Inflection at 40 (sign flip from negative to positive)
        assert result["inflection_points"] == [40]

    def test_positive_to_negative_sign_flip(self):
        """Test inflection point when trend changes from increasing to decreasing."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},  # +80
            30: {"request_throughput_avg": {"mean": 260.0}},  # +80
            40: {"request_throughput_avg": {"mean": 240.0}},  # -20 (sign flip!)
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [80.0, 80.0, -20.0]
        # Inflection at 40 (sign flip from positive to negative)
        assert result["inflection_points"] == [40]

    def test_two_values_no_inflection(self):
        """Test with only two values (one rate of change, no inflection possible)."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
        }

        result = analyze_trends(per_value_stats, [10, 20], "request_throughput_avg")

        assert result["rate_of_change"] == [80.0]
        assert result["inflection_points"] == []

    def test_single_value_empty_results(self):
        """Test with single value returns empty rate_of_change and inflection_points."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
        }

        result = analyze_trends(per_value_stats, [10], "request_throughput_avg")

        assert result["rate_of_change"] == []
        assert result["inflection_points"] == []

    def test_zero_rate_of_change(self):
        """Test with constant values (zero rate of change)."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 100.0}},
            30: {"request_throughput_avg": {"mean": 100.0}},
            40: {"request_throughput_avg": {"mean": 100.0}},
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [0.0, 0.0, 0.0]
        assert result["inflection_points"] == []

    def test_exactly_50_percent_magnitude_change_no_inflection(self):
        """Test that exactly 50% magnitude change does NOT trigger inflection."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 200.0}},  # +100
            30: {"request_throughput_avg": {"mean": 250.0}},  # +50 (exactly 50% of 100)
        }

        result = analyze_trends(per_value_stats, [10, 20, 30], "request_throughput_avg")

        assert result["rate_of_change"] == [100.0, 50.0]
        # No inflection (50% is the threshold, not > 50%)
        assert result["inflection_points"] == []

    def test_just_over_50_percent_magnitude_change_triggers_inflection(self):
        """Test that > 50% magnitude change triggers inflection."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 200.0}},  # +100
            30: {"request_throughput_avg": {"mean": 249.0}},  # +49 (51% drop)
        }

        result = analyze_trends(per_value_stats, [10, 20, 30], "request_throughput_avg")

        assert result["rate_of_change"] == [100.0, 49.0]
        # Inflection at 30 (magnitude change > 50%)
        assert result["inflection_points"] == [30]

    def test_negative_values_in_metrics(self):
        """Test that function works with negative metric values."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"custom_metric": {"mean": -100.0}},
            20: {"custom_metric": {"mean": -80.0}},  # +20
            30: {"custom_metric": {"mean": -60.0}},  # +20
            40: {
                "custom_metric": {"mean": -50.0}
            },  # +10 (exactly 50% drop, no inflection)
        }

        result = analyze_trends(per_value_stats, [10, 20, 30, 40], "custom_metric")

        assert result["rate_of_change"] == [20.0, 20.0, 10.0]
        # No inflection (50% is the threshold, not > 50%)
        assert result["inflection_points"] == []

    def test_mixed_trend_pattern(self):
        """Test mixed trend with both increases and decreases."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 150.0}},  # +50
            30: {"request_throughput_avg": {"mean": 140.0}},  # -10 (sign flip!)
            40: {"request_throughput_avg": {"mean": 180.0}},  # +40 (sign flip!)
            50: {"request_throughput_avg": {"mean": 170.0}},  # -10 (sign flip!)
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40, 50], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [50.0, -10.0, 40.0, -10.0]
        # Inflections at 30, 40, 50 (all sign flips)
        assert result["inflection_points"] == [30, 40, 50]

    def test_sweep_values_order_matters(self):
        """Test that sweep_values order determines the analysis order."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 200.0}},
            30: {"request_throughput_avg": {"mean": 150.0}},
        }

        # Different order produces different results
        result1 = analyze_trends(
            per_value_stats, [10, 20, 30], "request_throughput_avg"
        )
        result2 = analyze_trends(
            per_value_stats, [10, 30, 20], "request_throughput_avg"
        )

        assert result1["rate_of_change"] == [100.0, -50.0]
        assert result2["rate_of_change"] == [50.0, 50.0]
        assert result1["inflection_points"] == [30]  # Sign flip
        assert result2["inflection_points"] == []  # No inflection

    def test_large_magnitude_increase(self):
        """Test detection of large magnitude increase (> 50%)."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 150.0}},  # +50
            30: {"request_throughput_avg": {"mean": 250.0}},  # +100 (100% increase!)
        }

        result = analyze_trends(per_value_stats, [10, 20, 30], "request_throughput_avg")

        assert result["rate_of_change"] == [50.0, 100.0]
        # Inflection at 30 (rate doubled)
        assert result["inflection_points"] == [30]

    def test_realistic_throughput_saturation(self):
        """Test realistic scenario: throughput increases then saturates."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},  # +80
            30: {"request_throughput_avg": {"mean": 270.0}},  # +90
            40: {"request_throughput_avg": {"mean": 285.0}},  # +15 (saturation!)
            50: {"request_throughput_avg": {"mean": 290.0}},  # +5 (plateau)
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40, 50], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [80.0, 90.0, 15.0, 5.0]
        # Inflections at 40 (rate dropped 83%) and 50 (rate dropped 67%)
        assert result["inflection_points"] == [40, 50]

    def test_zero_to_nonzero_rate_change(self):
        """Test inflection when rate changes from zero to non-zero."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 100.0}},  # 0 change
            30: {"request_throughput_avg": {"mean": 150.0}},  # +50 (from 0!)
            40: {"request_throughput_avg": {"mean": 200.0}},  # +50
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [0.0, 50.0, 50.0]
        # Inflection at 30 (rate changed from 0 to 50, sign flip: 0 * 50 = 0 < 0 is False)
        # But magnitude check is skipped when prev_rate == 0
        # So this should NOT trigger inflection based on current implementation
        assert result["inflection_points"] == []

    def test_nonzero_to_zero_rate_change(self):
        """Test inflection when rate changes from non-zero to zero."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 150.0}},  # +50
            30: {"request_throughput_avg": {"mean": 200.0}},  # +50
            40: {"request_throughput_avg": {"mean": 200.0}},  # 0 (plateau!)
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [50.0, 50.0, 0.0]
        # Inflection at 40 (rate dropped from 50 to 0, 100% drop)
        assert result["inflection_points"] == [40]

    def test_very_small_rate_changes(self):
        """Test with very small floating point rate changes."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 100.001}},  # +0.001
            30: {"request_throughput_avg": {"mean": 100.002}},  # +0.001
            40: {"request_throughput_avg": {"mean": 100.003}},  # +0.001
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        assert result["rate_of_change"] == pytest.approx([0.001, 0.001, 0.001])
        # No inflection (consistent small changes)
        assert result["inflection_points"] == []

    def test_empty_sweep_values(self):
        """Test with empty sweep_values list."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {}

        result = analyze_trends(per_value_stats, [], "request_throughput_avg")

        assert result["rate_of_change"] == []
        assert result["inflection_points"] == []

    def test_floating_point_precision_in_magnitude_check(self):
        """Test that floating point precision doesn't cause spurious inflections."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 150.0}},  # +50.0
            30: {"request_throughput_avg": {"mean": 200.0}},  # +50.0 (exactly same)
            40: {"request_throughput_avg": {"mean": 250.0}},  # +50.0 (exactly same)
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [50.0, 50.0, 50.0]
        # No inflection (all rates identical)
        assert result["inflection_points"] == []

    def test_alternating_zero_and_nonzero_rates(self):
        """Test pattern with alternating zero and non-zero rate changes."""
        from aiperf.orchestrator.aggregation import analyze_trends

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 150.0}},  # +50
            30: {"request_throughput_avg": {"mean": 150.0}},  # 0
            40: {"request_throughput_avg": {"mean": 200.0}},  # +50
            50: {"request_throughput_avg": {"mean": 200.0}},  # 0
        }

        result = analyze_trends(
            per_value_stats, [10, 20, 30, 40, 50], "request_throughput_avg"
        )

        assert result["rate_of_change"] == [50.0, 0.0, 50.0, 0.0]
        # Inflections at 30 (sign flip: 50*0=0, not <0, but magnitude: |0-50|>0.5*50)
        # and 40 (sign flip: 0*50=0, not <0, and prev_rate==0 so no magnitude check)
        # and 50 (magnitude: |0-50|>0.5*50)
        assert result["inflection_points"] == [30, 50]


class TestSweepAggregation:
    """Tests for SweepAggregation class."""

    def test_compute_returns_dict_with_required_keys(self):
        """Test that compute() returns a dict with all required keys."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},
                "ttft_p99_ms": {"mean": 60.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        # Verify all required keys are present
        assert "metadata" in result
        assert "per_value_metrics" in result
        assert "best_configurations" in result
        assert "pareto_optimal" in result
        assert "trends" in result

    def test_metadata_section_structure(self):
        """Test that metadata section has correct structure."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
            30: {"request_throughput_avg": {"mean": 260.0}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30])

        metadata = result["metadata"]
        assert metadata["parameter_name"] == "concurrency"
        assert metadata["parameter_values"] == [10, 20, 30]
        assert metadata["num_values"] == 3

    def test_per_value_metrics_converts_int_keys_to_strings(self):
        """Test that per_value_metrics converts int keys to strings for JSON compatibility."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        per_value_metrics = result["per_value_metrics"]
        # Keys should be strings
        assert "10" in per_value_metrics
        assert "20" in per_value_metrics
        # Should not have int keys
        assert 10 not in per_value_metrics
        assert 20 not in per_value_metrics

    def test_per_value_metrics_preserves_stats_structure(self):
        """Test that per_value_metrics preserves the original stats structure."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0, "std": 5.0, "min": 95.0},
                "ttft_p99_ms": {"mean": 50.0, "std": 2.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10])

        metrics = result["per_value_metrics"]["10"]
        assert metrics["request_throughput_avg"]["mean"] == 100.0
        assert metrics["request_throughput_avg"]["std"] == 5.0
        assert metrics["request_throughput_avg"]["min"] == 95.0
        assert metrics["ttft_p99_ms"]["mean"] == 50.0
        assert metrics["ttft_p99_ms"]["std"] == 2.0

    def test_pareto_optimal_uses_identify_pareto_optimal_function(self):
        """Test that pareto_optimal section uses identify_pareto_optimal()."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        # Config 1: high throughput, high latency
        # Config 2: low throughput, low latency
        # Both should be Pareto optimal
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 100.0},
            },
            20: {
                "request_throughput_avg": {"mean": 50.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        assert sorted(result["pareto_optimal"]) == [10, 20]

    def test_single_value_sweep(self):
        """Test compute() with single sweep value."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10])

        assert result["metadata"]["num_values"] == 1
        assert result["metadata"]["parameter_values"] == [10]
        assert "10" in result["per_value_metrics"]
        assert result["pareto_optimal"] == [10]

    def test_empty_sweep_values(self):
        """Test compute() with empty sweep values."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        result = SweepAggregation.compute({}, [])

        assert result["metadata"]["num_values"] == 0
        assert result["metadata"]["parameter_values"] == []
        assert result["per_value_metrics"] == {}
        assert result["pareto_optimal"] == []

    def test_best_configurations_is_dict(self):
        """Test that best_configurations is a dict (implementation in task 8.8)."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        assert isinstance(result["best_configurations"], dict)

    def test_trends_is_dict(self):
        """Test that trends is a dict (implementation in task 8.7)."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        assert isinstance(result["trends"], dict)

    def test_compute_is_static_method(self):
        """Test that compute() is a static method (can be called without instance)."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
        }

        # Should be callable without creating an instance
        result = SweepAggregation.compute(per_value_stats, [10])
        assert result is not None

    def test_multiple_sweep_values_preserves_order(self):
        """Test that sweep values order is preserved in metadata."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            40: {"request_throughput_avg": {"mean": 100.0}},
            10: {"request_throughput_avg": {"mean": 180.0}},
            30: {"request_throughput_avg": {"mean": 260.0}},
            20: {"request_throughput_avg": {"mean": 340.0}},
        }

        # Provide values in specific order
        sweep_values = [10, 20, 30, 40]
        result = SweepAggregation.compute(per_value_stats, sweep_values)

        # Order should be preserved
        assert result["metadata"]["parameter_values"] == [10, 20, 30, 40]

    def test_large_sweep_values(self):
        """Test compute() with many sweep values."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        # Create 10 sweep values
        per_value_stats = {}
        sweep_values = []
        for i in range(10, 110, 10):
            per_value_stats[i] = {
                "request_throughput_avg": {"mean": float(i * 10)},
                "ttft_p99_ms": {"mean": float(i)},
            }
            sweep_values.append(i)

        result = SweepAggregation.compute(per_value_stats, sweep_values)

        assert result["metadata"]["num_values"] == 10
        assert len(result["per_value_metrics"]) == 10
        assert "10" in result["per_value_metrics"]
        assert "100" in result["per_value_metrics"]

    def test_compute_with_complex_stats_structure(self):
        """Test compute() preserves complex nested stats structure."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {
                    "mean": 100.0,
                    "std": 5.0,
                    "min": 95.0,
                    "max": 108.0,
                    "cv": 0.05,
                    "ci_low": 94.3,
                    "ci_high": 106.7,
                    "unit": "requests/sec",
                },
                "ttft_p99_ms": {
                    "mean": 50.0,
                    "std": 2.0,
                    "unit": "ms",
                },
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10])

        metrics = result["per_value_metrics"]["10"]
        # Verify all nested fields are preserved
        assert metrics["request_throughput_avg"]["mean"] == 100.0
        assert metrics["request_throughput_avg"]["std"] == 5.0
        assert metrics["request_throughput_avg"]["min"] == 95.0
        assert metrics["request_throughput_avg"]["max"] == 108.0
        assert metrics["request_throughput_avg"]["cv"] == 0.05
        assert metrics["request_throughput_avg"]["ci_low"] == 94.3
        assert metrics["request_throughput_avg"]["ci_high"] == 106.7
        assert metrics["request_throughput_avg"]["unit"] == "requests/sec"

    def test_trends_computed_for_key_metrics(self):
        """Test that trends are computed for key metrics (request_throughput_avg, ttft_p99_ms)."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},
                "ttft_p99_ms": {"mean": 60.0},
            },
            30: {
                "request_throughput_avg": {"mean": 270.0},
                "ttft_p99_ms": {"mean": 75.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30])

        trends = result["trends"]
        # Should have trends for both key metrics
        assert "request_throughput_avg" in trends
        assert "ttft_p99_ms" in trends

    def test_trends_structure_matches_analyze_trends_output(self):
        """Test that trends structure matches analyze_trends() output."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},
                "ttft_p99_ms": {"mean": 60.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        # Each trend should have rate_of_change and inflection_points
        for metric_key in ["request_throughput_avg", "ttft_p99_ms"]:
            assert "rate_of_change" in result["trends"][metric_key]
            assert "inflection_points" in result["trends"][metric_key]

    def test_trends_empty_for_single_value(self):
        """Test that trends is empty dict for single sweep value."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10])

        # No trends with single value
        assert result["trends"] == {}

    def test_trends_empty_for_empty_stats(self):
        """Test that trends is empty dict for empty stats."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        result = SweepAggregation.compute({}, [])

        assert result["trends"] == {}

    def test_trends_only_includes_present_metrics(self):
        """Test that trends only includes metrics present in all sweep values."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "custom_metric": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},
                # Missing custom_metric
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        trends = result["trends"]
        # Should have throughput trend (present in all and is a key metric)
        assert "request_throughput_avg" in trends
        # Should NOT have custom_metric trend (not a key metric)
        assert "custom_metric" not in trends

    def test_trends_skips_missing_key_metrics(self):
        """Test that trends skips key metrics that don't exist in the data."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        # Only has custom metrics, not the key metrics
        per_value_stats = {
            10: {"custom_metric": {"mean": 100.0}},
            20: {"custom_metric": {"mean": 180.0}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        trends = result["trends"]
        # Should be empty (no key metrics present)
        assert trends == {}

    def test_trends_rate_of_change_values(self):
        """Test that trends contain correct rate_of_change values."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
            30: {"request_throughput_avg": {"mean": 270.0}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30])

        rate_of_change = result["trends"]["request_throughput_avg"]["rate_of_change"]
        assert rate_of_change == [80.0, 90.0]

    def test_trends_inflection_points_detected(self):
        """Test that trends detect inflection points correctly."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},  # +80
            30: {"request_throughput_avg": {"mean": 270.0}},  # +90
            40: {"request_throughput_avg": {"mean": 285.0}},  # +15 (inflection!)
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30, 40])

        inflection_points = result["trends"]["request_throughput_avg"][
            "inflection_points"
        ]
        assert inflection_points == [40]

    def test_trends_with_both_metrics_having_different_patterns(self):
        """Test trends when throughput and latency have different patterns."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 30.0},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},  # +80
                "ttft_p99_ms": {"mean": 35.0},  # +5
            },
            30: {
                "request_throughput_avg": {"mean": 270.0},  # +90
                "ttft_p99_ms": {"mean": 50.0},  # +15 (inflection!)
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30])

        # Throughput: steady increase, no inflection
        throughput_trends = result["trends"]["request_throughput_avg"]
        assert throughput_trends["rate_of_change"] == [80.0, 90.0]
        assert throughput_trends["inflection_points"] == []

        # Latency: inflection at 30
        latency_trends = result["trends"]["ttft_p99_ms"]
        assert latency_trends["rate_of_change"] == [5.0, 15.0]
        assert latency_trends["inflection_points"] == [30]

    def test_trends_with_additional_metrics_beyond_key_metrics(self):
        """Test that trends only analyzes key metrics, not all available metrics."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
                "custom_metric_1": {"mean": 10.0},
                "custom_metric_2": {"mean": 20.0},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},
                "ttft_p99_ms": {"mean": 60.0},
                "custom_metric_1": {"mean": 15.0},
                "custom_metric_2": {"mean": 25.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        trends = result["trends"]
        # Should only have key metrics
        assert set(trends.keys()) == {"request_throughput_avg", "ttft_p99_ms"}
        # Should NOT have custom metrics
        assert "custom_metric_1" not in trends
        assert "custom_metric_2" not in trends

    def test_trends_with_realistic_throughput_saturation(self):
        """Test trends with realistic throughput saturation pattern."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 30.0},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},
                "ttft_p99_ms": {"mean": 35.0},
            },
            30: {
                "request_throughput_avg": {"mean": 270.0},
                "ttft_p99_ms": {"mean": 45.0},
            },
            40: {
                "request_throughput_avg": {"mean": 285.0},  # Saturation starts
                "ttft_p99_ms": {"mean": 70.0},  # Latency increases
            },
            50: {
                "request_throughput_avg": {"mean": 290.0},  # Plateau
                "ttft_p99_ms": {"mean": 120.0},  # Latency spikes
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30, 40, 50])

        # Throughput: inflections at saturation points
        throughput_trends = result["trends"]["request_throughput_avg"]
        assert 40 in throughput_trends["inflection_points"]
        assert 50 in throughput_trends["inflection_points"]

        # Latency: inflections as it increases
        latency_trends = result["trends"]["ttft_p99_ms"]
        assert len(latency_trends["inflection_points"]) > 0

    def test_trends_preserves_sweep_values_order(self):
        """Test that trends analysis respects sweep_values order."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 200.0}},
            30: {"request_throughput_avg": {"mean": 150.0}},
        }

        # Different order produces different trends
        result1 = SweepAggregation.compute(per_value_stats, [10, 20, 30])
        result2 = SweepAggregation.compute(per_value_stats, [10, 30, 20])

        # Different rate of change based on order
        assert result1["trends"]["request_throughput_avg"]["rate_of_change"] == [
            100.0,
            -50.0,
        ]
        assert result2["trends"]["request_throughput_avg"]["rate_of_change"] == [
            50.0,
            50.0,
        ]


class TestBestConfigurations:
    """Tests for best_configurations section in SweepAggregation.compute()."""

    def test_best_throughput_identified(self):
        """Test that best throughput configuration is correctly identified."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
            30: {"request_throughput_avg": {"mean": 260.0}},
            40: {"request_throughput_avg": {"mean": 350.2}},  # Best
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30, 40])

        best_throughput = result["best_configurations"]["best_throughput"]
        assert best_throughput["value"] == 40
        assert best_throughput["metric"] == 350.2

    def test_best_latency_identified(self):
        """Test that best latency configuration is correctly identified."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"ttft_p99_ms": {"mean": 120.5}},  # Best
            20: {"ttft_p99_ms": {"mean": 150.0}},
            30: {"ttft_p99_ms": {"mean": 180.0}},
            40: {"ttft_p99_ms": {"mean": 200.0}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30, 40])

        best_latency = result["best_configurations"]["best_latency_p99"]
        assert best_latency["value"] == 10
        assert best_latency["metric"] == 120.5

    def test_best_configurations_with_both_metrics(self):
        """Test best configurations when both throughput and latency are present."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},  # Best latency
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},
                "ttft_p99_ms": {"mean": 60.0},
            },
            30: {
                "request_throughput_avg": {"mean": 350.2},  # Best throughput
                "ttft_p99_ms": {"mean": 80.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30])

        # Best throughput at value 30
        assert result["best_configurations"]["best_throughput"]["value"] == 30
        assert result["best_configurations"]["best_throughput"]["metric"] == 350.2

        # Best latency at value 10
        assert result["best_configurations"]["best_latency_p99"]["value"] == 10
        assert result["best_configurations"]["best_latency_p99"]["metric"] == 50.0

    def test_best_configurations_includes_units(self):
        """Test that best configurations include unit fields."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0, "unit": "requests/sec"},
                "ttft_p99_ms": {"mean": 50.0, "unit": "ms"},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0, "unit": "requests/sec"},
                "ttft_p99_ms": {"mean": 60.0, "unit": "ms"},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        # Check units are included
        assert (
            result["best_configurations"]["best_throughput"]["unit"] == "requests/sec"
        )
        assert result["best_configurations"]["best_latency_p99"]["unit"] == "ms"

    def test_best_configurations_default_units_when_missing(self):
        """Test that default units are used when not present in stats."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},  # No unit
                "ttft_p99_ms": {"mean": 50.0},  # No unit
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},
                "ttft_p99_ms": {"mean": 60.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        # Check default units are used
        assert (
            result["best_configurations"]["best_throughput"]["unit"] == "requests/sec"
        )
        assert result["best_configurations"]["best_latency_p99"]["unit"] == "ms"

    def test_best_configurations_empty_when_no_stats(self):
        """Test that best_configurations is empty dict when no stats provided."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        result = SweepAggregation.compute({}, [])

        assert result["best_configurations"] == {}

    def test_best_configurations_only_throughput_when_latency_missing(self):
        """Test that only best_throughput is included when latency metric is missing."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 180.0}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        # Should have best_throughput
        assert "best_throughput" in result["best_configurations"]
        assert result["best_configurations"]["best_throughput"]["value"] == 20

        # Should NOT have best_latency_p99
        assert "best_latency_p99" not in result["best_configurations"]

    def test_best_configurations_only_latency_when_throughput_missing(self):
        """Test that only best_latency_p99 is included when throughput metric is missing."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"ttft_p99_ms": {"mean": 50.0}},
            20: {"ttft_p99_ms": {"mean": 60.0}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        # Should have best_latency_p99
        assert "best_latency_p99" in result["best_configurations"]
        assert result["best_configurations"]["best_latency_p99"]["value"] == 10

        # Should NOT have best_throughput
        assert "best_throughput" not in result["best_configurations"]

    def test_best_configurations_handles_partial_metric_presence(self):
        """Test that best configurations handles case where metric is missing in some values."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0},
                # Missing ttft_p99_ms
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        # Should have best_throughput (present in all)
        assert "best_throughput" in result["best_configurations"]

        # Should NOT have best_latency_p99 (not present in all)
        assert "best_latency_p99" not in result["best_configurations"]

    def test_best_configurations_single_value(self):
        """Test best configurations with single sweep value."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0},
                "ttft_p99_ms": {"mean": 50.0},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10])

        # Single value is best for both
        assert result["best_configurations"]["best_throughput"]["value"] == 10
        assert result["best_configurations"]["best_latency_p99"]["value"] == 10

    def test_best_configurations_with_equal_values(self):
        """Test best configurations when multiple values have same metric."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 100.0}},
            20: {"request_throughput_avg": {"mean": 100.0}},  # Equal
            30: {"request_throughput_avg": {"mean": 100.0}},  # Equal
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30])

        # Should pick one (implementation detail: max() picks first occurrence)
        assert result["best_configurations"]["best_throughput"]["value"] in [10, 20, 30]
        assert result["best_configurations"]["best_throughput"]["metric"] == 100.0

    def test_best_configurations_structure(self):
        """Test that best configurations have correct structure with value, metric, and unit."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 100.0, "unit": "requests/sec"},
                "ttft_p99_ms": {"mean": 50.0, "unit": "ms"},
            },
            20: {
                "request_throughput_avg": {"mean": 180.0, "unit": "requests/sec"},
                "ttft_p99_ms": {"mean": 60.0, "unit": "ms"},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20])

        # Check structure of best_throughput
        best_throughput = result["best_configurations"]["best_throughput"]
        assert "value" in best_throughput
        assert "metric" in best_throughput
        assert "unit" in best_throughput
        assert isinstance(best_throughput["value"], int)
        assert isinstance(best_throughput["metric"], float)
        assert isinstance(best_throughput["unit"], str)

        # Check structure of best_latency_p99
        best_latency = result["best_configurations"]["best_latency_p99"]
        assert "value" in best_latency
        assert "metric" in best_latency
        assert "unit" in best_latency
        assert isinstance(best_latency["value"], int)
        assert isinstance(best_latency["metric"], float)
        assert isinstance(best_latency["unit"], str)

    def test_best_configurations_with_negative_values(self):
        """Test best configurations work with negative metric values (edge case)."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"custom_metric": {"mean": -100.0}},
            20: {"custom_metric": {"mean": -50.0}},  # Highest (best for maximize)
            30: {"custom_metric": {"mean": -150.0}},  # Lowest (best for minimize)
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30])

        # No best configurations (custom_metric is not a key metric)
        assert result["best_configurations"] == {}

    def test_best_configurations_with_large_values(self):
        """Test best configurations with large metric values."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"request_throughput_avg": {"mean": 10000.0}},
            20: {"request_throughput_avg": {"mean": 50000.0}},
            30: {"request_throughput_avg": {"mean": 100000.0}},  # Best
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30])

        assert result["best_configurations"]["best_throughput"]["value"] == 30
        assert result["best_configurations"]["best_throughput"]["metric"] == 100000.0

    def test_best_configurations_with_small_differences(self):
        """Test best configurations with very small differences in metrics."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        per_value_stats = {
            10: {"ttft_p99_ms": {"mean": 50.001}},
            20: {"ttft_p99_ms": {"mean": 50.000}},  # Best (lowest)
            30: {"ttft_p99_ms": {"mean": 50.002}},
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30])

        assert result["best_configurations"]["best_latency_p99"]["value"] == 20
        assert result["best_configurations"]["best_latency_p99"]["metric"] == 50.000

    def test_best_configurations_realistic_scenario(self):
        """Test best configurations with realistic sweep data."""
        from aiperf.orchestrator.aggregation import SweepAggregation

        # Realistic scenario: throughput increases with concurrency, latency also increases
        per_value_stats = {
            10: {
                "request_throughput_avg": {"mean": 95.5, "unit": "requests/sec"},
                "ttft_p99_ms": {"mean": 45.2, "unit": "ms"},  # Best latency
            },
            20: {
                "request_throughput_avg": {"mean": 175.3, "unit": "requests/sec"},
                "ttft_p99_ms": {"mean": 52.8, "unit": "ms"},
            },
            30: {
                "request_throughput_avg": {"mean": 245.7, "unit": "requests/sec"},
                "ttft_p99_ms": {"mean": 68.5, "unit": "ms"},
            },
            40: {
                "request_throughput_avg": {
                    "mean": 298.2,
                    "unit": "requests/sec",
                },  # Best throughput
                "ttft_p99_ms": {"mean": 95.3, "unit": "ms"},
            },
        }

        result = SweepAggregation.compute(per_value_stats, [10, 20, 30, 40])

        # Best throughput at highest concurrency
        best_throughput = result["best_configurations"]["best_throughput"]
        assert best_throughput["value"] == 40
        assert best_throughput["metric"] == 298.2
        assert best_throughput["unit"] == "requests/sec"

        # Best latency at lowest concurrency
        best_latency = result["best_configurations"]["best_latency_p99"]
        assert best_latency["value"] == 10
        assert best_latency["metric"] == 45.2
        assert best_latency["unit"] == "ms"
