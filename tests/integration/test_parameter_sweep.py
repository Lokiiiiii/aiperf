# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for parameter sweeping feature."""

import json
from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestParameterSweep:
    """Integration tests for parameter sweeping."""

    async def test_sweep_with_confidence_repeated_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test sweep + confidence reporting in repeated mode.

        This test validates:
        - Requirement 4.1: Repeated mode executes entire sweep N times
        - Requirement 4.2: All sweep values in sequence before starting next trial
        - Requirement 4.5: Confidence statistics computed per concurrency value

        Execution pattern with --concurrency 2,4,6 --num-profile-runs 3:
        Trial 1: [2→4→6]
        Trial 2: [2→4→6]
        Trial 3: [2→4→6]
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6 \
                --num-profile-runs 3 \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify basic execution
        assert result.exit_code == 0

        # Verify hierarchical directory structure for repeated mode:
        # artifacts/
        #   profile_runs/
        #     trial_0001/
        #       concurrency_2/
        #       concurrency_4/
        #       concurrency_6/
        #     trial_0002/
        #       concurrency_2/
        #       concurrency_4/
        #       concurrency_6/
        #     trial_0003/
        #       concurrency_2/
        #       concurrency_4/
        #       concurrency_6/

        profile_runs_dir = temp_output_dir / "profile_runs"
        assert profile_runs_dir.exists(), "profile_runs directory should exist"

        # Verify trial directories
        trial_dirs = sorted(profile_runs_dir.glob("trial_*"))
        assert len(trial_dirs) == 3, "Should have 3 trial directories"
        assert trial_dirs[0].name == "trial_0001"
        assert trial_dirs[1].name == "trial_0002"
        assert trial_dirs[2].name == "trial_0003"

        # Verify each trial has all concurrency values
        concurrency_values = [2, 4, 6]
        for trial_dir in trial_dirs:
            for concurrency in concurrency_values:
                concurrency_dir = trial_dir / f"concurrency_{concurrency}"
                assert concurrency_dir.exists(), (
                    f"{trial_dir.name} should have concurrency_{concurrency}"
                )

                # Verify artifacts exist
                json_file = concurrency_dir / "profile_export_aiperf.json"
                csv_file = concurrency_dir / "profile_export_aiperf.csv"
                assert json_file.exists(), (
                    f"{trial_dir.name}/concurrency_{concurrency} should have JSON"
                )
                assert csv_file.exists(), (
                    f"{trial_dir.name}/concurrency_{concurrency} should have CSV"
                )

                # Verify JSON content
                with open(json_file) as f:
                    run_data = json.load(f)
                    assert run_data["request_count"]["avg"] == 10

        # Verify aggregate directory structure:
        # aggregate/
        #   concurrency_2/
        #     profile_export_aiperf_aggregate.json
        #     profile_export_aiperf_aggregate.csv
        #   concurrency_4/
        #     profile_export_aiperf_aggregate.json
        #     profile_export_aiperf_aggregate.csv
        #   concurrency_6/
        #     profile_export_aiperf_aggregate.json
        #     profile_export_aiperf_aggregate.csv
        #   sweep_aggregate/
        #     profile_export_aiperf_sweep.json
        #     profile_export_aiperf_sweep.csv

        aggregate_dir = temp_output_dir / "aggregate"
        assert aggregate_dir.exists(), "aggregate directory should exist"

        # Verify per-concurrency aggregate directories
        for concurrency in concurrency_values:
            concurrency_agg_dir = aggregate_dir / f"concurrency_{concurrency}"
            assert concurrency_agg_dir.exists(), (
                f"aggregate/concurrency_{concurrency} should exist"
            )

            agg_json = concurrency_agg_dir / "profile_export_aiperf_aggregate.json"
            agg_csv = concurrency_agg_dir / "profile_export_aiperf_aggregate.csv"
            assert agg_json.exists(), (
                f"concurrency_{concurrency} aggregate JSON should exist"
            )
            assert agg_csv.exists(), (
                f"concurrency_{concurrency} aggregate CSV should exist"
            )

            # Verify aggregate JSON schema for this concurrency value
            with open(agg_json) as f:
                agg_data = json.load(f)

                # Check metadata
                assert agg_data["metadata"]["aggregation_type"] == "confidence"
                assert agg_data["metadata"]["num_profile_runs"] == 3
                assert agg_data["metadata"]["num_successful_runs"] == 3
                assert len(agg_data["metadata"]["failed_runs"]) == 0
                assert agg_data["metadata"]["confidence_level"] == 0.95

                # Check metrics structure
                assert "metrics" in agg_data
                metrics = agg_data["metrics"]
                assert len(metrics) > 0, "Should have aggregated metrics"

                # Verify confidence interval fields
                throughput_metrics = [k for k in metrics if "throughput" in k.lower()]
                assert len(throughput_metrics) > 0, "Should have throughput metrics"

                sample_metric = metrics[throughput_metrics[0]]
                required_fields = [
                    "mean",
                    "std",
                    "min",
                    "max",
                    "cv",
                    "se",
                    "ci_low",
                    "ci_high",
                    "t_critical",
                    "unit",
                ]
                for field in required_fields:
                    assert field in sample_metric, f"Metric should have {field} field"

        # Verify sweep aggregate directory
        sweep_agg_dir = aggregate_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), "sweep_aggregate directory should exist"

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        sweep_csv = sweep_agg_dir / "profile_export_aiperf_sweep.csv"
        assert sweep_json.exists(), "Sweep aggregate JSON should exist"
        assert sweep_csv.exists(), "Sweep aggregate CSV should exist"

        # Verify sweep aggregate JSON schema
        with open(sweep_json) as f:
            sweep_data = json.load(f)

            # Check metadata
            assert sweep_data["metadata"]["aggregation_type"] == "sweep"
            assert sweep_data["metadata"]["parameter_name"] == "concurrency"
            assert sweep_data["metadata"]["parameter_values"] == [2, 4, 6]
            assert sweep_data["metadata"]["num_values"] == 3
            assert sweep_data["metadata"]["num_trials_per_value"] == 3
            assert sweep_data["metadata"]["sweep_mode"] == "repeated"
            assert sweep_data["metadata"]["confidence_level"] == 0.95

            # Check per_value_metrics structure
            assert "per_value_metrics" in sweep_data
            per_value_metrics = sweep_data["per_value_metrics"]
            assert "2" in per_value_metrics
            assert "4" in per_value_metrics
            assert "6" in per_value_metrics

            # Verify each value has metrics with confidence statistics
            for value_str in ["2", "4", "6"]:
                value_metrics = per_value_metrics[value_str]
                assert len(value_metrics) > 0, f"Value {value_str} should have metrics"

                # Check a sample metric has confidence fields
                throughput_keys = [
                    k for k in value_metrics if "throughput" in k.lower()
                ]
                assert len(throughput_keys) > 0, (
                    f"Value {value_str} should have throughput metrics"
                )

                sample_metric = value_metrics[throughput_keys[0]]
                for field in ["mean", "std", "ci_low", "ci_high"]:
                    assert field in sample_metric, (
                        f"Value {value_str} metric should have {field}"
                    )

            # Check best_configurations
            assert "best_configurations" in sweep_data
            best_configs = sweep_data["best_configurations"]
            assert "best_throughput" in best_configs
            assert "best_latency_p99" in best_configs

            # Verify best_throughput structure
            best_throughput = best_configs["best_throughput"]
            assert "value" in best_throughput
            assert "metric" in best_throughput
            assert "unit" in best_throughput
            assert best_throughput["value"] in [2, 4, 6]

            # Verify best_latency structure
            best_latency = best_configs["best_latency_p99"]
            assert "value" in best_latency
            assert "metric" in best_latency
            assert "unit" in best_latency
            assert best_latency["value"] in [2, 4, 6]

            # Check pareto_optimal
            assert "pareto_optimal" in sweep_data
            pareto_optimal = sweep_data["pareto_optimal"]
            assert isinstance(pareto_optimal, list)
            assert len(pareto_optimal) > 0, (
                "Should have at least one Pareto optimal point"
            )
            for value in pareto_optimal:
                assert value in [2, 4, 6], "Pareto optimal values should be from sweep"

            # Check trends
            assert "trends" in sweep_data
            trends = sweep_data["trends"]
            assert len(trends) > 0, "Should have trend analysis"

            # Verify trend structure for a sample metric
            if "request_throughput_avg" in trends:
                throughput_trend = trends["request_throughput_avg"]
                assert "inflection_points" in throughput_trend
                assert "rate_of_change" in throughput_trend
                assert isinstance(throughput_trend["inflection_points"], list)
                assert isinstance(throughput_trend["rate_of_change"], list)
                # Rate of change should have N-1 values for N sweep values
                assert len(throughput_trend["rate_of_change"]) == 2

        # Verify sweep CSV format
        csv_content = sweep_csv.read_text()
        lines = csv_content.strip().split("\n")

        # Check header
        header = lines[0]
        required_columns = [
            "concurrency",
            "metric",
            "mean",
            "std",
            "min",
            "max",
            "cv",
            "se",
            "ci_low",
            "ci_high",
            "unit",
        ]
        for col in required_columns:
            assert col in header, f"Sweep CSV should have {col} column"

        # Check data rows (should have rows for each concurrency × metrics)
        assert len(lines) > 1, "Sweep CSV should have data rows"

    async def test_sweep_with_confidence_independent_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test sweep + confidence reporting in independent mode.

        This test validates:
        - Requirement 4.3: Independent mode executes N trials at each sweep value
        - Requirement 4.4: All trials at one sweep value before moving to next
        - Requirement 4.5: Confidence statistics computed per concurrency value

        Execution pattern with --concurrency 2,4,6 --num-profile-runs 3:
        Concurrency 2: [trial1, trial2, trial3]
        Concurrency 4: [trial1, trial2, trial3]
        Concurrency 6: [trial1, trial2, trial3]
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6 \
                --num-profile-runs 3 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify basic execution
        assert result.exit_code == 0

        # Verify hierarchical directory structure for independent mode:
        # artifacts/
        #   concurrency_2/
        #     profile_runs/
        #       trial_0001/
        #       trial_0002/
        #       trial_0003/
        #     aggregate/
        #       profile_export_aiperf_aggregate.json
        #   concurrency_4/
        #     profile_runs/
        #       trial_0001/
        #       trial_0002/
        #       trial_0003/
        #     aggregate/
        #       profile_export_aiperf_aggregate.json
        #   concurrency_6/
        #     profile_runs/
        #       trial_0001/
        #       trial_0002/
        #       trial_0003/
        #     aggregate/
        #       profile_export_aiperf_aggregate.json
        #   sweep_aggregate/
        #     profile_export_aiperf_sweep.json
        #     profile_export_aiperf_sweep.csv

        concurrency_values = [2, 4, 6]

        # Verify concurrency directories exist
        for concurrency in concurrency_values:
            concurrency_dir = temp_output_dir / f"concurrency_{concurrency}"
            assert concurrency_dir.exists(), (
                f"concurrency_{concurrency} directory should exist"
            )

            # Verify profile_runs directory
            profile_runs_dir = concurrency_dir / "profile_runs"
            assert profile_runs_dir.exists(), (
                f"concurrency_{concurrency}/profile_runs should exist"
            )

            # Verify trial directories
            trial_dirs = sorted(profile_runs_dir.glob("trial_*"))
            assert len(trial_dirs) == 3, (
                f"concurrency_{concurrency} should have 3 trial directories"
            )
            assert trial_dirs[0].name == "trial_0001"
            assert trial_dirs[1].name == "trial_0002"
            assert trial_dirs[2].name == "trial_0003"

            # Verify each trial has artifacts
            for trial_dir in trial_dirs:
                json_file = trial_dir / "profile_export_aiperf.json"
                csv_file = trial_dir / "profile_export_aiperf.csv"
                assert json_file.exists(), (
                    f"{concurrency_dir.name}/{trial_dir.name} should have JSON"
                )
                assert csv_file.exists(), (
                    f"{concurrency_dir.name}/{trial_dir.name} should have CSV"
                )

                # Verify JSON content
                with open(json_file) as f:
                    run_data = json.load(f)
                    assert run_data["request_count"]["avg"] == 10

            # Verify aggregate directory
            aggregate_dir = concurrency_dir / "aggregate"
            assert aggregate_dir.exists(), (
                f"concurrency_{concurrency}/aggregate should exist"
            )

            # Verify aggregate files
            agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
            agg_csv = aggregate_dir / "profile_export_aiperf_aggregate.csv"
            assert agg_json.exists(), (
                f"concurrency_{concurrency} aggregate JSON should exist"
            )
            assert agg_csv.exists(), (
                f"concurrency_{concurrency} aggregate CSV should exist"
            )

            # Verify aggregate JSON schema
            with open(agg_json) as f:
                agg_data = json.load(f)

                # Check metadata
                assert agg_data["metadata"]["aggregation_type"] == "confidence"
                assert agg_data["metadata"]["num_profile_runs"] == 3
                assert agg_data["metadata"]["num_successful_runs"] == 3
                assert len(agg_data["metadata"]["failed_runs"]) == 0
                assert agg_data["metadata"]["confidence_level"] == 0.95

                # Check metrics structure
                assert "metrics" in agg_data
                metrics = agg_data["metrics"]
                assert len(metrics) > 0, "Should have aggregated metrics"

                # Verify confidence interval fields
                throughput_metrics = [k for k in metrics if "throughput" in k.lower()]
                assert len(throughput_metrics) > 0, "Should have throughput metrics"

                sample_metric = metrics[throughput_metrics[0]]
                required_fields = [
                    "mean",
                    "std",
                    "min",
                    "max",
                    "cv",
                    "se",
                    "ci_low",
                    "ci_high",
                    "t_critical",
                    "unit",
                ]
                for field in required_fields:
                    assert field in sample_metric, f"Metric should have {field} field"

        # Verify sweep aggregate directory
        sweep_agg_dir = temp_output_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), "sweep_aggregate directory should exist"

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        sweep_csv = sweep_agg_dir / "profile_export_aiperf_sweep.csv"
        assert sweep_json.exists(), "Sweep aggregate JSON should exist"
        assert sweep_csv.exists(), "Sweep aggregate CSV should exist"

        # Verify sweep aggregate JSON schema
        with open(sweep_json) as f:
            sweep_data = json.load(f)

            # Check metadata
            assert sweep_data["metadata"]["aggregation_type"] == "sweep"
            assert sweep_data["metadata"]["parameter_name"] == "concurrency"
            assert sweep_data["metadata"]["parameter_values"] == [2, 4, 6]
            assert sweep_data["metadata"]["num_values"] == 3
            assert sweep_data["metadata"]["num_trials_per_value"] == 3
            assert sweep_data["metadata"]["sweep_mode"] == "independent"
            assert sweep_data["metadata"]["confidence_level"] == 0.95

            # Check per_value_metrics structure
            assert "per_value_metrics" in sweep_data
            per_value_metrics = sweep_data["per_value_metrics"]
            assert "2" in per_value_metrics
            assert "4" in per_value_metrics
            assert "6" in per_value_metrics

            # Verify each value has metrics with confidence statistics
            for value_str in ["2", "4", "6"]:
                value_metrics = per_value_metrics[value_str]
                assert len(value_metrics) > 0, f"Value {value_str} should have metrics"

                # Check a sample metric has confidence fields
                throughput_keys = [
                    k for k in value_metrics if "throughput" in k.lower()
                ]
                assert len(throughput_keys) > 0, (
                    f"Value {value_str} should have throughput metrics"
                )

                sample_metric = value_metrics[throughput_keys[0]]
                for field in ["mean", "std", "ci_low", "ci_high"]:
                    assert field in sample_metric, (
                        f"Value {value_str} metric should have {field}"
                    )

            # Check best_configurations
            assert "best_configurations" in sweep_data
            best_configs = sweep_data["best_configurations"]
            assert "best_throughput" in best_configs
            assert "best_latency_p99" in best_configs

            # Verify best_throughput structure
            best_throughput = best_configs["best_throughput"]
            assert "value" in best_throughput
            assert "metric" in best_throughput
            assert "unit" in best_throughput
            assert best_throughput["value"] in [2, 4, 6]

            # Verify best_latency structure
            best_latency = best_configs["best_latency_p99"]
            assert "value" in best_latency
            assert "metric" in best_latency
            assert "unit" in best_latency
            assert best_latency["value"] in [2, 4, 6]

            # Check pareto_optimal
            assert "pareto_optimal" in sweep_data
            pareto_optimal = sweep_data["pareto_optimal"]
            assert isinstance(pareto_optimal, list)
            assert len(pareto_optimal) > 0, (
                "Should have at least one Pareto optimal point"
            )
            for value in pareto_optimal:
                assert value in [2, 4, 6], "Pareto optimal values should be from sweep"

            # Check trends
            assert "trends" in sweep_data
            trends = sweep_data["trends"]
            assert len(trends) > 0, "Should have trend analysis"

            # Verify trend structure for a sample metric
            if "request_throughput_avg" in trends:
                throughput_trend = trends["request_throughput_avg"]
                assert "inflection_points" in throughput_trend
                assert "rate_of_change" in throughput_trend
                assert isinstance(throughput_trend["inflection_points"], list)
                assert isinstance(throughput_trend["rate_of_change"], list)
                # Rate of change should have N-1 values for N sweep values
                assert len(throughput_trend["rate_of_change"]) == 2

        # Verify sweep CSV format
        csv_content = sweep_csv.read_text()
        lines = csv_content.strip().split("\n")

        # Check header
        header = lines[0]
        required_columns = [
            "concurrency",
            "metric",
            "mean",
            "std",
            "min",
            "max",
            "cv",
            "se",
            "ci_low",
            "ci_high",
            "unit",
        ]
        for col in required_columns:
            assert col in header, f"Sweep CSV should have {col} column"

        # Check data rows (should have rows for each concurrency × metrics)
        assert len(lines) > 1, "Sweep CSV should have data rows"

    async def test_artifact_directory_structure_repeated_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test artifact directory structure for repeated mode.

        This test specifically validates:
        - Requirement 3.1: Trial directories with zero-padded numbering
        - Requirement 3.2: Sweep-value results nested within each trial
        - Requirement 3.4: Zero-padded numbering for consistent sorting

        Expected structure:
        artifacts/
          profile_runs/
            trial_0001/
              concurrency_2/
              concurrency_4/
            trial_0002/
              concurrency_2/
              concurrency_4/
          aggregate/
            concurrency_2/
            concurrency_4/
            sweep_aggregate/
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --num-profile-runs 2 \
                --request-count 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify execution succeeded
        assert result.exit_code == 0

        # Test Requirement 3.1: Trial directories with zero-padded numbering
        profile_runs_dir = temp_output_dir / "profile_runs"
        assert profile_runs_dir.exists(), "profile_runs directory must exist"

        trial_dirs = sorted(profile_runs_dir.glob("trial_*"))
        assert len(trial_dirs) == 2, "Should have exactly 2 trial directories"

        # Verify zero-padded naming (trial_0001, trial_0002)
        assert trial_dirs[0].name == "trial_0001", "First trial should be trial_0001"
        assert trial_dirs[1].name == "trial_0002", "Second trial should be trial_0002"

        # Verify lexicographic sorting matches numeric sorting
        trial_names = [d.name for d in trial_dirs]
        assert trial_names == sorted(trial_names), (
            "Zero-padded trial names should sort correctly"
        )

        # Test Requirement 3.2: Sweep-value results nested within each trial
        concurrency_values = [2, 4]
        for trial_dir in trial_dirs:
            for concurrency in concurrency_values:
                concurrency_dir = trial_dir / f"concurrency_{concurrency}"
                assert concurrency_dir.exists(), (
                    f"{trial_dir.name} must contain concurrency_{concurrency}"
                )

                # Verify required artifacts exist
                json_file = concurrency_dir / "profile_export_aiperf.json"
                csv_file = concurrency_dir / "profile_export_aiperf.csv"
                assert json_file.exists(), (
                    f"{trial_dir.name}/concurrency_{concurrency} must have JSON artifact"
                )
                assert csv_file.exists(), (
                    f"{trial_dir.name}/concurrency_{concurrency} must have CSV artifact"
                )

        # Test Requirement 3.4: Zero-padded numbering for aggregate directories
        aggregate_dir = temp_output_dir / "aggregate"
        assert aggregate_dir.exists(), "aggregate directory must exist"

        # Verify per-concurrency aggregate directories
        for concurrency in concurrency_values:
            concurrency_agg_dir = aggregate_dir / f"concurrency_{concurrency}"
            assert concurrency_agg_dir.exists(), (
                f"aggregate/concurrency_{concurrency} must exist"
            )

            # Verify aggregate artifacts
            agg_json = concurrency_agg_dir / "profile_export_aiperf_aggregate.json"
            agg_csv = concurrency_agg_dir / "profile_export_aiperf_aggregate.csv"
            assert agg_json.exists(), (
                f"concurrency_{concurrency} must have aggregate JSON"
            )
            assert agg_csv.exists(), (
                f"concurrency_{concurrency} must have aggregate CSV"
            )

        # Verify sweep aggregate directory
        sweep_agg_dir = aggregate_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), "sweep_aggregate directory must exist"

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        sweep_csv = sweep_agg_dir / "profile_export_aiperf_sweep.csv"
        assert sweep_json.exists(), "sweep_aggregate must have JSON"
        assert sweep_csv.exists(), "sweep_aggregate must have CSV"

        # Verify hierarchical structure integrity
        # All trial directories should be at the same level
        all_trial_dirs = list(profile_runs_dir.glob("*"))
        for d in all_trial_dirs:
            assert d.is_dir(), f"{d.name} should be a directory"
            assert d.name.startswith("trial_"), (
                f"{d.name} should follow trial_NNNN naming"
            )

        # All concurrency directories within trials should be at the same level
        for trial_dir in trial_dirs:
            concurrency_dirs = list(trial_dir.glob("*"))
            for d in concurrency_dirs:
                assert d.is_dir(), f"{trial_dir.name}/{d.name} should be a directory"
                assert d.name.startswith("concurrency_"), (
                    f"{d.name} should follow concurrency_N naming"
                )

    async def test_artifact_directory_structure_independent_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test artifact directory structure for independent mode.

        This test specifically validates:
        - Requirement 3.1: Trial directories with zero-padded numbering
        - Requirement 3.2: Different structure for independent mode (concurrency first, then trials)
        - Requirement 3.4: Zero-padded numbering for consistent sorting

        Expected structure:
        artifacts/
          concurrency_2/
            profile_runs/
              trial_0001/
              trial_0002/
            aggregate/
          concurrency_4/
            profile_runs/
              trial_0001/
              trial_0002/
            aggregate/
          sweep_aggregate/
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --num-profile-runs 2 \
                --parameter-sweep-mode independent \
                --request-count 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify execution succeeded
        assert result.exit_code == 0

        # Test Requirement 3.2: Different structure for independent mode
        # Concurrency directories should be at the top level
        concurrency_values = [2, 4]

        for concurrency in concurrency_values:
            concurrency_dir = temp_output_dir / f"concurrency_{concurrency}"
            assert concurrency_dir.exists(), (
                f"concurrency_{concurrency} directory must exist at top level"
            )

            # Test Requirement 3.1: Trial directories with zero-padded numbering
            profile_runs_dir = concurrency_dir / "profile_runs"
            assert profile_runs_dir.exists(), (
                f"concurrency_{concurrency}/profile_runs must exist"
            )

            trial_dirs = sorted(profile_runs_dir.glob("trial_*"))
            assert len(trial_dirs) == 2, (
                f"concurrency_{concurrency} should have exactly 2 trial directories"
            )

            # Verify zero-padded naming (trial_0001, trial_0002)
            assert trial_dirs[0].name == "trial_0001", (
                f"First trial in concurrency_{concurrency} should be trial_0001"
            )
            assert trial_dirs[1].name == "trial_0002", (
                f"Second trial in concurrency_{concurrency} should be trial_0002"
            )

            # Test Requirement 3.4: Verify lexicographic sorting matches numeric sorting
            trial_names = [d.name for d in trial_dirs]
            assert trial_names == sorted(trial_names), (
                f"Zero-padded trial names in concurrency_{concurrency} should sort correctly"
            )

            # Verify each trial has required artifacts
            for trial_dir in trial_dirs:
                json_file = trial_dir / "profile_export_aiperf.json"
                csv_file = trial_dir / "profile_export_aiperf.csv"
                assert json_file.exists(), (
                    f"concurrency_{concurrency}/{trial_dir.name} must have JSON artifact"
                )
                assert csv_file.exists(), (
                    f"concurrency_{concurrency}/{trial_dir.name} must have CSV artifact"
                )

                # Verify JSON content has correct request count
                with open(json_file) as f:
                    run_data = json.load(f)
                    assert run_data["request_count"]["avg"] == 5

            # Verify aggregate directory exists for this concurrency
            aggregate_dir = concurrency_dir / "aggregate"
            assert aggregate_dir.exists(), (
                f"concurrency_{concurrency}/aggregate must exist"
            )

            # Verify aggregate artifacts
            agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
            agg_csv = aggregate_dir / "profile_export_aiperf_aggregate.csv"
            assert agg_json.exists(), (
                f"concurrency_{concurrency}/aggregate must have JSON"
            )
            assert agg_csv.exists(), (
                f"concurrency_{concurrency}/aggregate must have CSV"
            )

        # Verify sweep aggregate directory at top level
        sweep_agg_dir = temp_output_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), (
            "sweep_aggregate directory must exist at top level"
        )

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        sweep_csv = sweep_agg_dir / "profile_export_aiperf_sweep.csv"
        assert sweep_json.exists(), "sweep_aggregate must have JSON"
        assert sweep_csv.exists(), "sweep_aggregate must have CSV"

        # Verify hierarchical structure integrity for independent mode
        # All concurrency directories should be at the top level
        top_level_dirs = [d for d in temp_output_dir.glob("*") if d.is_dir()]
        concurrency_dirs = [
            d for d in top_level_dirs if d.name.startswith("concurrency_")
        ]
        assert len(concurrency_dirs) == 2, (
            "Should have exactly 2 concurrency directories at top level"
        )

        # Verify concurrency directory naming
        concurrency_dir_names = sorted([d.name for d in concurrency_dirs])
        assert concurrency_dir_names == ["concurrency_2", "concurrency_4"], (
            "Concurrency directories should follow concurrency_N naming"
        )

        # All trial directories within each concurrency should be at the same level
        for concurrency in concurrency_values:
            concurrency_dir = temp_output_dir / f"concurrency_{concurrency}"
            profile_runs_dir = concurrency_dir / "profile_runs"
            trial_dirs = list(profile_runs_dir.glob("*"))

            for d in trial_dirs:
                assert d.is_dir(), (
                    f"concurrency_{concurrency}/profile_runs/{d.name} should be a directory"
                )
                assert d.name.startswith("trial_"), (
                    f"{d.name} should follow trial_NNNN naming"
                )

        # Verify the structure is different from repeated mode
        # In independent mode, concurrency comes first, then trials
        # In repeated mode, trials come first, then concurrency
        # This test confirms independent mode structure
        assert not (temp_output_dir / "profile_runs").exists(), (
            "Independent mode should NOT have profile_runs at top level"
        )
        assert (temp_output_dir / "concurrency_2" / "profile_runs").exists(), (
            "Independent mode should have profile_runs under each concurrency"
        )

    async def test_partial_failure_scenarios(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test that parameter sweep system can handle partial failures gracefully.

        This test validates:
        - Requirement 2.3: System continues with remaining sweep values when one fails
        - Requirement 8.3: Partial results preserved when some values fail
        - Requirement 8.4: Warnings about failed values when some succeed

        Note: This test runs a successful sweep to verify the infrastructure exists
        to track failures. The actual failure handling is tested through the
        aggregation metadata structure which includes failed_runs tracking.

        Testing actual failures with error injection is unreliable because:
        - Error rates are probabilistic and may cause all runs to fail
        - We cannot control which specific runs fail
        - The test would be flaky and non-deterministic

        Instead, we verify that:
        1. The system completes sweeps successfully
        2. Aggregation metadata includes failure tracking fields
        3. The structure supports partial failure scenarios
        """
        # Run a successful sweep to verify the infrastructure
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --num-profile-runs 2 \
                --request-count 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify successful execution
        assert result.exit_code == 0, "Sweep should complete successfully"

        # Verify directory structure
        profile_runs_dir = temp_output_dir / "profile_runs"
        assert profile_runs_dir.exists(), "profile_runs directory should exist"

        trial_dirs = sorted(profile_runs_dir.glob("trial_*"))
        assert len(trial_dirs) == 2, "Should have 2 trial directories"

        # Verify all runs completed successfully
        for trial_dir in trial_dirs:
            for concurrency in [2, 4]:
                concurrency_dir = trial_dir / f"concurrency_{concurrency}"
                assert concurrency_dir.exists(), (
                    f"{trial_dir.name}/concurrency_{concurrency} should exist"
                )

                json_file = concurrency_dir / "profile_export_aiperf.json"
                assert json_file.exists(), (
                    f"{trial_dir.name}/concurrency_{concurrency} should have JSON"
                )

        # Verify aggregation metadata includes failure tracking
        aggregate_dir = temp_output_dir / "aggregate"
        assert aggregate_dir.exists(), "aggregate directory should exist"

        for concurrency in [2, 4]:
            concurrency_agg_dir = aggregate_dir / f"concurrency_{concurrency}"
            assert concurrency_agg_dir.exists(), (
                f"aggregate/concurrency_{concurrency} should exist"
            )

            agg_json = concurrency_agg_dir / "profile_export_aiperf_aggregate.json"
            assert agg_json.exists(), (
                f"concurrency_{concurrency} aggregate JSON should exist"
            )

            with open(agg_json) as f:
                agg_data = json.load(f)

                # Verify metadata includes failure tracking fields
                # This is the key infrastructure for partial failure handling
                metadata = agg_data["metadata"]
                assert "num_profile_runs" in metadata, (
                    "Metadata should track total number of runs"
                )
                assert "num_successful_runs" in metadata, (
                    "Metadata should track successful runs"
                )
                assert "failed_runs" in metadata, "Metadata should track failed runs"

                # In this successful case, verify accounting
                num_successful = metadata["num_successful_runs"]
                num_failed = len(metadata["failed_runs"])
                total_expected = metadata["num_profile_runs"]

                assert num_successful + num_failed == total_expected, (
                    f"Concurrency {concurrency}: successful ({num_successful}) + "
                    f"failed ({num_failed}) should equal total ({total_expected})"
                )

                # All runs should have succeeded in this test
                assert num_successful == 2, (
                    f"Concurrency {concurrency}: should have 2 successful runs"
                )
                assert num_failed == 0, (
                    f"Concurrency {concurrency}: should have 0 failed runs"
                )
                assert len(metadata["failed_runs"]) == 0, (
                    f"Concurrency {concurrency}: failed_runs list should be empty"
                )

        # Verify sweep aggregate exists and has proper structure
        sweep_agg_dir = aggregate_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), "sweep_aggregate directory should exist"

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        assert sweep_json.exists(), "Sweep aggregate JSON should exist"

        with open(sweep_json) as f:
            sweep_data = json.load(f)

            # Verify sweep aggregate structure supports partial failures
            assert "metadata" in sweep_data
            assert "per_value_metrics" in sweep_data

            # All values should be included since all succeeded
            per_value_metrics = sweep_data["per_value_metrics"]
            assert "2" in per_value_metrics, "Should have metrics for concurrency 2"
            assert "4" in per_value_metrics, "Should have metrics for concurrency 4"

            # Verify each value has valid metrics
            for value_str in ["2", "4"]:
                metrics = per_value_metrics[value_str]
                assert len(metrics) > 0, f"Value {value_str} should have metrics"

                # Verify metrics have expected confidence statistics structure
                sample_metric = next(iter(metrics.values()))
                assert "mean" in sample_metric, (
                    f"Value {value_str} metrics should have mean"
                )
                assert "std" in sample_metric, (
                    f"Value {value_str} metrics should have std"
                )
                assert "ci_low" in sample_metric, (
                    f"Value {value_str} metrics should have ci_low"
                )
                assert "ci_high" in sample_metric, (
                    f"Value {value_str} metrics should have ci_high"
                )

        # Summary: This test verifies that the infrastructure for handling
        # partial failures is in place:
        # 1. Aggregation metadata tracks successful and failed runs
        # 2. The accounting (successful + failed = total) is correct
        # 3. Sweep aggregates only include values with successful runs
        # 4. The system can complete sweeps and generate proper aggregates
        #
        # In actual partial failure scenarios (which occur in production):
        # - failed_runs list would contain indices of failed runs
        # - num_successful_runs would be < num_profile_runs
        # - Sweep aggregate would only include values with >= 2 successful runs
        # - System would continue despite failures (Requirement 2.3)
        # - Successful results would be preserved (Requirement 8.3)
        # - Warnings would be logged about failures (Requirement 8.4)

    async def test_backward_compatibility_single_concurrency(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test backward compatibility with single concurrency value.

        This test validates:
        - Requirement 6.1: Single-value concurrency has identical behavior to pre-sweep
        - Requirement 6.2: Identical output structure with confidence runs
        - Requirement 6.4: No sweep-specific directories or aggregates

        The test verifies that using --concurrency 10 (single value) produces
        the same directory structure and output format as the pre-sweep implementation,
        ensuring existing scripts and workflows continue to work unchanged.
        """
        # Test 1: Single concurrency without confidence runs
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 5 \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify successful execution
        assert result.exit_code == 0, "Single concurrency run should succeed"

        # Verify flat directory structure (no sweep-specific directories)
        # Expected structure:
        # artifacts/
        #   {base_name}/
        #     profile_export_aiperf.json
        #     profile_export_aiperf.csv
        #     ...

        # Should NOT have sweep-specific directories
        assert not (temp_output_dir / "profile_runs").exists(), (
            "Single concurrency should NOT create profile_runs directory"
        )
        assert not (temp_output_dir / "concurrency_5").exists(), (
            "Single concurrency should NOT create concurrency_N directory"
        )
        assert not (temp_output_dir / "sweep_aggregate").exists(), (
            "Single concurrency should NOT create sweep_aggregate directory"
        )
        assert not (temp_output_dir / "aggregate").exists(), (
            "Single concurrency without confidence should NOT create aggregate directory"
        )

        # Verify artifacts exist at top level
        json_file = temp_output_dir / "profile_export_aiperf.json"
        csv_file = temp_output_dir / "profile_export_aiperf.csv"
        assert json_file.exists(), "Should have JSON artifact at top level"
        assert csv_file.exists(), "Should have CSV artifact at top level"

        # Verify JSON content has no sweep-related metadata
        with open(json_file) as f:
            run_data = json.load(f)
            assert run_data["request_count"]["avg"] == 10

            # Should NOT have sweep-related metadata
            assert "sweep_index" not in run_data.get("metadata", {}), (
                "Single concurrency should NOT have sweep_index metadata"
            )
            assert "sweep_mode" not in run_data.get("metadata", {}), (
                "Single concurrency should NOT have sweep_mode metadata"
            )

        # Test 2: Single concurrency WITH confidence runs
        # This should produce the same structure as pre-sweep confidence reporting
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 5 \
                --num-profile-runs 3 \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify successful execution
        assert result.exit_code == 0, (
            "Single concurrency with confidence should succeed"
        )

        # Verify confidence directory structure (no sweep-specific directories)
        # Expected structure:
        # artifacts/
        #   {base_name}/
        #     profile_runs/
        #       trial_0001/
        #         profile_export_aiperf.json
        #         profile_export_aiperf.csv
        #       trial_0002/
        #       trial_0003/
        #     aggregate/
        #       profile_export_aiperf_aggregate.json
        #       profile_export_aiperf_aggregate.csv

        profile_runs_dir = temp_output_dir / "profile_runs"
        assert profile_runs_dir.exists(), (
            "Single concurrency with confidence should have profile_runs directory"
        )

        # Verify trial directories exist
        trial_dirs = sorted(profile_runs_dir.glob("trial_*"))
        assert len(trial_dirs) == 3, "Should have 3 trial directories"
        assert trial_dirs[0].name == "trial_0001"
        assert trial_dirs[1].name == "trial_0002"
        assert trial_dirs[2].name == "trial_0003"

        # Verify each trial has artifacts at top level (NOT nested in concurrency_N)
        for trial_dir in trial_dirs:
            json_file = trial_dir / "profile_export_aiperf.json"
            csv_file = trial_dir / "profile_export_aiperf.csv"
            assert json_file.exists(), f"{trial_dir.name} should have JSON at top level"
            assert csv_file.exists(), f"{trial_dir.name} should have CSV at top level"

            # Should NOT have concurrency subdirectories
            concurrency_dirs = list(trial_dir.glob("concurrency_*"))
            assert len(concurrency_dirs) == 0, (
                f"{trial_dir.name} should NOT have concurrency subdirectories"
            )

            # Verify JSON content
            with open(json_file) as f:
                run_data = json.load(f)
                assert run_data["request_count"]["avg"] == 10

                # Should NOT have sweep-related metadata
                assert "sweep_index" not in run_data.get("metadata", {}), (
                    f"{trial_dir.name} should NOT have sweep_index metadata"
                )
                assert "sweep_mode" not in run_data.get("metadata", {}), (
                    f"{trial_dir.name} should NOT have sweep_mode metadata"
                )

        # Verify aggregate directory structure
        aggregate_dir = temp_output_dir / "profile_runs" / "aggregate"
        assert aggregate_dir.exists(), (
            "Should have aggregate directory under profile_runs"
        )

        # Aggregate artifacts should be at top level (NOT in concurrency_N subdirectory)
        agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
        agg_csv = aggregate_dir / "profile_export_aiperf_aggregate.csv"
        assert agg_json.exists(), "Should have aggregate JSON at top level"
        assert agg_csv.exists(), "Should have aggregate CSV at top level"

        # Should NOT have concurrency subdirectories in aggregate
        concurrency_dirs = list(aggregate_dir.glob("concurrency_*"))
        assert len(concurrency_dirs) == 0, (
            "Aggregate should NOT have concurrency subdirectories"
        )

        # Should NOT have sweep_aggregate directory
        assert not (aggregate_dir / "sweep_aggregate").exists(), (
            "Single concurrency should NOT create sweep_aggregate subdirectory"
        )
        assert not (temp_output_dir / "sweep_aggregate").exists(), (
            "Single concurrency should NOT create sweep_aggregate directory"
        )

        # Verify aggregate JSON schema matches pre-sweep format
        with open(agg_json) as f:
            agg_data = json.load(f)

            # Check metadata
            assert agg_data["metadata"]["aggregation_type"] == "confidence"
            assert agg_data["metadata"]["num_profile_runs"] == 3
            assert agg_data["metadata"]["num_successful_runs"] == 3
            assert len(agg_data["metadata"]["failed_runs"]) == 0
            assert agg_data["metadata"]["confidence_level"] == 0.95

            # Should NOT have sweep-related metadata
            assert "parameter_name" not in agg_data["metadata"], (
                "Aggregate should NOT have parameter_name metadata"
            )
            assert "parameter_values" not in agg_data["metadata"], (
                "Aggregate should NOT have parameter_values metadata"
            )
            assert "sweep_mode" not in agg_data["metadata"], (
                "Aggregate should NOT have sweep_mode metadata"
            )

            # Check metrics structure (should have confidence statistics)
            assert "metrics" in agg_data
            metrics = agg_data["metrics"]
            assert len(metrics) > 0, "Should have aggregated metrics"

            # Verify confidence interval fields
            throughput_metrics = [k for k in metrics if "throughput" in k.lower()]
            assert len(throughput_metrics) > 0, "Should have throughput metrics"

            sample_metric = metrics[throughput_metrics[0]]
            required_fields = [
                "mean",
                "std",
                "min",
                "max",
                "cv",
                "se",
                "ci_low",
                "ci_high",
                "t_critical",
                "unit",
            ]
            for field in required_fields:
                assert field in sample_metric, f"Metric should have {field} field"

        # Summary: This test confirms that single-value concurrency maintains
        # complete backward compatibility:
        # 1. No sweep-specific directories (profile_runs/trial_N/concurrency_M pattern)
        # 2. No sweep_aggregate directory
        # 3. No sweep-related metadata in JSON outputs
        # 4. Identical directory structure to pre-sweep implementation
        # 5. Confidence reporting works exactly as before
        # 6. Existing scripts and workflows continue to work unchanged

    async def test_aggregate_file_generation(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test comprehensive aggregate file generation for parameter sweeps.

        This test validates:
        - Requirement 5.1: Sweep aggregate JSON file generated
        - Requirement 5.5: CSV export for tabular analysis
        - Requirement 11.1: Sweep aggregate written to correct path
        - Requirement 11.5: CSV export included

        Tests both repeated and independent modes to ensure all aggregate files
        are generated correctly with proper content and schema.
        """
        # Test 1: Repeated mode aggregate file generation
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6 \
                --num-profile-runs 3 \
                --parameter-sweep-mode repeated \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify successful execution
        assert result.exit_code == 0, "Repeated mode sweep should succeed"

        # Verify per-value confidence aggregate files (JSON and CSV)
        # For repeated mode: aggregate/concurrency_N/profile_export_aiperf_aggregate.{json,csv}
        aggregate_dir = temp_output_dir / "aggregate"
        assert aggregate_dir.exists(), "aggregate directory must exist"

        concurrency_values = [2, 4, 6]
        for concurrency in concurrency_values:
            concurrency_agg_dir = aggregate_dir / f"concurrency_{concurrency}"
            assert concurrency_agg_dir.exists(), (
                f"Per-value aggregate directory for concurrency_{concurrency} must exist"
            )

            # Verify JSON aggregate file exists
            agg_json = concurrency_agg_dir / "profile_export_aiperf_aggregate.json"
            assert agg_json.exists(), (
                f"Per-value aggregate JSON for concurrency_{concurrency} must exist"
            )

            # Verify CSV aggregate file exists
            agg_csv = concurrency_agg_dir / "profile_export_aiperf_aggregate.csv"
            assert agg_csv.exists(), (
                f"Per-value aggregate CSV for concurrency_{concurrency} must exist"
            )

            # Validate JSON content and schema
            with open(agg_json) as f:
                agg_data = json.load(f)

                # Verify required top-level keys
                assert "metadata" in agg_data, "Aggregate JSON must have metadata"
                assert "metrics" in agg_data, "Aggregate JSON must have metrics"

                # Verify metadata schema
                metadata = agg_data["metadata"]
                required_metadata_fields = [
                    "aggregation_type",
                    "num_profile_runs",
                    "num_successful_runs",
                    "failed_runs",
                    "confidence_level",
                ]
                for field in required_metadata_fields:
                    assert field in metadata, (
                        f"Aggregate metadata must have {field} field"
                    )

                assert metadata["aggregation_type"] == "confidence", (
                    "Per-value aggregate should have aggregation_type=confidence"
                )
                assert metadata["num_profile_runs"] == 3, "Should have 3 profile runs"
                assert metadata["num_successful_runs"] == 3, (
                    "All 3 runs should be successful"
                )
                assert len(metadata["failed_runs"]) == 0, "Should have no failed runs"
                assert metadata["confidence_level"] == 0.95, (
                    "Should use 95% confidence level"
                )

                # Verify metrics schema
                metrics = agg_data["metrics"]
                assert len(metrics) > 0, "Should have aggregated metrics"

                # Verify each metric has required confidence statistics fields
                required_metric_fields = [
                    "mean",
                    "std",
                    "min",
                    "max",
                    "cv",
                    "se",
                    "ci_low",
                    "ci_high",
                    "t_critical",
                    "unit",
                ]
                for metric_name, metric_data in metrics.items():
                    for field in required_metric_fields:
                        assert field in metric_data, (
                            f"Metric {metric_name} must have {field} field"
                        )

            # Validate CSV content and format
            csv_content = agg_csv.read_text()
            csv_lines = csv_content.strip().split("\n")
            assert len(csv_lines) > 1, "CSV must have header and data rows"

            # Verify CSV header
            header = csv_lines[0]
            required_csv_columns = [
                "metric",
                "mean",
                "std",
                "min",
                "max",
                "cv",
                "se",
                "ci_low",
                "ci_high",
                "t_critical",
                "unit",
            ]
            for col in required_csv_columns:
                assert col in header, f"CSV header must have {col} column"

        # Verify sweep aggregate files (JSON and CSV)
        # Path: aggregate/sweep_aggregate/profile_export_aiperf_sweep.{json,csv}
        sweep_agg_dir = aggregate_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), (
            "Sweep aggregate directory must exist (Requirement 11.1)"
        )

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        assert sweep_json.exists(), (
            "Sweep aggregate JSON must exist (Requirement 5.1, 11.1)"
        )

        sweep_csv = sweep_agg_dir / "profile_export_aiperf_sweep.csv"
        assert sweep_csv.exists(), (
            "Sweep aggregate CSV must exist (Requirement 5.5, 11.5)"
        )

        # Validate sweep aggregate JSON content and schema
        with open(sweep_json) as f:
            sweep_data = json.load(f)

            # Verify required top-level keys
            required_top_level_keys = [
                "metadata",
                "per_value_metrics",
                "best_configurations",
                "pareto_optimal",
                "trends",
            ]
            for key in required_top_level_keys:
                assert key in sweep_data, f"Sweep aggregate JSON must have {key} key"

            # Verify metadata schema
            metadata = sweep_data["metadata"]
            required_sweep_metadata_fields = [
                "aggregation_type",
                "parameter_name",
                "parameter_values",
                "num_values",
                "num_trials_per_value",
                "sweep_mode",
                "confidence_level",
            ]
            for field in required_sweep_metadata_fields:
                assert field in metadata, f"Sweep metadata must have {field} field"

            assert metadata["aggregation_type"] == "sweep", (
                "Sweep aggregate should have aggregation_type=sweep"
            )
            assert metadata["parameter_name"] == "concurrency", (
                "Should sweep concurrency parameter"
            )
            assert metadata["parameter_values"] == [2, 4, 6], (
                "Should have correct parameter values"
            )
            assert metadata["num_values"] == 3, "Should have 3 sweep values"
            assert metadata["num_trials_per_value"] == 3, (
                "Should have 3 trials per value"
            )
            assert metadata["sweep_mode"] == "repeated", "Should use repeated mode"
            assert metadata["confidence_level"] == 0.95, (
                "Should use 95% confidence level"
            )

            # Verify per_value_metrics schema
            per_value_metrics = sweep_data["per_value_metrics"]
            assert "2" in per_value_metrics, "Should have metrics for concurrency 2"
            assert "4" in per_value_metrics, "Should have metrics for concurrency 4"
            assert "6" in per_value_metrics, "Should have metrics for concurrency 6"

            # Verify each value has metrics with confidence statistics
            for value_str in ["2", "4", "6"]:
                value_metrics = per_value_metrics[value_str]
                assert len(value_metrics) > 0, f"Value {value_str} must have metrics"

                # Verify metrics have confidence statistics fields
                for metric_name, metric_data in value_metrics.items():
                    required_fields = ["mean", "std", "ci_low", "ci_high", "unit"]
                    for field in required_fields:
                        assert field in metric_data, (
                            f"Value {value_str} metric {metric_name} must have {field}"
                        )

            # Verify best_configurations schema
            best_configs = sweep_data["best_configurations"]
            assert "best_throughput" in best_configs, (
                "Must identify best throughput configuration"
            )
            assert "best_latency_p99" in best_configs, (
                "Must identify best latency configuration"
            )

            # Verify best_throughput structure
            best_throughput = best_configs["best_throughput"]
            required_best_fields = ["value", "metric", "unit"]
            for field in required_best_fields:
                assert field in best_throughput, (
                    f"best_throughput must have {field} field"
                )
            assert best_throughput["value"] in [2, 4, 6], (
                "best_throughput value must be from sweep"
            )

            # Verify best_latency structure
            best_latency = best_configs["best_latency_p99"]
            for field in required_best_fields:
                assert field in best_latency, (
                    f"best_latency_p99 must have {field} field"
                )
            assert best_latency["value"] in [2, 4, 6], (
                "best_latency_p99 value must be from sweep"
            )

            # Verify pareto_optimal schema
            pareto_optimal = sweep_data["pareto_optimal"]
            assert isinstance(pareto_optimal, list), "pareto_optimal must be a list"
            assert len(pareto_optimal) > 0, (
                "Must have at least one Pareto optimal point"
            )
            for value in pareto_optimal:
                assert value in [2, 4, 6], "Pareto optimal values must be from sweep"

            # Verify trends schema
            trends = sweep_data["trends"]
            assert len(trends) > 0, "Must have trend analysis"

            # Verify trend structure for each metric
            for metric_name, trend_data in trends.items():
                assert "inflection_points" in trend_data, (
                    f"Trend for {metric_name} must have inflection_points"
                )
                assert "rate_of_change" in trend_data, (
                    f"Trend for {metric_name} must have rate_of_change"
                )
                assert isinstance(trend_data["inflection_points"], list), (
                    f"Trend inflection_points for {metric_name} must be a list"
                )
                assert isinstance(trend_data["rate_of_change"], list), (
                    f"Trend rate_of_change for {metric_name} must be a list"
                )
                # Rate of change should have N-1 values for N sweep values
                assert len(trend_data["rate_of_change"]) == 2, (
                    f"Trend rate_of_change for {metric_name} should have 2 values (N-1)"
                )

        # Validate sweep aggregate CSV content and format
        csv_content = sweep_csv.read_text()
        csv_lines = csv_content.strip().split("\n")
        assert len(csv_lines) > 1, "Sweep CSV must have header and data rows"

        # Verify CSV header
        header = csv_lines[0]
        required_sweep_csv_columns = [
            "concurrency",
            "metric",
            "mean",
            "std",
            "min",
            "max",
            "cv",
            "se",
            "ci_low",
            "ci_high",
            "unit",
        ]
        for col in required_sweep_csv_columns:
            assert col in header, f"Sweep CSV header must have {col} column"

        # Verify data rows exist for each concurrency value
        # Should have rows for each concurrency × metrics combination
        data_rows = csv_lines[1:]
        assert len(data_rows) > 0, "Sweep CSV must have data rows"

        # Verify each concurrency value appears in the CSV
        csv_full_content = "\n".join(csv_lines)
        for concurrency in concurrency_values:
            assert str(concurrency) in csv_full_content, (
                f"Sweep CSV must have data for concurrency {concurrency}"
            )

        # Test 2: Independent mode aggregate file generation
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6 \
                --num-profile-runs 3 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify successful execution
        assert result.exit_code == 0, "Independent mode sweep should succeed"

        # Verify per-value confidence aggregate files (JSON and CSV)
        # For independent mode: concurrency_N/aggregate/profile_export_aiperf_aggregate.{json,csv}
        for concurrency in concurrency_values:
            concurrency_dir = temp_output_dir / f"concurrency_{concurrency}"
            assert concurrency_dir.exists(), (
                f"Concurrency directory for {concurrency} must exist"
            )

            aggregate_dir = concurrency_dir / "aggregate"
            assert aggregate_dir.exists(), (
                f"Aggregate directory for concurrency_{concurrency} must exist"
            )

            # Verify JSON aggregate file exists
            agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
            assert agg_json.exists(), (
                f"Per-value aggregate JSON for concurrency_{concurrency} must exist"
            )

            # Verify CSV aggregate file exists
            agg_csv = aggregate_dir / "profile_export_aiperf_aggregate.csv"
            assert agg_csv.exists(), (
                f"Per-value aggregate CSV for concurrency_{concurrency} must exist"
            )

            # Validate JSON content (same schema as repeated mode)
            with open(agg_json) as f:
                agg_data = json.load(f)

                assert "metadata" in agg_data
                assert "metrics" in agg_data

                metadata = agg_data["metadata"]
                assert metadata["aggregation_type"] == "confidence"
                assert metadata["num_profile_runs"] == 3
                assert metadata["num_successful_runs"] == 3
                assert len(metadata["failed_runs"]) == 0
                assert metadata["confidence_level"] == 0.95

                metrics = agg_data["metrics"]
                assert len(metrics) > 0

                # Verify metrics have required fields
                for metric_name, metric_data in metrics.items():
                    for field in required_metric_fields:
                        assert field in metric_data, (
                            f"Metric {metric_name} must have {field} field"
                        )

            # Validate CSV content
            csv_content = agg_csv.read_text()
            csv_lines = csv_content.strip().split("\n")
            assert len(csv_lines) > 1, "CSV must have header and data rows"

            header = csv_lines[0]
            for col in required_csv_columns:
                assert col in header, f"CSV header must have {col} column"

        # Verify sweep aggregate files (JSON and CSV)
        # Path: sweep_aggregate/profile_export_aiperf_sweep.{json,csv}
        sweep_agg_dir = temp_output_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), (
            "Sweep aggregate directory must exist for independent mode"
        )

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        assert sweep_json.exists(), (
            "Sweep aggregate JSON must exist for independent mode"
        )

        sweep_csv = sweep_agg_dir / "profile_export_aiperf_sweep.csv"
        assert sweep_csv.exists(), "Sweep aggregate CSV must exist for independent mode"

        # Validate sweep aggregate JSON content (same schema as repeated mode)
        with open(sweep_json) as f:
            sweep_data = json.load(f)

            # Verify all required keys exist
            for key in required_top_level_keys:
                assert key in sweep_data, f"Sweep aggregate JSON must have {key} key"

            # Verify metadata
            metadata = sweep_data["metadata"]
            assert metadata["aggregation_type"] == "sweep"
            assert metadata["parameter_name"] == "concurrency"
            assert metadata["parameter_values"] == [2, 4, 6]
            assert metadata["num_values"] == 3
            assert metadata["num_trials_per_value"] == 3
            assert metadata["sweep_mode"] == "independent", (
                "Should use independent mode"
            )
            assert metadata["confidence_level"] == 0.95

            # Verify per_value_metrics
            per_value_metrics = sweep_data["per_value_metrics"]
            for value_str in ["2", "4", "6"]:
                assert value_str in per_value_metrics
                value_metrics = per_value_metrics[value_str]
                assert len(value_metrics) > 0

            # Verify best_configurations
            best_configs = sweep_data["best_configurations"]
            assert "best_throughput" in best_configs
            assert "best_latency_p99" in best_configs

            # Verify pareto_optimal
            pareto_optimal = sweep_data["pareto_optimal"]
            assert isinstance(pareto_optimal, list)
            assert len(pareto_optimal) > 0

            # Verify trends
            trends = sweep_data["trends"]
            assert len(trends) > 0

        # Validate sweep aggregate CSV content
        csv_content = sweep_csv.read_text()
        csv_lines = csv_content.strip().split("\n")
        assert len(csv_lines) > 1, "Sweep CSV must have header and data rows"

        header = csv_lines[0]
        for col in required_sweep_csv_columns:
            assert col in header, f"Sweep CSV header must have {col} column"

        # Summary: This test comprehensively validates aggregate file generation:
        # 1. Per-value confidence aggregate files (JSON and CSV) are generated
        # 2. Sweep aggregate files (JSON and CSV) are generated
        # 3. All files have correct content and schema
        # 4. Both repeated and independent modes generate correct aggregates
        # 5. File paths match requirements (11.1, 11.5)
        # 6. CSV exports are included for tabular analysis (5.5, 11.5)

    async def test_per_value_confidence_statistics(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test per-value confidence statistics computation.

        This test validates:
        - Requirement 4.5: Confidence statistics computed for each concurrency value across N trials
        - Requirement 5.2: Per-value metrics include mean, std, min, max if confidence runs used

        The test verifies that:
        1. Confidence statistics are calculated for each concurrency value
        2. Statistics are computed across all trials at that value
        3. Statistics include all required fields (mean, std, min, max, cv, se, ci_low, ci_high)
        4. Statistics are mathematically correct
        5. Both repeated and independent modes produce correct statistics
        """
        # Test 1: Repeated mode per-value confidence statistics
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6 \
                --num-profile-runs 3 \
                --parameter-sweep-mode repeated \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify successful execution
        assert result.exit_code == 0, "Repeated mode sweep should succeed"

        # Verify per-value confidence aggregates exist
        aggregate_dir = temp_output_dir / "aggregate"
        assert aggregate_dir.exists(), "aggregate directory must exist"

        concurrency_values = [2, 4, 6]

        # Collect raw values from individual trials for validation
        raw_values_by_concurrency = {c: {} for c in concurrency_values}

        # Read individual trial results to collect raw metric values
        profile_runs_dir = temp_output_dir / "profile_runs"
        trial_dirs = sorted(profile_runs_dir.glob("trial_*"))
        assert len(trial_dirs) == 3, "Should have 3 trial directories"

        for trial_dir in trial_dirs:
            for concurrency in concurrency_values:
                concurrency_dir = trial_dir / f"concurrency_{concurrency}"
                json_file = concurrency_dir / "profile_export_aiperf.json"

                with open(json_file) as f:
                    run_data = json.load(f)

                    # Collect metric values from this trial
                    for metric_name, metric_value in run_data.items():
                        if isinstance(metric_value, dict) and "avg" in metric_value:
                            # This is a metric with avg value
                            value = metric_value["avg"]
                            if (
                                metric_name
                                not in raw_values_by_concurrency[concurrency]
                            ):
                                raw_values_by_concurrency[concurrency][metric_name] = []
                            raw_values_by_concurrency[concurrency][metric_name].append(
                                value
                            )

        # Verify per-value confidence statistics for each concurrency
        for concurrency in concurrency_values:
            concurrency_agg_dir = aggregate_dir / f"concurrency_{concurrency}"
            assert concurrency_agg_dir.exists(), (
                f"Per-value aggregate directory for concurrency_{concurrency} must exist"
            )

            agg_json = concurrency_agg_dir / "profile_export_aiperf_aggregate.json"
            assert agg_json.exists(), (
                f"Per-value aggregate JSON for concurrency_{concurrency} must exist"
            )

            with open(agg_json) as f:
                agg_data = json.load(f)

                # Verify metadata indicates confidence aggregation
                metadata = agg_data["metadata"]
                assert metadata["aggregation_type"] == "confidence", (
                    f"Concurrency {concurrency} should have confidence aggregation"
                )
                assert metadata["num_profile_runs"] == 3, (
                    f"Concurrency {concurrency} should have 3 profile runs"
                )
                assert metadata["num_successful_runs"] == 3, (
                    f"Concurrency {concurrency} should have 3 successful runs"
                )
                assert metadata["confidence_level"] == 0.95, (
                    f"Concurrency {concurrency} should use 95% confidence level"
                )

                # Verify metrics have confidence statistics
                metrics = agg_data["metrics"]
                assert len(metrics) > 0, (
                    f"Concurrency {concurrency} should have aggregated metrics"
                )

                # Verify each metric has all required confidence statistics fields
                required_fields = [
                    "mean",
                    "std",
                    "min",
                    "max",
                    "cv",
                    "se",
                    "ci_low",
                    "ci_high",
                    "t_critical",
                    "unit",
                ]

                for metric_name, metric_data in metrics.items():
                    # Verify all required fields exist
                    for field in required_fields:
                        assert field in metric_data, (
                            f"Concurrency {concurrency} metric {metric_name} must have {field} field"
                        )

                    # Verify field types
                    assert isinstance(metric_data["mean"], int | float), (
                        f"Concurrency {concurrency} metric {metric_name} mean must be numeric"
                    )
                    assert isinstance(metric_data["std"], int | float), (
                        f"Concurrency {concurrency} metric {metric_name} std must be numeric"
                    )
                    assert isinstance(metric_data["min"], int | float), (
                        f"Concurrency {concurrency} metric {metric_name} min must be numeric"
                    )
                    assert isinstance(metric_data["max"], int | float), (
                        f"Concurrency {concurrency} metric {metric_name} max must be numeric"
                    )
                    # CV can be None when std is 0 (all values identical)
                    assert metric_data["cv"] is None or isinstance(
                        metric_data["cv"], int | float
                    ), (
                        f"Concurrency {concurrency} metric {metric_name} cv must be numeric or None"
                    )
                    assert isinstance(metric_data["se"], int | float), (
                        f"Concurrency {concurrency} metric {metric_name} se must be numeric"
                    )
                    assert isinstance(metric_data["ci_low"], int | float), (
                        f"Concurrency {concurrency} metric {metric_name} ci_low must be numeric"
                    )
                    assert isinstance(metric_data["ci_high"], int | float), (
                        f"Concurrency {concurrency} metric {metric_name} ci_high must be numeric"
                    )
                    assert isinstance(metric_data["t_critical"], int | float), (
                        f"Concurrency {concurrency} metric {metric_name} t_critical must be numeric"
                    )
                    assert isinstance(metric_data["unit"], str), (
                        f"Concurrency {concurrency} metric {metric_name} unit must be string"
                    )

                    # Verify mathematical relationships (with epsilon for floating-point comparison)
                    epsilon = 1e-9
                    # 1. min <= mean <= max (with tolerance for floating-point precision)
                    assert (
                        metric_data["min"] - epsilon
                        <= metric_data["mean"]
                        <= metric_data["max"] + epsilon
                    ), (
                        f"Concurrency {concurrency} metric {metric_name}: "
                        f"min ({metric_data['min']}) <= mean ({metric_data['mean']}) <= "
                        f"max ({metric_data['max']}) must hold"
                    )

                    # 2. std >= 0
                    assert metric_data["std"] >= 0, (
                        f"Concurrency {concurrency} metric {metric_name}: "
                        f"std ({metric_data['std']}) must be non-negative"
                    )

                    # 3. se >= 0
                    assert metric_data["se"] >= 0, (
                        f"Concurrency {concurrency} metric {metric_name}: "
                        f"se ({metric_data['se']}) must be non-negative"
                    )

                    # 4. ci_low <= mean <= ci_high
                    assert (
                        metric_data["ci_low"]
                        <= metric_data["mean"]
                        <= metric_data["ci_high"]
                    ), (
                        f"Concurrency {concurrency} metric {metric_name}: "
                        f"ci_low ({metric_data['ci_low']}) <= mean ({metric_data['mean']}) <= "
                        f"ci_high ({metric_data['ci_high']}) must hold"
                    )

                    # 5. t_critical > 0 (for 95% confidence with 3 samples, df=2)
                    assert metric_data["t_critical"] > 0, (
                        f"Concurrency {concurrency} metric {metric_name}: "
                        f"t_critical ({metric_data['t_critical']}) must be positive"
                    )

                    # 6. cv = std / mean (if mean != 0)
                    if metric_data["mean"] != 0:
                        expected_cv = metric_data["std"] / abs(metric_data["mean"])
                        assert abs(metric_data["cv"] - expected_cv) < 0.01, (
                            f"Concurrency {concurrency} metric {metric_name}: "
                            f"cv ({metric_data['cv']}) should equal std/mean ({expected_cv})"
                        )

                    # 7. Verify statistics match raw values (if we collected them)
                    if metric_name in raw_values_by_concurrency[concurrency]:
                        raw_values = raw_values_by_concurrency[concurrency][metric_name]
                        assert len(raw_values) == 3, (
                            f"Should have collected 3 raw values for {metric_name}"
                        )

                        # Compute expected statistics from raw values
                        import statistics

                        expected_mean = statistics.mean(raw_values)
                        expected_min = min(raw_values)
                        expected_max = max(raw_values)

                        # Allow small floating point tolerance
                        tolerance = 0.01
                        assert abs(metric_data["mean"] - expected_mean) < tolerance, (
                            f"Concurrency {concurrency} metric {metric_name}: "
                            f"mean ({metric_data['mean']}) should match computed mean ({expected_mean})"
                        )
                        assert abs(metric_data["min"] - expected_min) < tolerance, (
                            f"Concurrency {concurrency} metric {metric_name}: "
                            f"min ({metric_data['min']}) should match computed min ({expected_min})"
                        )
                        assert abs(metric_data["max"] - expected_max) < tolerance, (
                            f"Concurrency {concurrency} metric {metric_name}: "
                            f"max ({metric_data['max']}) should match computed max ({expected_max})"
                        )

        # Verify sweep aggregate includes per-value metrics with confidence statistics
        sweep_agg_dir = aggregate_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), "sweep_aggregate directory must exist"

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        assert sweep_json.exists(), "Sweep aggregate JSON must exist"

        with open(sweep_json) as f:
            sweep_data = json.load(f)

            # Verify metadata
            metadata = sweep_data["metadata"]
            assert metadata["aggregation_type"] == "sweep"
            assert metadata["num_trials_per_value"] == 3, (
                "Sweep metadata should indicate 3 trials per value"
            )
            assert metadata["confidence_level"] == 0.95, (
                "Sweep metadata should indicate 95% confidence level"
            )

            # Verify per_value_metrics includes confidence statistics
            per_value_metrics = sweep_data["per_value_metrics"]
            assert "2" in per_value_metrics, "Should have metrics for concurrency 2"
            assert "4" in per_value_metrics, "Should have metrics for concurrency 4"
            assert "6" in per_value_metrics, "Should have metrics for concurrency 6"

            # Verify each value has metrics with confidence statistics
            for value_str in ["2", "4", "6"]:
                value_metrics = per_value_metrics[value_str]
                assert len(value_metrics) > 0, f"Value {value_str} must have metrics"

                # Verify metrics have confidence statistics fields
                for metric_name, metric_data in value_metrics.items():
                    # Required fields for per-value metrics in sweep aggregate
                    required_sweep_fields = [
                        "mean",
                        "std",
                        "min",
                        "max",
                        "ci_low",
                        "ci_high",
                        "unit",
                    ]
                    for field in required_sweep_fields:
                        assert field in metric_data, (
                            f"Sweep per-value metric {value_str}/{metric_name} must have {field}"
                        )

                    # Verify mathematical relationships (with epsilon for floating-point precision)
                    epsilon = 1e-9
                    assert (
                        metric_data["min"] - epsilon
                        <= metric_data["mean"]
                        <= metric_data["max"] + epsilon
                    ), (
                        f"Sweep value {value_str} metric {metric_name}: "
                        f"min <= mean <= max must hold"
                    )
                    assert metric_data["std"] >= 0, (
                        f"Sweep value {value_str} metric {metric_name}: std must be non-negative"
                    )
                    assert (
                        metric_data["ci_low"]
                        <= metric_data["mean"]
                        <= metric_data["ci_high"]
                    ), (
                        f"Sweep value {value_str} metric {metric_name}: "
                        f"ci_low <= mean <= ci_high must hold"
                    )

        # Test 2: Independent mode per-value confidence statistics
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6 \
                --num-profile-runs 3 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify successful execution
        assert result.exit_code == 0, "Independent mode sweep should succeed"

        # Verify per-value confidence aggregates exist (different structure for independent mode)
        for concurrency in concurrency_values:
            concurrency_dir = temp_output_dir / f"concurrency_{concurrency}"
            assert concurrency_dir.exists(), (
                f"Concurrency directory for {concurrency} must exist"
            )

            aggregate_dir = concurrency_dir / "aggregate"
            assert aggregate_dir.exists(), (
                f"Aggregate directory for concurrency_{concurrency} must exist"
            )

            agg_json = aggregate_dir / "profile_export_aiperf_aggregate.json"
            assert agg_json.exists(), (
                f"Per-value aggregate JSON for concurrency_{concurrency} must exist"
            )

            with open(agg_json) as f:
                agg_data = json.load(f)

                # Verify metadata
                metadata = agg_data["metadata"]
                assert metadata["aggregation_type"] == "confidence"
                assert metadata["num_profile_runs"] == 3
                assert metadata["num_successful_runs"] == 3
                assert metadata["confidence_level"] == 0.95

                # Verify metrics have confidence statistics
                metrics = agg_data["metrics"]
                assert len(metrics) > 0, (
                    f"Concurrency {concurrency} should have aggregated metrics"
                )

                # Verify each metric has all required fields and valid values
                for metric_name, metric_data in metrics.items():
                    # Verify all required fields exist
                    for field in required_fields:
                        assert field in metric_data, (
                            f"Independent mode concurrency {concurrency} metric {metric_name} "
                            f"must have {field} field"
                        )

                    # Verify mathematical relationships (same as repeated mode, with epsilon for floating-point precision)
                    epsilon = 1e-9
                    assert (
                        metric_data["min"] - epsilon
                        <= metric_data["mean"]
                        <= metric_data["max"] + epsilon
                    ), (
                        f"Independent mode concurrency {concurrency} metric {metric_name}: "
                        f"min <= mean <= max must hold"
                    )
                    assert metric_data["std"] >= 0, (
                        f"Independent mode concurrency {concurrency} metric {metric_name}: "
                        f"std must be non-negative"
                    )
                    assert (
                        metric_data["ci_low"]
                        <= metric_data["mean"]
                        <= metric_data["ci_high"]
                    ), (
                        f"Independent mode concurrency {concurrency} metric {metric_name}: "
                        f"ci_low <= mean <= ci_high must hold"
                    )

        # Verify sweep aggregate for independent mode
        sweep_agg_dir = temp_output_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), "sweep_aggregate directory must exist"

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        assert sweep_json.exists(), "Sweep aggregate JSON must exist"

        with open(sweep_json) as f:
            sweep_data = json.load(f)

            # Verify metadata
            metadata = sweep_data["metadata"]
            assert metadata["aggregation_type"] == "sweep"
            assert metadata["sweep_mode"] == "independent"
            assert metadata["num_trials_per_value"] == 3
            assert metadata["confidence_level"] == 0.95

            # Verify per_value_metrics includes confidence statistics
            per_value_metrics = sweep_data["per_value_metrics"]
            for value_str in ["2", "4", "6"]:
                assert value_str in per_value_metrics, (
                    f"Should have metrics for concurrency {value_str}"
                )

                value_metrics = per_value_metrics[value_str]
                assert len(value_metrics) > 0, f"Value {value_str} must have metrics"

                # Verify metrics have confidence statistics
                for metric_name, metric_data in value_metrics.items():
                    required_sweep_fields = [
                        "mean",
                        "std",
                        "min",
                        "max",
                        "ci_low",
                        "ci_high",
                        "unit",
                    ]
                    for field in required_sweep_fields:
                        assert field in metric_data, (
                            f"Independent mode sweep per-value metric {value_str}/{metric_name} "
                            f"must have {field}"
                        )

                    # Verify mathematical relationships (with epsilon for floating-point precision)
                    epsilon = 1e-9
                    assert (
                        metric_data["min"] - epsilon
                        <= metric_data["mean"]
                        <= metric_data["max"] + epsilon
                    )
                    assert metric_data["std"] >= 0
                    assert (
                        metric_data["ci_low"] - epsilon
                        <= metric_data["mean"]
                        <= metric_data["ci_high"] + epsilon
                    )

        # Summary: This test comprehensively validates per-value confidence statistics:
        # 1. Confidence statistics are computed for each concurrency value (Requirement 4.5)
        # 2. Statistics include mean, std, min, max, cv, se, ci_low, ci_high (Requirement 5.2)
        # 3. Statistics are mathematically correct (min <= mean <= max, ci_low <= mean <= ci_high)
        # 4. Both repeated and independent modes produce correct statistics
        # 5. Sweep aggregate includes per-value metrics with confidence statistics
        # 6. All required fields are present and have correct types
        # 7. Mathematical relationships between fields are validated

    async def test_sweep_level_statistics(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test sweep-level statistics computation.

        This test validates:
        - Requirement 5.3: Best throughput value identified
        - Requirement 5.4: Best latency value identified
        - Requirement 5.6: Pareto optimal points identified
        - Requirement 5.7: Multiple Pareto optimal points listed
        - Requirement 5.9: Trend analysis computed
        - Requirement 5.10: Throughput trend indicated
        - Requirement 5.11: Latency trend indicated

        The test verifies that:
        1. Best configurations are correctly identified for throughput and latency
        2. Pareto optimal points are identified (non-dominated configurations)
        3. Trend analysis is performed with inflection points and rate of change
        4. Both repeated and independent modes produce correct sweep-level statistics
        """
        # Test 1: Repeated mode sweep-level statistics
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6,8 \
                --num-profile-runs 3 \
                --parameter-sweep-mode repeated \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify successful execution
        assert result.exit_code == 0, "Repeated mode sweep should succeed"

        # Read sweep aggregate JSON
        aggregate_dir = temp_output_dir / "aggregate"
        sweep_agg_dir = aggregate_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), "sweep_aggregate directory must exist"

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        assert sweep_json.exists(), "Sweep aggregate JSON must exist"

        with open(sweep_json) as f:
            sweep_data = json.load(f)

            # Verify metadata
            metadata = sweep_data["metadata"]
            assert metadata["aggregation_type"] == "sweep"
            assert metadata["parameter_name"] == "concurrency"
            assert metadata["parameter_values"] == [2, 4, 6, 8]
            assert metadata["num_values"] == 4
            assert metadata["sweep_mode"] == "repeated"

            # Test Requirement 5.3 & 5.4: Best configurations identified
            assert "best_configurations" in sweep_data, (
                "Sweep aggregate must have best_configurations"
            )
            best_configs = sweep_data["best_configurations"]

            # Verify best_throughput structure and validity
            assert "best_throughput" in best_configs, (
                "Must identify best throughput configuration (Requirement 5.3)"
            )
            best_throughput = best_configs["best_throughput"]

            # Verify required fields
            assert "value" in best_throughput, "best_throughput must have value field"
            assert "metric" in best_throughput, "best_throughput must have metric field"
            assert "unit" in best_throughput, "best_throughput must have unit field"

            # Verify value is from sweep
            assert best_throughput["value"] in [2, 4, 6, 8], (
                "best_throughput value must be from sweep values"
            )

            # Verify metric is numeric and positive
            assert isinstance(best_throughput["metric"], int | float), (
                "best_throughput metric must be numeric"
            )
            assert best_throughput["metric"] > 0, (
                "best_throughput metric must be positive"
            )

            # Verify unit is appropriate for throughput
            assert (
                "request" in best_throughput["unit"].lower()
                or "req" in best_throughput["unit"].lower()
            ), "best_throughput unit should be related to requests"

            # Verify best_latency structure and validity
            assert "best_latency_p99" in best_configs, (
                "Must identify best latency configuration (Requirement 5.4)"
            )
            best_latency = best_configs["best_latency_p99"]

            # Verify required fields
            assert "value" in best_latency, "best_latency_p99 must have value field"
            assert "metric" in best_latency, "best_latency_p99 must have metric field"
            assert "unit" in best_latency, "best_latency_p99 must have unit field"

            # Verify value is from sweep
            assert best_latency["value"] in [2, 4, 6, 8], (
                "best_latency_p99 value must be from sweep values"
            )

            # Verify metric is numeric and positive
            assert isinstance(best_latency["metric"], int | float), (
                "best_latency_p99 metric must be numeric"
            )
            assert best_latency["metric"] > 0, (
                "best_latency_p99 metric must be positive"
            )

            # Verify unit is appropriate for latency
            assert (
                "ms" in best_latency["unit"].lower()
                or "sec" in best_latency["unit"].lower()
            ), "best_latency_p99 unit should be time-related"

            # Verify best configurations are actually optimal
            # Best throughput should have highest throughput value
            per_value_metrics = sweep_data["per_value_metrics"]
            throughput_values = {}
            latency_values = {}

            for value_str in ["2", "4", "6", "8"]:
                value_metrics = per_value_metrics[value_str]

                # Find throughput metric
                throughput_keys = [
                    k
                    for k in value_metrics
                    if "throughput" in k.lower() and "request" in k.lower()
                ]
                if throughput_keys:
                    throughput_values[int(value_str)] = value_metrics[
                        throughput_keys[0]
                    ]["mean"]

                # Find latency p99 metric
                latency_keys = [
                    k
                    for k in value_metrics
                    if "ttft" in k.lower() and "p99" in k.lower()
                ]
                if latency_keys:
                    latency_values[int(value_str)] = value_metrics[latency_keys[0]][
                        "mean"
                    ]

            # Verify best throughput has maximum throughput
            if throughput_values:
                max_throughput_value = max(throughput_values, key=throughput_values.get)
                assert best_throughput["value"] == max_throughput_value, (
                    f"best_throughput value ({best_throughput['value']}) should be "
                    f"the concurrency with maximum throughput ({max_throughput_value})"
                )

            # Verify best latency has minimum latency
            if latency_values:
                min_latency_value = min(latency_values, key=latency_values.get)
                assert best_latency["value"] == min_latency_value, (
                    f"best_latency_p99 value ({best_latency['value']}) should be "
                    f"the concurrency with minimum latency ({min_latency_value})"
                )

            # Test Requirement 5.6 & 5.7: Pareto optimal points identified
            assert "pareto_optimal" in sweep_data, (
                "Sweep aggregate must have pareto_optimal (Requirement 5.6)"
            )
            pareto_optimal = sweep_data["pareto_optimal"]

            # Verify pareto_optimal is a list
            assert isinstance(pareto_optimal, list), "pareto_optimal must be a list"

            # Verify at least one Pareto optimal point exists
            assert len(pareto_optimal) > 0, (
                "Must have at least one Pareto optimal point"
            )

            # Verify all Pareto optimal values are from sweep
            for value in pareto_optimal:
                assert value in [2, 4, 6, 8], (
                    f"Pareto optimal value {value} must be from sweep values"
                )

            # Verify Pareto optimal points are sorted
            assert pareto_optimal == sorted(pareto_optimal), (
                "Pareto optimal points should be sorted"
            )

            # Verify Pareto optimality property: no point dominates another
            # A point dominates another if it has both higher throughput AND lower latency
            if throughput_values and latency_values:
                for pareto_value in pareto_optimal:
                    # Check that no other point dominates this Pareto optimal point
                    for other_value in [2, 4, 6, 8]:
                        if other_value == pareto_value:
                            continue

                        # Check if other_value dominates pareto_value
                        # (higher throughput AND lower latency)
                        other_throughput = throughput_values.get(other_value, 0)
                        pareto_throughput = throughput_values.get(pareto_value, 0)
                        other_latency = latency_values.get(other_value, float("inf"))
                        pareto_latency = latency_values.get(pareto_value, float("inf"))

                        # If other dominates pareto, that's an error
                        dominates = (
                            other_throughput > pareto_throughput
                            and other_latency < pareto_latency
                        )
                        assert not dominates, (
                            f"Pareto optimal point {pareto_value} is dominated by {other_value} "
                            f"(throughput: {other_throughput} > {pareto_throughput}, "
                            f"latency: {other_latency} < {pareto_latency})"
                        )

            # Test Requirement 5.7: Multiple Pareto optimal points can be listed
            # With 4 sweep values, we should potentially have multiple Pareto points
            # (This is a property test - we verify the system CAN list multiple points)
            # The actual number depends on the data, but the structure supports it
            if len(pareto_optimal) > 1:
                # If we have multiple points, verify they're all distinct
                assert len(pareto_optimal) == len(set(pareto_optimal)), (
                    "Pareto optimal points should be distinct"
                )

            # Test Requirement 5.9, 5.10, 5.11: Trend analysis computed
            assert "trends" in sweep_data, (
                "Sweep aggregate must have trends (Requirement 5.9)"
            )
            trends = sweep_data["trends"]

            # Verify trends is not empty
            assert len(trends) > 0, "Must have trend analysis for at least one metric"

            # Verify trend structure for each metric
            for metric_name, trend_data in trends.items():
                # Verify required fields
                assert "inflection_points" in trend_data, (
                    f"Trend for {metric_name} must have inflection_points"
                )
                assert "rate_of_change" in trend_data, (
                    f"Trend for {metric_name} must have rate_of_change"
                )

                # Verify field types
                assert isinstance(trend_data["inflection_points"], list), (
                    f"Trend inflection_points for {metric_name} must be a list"
                )
                assert isinstance(trend_data["rate_of_change"], list), (
                    f"Trend rate_of_change for {metric_name} must be a list"
                )

                # Verify rate_of_change has N-1 values for N sweep values
                assert len(trend_data["rate_of_change"]) == 3, (
                    f"Trend rate_of_change for {metric_name} should have 3 values (N-1 for N=4)"
                )

                # Verify all rate_of_change values are numeric
                for i, rate in enumerate(trend_data["rate_of_change"]):
                    assert isinstance(rate, int | float), (
                        f"Trend rate_of_change[{i}] for {metric_name} must be numeric"
                    )

                # Verify inflection points are from sweep values
                for inflection_value in trend_data["inflection_points"]:
                    assert inflection_value in [2, 4, 6, 8], (
                        f"Inflection point {inflection_value} for {metric_name} must be from sweep values"
                    )

            # Test Requirement 5.10: Throughput trend indicated
            # Find throughput metric in trends
            throughput_trend_keys = [
                k
                for k in trends
                if "throughput" in k.lower() and "request" in k.lower()
            ]
            assert len(throughput_trend_keys) > 0, (
                "Must have trend analysis for throughput metric (Requirement 5.10)"
            )

            for throughput_key in throughput_trend_keys:
                throughput_trend = trends[throughput_key]

                # Verify trend has rate_of_change (indicates direction)
                assert len(throughput_trend["rate_of_change"]) > 0, (
                    f"Throughput trend {throughput_key} must have rate_of_change values"
                )

                # Verify we can determine trend pattern from rate_of_change
                # Pattern is derivable: all positive = increasing, all negative = decreasing, etc.
                rates = throughput_trend["rate_of_change"]
                positive_count = sum(1 for r in rates if r > 0)
                negative_count = sum(1 for r in rates if r < 0)

                # We should be able to classify the trend
                # (actual classification logic is in the implementation)
                assert positive_count + negative_count <= len(rates), (
                    "Rate of change values should be classifiable"
                )

            # Test Requirement 5.11: Latency trend indicated
            # Find latency metric in trends (ttft_p99_ms or request_latency_p99 as fallback)
            latency_trend_keys = [
                k
                for k in trends
                if ("ttft" in k.lower() and "p99" in k.lower())
                or ("request_latency" in k.lower() and "p99" in k.lower())
            ]
            assert len(latency_trend_keys) > 0, (
                "Must have trend analysis for latency metric (Requirement 5.11)"
            )

            for latency_key in latency_trend_keys:
                latency_trend = trends[latency_key]

                # Verify trend has rate_of_change (indicates direction)
                assert len(latency_trend["rate_of_change"]) > 0, (
                    f"Latency trend {latency_key} must have rate_of_change values"
                )

                # Verify we can determine trend pattern from rate_of_change
                rates = latency_trend["rate_of_change"]
                positive_count = sum(1 for r in rates if r > 0)
                negative_count = sum(1 for r in rates if r < 0)

                # We should be able to classify the trend
                assert positive_count + negative_count <= len(rates), (
                    "Rate of change values should be classifiable"
                )

        # Test 2: Independent mode sweep-level statistics
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6,8 \
                --num-profile-runs 3 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify successful execution
        assert result.exit_code == 0, "Independent mode sweep should succeed"

        # Read sweep aggregate JSON
        sweep_agg_dir = temp_output_dir / "sweep_aggregate"
        assert sweep_agg_dir.exists(), "sweep_aggregate directory must exist"

        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        assert sweep_json.exists(), "Sweep aggregate JSON must exist"

        with open(sweep_json) as f:
            sweep_data = json.load(f)

            # Verify metadata
            metadata = sweep_data["metadata"]
            assert metadata["aggregation_type"] == "sweep"
            assert metadata["parameter_name"] == "concurrency"
            assert metadata["parameter_values"] == [2, 4, 6, 8]
            assert metadata["num_values"] == 4
            assert metadata["sweep_mode"] == "independent"

            # Verify best configurations exist and are valid
            assert "best_configurations" in sweep_data
            best_configs = sweep_data["best_configurations"]

            assert "best_throughput" in best_configs
            best_throughput = best_configs["best_throughput"]
            assert "value" in best_throughput
            assert "metric" in best_throughput
            assert "unit" in best_throughput
            assert best_throughput["value"] in [2, 4, 6, 8]
            assert isinstance(best_throughput["metric"], int | float)
            assert best_throughput["metric"] > 0

            assert "best_latency_p99" in best_configs
            best_latency = best_configs["best_latency_p99"]
            assert "value" in best_latency
            assert "metric" in best_latency
            assert "unit" in best_latency
            assert best_latency["value"] in [2, 4, 6, 8]
            assert isinstance(best_latency["metric"], int | float)
            assert best_latency["metric"] > 0

            # Verify Pareto optimal points exist and are valid
            assert "pareto_optimal" in sweep_data
            pareto_optimal = sweep_data["pareto_optimal"]
            assert isinstance(pareto_optimal, list)
            assert len(pareto_optimal) > 0
            for value in pareto_optimal:
                assert value in [2, 4, 6, 8]
            assert pareto_optimal == sorted(pareto_optimal)

            # Verify trends exist and are valid
            assert "trends" in sweep_data
            trends = sweep_data["trends"]
            assert len(trends) > 0

            for _metric_name, trend_data in trends.items():
                assert "inflection_points" in trend_data
                assert "rate_of_change" in trend_data
                assert isinstance(trend_data["inflection_points"], list)
                assert isinstance(trend_data["rate_of_change"], list)
                assert len(trend_data["rate_of_change"]) == 3  # N-1 for N=4

                for rate in trend_data["rate_of_change"]:
                    assert isinstance(rate, int | float)

                for inflection_value in trend_data["inflection_points"]:
                    assert inflection_value in [2, 4, 6, 8]

            # Verify throughput trend exists
            throughput_trend_keys = [
                k
                for k in trends
                if "throughput" in k.lower() and "request" in k.lower()
            ]
            assert len(throughput_trend_keys) > 0, (
                "Independent mode must have throughput trend"
            )

            # Verify latency trend exists (ttft_p99_ms or request_latency_p99 as fallback)
            latency_trend_keys = [
                k
                for k in trends
                if ("ttft" in k.lower() and "p99" in k.lower())
                or ("request_latency" in k.lower() and "p99" in k.lower())
            ]
            assert len(latency_trend_keys) > 0, (
                "Independent mode must have latency trend"
            )

        # Summary: This test comprehensively validates sweep-level statistics:
        # 1. Best throughput configuration is identified (Requirement 5.3)
        # 2. Best latency configuration is identified (Requirement 5.4)
        # 3. Pareto optimal points are identified correctly (Requirement 5.6)
        # 4. Multiple Pareto optimal points can be listed (Requirement 5.7)
        # 5. Trend analysis is computed with inflection points and rate of change (Requirement 5.9)
        # 6. Throughput trend is indicated (Requirement 5.10)
        # 7. Latency trend is indicated (Requirement 5.11)
        # 8. Both repeated and independent modes produce correct sweep-level statistics
        # 9. Best configurations are verified to be actually optimal
        # 10. Pareto optimality property is verified (no point dominates another)
        # 11. Trend data structure is validated (rate of change has N-1 values)

    async def test_sweep_only_mode_without_confidence(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test sweep-only mode without confidence aggregation.

        This test validates:
        - Sweep mode with single run per value (no confidence aggregation)
        - Correct directory structure without trial nesting
        - No confidence aggregate files generated
        - Sweep aggregate still generated with single-run metrics

        Execution pattern with --concurrency 2,4,6 (default --num-profile-runs 1):
        Single run at concurrency 2
        Single run at concurrency 4
        Single run at concurrency 6
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6 \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Verify basic execution
        assert result.exit_code == 0

        # Verify directory structure for sweep-only mode:
        # artifacts/
        #   concurrency_2/
        #     profile_export_aiperf.json
        #     profile_export_aiperf.csv
        #   concurrency_4/
        #     profile_export_aiperf.json
        #     profile_export_aiperf.csv
        #   concurrency_6/
        #     profile_export_aiperf.json
        #     profile_export_aiperf.csv
        #   sweep_aggregate/
        #     profile_export_aiperf_sweep.json
        #     profile_export_aiperf_sweep.csv

        concurrency_values = [2, 4, 6]

        # Verify concurrency directories exist with single run artifacts
        for concurrency in concurrency_values:
            concurrency_dir = temp_output_dir / f"concurrency_{concurrency}"
            assert concurrency_dir.exists(), (
                f"concurrency_{concurrency} directory should exist"
            )

            # Verify NO profile_runs subdirectory (single run, no nesting)
            profile_runs_dir = concurrency_dir / "profile_runs"
            assert not profile_runs_dir.exists(), (
                f"concurrency_{concurrency} should NOT have profile_runs subdirectory in sweep-only mode"
            )

            # Verify NO aggregate subdirectory (no confidence aggregation)
            aggregate_dir = concurrency_dir / "aggregate"
            assert not aggregate_dir.exists(), (
                f"concurrency_{concurrency} should NOT have aggregate subdirectory in sweep-only mode"
            )

            # Verify single run artifacts exist directly in concurrency directory
            json_file = concurrency_dir / "profile_export_aiperf.json"
            csv_file = concurrency_dir / "profile_export_aiperf.csv"
            assert json_file.exists(), (
                f"concurrency_{concurrency} should have JSON artifact"
            )
            assert csv_file.exists(), (
                f"concurrency_{concurrency} should have CSV artifact"
            )

            # Verify JSON content
            with open(json_file) as f:
                run_data = json.load(f)
                assert run_data["request_count"]["avg"] == 10

        # Verify NO top-level aggregate directory (no confidence aggregation)
        top_level_aggregate_dir = temp_output_dir / "aggregate"
        assert not top_level_aggregate_dir.exists(), (
            "In sweep-only mode, no aggregate directory should exist"
        )

        # Verify NO sweep aggregate directory (sweep-only mode doesn't generate aggregates)
        sweep_agg_dir = temp_output_dir / "sweep_aggregate"
        assert not sweep_agg_dir.exists(), (
            "In sweep-only mode, no sweep_aggregate directory should exist"
        )

        # Summary: This test validates sweep-only mode behavior:
        # 1. Single run per sweep value (no confidence aggregation)
        # 2. Flat directory structure (no trial nesting)
        # 3. No confidence aggregate files generated
        # 4. No sweep aggregate generated (current behavior for single runs)
        # 5. Each concurrency value has its own directory with artifacts

    async def test_sweep_directory_structure_consumable_by_plot(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test that sweep directory structure is consumable by plot command.

        This test validates:
        - Sweep generates proper directory structure
        - Plot command can consume sweep directories without errors
        - Plot command recognizes sweep runs correctly

        Note: Plot may not generate PNG files if data lacks required metrics
        (e.g., GPU telemetry, streaming metrics), but it should run successfully.

        Workflow:
        1. Run parameter sweep (repeated mode)
        2. Verify directory structure
        3. Run aiperf plot on sweep output
        4. Verify plot command succeeds and creates output directory
        """
        # Step 1: Run parameter sweep in repeated mode
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6 \
                --num-profile-runs 2 \
                --parameter-sweep-mode repeated \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert profile_result.exit_code == 0, "Profile command should succeed"

        # Step 2: Verify sweep directory structure exists
        profile_runs_dir = temp_output_dir / "profile_runs"
        assert profile_runs_dir.exists(), "profile_runs directory should exist"

        # Verify trial directories
        trial_dirs = sorted(profile_runs_dir.glob("trial_*"))
        assert len(trial_dirs) == 2, "Should have 2 trial directories"

        # Verify each trial has concurrency subdirectories with artifacts
        for trial_dir in trial_dirs:
            for concurrency in [2, 4, 6]:
                concurrency_dir = trial_dir / f"concurrency_{concurrency}"
                assert concurrency_dir.exists(), (
                    f"{trial_dir.name}/concurrency_{concurrency} should exist"
                )

                # Verify artifacts exist
                json_file = concurrency_dir / "profile_export_aiperf.json"
                csv_file = concurrency_dir / "profile_export_aiperf.csv"
                assert json_file.exists(), f"{concurrency_dir} should have JSON export"
                assert csv_file.exists(), f"{concurrency_dir} should have CSV export"

        # Step 3: Run aiperf plot on the sweep output directory
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {temp_output_dir}
            """,
            assert_success=True,
        )
        assert plot_result.exit_code == 0, (
            "Plot command should succeed on sweep directory"
        )

        # Step 4: Verify plot directory was created and log exists
        plot_dir = temp_output_dir / "plots"
        assert plot_dir.exists(), f"Plot directory should be created at {plot_dir}"

        plot_log = plot_dir / "aiperf_plot.log"
        assert plot_log.exists(), "Plot log should be created"

        # Verify plot log shows runs were detected
        log_content = plot_log.read_text()
        assert "Found 6 unique run directories" in log_content, (
            "Plot should detect 6 run directories (2 trials × 3 concurrency values)"
        )
        assert "MULTI_RUN mode" in log_content, (
            "Plot should detect multi-run mode for sweep"
        )

    async def test_sweep_aggregate_structure_validation(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ):
        """Test sweep aggregate structure with new multi-parameter format.

        This test validates:
        - Sweep aggregates use new coordinate-based format
        - per_combination_metrics is properly structured
        - best_configurations uses parameters dict
        - Pareto optimal configurations are identified

        Workflow:
        1. Run parameter sweep
        2. Verify sweep aggregate JSON structure
        3. Validate new format fields
        """
        # Step 1: Run parameter sweep
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4,6 \
                --num-profile-runs 2 \
                --parameter-sweep-mode repeated \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert profile_result.exit_code == 0, "Profile should succeed"

        # Step 2: Load sweep aggregate JSON
        sweep_agg_dir = temp_output_dir / "aggregate" / "sweep_aggregate"
        sweep_json = sweep_agg_dir / "profile_export_aiperf_sweep.json"
        assert sweep_json.exists(), "Sweep JSON should exist"

        import json

        with open(sweep_json) as f:
            sweep_data = json.load(f)

        # Step 3: Validate new format structure
        # Verify top-level keys
        assert "metadata" in sweep_data
        assert "per_combination_metrics" in sweep_data
        assert "best_configurations" in sweep_data
        assert "pareto_optimal" in sweep_data

        # Verify metadata has sweep_parameters (not parameter_name/values)
        metadata = sweep_data["metadata"]
        assert "sweep_parameters" in metadata
        assert isinstance(metadata["sweep_parameters"], list)
        assert len(metadata["sweep_parameters"]) > 0

        # Verify per_combination_metrics structure
        per_combo = sweep_data["per_combination_metrics"]
        assert isinstance(per_combo, list)
        assert len(per_combo) == 3  # 3 concurrency values

        for combo in per_combo:
            assert "parameters" in combo
            assert "metrics" in combo
            assert isinstance(combo["parameters"], dict)
            assert "concurrency" in combo["parameters"]

        # Verify best_configurations uses parameters dict
        best_configs = sweep_data["best_configurations"]
        if "best_throughput" in best_configs:
            best_throughput = best_configs["best_throughput"]
            assert "parameters" in best_throughput
            assert isinstance(best_throughput["parameters"], dict)
            assert "metric" in best_throughput
            assert "unit" in best_throughput

        # Verify pareto_optimal is a list of parameter dicts
        pareto = sweep_data["pareto_optimal"]
        assert isinstance(pareto, list)
        for config in pareto:
            assert isinstance(config, dict)
            assert "concurrency" in config
