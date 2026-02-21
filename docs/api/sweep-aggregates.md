<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Sweep Aggregate API Reference

Complete API documentation for parameter sweep aggregate outputs, including JSON schema, CSV format, and programmatic analysis examples.

## Overview

When running parameter sweeps with AIPerf (e.g., `--concurrency 10,20,30`), the system generates sweep aggregate files that summarize performance across all parameter values. These aggregates enable:

- Comparison of performance across parameter values
- Identification of optimal configurations
- Pareto frontier analysis for multi-objective optimization
- Trend analysis showing how metrics change with parameter values

## Output Files

Sweep aggregates are written to the `sweep_aggregate/` directory within your artifacts:

```
artifacts/
  {benchmark_name}/
    sweep_aggregate/
      profile_export_aiperf_sweep.json    # Structured data for programmatic analysis
      profile_export_aiperf_sweep.csv     # Tabular format for spreadsheet analysis
```

---

## JSON Schema

### Top-Level Structure

```json
{
  "metadata": { ... },
  "per_value_metrics": { ... },
  "best_configurations": { ... },
  "pareto_optimal": [ ... ],
  "trends": { ... }
}
```

### Metadata Section

Contains information about the sweep configuration and execution.

```json
{
  "metadata": {
    "aggregation_type": "sweep",
    "parameter_name": "concurrency",
    "parameter_values": [10, 20, 30, 40],
    "num_values": 4,
    "num_trials_per_value": 5,
    "sweep_mode": "repeated",
    "confidence_level": 0.95,
    "benchmark_config": { ... }
  }
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `aggregation_type` | string | Always `"sweep"` for sweep aggregates |
| `parameter_name` | string | Name of the parameter being swept (e.g., `"concurrency"`) |
| `parameter_values` | array[int] | List of parameter values tested |
| `num_values` | int | Number of parameter values in the sweep |
| `num_trials_per_value` | int | Number of trials executed per parameter value (from `--num-profile-runs`) |
| `sweep_mode` | string | Execution mode: `"repeated"` or `"independent"` |
| `confidence_level` | float | Confidence level for statistical intervals (e.g., 0.95 for 95%) |
| `benchmark_config` | object | Full benchmark configuration used |

### Per-Value Metrics Section

Contains aggregated metrics for each parameter value. When multiple trials are run (`--num-profile-runs > 1`), includes confidence statistics.

```json
{
  "per_value_metrics": {
    "10": {
      "request_throughput_avg": {
        "mean": 100.5,
        "std": 5.2,
        "min": 95.0,
        "max": 108.0,
        "cv": 0.052,
        "se": 2.3,
        "ci_low": 94.3,
        "ci_high": 106.7,
        "t_critical": 2.776,
        "unit": "requests/sec"
      },
      "ttft_p99_ms": {
        "mean": 120.5,
        "std": 8.1,
        "min": 110.2,
        "max": 132.8,
        "cv": 0.067,
        "se": 3.6,
        "ci_low": 111.5,
        "ci_high": 129.5,
        "t_critical": 2.776,
        "unit": "ms"
      }
    },
    "20": { ... },
    "30": { ... },
    "40": { ... }
  }
}
```

**Metric Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mean` | float | Mean value across trials |
| `std` | float | Standard deviation across trials |
| `min` | float | Minimum value observed |
| `max` | float | Maximum value observed |
| `cv` | float | Coefficient of variation (std/mean) |
| `se` | float | Standard error of the mean |
| `ci_low` | float | Lower bound of confidence interval |
| `ci_high` | float | Upper bound of confidence interval |
| `t_critical` | float | Critical t-value used for confidence interval |
| `unit` | string | Unit of measurement |

**Note:** For single-trial sweeps (`--num-profile-runs 1`), only `mean` and `unit` fields are present.

### Best Configurations Section

Identifies the parameter values that achieved the best performance for key metrics.

```json
{
  "best_configurations": {
    "best_throughput": {
      "value": 40,
      "metric": 350.2,
      "unit": "requests/sec"
    },
    "best_latency_p99": {
      "value": 10,
      "metric": 120.5,
      "unit": "ms"
    }
  }
}
```

**Configuration Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `value` | int | Parameter value that achieved best performance |
| `metric` | float | The metric value achieved |
| `unit` | string | Unit of measurement |

**Available Configurations:**

- `best_throughput`: Highest `request_throughput_avg`
- `best_latency_p99`: Lowest `ttft_p99_ms`

### Pareto Optimal Section

Lists parameter values that are Pareto optimal - configurations where no other configuration is strictly better on all objectives simultaneously.

```json
{
  "pareto_optimal": [10, 30, 40]
}
```

**Default Objectives:**
- Maximize: `request_throughput_avg` (throughput)
- Minimize: `ttft_p99_ms` (latency)

A configuration is Pareto optimal if:
- No other configuration has both higher throughput AND lower latency
- It represents a valid trade-off point on the efficiency frontier

**Example Interpretation:**
```
Concurrency 10: Low latency, moderate throughput (latency-optimized)
Concurrency 30: Balanced latency and throughput
Concurrency 40: High throughput, higher latency (throughput-optimized)
```

### Trends Section

Analyzes how metrics change across parameter values, identifying patterns and inflection points.

```json
{
  "trends": {
    "request_throughput_avg": {
      "inflection_points": [30],
      "rate_of_change": [75.5, 85.2, 15.3]
    },
    "ttft_p99_ms": {
      "inflection_points": [],
      "rate_of_change": [10.2, 15.5, 20.1]
    }
  }
}
```

**Trend Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `inflection_points` | array[int] | Parameter values where trend changes significantly |
| `rate_of_change` | array[float] | Change in metric between consecutive parameter values |

**Interpreting Rate of Change:**

The `rate_of_change` array has N-1 values for N parameter values:
- `rate_of_change[0]`: Change from value 1 to value 2
- `rate_of_change[1]`: Change from value 2 to value 3
- etc.

**Pattern Detection:**
- All positive → Increasing trend
- All negative → Decreasing trend
- Near zero → Plateau
- Mixed signs → Non-monotonic behavior

**Inflection Points:**

Detected when:
- Sign flip in rate of change (increasing → decreasing or vice versa)
- Magnitude change > 50% between consecutive rates

**Example:**
```json
{
  "request_throughput_avg": {
    "rate_of_change": [80.0, 90.0, 15.0]
  }
}
```
Interpretation: Throughput increases steadily from value 1→2 (+80) and 2→3 (+90), then plateaus from 3→4 (+15), indicating diminishing returns.

---

## CSV Format

The CSV export provides a tabular view optimized for spreadsheet analysis and plotting.

### Structure

The CSV file contains multiple sections separated by blank lines:

1. **Per-Value Metrics Table** (main data)
2. **Best Configurations**
3. **Pareto Optimal Points**
4. **Trends**
5. **Metadata**

### Per-Value Metrics Table

The first section is a wide-format table with one row per parameter value:

```csv
parameter_value,request_throughput_avg_mean,request_throughput_avg_std,request_throughput_avg_min,request_throughput_avg_max,request_throughput_avg_cv,ttft_p99_ms_mean,ttft_p99_ms_std,ttft_p99_ms_min,ttft_p99_ms_max,ttft_p99_ms_cv
10,100.50,5.20,95.00,108.00,0.0520,120.50,8.10,110.20,132.80,0.0672
20,180.30,8.50,170.00,195.00,0.0471,135.20,9.30,125.00,148.00,0.0688
30,270.80,12.10,255.00,290.00,0.0447,155.80,11.20,142.00,172.00,0.0719
40,285.50,15.30,265.00,310.00,0.0536,180.30,13.50,165.00,200.00,0.0749
```

**Columns:**
- `parameter_value`: The parameter value (e.g., concurrency level)
- `{metric}_mean`: Mean value across trials
- `{metric}_std`: Standard deviation
- `{metric}_min`: Minimum value
- `{metric}_max`: Maximum value
- `{metric}_cv`: Coefficient of variation

### Best Configurations Section

```csv
Best Configurations
Metric,Best Value,Metric Value,Unit
Best Throughput,40,285.50,requests/sec
Best Latency P99,10,120.50,ms
```

### Pareto Optimal Section

```csv
Pareto Optimal Points
Parameter Values
10
30
40
```

### Trends Section

```csv
Trends
Metric: request_throughput_avg
Inflection Points,30
Rate of Change,79.80, 90.50, 14.70

Metric: ttft_p99_ms
Inflection Points,None
Rate of Change,14.70, 20.60, 24.50
```

### Metadata Section

```csv
Metadata
Aggregation Type,sweep
Total Runs,12
Successful Runs,12
Parameter Name,concurrency
Parameter Values,"10, 20, 30, 40"
Number of Values,4
```

---

## Artifact Directory Structure

### Repeated Mode (`--parameter-sweep-mode repeated`)

Default mode where the full sweep is executed N times:

```
artifacts/
  {benchmark_name}/
    profile_runs/
      trial_0001/
        concurrency_10/
          profile_export_aiperf.json
          profile_export_aiperf.csv
        concurrency_20/
          profile_export_aiperf.json
          profile_export_aiperf.csv
        concurrency_30/
          profile_export_aiperf.json
          profile_export_aiperf.csv
      trial_0002/
        concurrency_10/
        concurrency_20/
        concurrency_30/
      trial_0003/
        concurrency_10/
        concurrency_20/
        concurrency_30/
    aggregate/
      concurrency_10/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
      concurrency_20/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
      concurrency_30/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
      sweep_aggregate/
        profile_export_aiperf_sweep.json
        profile_export_aiperf_sweep.csv
```

**Execution Pattern:**
```
Trial 1: [10 → 20 → 30]
Trial 2: [10 → 20 → 30]
Trial 3: [10 → 20 → 30]
```

### Independent Mode (`--parameter-sweep-mode independent`)

All trials at each parameter value before moving to the next:

```
artifacts/
  {benchmark_name}/
    concurrency_10/
      profile_runs/
        trial_0001/
          profile_export_aiperf.json
          profile_export_aiperf.csv
        trial_0002/
        trial_0003/
      aggregate/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
    concurrency_20/
      profile_runs/
        trial_0001/
        trial_0002/
        trial_0003/
      aggregate/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
    concurrency_30/
      profile_runs/
        trial_0001/
        trial_0002/
        trial_0003/
      aggregate/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
    sweep_aggregate/
      profile_export_aiperf_sweep.json
      profile_export_aiperf_sweep.csv
```

**Execution Pattern:**
```
Concurrency 10: [trial1, trial2, trial3]
Concurrency 20: [trial1, trial2, trial3]
Concurrency 30: [trial1, trial2, trial3]
```

### Single-Trial Sweep

When `--num-profile-runs 1` (or omitted), no trial directories are created:

```
artifacts/
  {benchmark_name}/
    concurrency_10/
      profile_export_aiperf.json
      profile_export_aiperf.csv
    concurrency_20/
      profile_export_aiperf.json
      profile_export_aiperf.csv
    concurrency_30/
      profile_export_aiperf.json
      profile_export_aiperf.csv
    sweep_aggregate/
      profile_export_aiperf_sweep.json
      profile_export_aiperf_sweep.csv
```

---

## Programmatic Analysis Examples

### Example 1: Load and Inspect Sweep Results

```python
import json
from pathlib import Path

# Load sweep aggregate
sweep_file = Path("artifacts/my_benchmark/sweep_aggregate/profile_export_aiperf_sweep.json")
with open(sweep_file) as f:
    sweep_data = json.load(f)

# Inspect metadata
metadata = sweep_data["metadata"]
print(f"Parameter: {metadata['parameter_name']}")
print(f"Values tested: {metadata['parameter_values']}")
print(f"Trials per value: {metadata['num_trials_per_value']}")
print(f"Sweep mode: {metadata['sweep_mode']}")
```

### Example 2: Find Optimal Configuration

```python
# Get best configurations
best_configs = sweep_data["best_configurations"]

best_throughput = best_configs["best_throughput"]
print(f"Best throughput: {best_throughput['metric']:.2f} {best_throughput['unit']}")
print(f"  at {metadata['parameter_name']}={best_throughput['value']}")

best_latency = best_configs["best_latency_p99"]
print(f"Best latency: {best_latency['metric']:.2f} {best_latency['unit']}")
print(f"  at {metadata['parameter_name']}={best_latency['value']}")
```

### Example 3: Analyze Pareto Frontier

```python
# Get Pareto optimal points
pareto_optimal = sweep_data["pareto_optimal"]
print(f"Pareto optimal configurations: {pareto_optimal}")

# Extract metrics for Pareto points
per_value_metrics = sweep_data["per_value_metrics"]

print("\nPareto Frontier:")
for value in pareto_optimal:
    metrics = per_value_metrics[str(value)]
    throughput = metrics["request_throughput_avg"]["mean"]
    latency = metrics["ttft_p99_ms"]["mean"]
    print(f"  {metadata['parameter_name']}={value}: "
          f"{throughput:.1f} req/s, {latency:.1f} ms p99")
```

### Example 4: Detect Performance Trends

```python
# Analyze trends
trends = sweep_data["trends"]

throughput_trend = trends.get("request_throughput_avg", {})
rate_of_change = throughput_trend.get("rate_of_change", [])
inflection_points = throughput_trend.get("inflection_points", [])

# Determine pattern
if all(r > 0 for r in rate_of_change):
    pattern = "increasing"
elif all(r < 0 for r in rate_of_change):
    pattern = "decreasing"
elif all(abs(r) < 5 for r in rate_of_change):
    pattern = "plateau"
else:
    pattern = "mixed"

print(f"Throughput trend: {pattern}")
if inflection_points:
    print(f"Inflection points at: {inflection_points}")
```

### Example 5: Compare Confidence Intervals

```python
import matplotlib.pyplot as plt

# Extract data for plotting
param_values = metadata["parameter_values"]
throughputs = []
ci_lows = []
ci_highs = []

for value in param_values:
    metrics = per_value_metrics[str(value)]
    tp = metrics["request_throughput_avg"]
    throughputs.append(tp["mean"])
    ci_lows.append(tp["ci_low"])
    ci_highs.append(tp["ci_high"])

# Plot with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(param_values, throughputs, 'o-', label='Mean Throughput')
plt.fill_between(param_values, ci_lows, ci_highs, alpha=0.3, label='95% CI')
plt.xlabel(metadata["parameter_name"].title())
plt.ylabel('Throughput (requests/sec)')
plt.title('Throughput vs Concurrency')
plt.legend()
plt.grid(True)
plt.savefig('throughput_sweep.png')
```

### Example 6: Export to Pandas DataFrame

```python
import pandas as pd

# Convert per-value metrics to DataFrame
rows = []
for value_str, metrics in per_value_metrics.items():
    row = {"parameter_value": int(value_str)}
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict):
            row[f"{metric_name}_mean"] = metric_data.get("mean")
            row[f"{metric_name}_std"] = metric_data.get("std")
            row[f"{metric_name}_cv"] = metric_data.get("cv")
        else:
            row[metric_name] = metric_data
    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values("parameter_value")

# Analyze
print(df[["parameter_value", "request_throughput_avg_mean", "ttft_p99_ms_mean"]])

# Export
df.to_csv("sweep_analysis.csv", index=False)
```

### Example 7: Identify Diminishing Returns

```python
# Calculate efficiency (throughput per unit of parameter)
param_values = metadata["parameter_values"]
efficiencies = []

for value in param_values:
    metrics = per_value_metrics[str(value)]
    throughput = metrics["request_throughput_avg"]["mean"]
    efficiency = throughput / value
    efficiencies.append(efficiency)

# Find point of diminishing returns (where efficiency drops significantly)
threshold = 0.8  # 20% drop
for i in range(1, len(efficiencies)):
    if efficiencies[i] < threshold * efficiencies[i-1]:
        print(f"Diminishing returns detected at {metadata['parameter_name']}={param_values[i]}")
        print(f"  Efficiency dropped from {efficiencies[i-1]:.2f} to {efficiencies[i]:.2f}")
        break
```

### Example 8: Multi-Objective Decision Making

```python
# Score configurations based on weighted objectives
weights = {
    "throughput": 0.6,  # 60% weight on throughput
    "latency": 0.4,     # 40% weight on latency
}

# Normalize metrics to [0, 1] range
throughputs = [per_value_metrics[str(v)]["request_throughput_avg"]["mean"]
               for v in param_values]
latencies = [per_value_metrics[str(v)]["ttft_p99_ms"]["mean"]
             for v in param_values]

max_tp = max(throughputs)
min_lat = min(latencies)
max_lat = max(latencies)

scores = []
for i, value in enumerate(param_values):
    # Normalize: higher is better for both
    tp_score = throughputs[i] / max_tp
    lat_score = 1 - (latencies[i] - min_lat) / (max_lat - min_lat)

    # Weighted combination
    score = weights["throughput"] * tp_score + weights["latency"] * lat_score
    scores.append((value, score))

# Find best configuration
best_value, best_score = max(scores, key=lambda x: x[1])
print(f"Best configuration for given weights: {metadata['parameter_name']}={best_value}")
print(f"  Score: {best_score:.3f}")
```

---

## See Also

- [Parameter Sweeping Tutorial](../tutorials/parameter-sweeping.md) - User guide with examples
- [Multi-Run Confidence Tutorial](../tutorials/multi-run-confidence.md) - Understanding confidence statistics
- [Working with Profile Exports](../tutorials/working-with-profile-exports.md) - General export analysis
- [CLI Options Reference](../cli_options.md) - Complete CLI documentation
