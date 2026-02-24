# Test Coverage Improvement Plan

## Overview

This document tracks the effort to improve test coverage for the parameter sweep feature. The Codecov report identified 268 lines missing coverage across 6 files.

**Current Status**: 75% overall coverage, 49% for orchestrator.py (the main gap)

**Goal**: Achieve 85%+ coverage for all parameter sweep related files

---

## Coverage Summary by File

| File | Current Coverage | Missing Lines | Priority | Change |
|------|-----------------|---------------|----------|--------|
| orchestrator.py | 63% | 153 miss + 21 partial | HIGH | +14% ⬆️ |
| cli_runner.py | 75% | 28 miss + 9 partial | MEDIUM | No change |
| aggregate_sweep_csv_exporter.py | 88% | 6 miss + 6 partial | LOW | No change |
| sweep.py | 96% | 3 miss + 2 partial | LOW | No change |
| loadgen_config.py | 94% | 7 miss + 6 partial | LOW | -1% ⬇️ |
| user_config.py | 92% | 24 miss + 22 partial | LOW | No change |

**Overall Progress**: orchestrator.py improved from 49% → 63% (+14%) due to new aggregation and sweep aggregation unit tests.

---

## Priority 1: orchestrator.py (49% → 85% target)

**File**: `src/aiperf/orchestrator/orchestrator.py`
**Missing**: 221 lines + 17 partial branches
**Current Tests**: Integration tests only, minimal unit tests

### Missing Coverage Analysis

#### 1. Aggregation and Export Methods (Lines 80-157)
**Lines**: 80-85, 97-109, 124-157
**Functions**: `_aggregate_and_export`, `_export_confidence_aggregate`, `_export_sweep_aggregates`
**Reason**: These are only tested via integration tests
**Test Needed**: Unit tests for:
- Confidence-only aggregation export
- Sweep aggregation export
- Mixed sweep + confidence export
- Error handling when aggregation fails

#### 2. Sweep Execution Logic (Lines 172-303)
**Lines**: 172-303
**Functions**: `_execute_parameter_sweep`, `_collect_failed_sweep_values`
**Reason**: Complex sweep orchestration logic not unit tested
**Test Needed**: Unit tests for:
- Parameter sweep execution with different strategies
- Failed run collection and reporting
- Sweep value iteration
- Error handling during sweep execution

#### 3. Per-Value Export Logic (Lines 329-383)
**Lines**: 329-357, 371-383
**Functions**: `_export_per_value_aggregates`, `export_all_per_value`
**Reason**: Async export logic not unit tested
**Test Needed**: Unit tests for:
- Per-value aggregate export
- Directory creation
- Concurrent export handling
- Export failure scenarios

#### 4. Result Aggregation (Lines 400-433)
**Lines**: 400-402, 422-433
**Functions**: `aggregate_results`
**Reason**: Aggregation routing logic not fully tested
**Test Needed**: Unit tests for:
- Confidence-only aggregation
- Sweep-only aggregation
- Mixed mode aggregation
- Empty results handling

#### 5. Strategy Execution (Lines 499-635)
**Lines**: 499-505, 538-635
**Functions**: `_execute_with_strategy`, `_execute_single_run`
**Reason**: Core execution logic tested via integration only
**Test Needed**: Unit tests for:
- Strategy execution with mocks
- Single run execution
- Error handling
- Result collection

#### 6. Sweep Aggregation Computation (Lines 662-760)
**Lines**: 662-760
**Functions**: `_compute_sweep_aggregate`
**Reason**: Complex aggregation logic not unit tested
**Test Needed**: Unit tests for:
- Sweep aggregate computation
- Per-value aggregate grouping
- Best configuration identification
- Pareto optimal calculation

### Test Files to Create/Enhance

1. **tests/unit/orchestrator/test_orchestrator_aggregation.py** ✅ **COMPLETED**
   - Test `_aggregate_and_export` routing logic ✅
   - Test `_export_confidence_aggregate` ✅
   - Test `_export_sweep_aggregates` ✅
   - Test per-value aggregate export (within sweep export) ✅
   - **Status**: All 14 tests passing
   - **Commit**: 82932dc6 (original), verified passing

2. **tests/unit/orchestrator/test_orchestrator_sweep_execution.py** (NEW)
   - Test `_execute_parameter_sweep`
   - Test `_collect_failed_sweep_values`
   - Test sweep iteration logic
   - Test error handling

3. **tests/unit/orchestrator/test_orchestrator.py** (ENHANCE)
   - Add tests for `aggregate_results`
   - Add tests for `_execute_with_strategy`
   - Add tests for `_execute_single_run`

4. **tests/unit/orchestrator/test_orchestrator_sweep_aggregation.py** ✅ **COMPLETED**
   - Test `_compute_sweep_aggregate` ✅
   - Test per-value grouping ✅
   - Test best configuration logic ✅
   - Test Pareto optimal calculation ✅
   - **Status**: All 9 tests passing, coverage improved +1%
   - **Commit**: b00c6bc5

---

## Priority 2: cli_runner.py (75% → 90% target)

**File**: `src/aiperf/cli_runner.py`
**Missing**: 28 lines + 9 partial branches
**Current Tests**: `tests/unit/test_cli_runner_aggregation.py` exists but incomplete

### Missing Coverage Analysis

#### Lines 116-118, 135-146
**Function**: Sweep aggregation setup in `run_benchmark`
**Reason**: Sweep-specific CLI logic not fully tested
**Test Needed**: Unit tests for:
- Sweep mode detection
- Sweep aggregation configuration
- Sweep-specific validation

#### Lines 197, 206, 219-229
**Function**: Result aggregation and export
**Reason**: Aggregation routing not fully tested
**Test Needed**: Unit tests for:
- Confidence-only aggregation
- Sweep aggregation
- Mixed mode aggregation
- Export path handling

#### Lines 266-270
**Function**: Error handling
**Reason**: Error paths not tested
**Test Needed**: Unit tests for:
- Aggregation failures
- Export failures
- Invalid configuration handling

### Test Files to Enhance

1. **tests/unit/test_cli_runner_aggregation.py** (ENHANCE)
   - Add sweep mode tests
   - Add aggregation routing tests
   - Add error handling tests

---

## Priority 3: aggregate_sweep_csv_exporter.py (88% → 95% target)

**File**: `src/aiperf/exporters/aggregate/aggregate_sweep_csv_exporter.py`
**Missing**: 6 lines + 6 partial branches
**Current Tests**: `tests/unit/exporters/aggregate/test_sweep_exporters.py` exists

### Missing Coverage Analysis

#### Lines 63, 102, 171, 174, 176, 178
**Functions**: Edge cases in CSV formatting
**Reason**: Specific formatting edge cases not tested
**Test Needed**: Unit tests for:
- Empty metric values
- Missing units
- Null/None handling
- Special characters in parameter names

### Test Files to Enhance

1. **tests/unit/exporters/aggregate/test_sweep_exporters.py** (ENHANCE)
   - Add edge case tests for CSV formatting
   - Add tests for missing/null values
   - Add tests for special characters

---

## Priority 4: sweep.py (96% → 98% target)

**File**: `src/aiperf/orchestrator/aggregation/sweep.py`
**Missing**: 3 lines + 2 partial branches
**Current Tests**: `tests/unit/orchestrator/aggregation/test_sweep.py` exists

### Missing Coverage Analysis

#### Lines 233, 268-271
**Functions**: Edge cases in Pareto optimal calculation
**Reason**: Specific edge cases not tested
**Test Needed**: Unit tests for:
- Empty metrics handling
- Single point Pareto optimal
- All points Pareto optimal

### Test Files to Enhance

1. **tests/unit/orchestrator/aggregation/test_sweep.py** (ENHANCE)
   - Add Pareto optimal edge case tests

---

## Priority 5: loadgen_config.py (95% → 98% target)

**File**: `src/aiperf/common/config/loadgen_config.py`
**Missing**: 6 lines + 5 partial branches

### Missing Coverage Analysis

#### Lines 110, 134, 700, 719-722, 758
**Functions**: Validation edge cases
**Reason**: Specific validation paths not tested
**Test Needed**: Unit tests for:
- Edge cases in concurrency list parsing
- Validation error messages
- Boundary conditions

### Test Files to Enhance

1. **tests/unit/common/config/test_loadgen_config_validators.py** (ENHANCE)
   - Add edge case validation tests

---

## Priority 6: user_config.py (92% → 95% target)

**File**: `src/aiperf/common/config/user_config.py`
**Missing**: 24 lines + 22 partial branches

### Missing Coverage Analysis

**Lines**: Various validation and edge case paths
**Reason**: Complex configuration validation not fully tested
**Test Needed**: Unit tests for:
- Sweep-specific validation
- Configuration edge cases
- Error message formatting

### Test Files to Enhance

1. **tests/unit/common/config/test_user_config.py** (ENHANCE)
   - Add sweep validation tests
   - Add edge case tests

---

## Implementation Plan

### Phase 1: High Priority (orchestrator.py)
**Goal**: 49% → 70% coverage
**Estimated Effort**: 3-4 hours
**Tasks**:
1. Create `test_orchestrator_aggregation.py` - Test export methods
2. Create `test_orchestrator_sweep_execution.py` - Test sweep execution
3. Enhance `test_orchestrator.py` - Test core methods

### Phase 2: Medium Priority (cli_runner.py)
**Goal**: 75% → 90% coverage
**Estimated Effort**: 1-2 hours
**Tasks**:
1. Enhance `test_cli_runner_aggregation.py` - Add sweep mode tests

### Phase 3: Low Priority (Remaining Files)
**Goal**: Bring all files to 95%+ coverage
**Estimated Effort**: 2-3 hours
**Tasks**:
1. Add edge case tests to existing test files
2. Add error handling tests
3. Add boundary condition tests

---

## Testing Strategy

### Unit Test Principles
1. **Mock External Dependencies**: Use mocks for file I/O, async operations
2. **Test One Thing**: Each test should verify one specific behavior
3. **Use Fixtures**: Create reusable test data fixtures
4. **Test Edge Cases**: Focus on error paths and boundary conditions
5. **Fast Execution**: Unit tests should run in milliseconds

### Coverage Goals
- **Critical Paths**: 100% coverage (main execution flows)
- **Error Handling**: 90%+ coverage (error paths)
- **Edge Cases**: 85%+ coverage (boundary conditions)
- **Overall**: 85%+ coverage per file

---

## Progress Tracking

### Completed
- ✅ Coverage analysis complete
- ✅ Tracking document created
- ✅ Test plan defined
- ✅ Phase 1 started: orchestrator.py unit tests (49% → 64%)
  - ✅ test_orchestrator_aggregation.py created (14 tests)
  - ✅ Aggregation routing tests
  - ✅ Export method tests
  - ✅ Failed sweep value collection tests
  - ✅ Test naming convention improvements (commit 4e008f3d)

### In Progress
- ⏳ Phase 1: orchestrator.py unit tests (target: 70%+)
  - Need: Sweep execution tests (_execute_parameter_sweep)
  - Need: Strategy execution tests (_execute_with_strategy, _execute_single_run)
  - Need: Sweep aggregation computation tests (_compute_sweep_aggregate)

### Todo
- ⬜ Phase 1: Complete remaining orchestrator.py tests
- ⬜ Phase 2: cli_runner.py unit tests
- ⬜ Phase 3: Edge case tests for remaining files
- ⬜ Final coverage verification

---

## Notes

- Integration tests provide good end-to-end coverage but don't catch edge cases
- Unit tests are needed for error handling, edge cases, and boundary conditions
- Focus on testing the "unhappy paths" that integration tests don't cover
- Use mocks extensively to isolate units under test
- Aim for fast, focused tests that can run in CI

---

## References

- Codecov Report: https://codecov.io/gh/NVIDIA/aiperf
- Integration Tests: `tests/integration/test_parameter_sweep.py`
- Existing Unit Tests: `tests/unit/orchestrator/`, `tests/unit/exporters/aggregate/`
