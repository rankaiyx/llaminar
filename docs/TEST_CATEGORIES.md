# Llaminar Test Categories Reference

**Last Updated**: Post-reorganization (123 tests)  
**Related Documents**: [TASK_PROFILE_VALIDATION.md](../TASK_PROFILE_VALIDATION.md), [SMOKE_TEST_COMPLETE.md](../SMOKE_TEST_COMPLETE.md)

---

## Quick Reference

| Category | Count | Runtime | Pass Rate | Use Case |
|----------|-------|---------|-----------|----------|
| **Smoke** | 14 | 1.16s | 100% | Every build, pre-commit |
| **Unit** | 99 | 2m30s | 89% | Pull requests, development |
| **Parity** | 4 | 24.5s | 75% | Cross-implementation validation |
| **Integration** | 26 | 3m0s | 81% | Major features, releases |
| **Full** | 123 | 3m0s | 86% | Pre-release, comprehensive |

---

## Test Naming Conventions

### Terminology

**"Parity" Tests** - Reserved for cross-implementation validation:
- `test_parity_framework.cpp` - Validates against llama.cpp and PyTorch
- `test_abstract_pipeline_parity.cpp` - Pipeline interface consistency
- Used ONLY when comparing Llaminar output to external reference implementations

**"Correctness" Tests** - Internal validation and unit tests:
- `test_*_correctness.cpp` - Self-contained correctness checks
- Mathematical properties (e.g., attention mask symmetry)
- Component contracts (e.g., RMSNorm invariants)
- Internal consistency (e.g., incremental vs batch decode)

**Historical Context**: 16 tests were renamed from `*_parity` to `*_correctness` to eliminate confusion. Only true cross-implementation validation tests should use "parity" naming.

---

## Test Categories by Purpose

### 1. Infrastructure Tests
**Purpose**: System initialization, configuration, topology

| Test | Category | Runtime | Purpose |
|------|----------|---------|---------|
| BasicTest | Smoke | ~0.05s | MPI initialization |
| NumaTest | Smoke | ~0.05s | NUMA topology detection |
| PipelineFactoryTest | Smoke | ~0.05s | Pipeline registration system |

**Usage**: Run on every build to catch environment issues early.

---

### 2. Kernel Tests
**Purpose**: Core compute kernels (Linear, Attention, RMSNorm, Softmax, RoPE)

#### Fast Kernel Tests (Smoke)
| Test | Runtime | Coverage |
|------|---------|----------|
| MPILinearKernelTest | ~0.20s | Distributed linear projection |
| MPIRMSNormKernelTest | ~0.15s | Distributed RMSNorm |
| MPIAttentionKernelTest | ~0.20s | Distributed attention mechanism |
| MPISoftmaxCorrectnessTest | ~0.05s | Softmax correctness validation |
| RMSNormCoreCorrectness | ~0.05s | RMSNorm core logic |
| SoftmaxCoreCorrectness | ~0.05s | Softmax core logic |
| LinearOrientationCorrectnessTest | ~0.05s | Linear orientation handling |

#### Comprehensive Kernel Tests (Unit)
| Category | Tests | Total Time | Coverage |
|----------|-------|------------|----------|
| attention | 1 | 0.70s | Attention kernel variants |
| rmsnorm | 1 | 2.02s | RMSNorm implementations |
| rope | 1 | 0.40s | RoPE (Rotary Position Embedding) |
| kernel | 3 | 7.57s | General kernel infrastructure |

**Usage**: 
- Smoke kernel tests: Every build
- Full kernel tests: Pull requests, kernel changes

---

### 3. Tensor & Distribution Tests
**Purpose**: Tensor partitioning, sharding, distributed operations

| Category | Tests | Runtime | Purpose |
|----------|-------|---------|---------|
| tp (tensor parallelism) | 33 | 46.34s | Tensor partition strategies |
| tp2 | 6 | 6.33s | 2-way tensor parallelism |
| tp3 | 6 | 5.25s | 3-way tensor parallelism |
| tp4 | 6 | 5.40s | 4-way tensor parallelism |
| sharding | 9 | 26.49s | Weight sharding correctness |

**Quick Tests (Smoke)**:
- TPPartitionSpecTest (~0.05s)

**Usage**: Run when modifying tensor distribution or MPI logic.

---

### 4. COSMA Tests
**Purpose**: Distributed matrix multiplication (COSMA library integration)

| Category | Tests | Runtime | Coverage |
|----------|-------|---------|----------|
| cosma | 16 | 18.54s | COSMA integration, layouts, strategies |
| gemm | 17 | 17.01s | General matrix multiply correctness |
| matrix | 2 | 18.19s | Large matrix operations |

**Quick Tests (Smoke)**:
- LargeMatmulPlanTest (~0.05s) - COSMA matrix planning

**Known Issues**: 
- CosmaInLayoutElementwiseMultiTest - Layout handling issue
- TPGemmCorrectness_Matrix - Gemm matrix parity

**Usage**: Run when modifying COSMA integration or matmul backend selection.

---

### 5. Model Loading & Quantization Tests
**Purpose**: GGUF loading, weight role classification, quantization formats

| Test Category | Tests | Runtime | Purpose |
|---------------|-------|---------|---------|
| dequant | 4 | 3.63s | Dequantization correctness |
| logging (ModelLoadRoleTags) | 10 | 300.53s | Weight role classification per quant format |

**Quick Tests (Smoke)**:
- DequantTest (~0.10s)
- WeightRoleClassification (~0.05s)

**Known Issues**: 7 ModelLoadRoleTags tests fail (missing model files)

**Usage**: 
- Smoke: Every build (basic dequant + role classification)
- Full: When adding quantization format support

---

### 6. Pipeline & Integration Tests
**Purpose**: End-to-end pipeline execution, multi-component scenarios

| Category | Tests | Runtime | Purpose |
|----------|-------|---------|---------|
| qwen | 1 | 3.18s | Qwen pipeline integration |
| integration | 2 | 4.51s | Multi-component integration |
| prefill | 1 | 1.13s | Prefill operation validation |
| simulation | 2 | 1.45s | Simulated execution scenarios |

**Known Issues**:
- QwenPipelineTest - Pipeline failures
- CosmaPrefillAttentionIntegrationTest - COSMA integration
- PrefillAttentionGoldenTest - Timeout (tolerance issues)

**Usage**: Integration test profile for major features.

---

### 7. Parity Tests (Cross-Implementation)
**Purpose**: Validate against reference implementations (llama.cpp, PyTorch)

| Test Suite | Tests | Runtime | Reference |
|------------|-------|---------|-----------|
| ParityFramework | 4 | ~22s | llama.cpp, PyTorch |
| AbstractPipelineParity | 1 | ~2s | Internal pipeline consistency |

**Test Breakdown**:
1. `BasicSnapshotCapture` - Snapshot capture mechanism
2. `SnapshotComparison` - Snapshot comparison infrastructure
3. `DistributedPipelineVsLlamaCpp` - ❌ Tolerance issue (max_abs=24.1 vs 0.002)
4. `DistributedPipelineVsPyTorchReference` - ⏭️ Requires PYTORCH_SNAPSHOT_DIR

**Known Issues**:
- llama.cpp parity: Numerical divergence exceeds tolerance
- PyTorch parity: Environment setup required

**Usage**: 
- Run before releases
- Run after major algorithm changes
- Consider relaxing tolerances or making informational

**Setup**:
```bash
# PyTorch parity requires snapshot directory
export PYTORCH_SNAPSHOT_DIR=/path/to/snapshots

# Run parity tests
ctest --test-dir build --verbose -R "ParityFramework"
```

---

### 8. Correctness Tests (Internal Validation)
**Purpose**: Self-consistency checks, mathematical properties

| Category | Tests | Purpose |
|----------|-------|---------|
| correctness | 5 | Component internal consistency |
| parity (internal) | 45 | Self-consistency validation (NOT cross-implementation) |

**Examples**:
- `test_rmsnorm_core_correctness.cpp` - RMSNorm invariants
- `test_softmax_core_correctness.cpp` - Softmax properties
- `test_linear_orientation_correctness.cpp` - Orientation handling
- `test_incremental_decode_correctness.cpp` - Batch vs incremental equivalence
- `test_rope_recurrence_correctness.cpp` - RoPE recurrence relation

**Usage**: Unit test profile (comprehensive internal validation).

---

### 9. Stress & Edge Case Tests
**Purpose**: Boundary conditions, resource limits, unusual inputs

| Category | Tests | Runtime | Purpose |
|----------|-------|---------|---------|
| stress | 2 | 4.47s | Resource limits, large operations |
| ragged | 8 | 7.21s | Ragged tensor handling |
| negative | 2 | 3.57s | Negative test cases |
| extended | 1 | 3.75s | Extended scenarios |

**Usage**: Full test suite for comprehensive validation.

---

### 10. MLP & Layer Tests
**Purpose**: Multi-layer perceptron, layer-by-layer validation

| Category | Tests | Runtime | Coverage |
|----------|-------|---------|----------|
| mlp | 13 | 26.60s | MLP layer correctness |
| embedding | 2 | 7.23s | Embedding layer handling |

**Known Issues**:
- EmbeddingStandaloneTest - 25s timeout

**Usage**: Unit test profile includes MLP tests.

---

## VSCode Tasks

All test profiles are available as VSCode tasks. Access via:
- `Ctrl+Shift+P` → "Tasks: Run Task" → Select test profile

### Available Tasks
1. **test: smoke tests** - 14 tests, ~1s
2. **test: unit tests** - 99 tests, ~2.5min
3. **test: parity integration** - 4 tests, ~25s
4. **test: integration tests** - 26 tests, ~3min
5. **test: run all tests** - 123 tests, ~3min

See `.vscode/tasks.json` for complete task definitions.

---

## Command Line Usage

### Smoke Tests (1.16s)
```bash
ctest --test-dir build --output-on-failure --parallel \
  -R "^(BasicTest|NumaTest|PipelineFactoryTest|DequantTest|TPPartitionSpecTest|LargeMatmulPlanTest|WeightRoleClassification|MPILinearKernelTest|MPIRMSNormKernelTest|MPIAttentionKernelTest|MPISoftmaxCorrectnessTest|RMSNormCoreCorrectness|SoftmaxCoreCorrectness|LinearOrientationCorrectnessTest)$"
```

### Unit Tests (2m30s)
```bash
ctest --test-dir build --output-on-failure --parallel \
  -E "(Integration|ParityFramework|Incremental|Qwen|Prefill|.*Stress.*)"
```

### Parity Integration (24.5s)
```bash
ctest --test-dir build --output-on-failure --verbose \
  -R "(ParityFramework|AbstractPipelineParity)"
```

### Integration Tests (3m0s)
```bash
ctest --test-dir build --output-on-failure --verbose \
  -R "(Integration|Incremental|Qwen|Prefill|End2End|KVCache)"
```

### Full Test Suite (3m0s)
```bash
ctest --test-dir build --output-on-failure --parallel
```

---

## Test Selection Guide

### When to Run Which Tests

| Scenario | Recommended Profile | Reasoning |
|----------|---------------------|-----------|
| **Every code change** | Smoke | 1.16s - catches major breakage |
| **Before commit** | Smoke | Fast feedback, pre-commit hook |
| **During development** | Unit | 2m30s - comprehensive kernel validation |
| **Pull request** | Unit + Integration | Validates changes without full suite |
| **Before merging** | Full | Complete validation |
| **Release candidate** | Full + Parity | Cross-implementation validation |
| **Kernel changes** | Smoke → Unit → Full | Incremental validation |
| **MPI/distributed changes** | Unit (MPI tests) → Integration | Focus on distributed operations |
| **Quantization changes** | Dequant tests → Full | Focus on loading tests |
| **Pipeline changes** | Integration → Parity | Multi-component + cross-validation |

---

## Failure Investigation Guide

### Common Failure Patterns

#### 1. ModelLoadRoleTags Failures (7 tests)
**Symptom**: All Q*_K format tests fail with logging errors  
**Cause**: Missing model files in `models/` directory  
**Fix**: Add small quantized test models or make optional  
**Severity**: Low (doesn't affect core functionality)

#### 2. Tolerance Exceeded Failures (3 tests)
**Symptom**: `max_abs` or `rel_l2` exceeds threshold  
**Examples**: PrefillAttentionGolden, ParityFramework (llama.cpp), TPGemm  
**Cause**: Numerical divergence (quantization, FP32 accumulation, algorithm differences)  
**Fix**: Relax tolerances or investigate layer-by-layer divergence  
**Severity**: Medium (may indicate real issue or just tolerance tuning)

#### 3. COSMA Integration Failures (3 tests)
**Symptom**: COSMA prefill or layout tests fail  
**Cause**: Distributed matrix multiplication integration issues  
**Fix**: Debug COSMA strategy selection and layout handling  
**Severity**: High (affects distributed prefill performance)

#### 4. Timeout Failures (2 tests)
**Symptom**: Test exceeds time limit (60s)  
**Examples**: EmbeddingStandaloneTest (25s), PrefillAttentionGoldenTest  
**Cause**: Expensive computation or infinite loop  
**Fix**: Optimize or disable permanently  
**Severity**: Medium (may indicate performance regression)

#### 5. Kernel Correctness Failures (2 tests)
**Symptom**: Core kernel validation fails  
**Examples**: AttentionMicroTest, IncrementalDecodeCorrectnessMulti  
**Cause**: Algorithm bug or numerical precision issue  
**Fix**: Debug kernel implementation  
**Severity**: **Critical** (affects core functionality)

---

## CI/CD Integration

### Recommended GitHub Actions Workflow

```yaml
name: Llaminar Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  smoke:
    name: Smoke Tests
    runs-on: ubuntu-latest
    timeout-minutes: 2
    steps:
      - uses: actions/checkout@v3
      - name: Configure
        run: cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
      - name: Build
        run: cmake --build build --parallel
      - name: Run Smoke Tests
        run: |
          ctest --test-dir build --output-on-failure --parallel \
            -R "^(BasicTest|NumaTest|PipelineFactoryTest|DequantTest|...)"

  unit:
    name: Unit Tests
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v3
      - name: Configure
        run: cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
      - name: Build
        run: cmake --build build --parallel
      - name: Run Unit Tests
        run: |
          ctest --test-dir build --output-on-failure --parallel \
            -E "(Integration|ParityFramework|Incremental|Qwen|Prefill|.*Stress.*)"

  integration:
    name: Integration Tests
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v3
      - name: Configure
        run: cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
      - name: Build
        run: cmake --build build --parallel
      - name: Run Integration Tests
        run: |
          ctest --test-dir build --output-on-failure --verbose \
            -R "(Integration|Incremental|Qwen|Prefill|End2End|KVCache)"

  full:
    name: Full Test Suite
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v3
      - name: Configure
        run: cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
      - name: Build
        run: cmake --build build --parallel
      - name: Run Full Suite
        run: ctest --test-dir build --output-on-failure --parallel
```

### Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "🧪 Running smoke tests before commit..."

ctest --test-dir build --parallel \
  -R "^(BasicTest|NumaTest|PipelineFactoryTest|DequantTest|TPPartitionSpecTest|LargeMatmulPlanTest|WeightRoleClassification|MPILinearKernelTest|MPIRMSNormKernelTest|MPIAttentionKernelTest|MPISoftmaxCorrectnessTest|RMSNormCoreCorrectness|SoftmaxCoreCorrectness|LinearOrientationCorrectnessTest)$"

if [ $? -ne 0 ]; then
  echo "❌ Smoke tests failed! Fix tests before committing."
  exit 1
fi

echo "✅ All smoke tests passed!"
exit 0
```

---

## Maintenance Guidelines

### Adding New Tests

1. **Choose appropriate naming**:
   - `test_*_correctness.cpp` for internal validation
   - `test_*_parity.cpp` ONLY for cross-implementation validation
   - `test_*.cpp` for integration or specialized tests

2. **Add appropriate labels** in CMakeLists.txt:
   ```cmake
   add_test(NAME MyNewTest ...)
   set_tests_properties(MyNewTest PROPERTIES LABELS "kernel;correctness;unit")
   ```

3. **Consider smoke test inclusion** if:
   - Runtime < 0.5s
   - Tests critical infrastructure (MPI, NUMA, pipeline)
   - Broad coverage with minimal cost

4. **Update documentation**:
   - Add to appropriate category in this file
   - Update test counts in TASK_PROFILE_VALIDATION.md
   - Update VSCode task filters if needed

### Removing Tests

1. **Mark as deprecated** first (add `DISABLED_` prefix to test name)
2. **Document reason** for removal in commit message
3. **Update CMakeLists.txt** to remove target
4. **Update test counts** in documentation
5. **Verify VSCode tasks** still work with new count

### Modifying Test Categories

1. **Update `.vscode/tasks.json`** with new filters
2. **Test all 5 profiles** to ensure no regressions
3. **Update documentation** with new timings/counts
4. **Consider backward compatibility** for CI/CD workflows

---

## Glossary

- **Smoke Test**: Fast sanity check (typically <1s per test)
- **Unit Test**: Component-level validation (typically <5s per test)
- **Integration Test**: Multi-component scenario (typically >5s)
- **Parity Test**: Cross-implementation validation (external reference)
- **Correctness Test**: Internal consistency validation (self-contained)
- **Golden Test**: Comparison against pre-computed reference output
- **Stress Test**: Resource limits and boundary conditions

---

## Related Documentation

- [TASK_PROFILE_VALIDATION.md](../TASK_PROFILE_VALIDATION.md) - Detailed validation report for all 5 test profiles
- [SMOKE_TEST_COMPLETE.md](../SMOKE_TEST_COMPLETE.md) - Comprehensive smoke test documentation
- [PARITY_FRAMEWORK_SUMMARY.md](../PARITY_FRAMEWORK_SUMMARY.md) - Parity testing framework details
- [.github/copilot-instructions.md](../.github/copilot-instructions.md) - Testing guidelines section

---

**Maintained by**: GitHub Copilot (David Sanftenberg)  
**Last Validation**: Post-reorganization test audit  
**Test Suite Version**: 123 tests (86% pass rate)
