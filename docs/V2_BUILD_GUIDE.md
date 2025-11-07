# V2 Build Configuration Quick Reference

## TL;DR

```bash
# Development (debugging, GDB)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug

# Production (maximum performance)
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release

# E2E Parity Testing (optimized + validation)
./build_e2e_release.sh test
```

## Build Types

| Build | Optimization | Snapshots | Use When |
|-------|--------------|-----------|----------|
| **Debug** | `-O0 -g` | ✅ ON | Developing, debugging with GDB |
| **Release** | `-O3 -DNDEBUG` | ❌ OFF | Deploying, benchmarking performance |
| **E2ERelease** | `-O3 -DNDEBUG` | ✅ ON | Running E2E parity tests |

## Common Commands

### Building

```bash
# Debug
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug
cmake --build build_v2 --parallel

# Release
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# E2ERelease (recommended: use helper script)
./build_e2e_release.sh
```

### Testing

```bash
# Unit tests (Debug build)
cd build_v2 && ctest -R "V2_Unit_" --output-on-failure

# E2E parity tests (E2ERelease build)
./build_e2e_release.sh test

# Or manually:
./build_v2_e2e_release/tests/v2/v2_test_qwen2_fp32_parity
```

### Cleaning

```bash
# Remove specific build
rm -rf build_v2           # Debug
rm -rf build_v2_release   # Release
rm -rf build_v2_e2e_release  # E2ERelease

# Clean and rebuild E2ERelease
./build_e2e_release.sh clean
```

## When to Use Each Build

### Use Debug When:
- ✅ Developing new features
- ✅ Debugging with GDB
- ✅ Running unit tests during development
- ✅ Iterating quickly (faster compile, no optimization)

### Use Release When:
- ✅ Deploying to production
- ✅ Benchmarking performance
- ✅ Measuring inference speed
- ✅ Maximum throughput needed

### Use E2ERelease When:
- ✅ Running E2E parity tests
- ✅ Validating PyTorch ground truth
- ✅ Testing optimized code paths
- ✅ Pre-commit/CI validation

## Performance Impact

**Snapshot Overhead (E2ERelease vs Release):**
- Memory: ~11 MB additional
- CPU: ~0.3ms per snapshot operation
- **Only matters during testing** - not present in standard Release

**Build Time:**
- Debug: ~45s (clean), ~5s (incremental)
- Release: ~60s (clean), ~7s (incremental)
- E2ERelease: ~60s (clean), ~7s (incremental)

## Troubleshooting

### "undefined reference to getSnapshot()"

**Problem:** Trying to run E2E tests with Release build.

**Solution:** Use E2ERelease build instead:
```bash
./build_e2e_release.sh test
```

### E2E Tests Taking Too Long

**Problem:** E2E tests slow in Debug build due to no optimization.

**Solution:** Use E2ERelease (optimized):
```bash
# Debug: ~300s for 5 tests
# E2ERelease: ~130s for 5 tests (2.3× faster)
./build_e2e_release.sh test
```

### Release Build Fails with Snapshot Code

**Problem:** Accidentally enabled snapshots in Release.

**Solution:** Clean rebuild without `-DENABLE_SNAPSHOTS=ON`:
```bash
rm -rf build_v2_release
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel
```

## Build Matrix Summary

```
                   ┌─────────┬─────────┬────────────┐
                   │ Debug   │ Release │ E2ERelease │
┌──────────────────┼─────────┼─────────┼────────────┤
│ Optimization     │ -O0     │ -O3     │ -O3        │
│ Debug Symbols    │ -g      │ (none)  │ (none)     │
│ NDEBUG           │ (none)  │ ✓       │ ✓          │
│ Snapshots        │ ✓       │ ✗       │ ✓          │
│ Use Case         │ Dev     │ Prod    │ Test       │
└──────────────────┴─────────┴─────────┴────────────┘
```

## See Also

- **Full Documentation:** `changelog/2025-11-07-e2e-release-build-config.md`
- **Verification Results:** `changelog/2025-11-07-build-config-verification.md`
- **V2 Architecture:** `.github/instructions/llaminar-v2-architecture.instructions.md`
