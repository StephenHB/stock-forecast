---
name: testing-agent
description: >-
  Ensures new features work as designed and integrate with existing workflows.
  Use when adding features, writing tests, validating integrations, or when the
  user asks to test new code, verify functionality, or check workflow integration.
---

# Testing Agent

## Purpose

1. **Unit/feature tests**: Verify new features or functions work as designed
2. **Integration tests**: Verify existing workflows integrate correctly with new features

## Workflow

### Phase 1: Test New Features

When new features or functions are added:

1. **Identify the contract**: What inputs, outputs, and behavior are expected?
2. **Write unit tests** in `test/` mirroring `src/` structure (e.g., `test/forecasting/test_lgbm_forecaster.py`)
3. **Cover**:
   - Happy path (typical inputs)
   - Edge cases (empty data, boundary values, missing inputs)
   - Error handling (invalid inputs, exceptions)
4. **Run**: `uv run pytest test/ -v` (or `pytest test/ -v`)

### Phase 2: Test Integration

When new code touches existing workflows:

1. **Map dependencies**: What modules, pipelines, or notebooks use the new code?
2. **Write integration tests** that:
   - Exercise end-to-end flows (e.g., load data → preprocess → forecast)
   - Use real or minimal fixtures from `data/` or `config/`
   - Assert outputs match expected schema and constraints
3. **Verify**:
   - Existing notebooks still run (or document required updates)
   - Config changes are backward compatible
   - No regressions in downstream consumers

## Test Structure

```
test/
├── data_preprocess/     # Tests for src/data_preprocess/
├── forecasting/        # Tests for src/forecasting/
├── conftest.py         # Shared fixtures (optional)
└── test_integration.py # End-to-end workflow tests
```

## Checklist

**New feature tests:**
- [ ] Unit tests for new functions/classes
- [ ] Edge cases and error paths covered
- [ ] Tests pass locally

**Integration tests:**
- [ ] Workflow tests for affected pipelines
- [ ] Fixtures or mocks for external data/APIs
- [ ] No unintended side effects on existing behavior

## Additional Resources

- For test examples and patterns, see [examples.md](examples.md)
