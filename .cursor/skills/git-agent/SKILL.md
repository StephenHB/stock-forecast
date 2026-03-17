---
name: git-agent
description: >-
  Helps with git commit, push, and pull requests when there are significant changes.
  Use when the user asks to commit, push, create a PR, or when substantial code changes
  are complete and need to be saved or shared.
---

# Git Agent

## Branch Policy

- **Do not modify code on `main`**. The main branch is for merged, reviewed code only.
- **Always work on a feature branch** when starting new tasks. Create a branch before making changes:
  ```bash
  git checkout main
  git pull origin main
  git checkout -b <branch-name>
  ```
- **Branch naming**: Use descriptive names (e.g., `feat/add-tests`, `fix/loader-validation`, `refactor/forecasting-pipeline`).

## When to Act

Trigger when:
- User asks to commit, push, or create a PR
- Substantial changes are complete (new features, refactors, bug fixes)
- User says "save my work", "push changes", or similar

## Workflow

### 1. Assess Changes

```bash
git status
git diff --stat
```

- **Significant**: New features, multiple files, refactors, bug fixes → proceed
- **Trivial**: Typo fixes, single-line tweaks → ask before committing

### 2. Testing Approval (Required)

Before any commit, push, or PR, obtain approval from the testing-agent:

1. **Invoke testing-agent** to validate the changes:
   - New features/functions: Run or add unit tests; verify they pass
   - Integration: Run or add integration tests; verify existing workflows still work
2. **Run tests**: `pytest tests/ -v` (or `uv run pytest tests/ -v`)
3. **Approval**: Proceed only when tests pass. If tests fail or are missing, fix or add tests first—do not commit until testing-agent approves.

**Skip** only for docs-only or config-only changes (e.g., README, YAML) where no code behavior changes.

### 3. Commit

1. Stage: `git add <files>` or `git add -A` for all
2. Write a clear commit message:

```
<type>(<scope>): <subject>

[optional body]
```

**Types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

**Examples**:
- `feat(forecasting): add LGBM backtester`
- `fix(data_preprocess): handle missing dates in loader`
- `refactor(skills): split AI_README into agent skills`

3. Commit: `git commit -m "message"`

### 4. Push

```bash
git push origin <branch-name>
```

If upstream not set: `git push -u origin <branch-name>`

### 5. Pull Request

When pushing a feature branch (not `main`):

1. Push the branch (step 4)
2. Open PR: Provide GitHub/GitLab URL or instruct user to open from remote
3. PR title: Same as commit message subject
4. PR body: Brief summary of changes, what was added/fixed, any notes

## Checklist

- [ ] Changes are significant enough to commit
- [ ] **Testing-agent approval**: Tests run and pass (or skipped for docs/config-only)
- [ ] Commit message follows type(scope): subject format
- [ ] Pushed to correct branch
- [ ] PR created if on feature branch (not main)

## Additional Resources

- For commit message examples and PR template, see [examples.md](examples.md)
