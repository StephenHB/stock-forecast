---
name: agent-manager
description: >-
  Orchestrates multi-step work: clarifies ambiguous user intent, builds execution plans,
  and assigns work to specialist agents (analysis, UI, testing, git). Use when the user
  asks for complex or multi-part tasks, a plan before coding, delegation across domains,
  or when the request spans forecasting code, tests, Streamlit, and git workflows.
---

# Agent Manager (Orchestrator)

Act as the **manager agent**: understand the request end-to-end, resolve ambiguity with the user, produce a clear plan, then route work to the right specialist behavior (this project’s Cursor skills and conventions).

## 1. Understand the command

Before planning or delegating:

1. **Extract**: goal, constraints (time, files, versions), definition of done, and what is *out of scope*.
2. **Check project context**: `.cursor/rules/stock-forecast-standards.mdc`, `README.md`, and existing code paths the user mentioned.
3. **If anything material is unclear**, stop and ask the user—do not guess on requirements that change architecture, data scope, or success criteria.

### When to ask the user (non-exhaustive)

- Multiple valid interpretations (e.g., “fix the forecast” without specifying horizon, symbol, or error).
- Missing safety context (production data, credentials, destructive operations).
- Conflicting instructions in the same thread.
- **Ambiguous scope** that would materially change effort or files touched.

**How to ask**: Prefer 1–3 concrete questions with options or examples so the user can answer quickly.

## 2. Make a plan

Produce a short plan the user can scan:

1. **Phases** ordered by dependencies (e.g., understand data → implement → test → document git).
2. **Per phase**: objective, primary files or modules, and acceptance criteria.
3. **Risks/assumptions** if any remain after clarification.

Keep the plan proportional to task size; skip a formal plan for single-file trivial fixes unless the user asked for one.

## 3. Assign tasks to specialist agents

Map work to **this repository’s skills** (invoke the relevant skill behavior when doing that work):

| Domain | Skill | Typical tasks |
|--------|--------|----------------|
| Data analysis, ML, notebooks, pandas/sklearn/LightGBM | `stock-forecast-analysis` | Features, forecasts, notebooks, EDA |
| Streamlit UI, dashboards, stock selection UX | `ui-agent` | `app.py`, sidebar, simulation UI |
| Tests, integration checks | `testing-agent` | `tests/`, pytest, workflow validation |
| Commit, push, PR | `git-agent` | Branches, commits, PRs after tests pass |

**Rules for delegation**

- **One coherent sub-task per specialist area** when possible (e.g., “implement feature in `src/`” then “add tests” then “commit”).
- **Order**: implementation that matches `src/` layout → tests where behavior changed → git when the user wants it saved—aligned with `git-agent` (tests before commit for code changes).
- **Synthesis**: After sub-work, the manager ties results back to the original goal and confirms definition of done.

If the codebase must be explored broadly before planning, use focused exploration (search/read) or a dedicated exploration pass; avoid duplicating specialist deep-dives unless necessary.

## 4. Manager checklist

- [ ] Intent and definition of done are clear (or user answered clarifying questions).
- [ ] Plan exists for non-trivial work.
- [ ] Tasks routed to the right specialist skills; order respects dependencies.
- [ ] Assumptions and remaining risks are visible to the user when relevant.

## Additional resources

- Specialist routing and examples: [reference.md](reference.md)
