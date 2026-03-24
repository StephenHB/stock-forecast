# Agent Manager — reference

## Specialist skills (project)

| Skill | Path | Invoke when |
|-------|------|-------------|
| stock-forecast-analysis | `.cursor/skills/stock-forecast-analysis/` | Notebooks, EDA, features, forecasting, sklearn/LightGBM |
| ui-agent | `.cursor/skills/ui-agent/` | Streamlit, `app.py`, dashboards |
| testing-agent | `.cursor/skills/testing-agent/` | New tests, pytest, integration checks |
| git-agent | `.cursor/skills/git-agent/` | Commit, push, PR; branch policy |

## Task → owner heuristic

- **Touches `app.py` or Streamlit UX** → ui-agent (may pair with analysis for pipeline wiring).
- **Touches `src/forecasting/` or `src/feature_engineering/`** → stock-forecast-analysis; add tests via testing-agent.
- **Touches `tests/` only** → testing-agent.
- **User asked to save or open PR** → git-agent after testing-agent approval for code changes.

## Plan template (optional)

```markdown
## Goal
[One sentence]

## Phases
1. **Phase A** — [objective] → files: `...` → done when: [...]
2. **Phase B** — ...

## Delegation
- Analysis/UI/Test/Git as needed per table above
```
