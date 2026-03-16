---
name: ui-agent
description: >-
  Builds interactive UIs for the stock-forecast project. Use when creating
  Streamlit apps, dashboards, stock selection interfaces, backtesting
  visualizations, or when the user asks for a UI, dashboard, or interactive app.
---

# UI Agent

## Purpose

Build interactive UIs that let users:
- Select stocks to predict (multi-select)
- Set forecast horizon (n days)
- View backtesting results (sidebar, default 2 years)
- Visualize predictions, metrics, and historical data

## Tech Stack (POC)

- **Streamlit** for rapid prototyping
- Built-in components: `st.multiselect`, `st.slider`, `st.sidebar`, `st.metric`, `st.line_chart`
- Optional: `plotly` for richer charts

## UI Layout Pattern

```
+------------------+------------------------+
| Sidebar          | Main content           |
| - Stock multi-   | - Predictions chart    |
|   select         | - Historical prices    |
| - Forecast days  | - Backtest metrics     |
| - Backtest years | - Summary cards        |
| - Run button     |                        |
+------------------+------------------------+
```

## Key Components

1. **Stock selection**: `st.multiselect` with config stocks
2. **Forecast horizon**: `st.slider` or `st.number_input` (days)
3. **Backtest config**: Sidebar, default 2 years
4. **Results**: `st.metric` for MAPE/R², `st.dataframe` for tables, `st.line_chart` for series
5. **Loading**: `st.spinner` during data fetch and model run

## Conventions

- Put config/inputs in sidebar
- Show most important metrics prominently
- Use `st.cache_data` for expensive data loading
- Handle empty selection gracefully

## New Features Workflow

When new functions or features are added to the project, **always** follow this process:

### 1. Research visualization

- **Work with the research agent** to find the best way to visualize the new features in the UI
- Search for: "how to visualize [feature type] in Streamlit/dashboard", "best practices for [data type] visualization"
- Consider: tables vs charts, color coding, drill-down vs summary, placement (sidebar vs main)

### 2. Review and modify UI

- **Review the existing UI** (`app.py`) and identify where the new features fit
- Modify the UI to integrate the new features:
  - Add new sections, expanders, or tabs as needed
  - Follow existing patterns (e.g., `st.dataframe` for tables, `st.caption` for descriptions)
  - Ensure layout remains clear and consistent

### 3. Execute and verify

- **Run the app**: `OMP_NUM_THREADS=1 streamlit run app.py`
- Verify the new features render correctly
- Check for errors, empty states, and edge cases

### 4. Checklist for future features

- [ ] Consult research agent for visualization approach
- [ ] Review existing UI structure
- [ ] Add/modify UI components for new features
- [ ] Run app and verify
- [ ] Update this workflow if the process changes
