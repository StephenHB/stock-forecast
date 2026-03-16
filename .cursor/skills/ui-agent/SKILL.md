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
