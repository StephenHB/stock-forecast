# LGBM Information Leakage Audit

**Branch:** `feat/lgbm-2stage`  
**Purpose:** Ensure no input features use data from dates >= target variable date.

---

## Target Variable Definition

- **Daily:** `target_{N}d = Close.shift(-N)` → for row at date `t`, target is Close at `t+N`
- **Weekly:** `target_{N}w = Close.shift(-N)` → for row at date `t`, target is Close at `t+N` weeks

**Verdict:** Target correctly uses future price only. No leakage in target definition.

---

## Feature Audit

### 1. Lag Features (`close_lag_1`, `close_lag_2`, etc.)

- **Source:** `Close.shift(lag)` with `lag > 0`
- **At row t:** Uses Close at `t-1`, `t-2`, ...
- **Verdict:** ✅ Causal. Only past data.

### 2. Rolling Features (`close_ma_5`, `close_std_10`, etc.)

- **Source:** `Close.rolling(window=w).mean()` with `center=False` (default)
- **At row t:** Uses Close from `t-w+1` to `t` (current and past)
- **Verdict:** ✅ Causal. Close at `t` is known when predicting Close at `t+N`.

### 3. Volume Features (`volume_ma_5`, `volume_ratio`)

- **Source:** Rolling on Volume
- **Verdict:** ✅ Causal.

### 4. Time Features (`day_of_week`, `month`, `week_of_year`)

- **Source:** Derived from index date at row `t`
- **Verdict:** ✅ Causal. Date at `t` is known.

### 5. Trend and Seasonality (Prophet / MA)

**Moving Average (fallback):**
- **Trend:** `Close.rolling(window).mean()` → causal
- **Seasonality:** `Close - trend` → causal
- **Verdict:** ✅ No leakage.

**Prophet (when data >= 60 rows):**
- **In-sample:** Prophet fits on full series; fitted values at row `t` use the entire fit (including future rows). The seasonal component is a smooth function; leakage is limited but present.
- **Backtesting:** For strict no-leakage, use MA fallback or fit Prophet per train window only (not yet implemented).
- **Single forecast:** Using last row's Prophet trend/seasonality is OK—we predict from known last date.
- **Verdict:** ⚠️ Minor in-sample leakage for Prophet in backtest. Use MA for strict backtest, or accept for production.

### 6. Volatility Features (DailyVolatilityFeatures)

- **Source:** Returns, rolling vol, ADR—all use past/current data
- **Verdict:** ✅ Causal.

---

## Summary

| Feature Type      | Leakage | Notes                          |
|-------------------|---------|--------------------------------|
| Lag               | None    | shift(positive) only           |
| Rolling           | None    | past + current                 |
| Time              | None    | from row date                  |
| Trend/Season (MA) | None    | causal rolling                 |
| Trend/Season (Prophet) | Minor | in-sample fit uses full series |

**Recommendation:** For backtesting, consider `use_prophet=False` or fit Prophet per train window. For production forecast, current implementation is acceptable.
