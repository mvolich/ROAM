# ROAM — Regime Optimised Allocation Model

Streamlit-based fixed-income portfolio optimiser. ROAM blends **baseline carry & roll-down**, a **macro regime overlay**, and **hard portfolio constraints** to propose allocations across fixed-income sleeves. Optimisation is solved with **cvxpy** under **VaR/CVaR** risk controls.

---

## Quick Start (Windows / Cursor)

> Prereqs: Python 3.11+, Git installed.

1. **Open this folder in Cursor**
   - File → *Open Folder…* → select the repo.
   - Settings → **AI** → enable “Use / Index codebase”.

2. **Create & select a local virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   When Cursor prompts “Use this environment for the workspace?” → **Yes**.

3. **Install packages required by `app.py`**
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   pip install streamlit pandas numpy plotly cvxpy ecos osqp openpyxl
   ```

4. **Confirm cvxpy can see solvers**
   ```powershell
   python - << 'PY'
import cvxpy as cp
print("Installed solvers:", cp.installed_solvers())
PY
   ```
   Expect to see at least `ECOS` and `OSQP` listed.

5. **Run**
   ```powershell
   streamlit run app.py
   ```

If you get a solver error, see **Troubleshooting**.

---

## What ROAM Does (architecture in brief)

1. **Load & validate input workbook**  
   Two sheets are merged by `Bloomberg_Ticker`. Missing or malformed fields are flagged in-app.
2. **Baseline expected return**  
   `ER_base = Yield_Hedged_Pct + (Roll_Down_bps_1Y / 10000) * 100` → expressed in annual %.
3. **Macro overlay**  
   Regime presets (Benign / Moderate / Severe) map **rate** shocks (2y/5y/10y/20y/30y) via **KRD** and **spread** shocks (IG/HY/AT1/EM) via **OASD**; **ER_macro_add** is added to `ER_base`.
4. **Optimisation**  
   Choose an objective (below). Hard constraints enforce fund corridors, factor budgets, turnover, and VaR/CVaR bands. Solved with cvxpy.
5. **Diagnostics & output**  
   Weights, contribution to return/risk, cap usage gauges, scenario P&L distribution; optional diff vs prior weights.

---

## Input Workbook Specification

Provide **one Excel file** with **two sheets**:

### 1) `Optimiser_Input` (required columns)

| Column | Type | Meaning |
|---|---|---|
| `Bloomberg_Ticker` | str | Unique key for the sleeve/segment (ticker-like). |
| `Name` | str | Human-readable label. |
| `Instrument_Type` | str | e.g., Govvies, Corps, AT1, EM, Cash, Hybrid. |
| `Yield_Hedged_Pct` | float | Annual yield in **percent** (e.g., 5.10 for 5.10%). |
| `Roll_Down_bps_1Y` | float | 1-year roll-down in **basis points** (e.g., 35 for 0.35%). |
| `OAD_Years` | float | Option-adjusted duration (years). |
| `OASD_Years` | float | Spread duration (years). Used for spread shock mapping. |
| `KRD_2y` `KRD_5y` `KRD_10y` `KRD_20y` `KRD_30y` | float | Key-rate DV01 weights or duration contributions. **Units must be consistent** with how macro shocks are applied. If `KRD_20y` absent, code infers or redistributes (implementation-specific). |
| `Include` | bool/int | 1/True to allow allocation; 0/False to exclude. |

> Notes  
> • `Yield_Hedged_Pct` and `Roll_Down_bps_1Y` together form the baseline ER.  
> • KRDs are used for rate shock translation; OASD for spread shock translation.  
> • Keep NaNs to a minimum. The app guards some gaps but correctness is on the input.

### 2) `MetaData` (required columns)

| Column | Type | Meaning |
|---|---|
| `Bloomberg_Ticker` | str | Must match `Optimiser_Input`. |
| `Is_Non_IG` | bool/int | 1 if sub-IG, else 0. |
| `Is_EM` | bool/int | 1 if EM exposure, else 0. |
| `Is_AT1` | bool/int | 1 if Additional Tier 1, else 0. |
| `Is_T2` | bool/int | 1 if Tier 2, else 0. |
| `Is_Hybrid` | bool/int | 1 if hybrid capital, else 0. |
| `Is_Cash` | bool/int | 1 if cash-like sleeve, else 0. |

### Optional: Previous Weights CSV

| Column | Example |
|---|---|
| `Name` or `Bloomberg_Ticker` | Must match. |
| `Weight` | 0.00–1.00 (fractions) or 0–100 (%) depending on app setting. |

---

## Objectives

- **Max Excess Return / CVaR** — maximise `(ER − r_f) / CVaR_α`.  
- **Risk-adjusted Return (ER − λ·CVaR)**.  
- **Min VaR for Target Return Percentile** — minimise `VaR_α` subject to `ER ≥ q_p(ER)`; auto-relaxes if infeasible.  
- **Max Return** — maximise `ER` subject to caps, budgets, and CVaR cap.

---

## Constraints & Risk Controls

- Corridors: Non-IG, EM, AT1, Hybrid, Cash.  
- Factor budgets: |KRD10y|, |Twist(30y−2y)|, sDV01 IG/HY.  
- Risk: VaR99 and CVaR99 at 1M.  
- Turnover: penalty and/or max turnover.  
- Feasibility: app can auto-relax return-percentile target.

---

## Macro Regimes

Preset 1M 99% shocks:
- Rates (2y/5y/10y/20y/30y) → via KRD.  
- Spreads (IG/HY/AT1/EM) → via OASD.

---

## Troubleshooting

- **ECOS not installed**: `pip install ecos osqp scs`  
- **Wrong interpreter**: Select `.venv\\Scripts\\python.exe` in Cursor.  
- **Infeasible**: relax corridors/budgets, lower target return percentile, widen VaR band.  
- **Merge drops rows**: check `Bloomberg_Ticker` matches exactly across sheets.

---

## Repository Layout

```
ROAM/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ .vscode/settings.json   (optional: pin interpreter)
```

---

## License

MIT
