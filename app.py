# Rubrics Fixed Income Optimiser (Streamlit)
# ------------------------------------------
# Forward-looking, factor & scenario-aware optimisation with fund-specific corridors and VaR controls.
# Open with: streamlit run app.py
#
# Inputs: Optimiser_Input_Final_v9.xlsx (sheet "Optimiser_Input" + "MetaData")
# Author: GPT-5 Pro (assistant)

import io
import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import time
APP_BUILD = f"macro-expret-fix-2025-09-02-cloud-fix-2-{int(time.time())}"

# Silence DPP warnings after making problems DPP-compliant
try:
    from cvxpy.reductions.solvers.solving_chain import DPP_ERROR_MSG
    warnings.filterwarnings("ignore", message=DPP_ERROR_MSG)
except Exception:
    pass

# --- Rubrics branding & theme ---
RB_COLORS = {
    "blue":   "#001E4F",  # Rubrics Blue
    "medblue":"#2C5697",  # Rubrics Medium Blue
    "ltblue": "#7BA4DB",  # Rubrics Light Blue
    "grey":   "#D8D7DF",  # Rubrics Grey
    "orange": "#CF4520",  # Rubrics Orange
}
FUND_COLOR = {"GFI": RB_COLORS["blue"], "GCF": RB_COLORS["medblue"], "EYF": RB_COLORS["ltblue"]}

from contextlib import contextmanager

PERF_KEY = "_perf_events"
def _perf_reset():
    st.session_state[PERF_KEY] = []
def _perf_add(step, ms, **meta):
    st.session_state.setdefault(PERF_KEY, [])
    st.session_state[PERF_KEY].append({"step": step, "ms": float(ms), **({k:v for k,v in meta.items() if v is not None})})
@contextmanager
def perf_step(name, **meta):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _perf_add(name, (time.perf_counter()-t0)*1000.0, **meta)

def inject_brand_css():
    st.markdown("""
    <style>
      /* Fonts */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
      :root{ --rb-blue:#001E4F; --rb-mblue:#2C5697; --rb-lblue:#7BA4DB; --rb-grey:#D8D7DF; --rb-orange:#CF4520; }
      html, body, .stApp, [class*="css"] { background-color:#f8f9fa; font-family:Inter, "Segoe UI", Roboto, Arial, sans-serif !important; color:#0b0c0c; }
      .stTabs [data-baseweb="tab-list"]{ display:flex !important; width:100% !important; gap:12px; border-bottom:none; }
      .stTabs [data-baseweb="tab"]{ background-color:var(--rb-grey); border-radius:4px 4px 0 0; color:var(--rb-blue); font-weight:600; min-width:180px; text-align:center; padding:8px 16px; }
      .stTabs [aria-selected="true"]{ background-color:var(--rb-mblue)!important; color:#fff!important; border-bottom:3px solid rgb(207,69,32)!important; }
      .stTabs [data-baseweb="tab-list"] > [data-baseweb="tab"]:last-child{ margin-left:auto !important; }

      .stButton > button, .stDownloadButton > button { background-color:var(--rb-mblue); color:#fff; border:none; border-radius:4px; padding:8px 16px; font-weight:600; }
      .stButton > button:hover, .stDownloadButton > button:hover { background-color:var(--rb-blue); }

      .stSlider [role="slider"]{ background-color:var(--rb-orange)!important; }
      .stSlider [data-baseweb="slider"] div[aria-hidden="true"]{ background-color:var(--rb-orange)!important; }

      h1, h2, h3, h4, h5, h6 { color:var(--rb-blue)!important; font-weight:700; }

      .rb-title { display:flex; align-items:center; justify-content:space-between; margin:.4rem 0 .2rem 0; }
      .rb-title .rb-label { font-weight:600; color:var(--rb-blue); }
      .rb-help { color:var(--rb-mblue); cursor:help; font-weight:700; user-select:none; }
      .rb-help:hover { color:var(--rb-orange); }

      /* Allow full-page scroll */
      html, body { height:auto!important; overflow-y:auto!important; }
      .stApp, [data-testid="stAppViewContainer"] { height:auto!important; min-height:100vh!important; overflow-y:auto!important; overflow-x:hidden; }
      section.main, [data-testid="stMain"] { height:auto!important; overflow:visible!important; }
      .block-container { padding-bottom:6rem!important; }
      .stColumn, .stExpander, [data-testid="column"], [data-testid="stVerticalBlock"] { overflow:visible!important; }

      /* Force white background on dropdowns */
      div[data-baseweb="select"] > div {
        background-color: #fff !important;
        border: 1px solid var(--rb-grey);
        border-radius: 4px;
        color: #000;
      }
      div[data-baseweb="select"] > div:hover { border-color: var(--rb-mblue); }

    """, unsafe_allow_html=True)

def title_with_help(label: str, help_text: str):
    st.markdown(
        f'<div class="rb-title"><div class="rb-label">{label}</div>'
        f'<div class="rb-help" title="{help_text}">ⓘ</div></div>',
        unsafe_allow_html=True
    )

BRAND_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        colorway=["#001E4F","#2C5697","#7BA4DB","#D8D7DF","#CF4520"],
        font=dict(family="Ringside, Inter, Segoe UI, Roboto, Arial, sans-serif"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", title=dict(font=dict(size=16))
    )
)
pio.templates["rubrics"] = BRAND_TEMPLATE
pio.templates.default = "rubrics"
plotly_default_config = {"displaylogo": False, "responsive": True}

# Try to import cvxpy; if missing, provide a graceful message
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception as e:
    CVXPY_AVAILABLE = False
    CVXPY_ERROR = str(e)

# ----------------------------- Solver -----------------------------
@dataclass
class ParamCVaRSolver:
    N: int; S: int
    w: 'cp.Variable'; z: 'cp.Variable'; alpha: 'cp.Variable'
    p_mu: 'cp.Parameter'; target_ret: 'cp.Parameter'; cvar_cap: 'cp.Parameter'
    p_cvar_alpha: 'cp.Parameter'; p_var_floor: 'cp.Parameter'
    prev_w: 'cp.Parameter'; max_turn: 'cp.Parameter'; turn_pen: 'cp.Parameter'
    p_max_non_ig: 'cp.Parameter'; p_max_em: 'cp.Parameter'; p_max_hybrid: 'cp.Parameter'; p_max_cash: 'cp.Parameter'; p_max_at1: 'cp.Parameter'
    p_min_non_ig: 'cp.Parameter'; p_min_em: 'cp.Parameter'; p_min_hybrid: 'cp.Parameter'; p_min_cash: 'cp.Parameter'; p_min_at1: 'cp.Parameter'
    p_lim_krd10: 'cp.Parameter'; p_lim_twist: 'cp.Parameter'; p_lim_sdv_ig: 'cp.Parameter'; p_lim_sdv_hy: 'cp.Parameter'
    p_min_krd10: 'cp.Parameter'; p_max_krd10: 'cp.Parameter'; p_min_twist: 'cp.Parameter'; p_max_twist: 'cp.Parameter'
    p_min_sdv_ig: 'cp.Parameter'; p_min_sdv_hy: 'cp.Parameter'
    p_w_min: 'cp.Parameter'; p_w_max: 'cp.Parameter'
    prob_min_cvar: 'cp.Problem'; prob_max_ret: 'cp.Problem'
    sharpe_lambda: 'cp.Parameter'; prob_max_sharpe: 'cp.Problem'

@st.cache_resource(show_spinner=False)
def get_compiled_solver(mu, pnl, tags, factor_matrix) -> ParamCVaRSolver:
    if not CVXPY_AVAILABLE:
        raise RuntimeError("CVXPY not available")
    mu = np.asarray(mu, dtype=np.float64)
    pnl = np.ascontiguousarray(np.asarray(pnl, dtype=np.float32))    # S x N
    factor_matrix = np.asarray(factor_matrix, dtype=np.float32)
    N, S = len(mu), pnl.shape[0]

    w = cp.Variable(N, nonneg=True)
    z = cp.Variable(S, nonneg=True)
    alpha = cp.Variable()

    p_mu = cp.Parameter(N)
    target_ret = cp.Parameter()
    cvar_cap = cp.Parameter()
    p_cvar_alpha = cp.Parameter(nonneg=True)
    p_var_floor = cp.Parameter(nonneg=True)
    prev_w = cp.Parameter(N)
    max_turn = cp.Parameter()
    turn_pen = cp.Parameter(nonneg=True)
    sharpe_lambda = cp.Parameter(nonneg=True)

    p_w_min = cp.Parameter(N)
    p_w_max = cp.Parameter(N)

    p_max_non_ig = cp.Parameter(); p_max_em = cp.Parameter(); p_max_hybrid = cp.Parameter(); p_max_cash = cp.Parameter(); p_max_at1 = cp.Parameter()
    p_min_non_ig = cp.Parameter(nonneg=True); p_min_em = cp.Parameter(nonneg=True); p_min_hybrid = cp.Parameter(nonneg=True); p_min_cash = cp.Parameter(nonneg=True); p_min_at1 = cp.Parameter(nonneg=True)

    p_lim_krd10 = cp.Parameter(); p_lim_twist = cp.Parameter(); p_lim_sdv_ig = cp.Parameter(); p_lim_sdv_hy = cp.Parameter()
    p_min_krd10 = cp.Parameter(); p_max_krd10 = cp.Parameter(); p_min_twist = cp.Parameter(); p_max_twist = cp.Parameter()
    p_min_sdv_ig = cp.Parameter(nonneg=True); p_min_sdv_hy = cp.Parameter(nonneg=True)

    is_non_ig = np.asarray(tags["is_non_ig"], dtype=np.float32)
    is_em     = np.asarray(tags["is_em"],     dtype=np.float32)
    is_hybrid = np.asarray(tags.get("is_hybrid", np.zeros(N, dtype=bool)), dtype=np.float32)
    is_tbill  = np.asarray(tags["is_tbill"],  dtype=np.float32)
    is_at1    = np.asarray(tags["is_at1"],    dtype=np.float32)
    is_ig     = np.asarray(tags["is_ig"],     dtype=np.float32)
    is_hy     = (1.0 - is_ig)

    losses = -(pnl @ w)
    cvar_expr = alpha + (1.0/((1.0 - p_cvar_alpha) * S)) * cp.sum(z)
    ridge = 1e-6

    # factor matrix columns: 2y,5y,10y,20y,30y,OASD
    krd10_term = factor_matrix[:, 2] @ w
    twist_term = (factor_matrix[:, 4] - factor_matrix[:, 0]) @ w
    s_ig_term  = (factor_matrix[:, 5] * is_ig) @ w
    s_hy_term  = (factor_matrix[:, 5] * is_hy) @ w

    constraints = [
        cp.sum(w) == 1,
        z >= losses - alpha,
        cvar_expr <= cvar_cap,
        alpha >= p_var_floor,

        w >= p_w_min,
        w <= p_w_max,

        is_non_ig @ w <= p_max_non_ig,
        is_em     @ w <= p_max_em,
        is_hybrid @ w <= p_max_hybrid,
        is_tbill  @ w <= p_max_cash,
        is_at1    @ w <= p_max_at1,

        is_non_ig @ w >= p_min_non_ig,
        is_em     @ w >= p_min_em,
        is_hybrid @ w >= p_min_hybrid,
        is_tbill  @ w >= p_min_cash,
        is_at1    @ w >= p_min_at1,

        krd10_term <= p_max_krd10,  krd10_term >= p_min_krd10,
        twist_term <= p_max_twist,  twist_term >= p_min_twist,
        s_ig_term  <= p_lim_sdv_ig, s_ig_term  >= p_min_sdv_ig,
        s_hy_term  <= p_lim_sdv_hy, s_hy_term  >= p_min_sdv_hy,

        cp.norm1(w - prev_w) <= max_turn,
        p_mu @ w >= target_ret,
    ]

    # Treat turnover penalty as ANNUAL; Min-CVaR uses monthly CVaR, so scale penalty by 1/12 here.
    obj_min_cvar = cp.Minimize(
        cvar_expr + (turn_pen/12.0)*cp.norm1(w-prev_w) + ridge*cp.sum_squares(w)
    )
    obj_max_ret  = cp.Maximize(p_mu @ w - turn_pen*cp.norm1(w-prev_w) - ridge*cp.sum_squares(w))
    obj_max_sharpe = cp.Maximize((p_mu @ w) - sharpe_lambda*(12.0*cvar_expr) - turn_pen*cp.norm1(w-prev_w) - ridge*cp.sum_squares(w))

    prob_min_cvar   = cp.Problem(obj_min_cvar,   constraints)
    prob_max_ret    = cp.Problem(obj_max_ret,    constraints)
    prob_max_sharpe = cp.Problem(obj_max_sharpe, constraints)

    p_mu.value = mu
    p_cvar_alpha.value = 0.99
    return ParamCVaRSolver(
        N=len(mu), S=S, w=w, z=z, alpha=alpha,
        p_mu=p_mu, target_ret=target_ret, cvar_cap=cvar_cap, p_cvar_alpha=p_cvar_alpha, p_var_floor=p_var_floor,
        prev_w=prev_w, max_turn=max_turn, turn_pen=turn_pen,
        p_max_non_ig=p_max_non_ig, p_max_em=p_max_em, p_max_hybrid=p_max_hybrid, p_max_cash=p_max_cash, p_max_at1=p_max_at1,
        p_min_non_ig=p_min_non_ig, p_min_em=p_min_em, p_min_hybrid=p_min_hybrid, p_min_cash=p_min_cash, p_min_at1=p_min_at1,
        p_lim_krd10=p_lim_krd10, p_lim_twist=p_lim_twist, p_lim_sdv_ig=p_lim_sdv_ig, p_lim_sdv_hy=p_lim_sdv_hy,
        p_min_krd10=p_min_krd10, p_max_krd10=p_max_krd10, p_min_twist=p_min_twist, p_max_twist=p_max_twist,
        p_min_sdv_ig=p_min_sdv_ig, p_min_sdv_hy=p_min_sdv_hy,
        p_w_min=p_w_min, p_w_max=p_w_max,
        prob_min_cvar=prob_min_cvar, prob_max_ret=prob_max_ret,
        sharpe_lambda=sharpe_lambda, prob_max_sharpe=prob_max_sharpe
    )

# ----------------------------- Configuration & constants -----------------------------
st.set_page_config(
    page_title="Regime Optimised Allocation Model",
    page_icon="https://rubricsam.com/wp-content/uploads/2021/01/cropped-rubrics-logo-tight.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  * { font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif !important; }

  /* Sidebar full height + natural scroll */
  [data-testid="stSidebar"] {
      min-height: 100vh !important;
      height: auto !important;
      overflow-y: visible !important;
  }
  [data-testid="stSidebarContent"] {
      min-height: 100vh !important;
      height: auto !important;
  }
</style>
""", unsafe_allow_html=True)
inject_brand_css()

INPUT_SHEET = "Optimiser_Input"
METADATA_SHEET = "MetaData"

FUND_CONSTRAINTS = {
    "GFI": { "max_non_ig": 0.25, "max_em": 0.30, "max_hybrid": 0.15, "max_cash": 0.20, "max_at1": 0.15 },
    "GCF": { "max_non_ig": 0.10, "max_em": 0.35,                         "max_cash": 0.20, "max_at1": 0.10 },
    "EYF": { "max_non_ig": 1.00, "max_em": 1.00,                         "max_cash": 0.20, "max_at1": 0.00 },
}
VAR99_CAP = { "GFI": 0.050, "GCF": 0.055, "EYF": 0.100 }

DEFAULT_SEED = 42
DEFAULT_DRAWS = 2000
DEFAULT_RATES  = {"2y": 30.0, "5y": 25.0, "10y": 20.0, "30y": 15.0}
DEFAULT_SPREAD = {"IG": 100.0, "HY": 250.0, "AT1": 350.0, "EM": 250.0}
FACTOR_BUDGETS_DEFAULT = {"limit_krd10y": 0.75, "limit_twist": 0.40, "limit_sdv01_ig": 3.0, "limit_sdv01_hy": 1.5}
TURNOVER_DEFAULTS = { "penalty_bps_per_100": 15.0, "max_turnover": 0.25 }
EPS = 1e-6

RATES_BP99  = { "2y": 60.0, "5y": 50.0, "10y": 45.0, "20y": 45.0, "30y": 40.0 }
SPREAD_BP99 = { "IG": 100.0, "HY": 200.0, "AT1": 350.0, "EM": 250.0 }

def _nearest_psd(A, eps=1e-10):
    """Higham-like nearest PSD projection for symmetric A."""
    B = (A + A.T) / 2.0
    w, V = np.linalg.eigh(B)
    w_clipped = np.clip(w, eps, None)
    return (V * w_clipped) @ V.T

def _mv_sample(mean, cov, n, kind="normal", nu=7, rng=None):
    """
    Samples n draws from N(mean,cov) or multivariate t_ν with scale cov.
    mean: (d,), cov: (d,d). For t, uses normal/chi-square mix.
    """
    rng = np.random.default_rng() if rng is None else rng
    d = mean.shape[0]
    if kind == "t":
        # multivariate t via normal / sqrt(chi2/nu)
        g = rng.standard_normal(size=(n, d))
        # Cholesky with PSD guard
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(_nearest_psd(cov))
        z = g @ L.T
        chi = rng.chisquare(df=nu, size=(n, 1))
        scale = np.sqrt(nu / chi)  # heavy-tail scale per draw
        return mean + z * scale
    else:
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(_nearest_psd(cov))
        g = rng.standard_normal(size=(n, d))
        return mean + g @ L.T

def spacer(h=1):
    for _ in range(h): st.write("")

# ----------------------------- Defaults/session helpers -----------------------------
def _user_defaults():
    return st.session_state.setdefault("user_defaults", {"globals": {}, "funds": {}})
def _save_global_defaults(settings: dict):
    ud = _user_defaults(); ud["globals"] = settings; st.session_state["user_defaults"] = ud
def _save_fund_defaults(fund: str, settings: dict):
    ud = _user_defaults(); ud["funds"][fund] = settings; st.session_state["user_defaults"] = ud
def _get_global_default(key: str, fallback): return _user_defaults().get("globals", {}).get(key, fallback)
def _get_fund_default(fund: str, key: str, fallback): return _user_defaults().get("funds", {}).get(fund, {}).get(key, fallback)

# -----------------------------
# Optimiser_Input validation
# -----------------------------
from dataclasses import dataclass
import textwrap

# Required and optional columns after rename/synonym resolution
REQUIRED_MAIN_COLS = [
    "Name",              # unique sleeve name
    "Include",           # bool or 0/1
    "Weight_Min",        # fraction [0,1]
    "Weight_Max",        # fraction [0,1]
    "Yield_Hedged_Pct",  # annual pct, e.g., 3.2 means 3.2%
    "OASD_Years",        # spread DV01-equivalent exposure in years (>=0)
    "KRD_2y",
    "KRD_5y",
    "KRD_10y",
]

# Optional but recommended; used in several paths
OPTIONAL_MAIN_COLS = [
    "KRD_20y",
    "Roll_Down_bps_1Y",  # treated as PERCENT in this app (team decision)
    "Category",          # IG/HY/EM/AT1/etc. if present
]

PERCENT_FRACTION_COLS = [
    "Weight_Min", "Weight_Max"
]

NUMERIC_NONNEG_COLS = [
    "OASD_Years", "KRD_2y", "KRD_5y", "KRD_10y"
]

KRD_ALLOWED_RANGE = (-10.0, 10.0)  # sane guardrails per year of rate move

@dataclass
class InputCheckResult:
    df: "pd.DataFrame"
    warnings: list

def _fail(msg: str):
    import streamlit as st
    st.error(msg)
    st.stop()

def _validate_optimizer_input(df: "pd.DataFrame") -> InputCheckResult:
    import numpy as np
    import pandas as pd

    if df is None or df.empty:
        _fail("Optimiser_Input is empty after ingest/rename.")

    cols = list(df.columns)
    missing = [c for c in REQUIRED_MAIN_COLS if c not in cols]
    
    if "Weight_Min" not in cols:
        df["Weight_Min"] = 0.0
        missing = [c for c in missing if c != "Weight_Min"]
    if "Weight_Max" not in cols:
        df["Weight_Max"] = 1.0
        missing = [c for c in missing if c != "Weight_Max"]
    # Defaults are applied silently, no warnings shown
    
    if missing:
        head = ", ".join(cols[:12])
        _fail(
            "Optimiser_Input is missing required columns: "
            f"{missing}\n\nFound columns (first 12): {head}\n"
            "Check the sheet name and column headers (case-insensitive synonyms must map to these names)."
        )

    # Coerce dtypes
    df = df.copy()

    # Include -> boolean
    if not pd.api.types.is_bool_dtype(df["Include"]):
        # Accept 1/0, 'TRUE'/'FALSE', etc.
        df["Include"] = df["Include"].astype(str).str.strip().str.lower().map(
            {"1": True, "true": True, "yes": True, "y": True, "0": False, "false": False, "no": False, "n": False}
        ).fillna(False)

    # Ensure Name uniqueness and non-empty
    if df["Name"].isna().any() or (df["Name"].astype(str).str.strip() == "").any():
        _fail("Column 'Name' contains empty values. Every row must have a non-empty unique Name.")
    if df["Name"].duplicated().any():
        dups = df.loc[df["Name"].duplicated(), "Name"].tolist()
        _fail(f"Duplicate Names not allowed: {dups}")

    # Bounds and numeric coercions
    warnings = []

    # Percent-as-fraction bounds for min/max
    for c in PERCENT_FRACTION_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad = df[c].isna() | (df[c] < 0) | (df[c] > 1)
        if bad.any():
            rows = df.index[bad].tolist()
            _fail(f"'{c}' must be within [0,1]. Bad rows (0-based): {rows}")

    if (df["Weight_Min"] > df["Weight_Max"]).any():
        rows = df.index[(df["Weight_Min"] > df["Weight_Max"])].tolist()
        _fail(f"'Weight_Min' cannot exceed 'Weight_Max'. Violations at rows: {rows}")

    # Ensure types and feasible bounds
    df["Weight_Min"] = pd.to_numeric(df["Weight_Min"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["Weight_Max"] = pd.to_numeric(df["Weight_Max"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    bad = df["Weight_Min"] > df["Weight_Max"]
    if bad.any():
        rows = df.index[bad].tolist()
        _fail(f"`Weight_Min` cannot exceed `Weight_Max`. Rows: {rows}")

    # Respect 'Include' — excluded sleeves must have max=0 and min=0
    include_mask = pd.api.types.is_bool_dtype(df["Include"])
    if not include_mask:
        # Convert Include to boolean if not already
        df["Include"] = df["Include"].astype(str).str.strip().str.lower().map(
            {"1": True, "true": True, "yes": True, "y": True, "0": False, "false": False, "no": False, "n": False}
        ).fillna(False)
    include_mask = df["Include"].astype(bool)
    df.loc[~include_mask, ["Weight_Min","Weight_Max"]] = 0.0

    # Non-negativity checks
    for c in NUMERIC_NONNEG_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            rows = df.index[df[c].isna()].tolist()
            _fail(f"'{c}' contains non-numeric values. Bad rows: {rows}")
        if (df[c] < 0).any():
            rows = df.index[(df[c] < 0)].tolist()
            _fail(f"'{c}' must be non-negative. Bad rows: {rows}")

    # Sanity guardrails for KRDs (helps catch unit mistakes)
    for c in ["KRD_2y", "KRD_5y", "KRD_10y"]:
        lo, hi = KRD_ALLOWED_RANGE
        out = (df[c] < lo) | (df[c] > hi)
        if out.any():
            rows = df.index[out].tolist()
            _fail(f"'{c}' outside plausible range {lo}..{hi}. Check units. Bad rows: {rows}")

    # Optional columns: coerce if present
    if "KRD_20y" in df.columns:
        df["KRD_20y"] = pd.to_numeric(df["KRD_20y"], errors="coerce").fillna(0.0)

    if "Roll_Down_bps_1Y" in df.columns:
        # Team agreement: it's provided as PERCENT (e.g., +0.40 means +0.40%).
        df["Roll_Down_bps_1Y"] = pd.to_numeric(df["Roll_Down_bps_1Y"], errors="coerce").fillna(0.0)
        # Heuristic check: if values look like big numbers, they might be bps by mistake.
        if df["Roll_Down_bps_1Y"].abs().gt(50.0).any():
            warnings.append(
                "Roll_Down_bps_1Y has magnitudes > 50. Interpreted as PERCENT by design; "
                "if your sheet is in bps, convert before loading (e.g., 40 bps -> 0.40)."
            )

    # Yield hedged: numeric; allow negative but warn if absurd
    df["Yield_Hedged_Pct"] = pd.to_numeric(df["Yield_Hedged_Pct"], errors="coerce")
    if df["Yield_Hedged_Pct"].isna().any():
        rows = df.index[df["Yield_Hedged_Pct"].isna()].tolist()
        _fail(f"'Yield_Hedged_Pct' contains non-numeric values. Bad rows: {rows}")
    if df["Yield_Hedged_Pct"].abs().gt(40.0).any():
        warnings.append("Yield_Hedged_Pct has values with |value| > 40%. Check units and hedging columns.")

    # OASD_Years plausibility (can be 0..~10 normally)
    if df["OASD_Years"].gt(15.0).any():
        warnings.append("OASD_Years > 15 detected. Check whether this column encodes years (not bps).")

    return InputCheckResult(df=df, warnings=warnings)

# ----------------------------- Ingest & tagging -----------------------------
# Legacy required columns for metadata compatibility
LEGACY_REQUIRED_MAIN_COLS = [
    "Bloomberg_Ticker","Name","Instrument_Type",
    "Yield_Hedged_Pct","Roll_Down_bps_1Y","OAD_Years","OASD_Years",
    "KRD_2y","KRD_5y","KRD_10y","KRD_30y","Include"
]
REQUIRED_META_COLS = [
    "Bloomberg_Ticker","Is_Non_IG", "Is_EM", "Is_AT1", "Is_T2", "Is_Hybrid", "Is_Cash"
]
MAIN_SYNONYMS = {"bloomberg_ticker": ["ticker","bbg_ticker"], "instrument_type": ["type","instr_type"],
                 "yield_hedged_pct": ["yield_hedged","yield_hedged_percent","yield_hedged_%"],
                 "roll_down_bps_1y": ["roll_down_bps","rolldown_bps_1y"], "oasd_years": ["spread_dur","sdur","asd_years"]}
META_SYNONYMS = {"bloomberg_ticker": ["ticker","bbg_ticker"], "is_non_ig": ["non_ig","isnonig","is_high_yield"],
                 "is_em": ["is_emhc","is_em_hc","is_emerging"], "is_at1": ["is_bank_at1","is_additional_tier1"],
                 "is_t2": ["is_bank_t2","is_tier2"], "is_hybrid": ["is_global_hybrid","is_hybrids"], "is_cash": ["is_tbill","is_cash_like"]}

def _to_bool(s):
    s = pd.Series(s)
    sn = pd.to_numeric(s, errors="coerce")
    out = pd.Series(False, index=s.index)
    num_mask = ~sn.isna()
    out[num_mask] = sn[num_mask] != 0
    str_map = {"true":True,"t":True,"1":True,"yes":True,"y":True,"false":False,"f":False,"0":False,"no":False,"n":False}
    str_mask = ~num_mask
    out[str_mask] = s[str_mask].astype(str).str.strip().str.lower().map(str_map).fillna(False)
    return out.astype(bool)

def _rename_with_synonyms(df, required, synonyms):
    low = {c.lower().strip(): c for c in df.columns}; ren = {}
    for need in required:
        ln = need.lower()
        if ln in low and low[ln] != need: ren[low[ln]] = need; continue
        for alt in synonyms.get(ln, []):
            if alt in low: ren[low[alt]] = need; break
    if ren: df = df.rename(columns=ren)
    if [c for c in required if c not in df.columns]:
        if "Bloomberg_Ticker" not in df.columns:
            raise ValueError("Missing required columns for main sheet")
    return df

def merge_meta_robust(df_main: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    a = df_main.copy(); b = df_meta.copy()
    for col in ["Bloomberg_Ticker", "Name"]:
        if col in a.columns: a[col + "_norm"] = a[col].astype(str).str.strip().str.upper()
        if col in b.columns: b[col + "_norm"] = b[col].astype(str).str.strip().str.upper()
    if "Bloomberg_Ticker_norm" in a.columns and "Bloomberg_Ticker_norm" in b.columns:
        a = a.merge(b.drop_duplicates(subset=["Bloomberg_Ticker_norm"]), on="Bloomberg_Ticker_norm", how="left", suffixes=("", "_meta"))
    else:
        a = a.merge(b, on="Bloomberg_Ticker", how="left", suffixes=("", "_meta"))

    cls = ["Is_AT1","Is_EM","Is_Non_IG","Is_Hybrid","Is_T2","Is_Cash"]
    have_all_cls = all(c in a.columns for c in cls)
    missing = a[cls].isna().all(axis=1) if have_all_cls else pd.Series(True, index=a.index)
    if missing.any() and "Name_norm" in a.columns and "Name_norm" in b.columns:
        m2 = (a.loc[missing, ["Name_norm"]]
              .merge(b.drop_duplicates(subset=["Name_norm"]), on="Name_norm", how="left", suffixes=("", "_byname")))
        for c in cls + ["Credit_Quality"]:
            src = c + "_byname"
            if c in a.columns and src in m2.columns: a.loc[missing, c] = a.loc[missing, c].combine_first(m2[src])
    drop_cols = [c for c in a.columns if c.endswith("_norm")]
    return a.drop(columns=drop_cols, errors="ignore")

@st.cache_data(show_spinner=False)
def load_joined_input(uploaded_file_bytes: bytes, path: str = None) -> pd.DataFrame:
    if uploaded_file_bytes is None: raise ValueError("No file provided")
    bio = io.BytesIO(uploaded_file_bytes); xls = pd.ExcelFile(bio, engine="openpyxl")
    df_main = pd.read_excel(xls, sheet_name="Optimiser_Input")
    df_meta = pd.read_excel(xls, sheet_name="MetaData")
    df_main = _rename_with_synonyms(df_main, LEGACY_REQUIRED_MAIN_COLS, MAIN_SYNONYMS)
    df_meta = _rename_with_synonyms(df_meta, REQUIRED_META_COLS, META_SYNONYMS)
    
    # Validate Optimiser_Input strictly (no UI warnings)
    try:
        _check = _validate_optimizer_input(df_main)
        df_main = _check.df
        # intentionally do not display _check.warnings
    except Exception as _e:
        # _validate_optimizer_input already calls st.error+st.stop on known issues,
        # but we also catch unexpected exceptions here.
        import traceback
        st.error("Unexpected error while validating Optimiser_Input.")
        st.code("".join(traceback.format_exception_only(type(_e), _e)))
        st.stop()
    
    for c in ["Yield_Hedged_Pct","Roll_Down_bps_1Y","OAD_Years","OASD_Years","KRD_2y","KRD_5y","KRD_10y","KRD_20y","KRD_30y"]:
        if c in df_main.columns: df_main[c] = pd.to_numeric(df_main[c], errors="coerce").fillna(0.0)
    df = merge_meta_robust(df_main, df_meta)
    # Base expected return (annual %): carry + 1Y roll (Roll_Down_bps_1Y is treated as %; team is aware of unit quirk)
    df["ExpRet_pct"] = df.get("Yield_Hedged_Pct", 0.0) + df.get("Roll_Down_bps_1Y", 0.0)
    if "Include" in df.columns: df = df[_to_bool(df["Include"])].copy()

    def _bool(col, fallback=None):
        if col in df.columns: return _to_bool(df[col])
        if fallback is not None: return pd.Series(fallback, index=df.index).fillna(False).astype(bool)
        return pd.Series(False, index=df.index)

    is_at1    = _bool("Is_AT1",    df["Instrument_Type"].str.upper().str.contains("AT1", na=False))
    is_t2     = _bool("Is_T2",     df["Instrument_Type"].str.upper().str.contains("T2",  na=False))
    is_em     = _bool("Is_EM",     df["Instrument_Type"].str.upper().eq("EM"))
    is_hybrid = _bool("Is_Hybrid", df["Name"].str.upper().str.contains("HYBRID", na=False))
    is_cash   = _bool("Is_Cash",   df["Instrument_Type"].str.upper().eq("CASH") | df["Name"].str.upper().str.contains("T-?BILL", regex=True, na=False))
    is_non_ig = _bool("Is_Non_IG", is_at1 | is_t2 | is_em)
    df["Is_AT1"]=is_at1; df["Is_T2"]=is_t2; df["Is_EM"]=is_em; df["Is_Hybrid"]=is_hybrid; df["Is_Cash"]=is_cash; df["Is_Non_IG"]=is_non_ig
    df["Is_IG"] = ~df["Is_Non_IG"]
    df.reset_index(drop=True, inplace=True)
    return df

def build_tags_from_meta(df: pd.DataFrame) -> dict:
    return {
        "is_tbill":  df["Is_Cash"].values.astype(bool),
        "is_at1":    df["Is_AT1"].values.astype(bool),
        "is_t2":     df["Is_T2"].values.astype(bool),
        "is_hybrid": df["Is_Hybrid"].values.astype(bool),
        "is_em":     df["Is_EM"].values.astype(bool),
        "is_non_ig": df["Is_Non_IG"].values.astype(bool),
        "is_ig":     df["Is_IG"].values.astype(bool),
        "is_hy_rating": (df["Is_Non_IG"] & ~df["Is_AT1"] & ~df["Is_EM"]).values.astype(bool),
    }

# ----------------------------- Scenarios & risk -----------------------------
def bp99_to_sigma(bp99: float) -> float: return (bp99 / 10000.0) / 2.33

@st.cache_data(show_spinner=False)
def simulate_mc_draws(n_draws: int, seed: int, rates_bp99: dict, spreads_bp99: dict, 
                     use_corr: bool = False, use_tails: bool = False, nu_df: int = 7,
                     rho_curve: float = 0.85, rho_credit: float = 0.60, rho_rates_credit: float = -0.30) -> dict:
    rng = np.random.default_rng(seed)
    sig_r = {k: bp99_to_sigma(v) for k,v in rates_bp99.items()}
    sig_s = {k: bp99_to_sigma(v) for k,v in spreads_bp99.items()}
    
    # Build arrays for rates and spreads
    tenor_labels = ["2y","5y","10y","20y","30y"]
    rate_sigmas = np.array([sig_r[k] for k in tenor_labels])
    # Handle 20y fallback
    rate_sigmas[3] = sig_r.get("20y", sig_r["10y"])
    
    spread_sigmas = np.array([sig_s["IG"], sig_s["HY"], sig_s["AT1"], sig_s["EM"]])
    
    n_r = len(tenor_labels)
    n_s = len(spread_sigmas)
    d = n_r + n_s

    # Means are zero for shocks (macro overlay added elsewhere)
    mean = np.zeros(d)

    if not use_corr:
        # Old behavior: independent normals (or t if requested), diagonal cov
        sig = np.concatenate([rate_sigmas, spread_sigmas])
        cov = np.diag(sig**2)
    else:
        # ---- Build correlated covariance ----
        # 1) Curve block Σ_rr via AR(1)-like decay by tenor distance
        # Define tenor indices as [0,1,2,3,4]; corr_ij = ρ_curve ** |i-j|
        idx = np.arange(n_r)
        C_rr = rho_curve ** np.abs(idx[:,None] - idx[None,:])
        Σ_rr = (rate_sigmas[:,None] * rate_sigmas[None,:]) * C_rr

        # 2) Credit block Σ_ss with single-factor clustering
        # Off-diagonals = ρ_credit, diagonals = 1
        C_ss = np.full((n_s, n_s), rho_credit)
        np.fill_diagonal(C_ss, 1.0)
        Σ_ss = (spread_sigmas[:,None] * spread_sigmas[None,:]) * C_ss

        # 3) Cross block Σ_rs linking rates (use 10y as pivot) to all credit sleeves
        # We map a single ρ_rc to all pairs for simplicity; weight more to 10y.
        w_curve = np.array([0.2, 0.6, 1.0, 0.6, 0.2])  # emphasize belly/10y
        rs = np.outer(rate_sigmas * w_curve, spread_sigmas) * rho_rates_credit
        Σ_rs = rs
        Σ_sr = Σ_rs.T

        # 4) Assemble full Σ
        top = np.concatenate([Σ_rr, Σ_rs], axis=1)
        bot = np.concatenate([Σ_sr, Σ_ss], axis=1)
        cov = np.concatenate([top, bot], axis=0)

        # Guard for PSD
        cov = _nearest_psd(cov)

    # ---- Sample shocks ----
    kind = "t" if use_tails else "normal"
    draws = _mv_sample(mean, cov, n_draws, kind=kind, nu=nu_df if use_tails else 7, rng=rng)
    # Split back into rate & spread shocks
    rate_draws = draws[:, :n_r]
    spread_draws = draws[:, n_r:]
    
    # Return in the same format as before
    d2, d5, d10, d20, d30 = rate_draws.T
    dig, dhy, dat1, dem = spread_draws.T
    
    return {"d2": d2, "d5": d5, "d10": d10, "d20": d20, "d30": d30, "dig": dig, "dhy": dhy, "dat1": dat1, "dem": dem}

@st.cache_data(show_spinner=False)
def build_asset_pnl_matrix(df: pd.DataFrame, tags: dict, mc: dict) -> np.ndarray:
    krd_cols = ["KRD_2y","KRD_5y","KRD_10y","KRD_20y","KRD_30y"]
    krd = df.reindex(columns=krd_cols, fill_value=0.0)[krd_cols].values
    sdur = df["OASD_Years"].values
    N = len(df); S = len(mc["d2"])
    is_at1 = tags["is_at1"]; is_em = tags["is_em"]; is_hy = tags["is_hy_rating"] | is_at1 | is_em

    spread = np.zeros((S, N))
    spread += mc["dig"].reshape(-1,1)
    spread[:, is_hy] = mc["dhy"].reshape(-1,1)
    spread[:, is_em]  = mc["dem"].reshape(-1,1)
    spread[:, is_at1] = mc["dat1"].reshape(-1,1)

    dy = np.vstack([mc["d2"],mc["d5"],mc["d10"],mc["d20"],mc["d30"]]).T
    rate_pnl   = -(dy @ krd.T)
    credit_pnl = -(spread * sdur.reshape(1,-1))
    return rate_pnl + credit_pnl

def var_cvar_from_pnl(port_pnl: np.ndarray, alpha: float = 0.99) -> tuple[float,float]:
    losses = -port_pnl
    if losses.size == 0: return 0.0, 0.0
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = tail.mean() if tail.size else var
    return float(var), float(cvar)

def extract_risk_free_rate(df: pd.DataFrame) -> float:
    name = df.get("Name", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
    tick = df.get("Bloomberg_Ticker", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
    mask = tick.eq("LD20TRUU") | name.str.contains(r"\bT-?BILL\b|US T-?BILL|LD20TRUU", regex=True, na=False)
    row = df.loc[mask]
    if not row.empty and "Yield_Hedged_Pct" in row:
        return float(pd.to_numeric(row["Yield_Hedged_Pct"].iloc[0], errors="coerce")) / 100.0
    return 0.0

def classical_sharpe_from_scenarios(port_pnl_monthly: np.ndarray, exp_return_annual_dec: float, rf_annual_dec: float) -> float:
    std_m = float(np.std(port_pnl_monthly, ddof=1))
    vol_ann = std_m * math.sqrt(12.0)
    excess = float(exp_return_annual_dec) - float(rf_annual_dec)
    return (excess / vol_ann) if vol_ann > 1e-12 else float("nan")

def target_return_from_percentile(mu_vec_dec: np.ndarray, pctile: int) -> float:
    p = max(1, min(100, int(pctile)))
    return float(np.percentile(np.asarray(mu_vec_dec, dtype=float), p))

# ---------- Macro drift (expected return) helper ----------
def compute_macro_add_percent(df: pd.DataFrame, tags: dict, preset_name: str,
                              rates_bp99: dict, spreads_bp99: dict,
                              rate_sign: int, spread_sign: int, k: float) -> np.ndarray:
    """
    Per-asset expected macro add in ANNUAL PERCENT:
      -(KRD · E[Δy]) - (OASD · E[Δspread]).
    We use a small fraction k of 1M 99%-tail magnitudes with scenario sign.
    In current environment, rates are assumed to FALL in all three presets; spreads tighten in benign, widen in stress.
    """
    rate_sign = float(np.sign(rate_sign) or -1.0)
    spread_sign = float(np.sign(spread_sign) or -1.0)
    k = float(max(0.01, min(0.50, k)))

    dy_dec = np.array([
        rate_sign*k*rates_bp99.get("2y",0)/10000,
        rate_sign*k*rates_bp99.get("5y",0)/10000,
        rate_sign*k*rates_bp99.get("10y",0)/10000,
        rate_sign*k*rates_bp99.get("20y",rates_bp99.get("10y",0))/10000,
        rate_sign*k*rates_bp99.get("30y",0)/10000
    ])
    K = df.reindex(columns=["KRD_2y","KRD_5y","KRD_10y","KRD_20y","KRD_30y"], fill_value=0.0).values
    rate_mu_m = -(K @ dy_dec)  # monthly decimal

    is_at1 = np.asarray(tags["is_at1"], bool)
    is_em  = np.asarray(tags["is_em"],  bool)
    is_ig  = np.asarray(tags["is_ig"],  bool)
    is_hy  = (~is_ig) | is_at1 | is_em

    dIG  = spread_sign*k*spreads_bp99.get("IG",0)/10000
    dHY  = spread_sign*k*spreads_bp99.get("HY",0)/10000
    dAT1 = spread_sign*k*spreads_bp99.get("AT1",0)/10000
    dEM  = spread_sign*k*spreads_bp99.get("EM",0)/10000

    # Government bonds should have zero spread sensitivity
    if "Is_Govie" in df.columns:
        is_government = df["Is_Govie"].astype(bool).values
    else:
        # Broader regex: add Bund, Gilt, JGB, OAT, BTP, UST, Sovereign, etc.
        is_government = df["Name"].str.contains("Treasury|Government|Govt|Bund|Gilt|JGB|OAT|BTP|Sov|Sovereign|UST", case=False, na=False).values

    dspread_m = np.where(is_at1,dAT1, np.where(is_em,dEM, np.where(is_hy,dHY,dIG)))
    dspread_m = np.where(is_government, 0.0, dspread_m)
    cred_mu_m = -(df["OASD_Years"].values * dspread_m)

    macro_annual_pct = (rate_mu_m + cred_mu_m)*12.0*100.0
    return np.clip(macro_annual_pct, -30.0, 30.0)

# ----------------------------- Optimiser wrappers -----------------------------
def _budget_headroom(df: pd.DataFrame, w: np.ndarray, budgets: dict, tags: dict) -> dict:
    krd10 = float(df["KRD_10y"].values @ w)
    twist = float((df["KRD_30y"].values - df["KRD_2y"].values) @ w)
    oasd  = df["OASD_Years"].values
    is_ig = tags["is_ig"]
    s_ig  = float(np.sum(oasd * w * is_ig))
    s_hy  = float(np.sum(oasd * w * (~is_ig)))
    lim_krd10 = budgets.get("limit_krd10y", 0.75)
    lim_twist = budgets.get("limit_twist", 0.40)
    lim_ig    = budgets.get("limit_sdv01_ig", 3.0)
    lim_hy    = budgets.get("limit_sdv01_hy", 1.5)
    return {
        "limit_krd10y": {"used": abs(krd10), "cap": lim_krd10, "headroom": lim_krd10 - abs(krd10), "binding": abs(krd10) >= lim_krd10 - EPS},
        "limit_twist":  {"used": abs(twist), "cap": lim_twist, "headroom": lim_twist - abs(twist), "binding": abs(twist) >= lim_twist - EPS},
        "limit_sdv01_ig": {"used": s_ig, "cap": lim_ig, "headroom": lim_ig - s_ig, "binding": s_ig >= lim_ig - EPS},
        "limit_sdv01_hy": {"used": s_hy, "cap": lim_hy, "headroom": lim_hy - s_hy, "binding": s_hy >= lim_hy - EPS},
    }

def _cap_headroom(weights: np.ndarray, tags: dict, caps: dict) -> dict:
    m = {"max_non_ig": ("Non-IG", tags["is_non_ig"]), "max_em": ("EM", tags["is_em"]),
         "max_hybrid": ("Hybrid", tags.get("is_hybrid", np.zeros_like(weights, dtype=bool))),
         "max_cash": ("Cash", tags["is_tbill"]), "max_at1": ("AT1", tags["is_at1"])}
    out = {}
    for k,(lbl,mask) in m.items():
        if k in caps:
            used = float(mask.astype(float) @ weights); cap  = float(caps[k])
            out[k] = {"label": lbl, "used": used, "cap": cap, "headroom": cap - used,
                      "binding": (cap > 0 and used >= cap - EPS) or (cap == 0 and used > EPS)}
    return out

def solve_max_excess_over_cvar(df, tags, mu, pnl_matrix, fund, params, prev_w=None, iters=12):
    import time as _t, numpy as _np, cvxpy as cp
    X = df.reindex(columns=["KRD_2y","KRD_5y","KRD_10y","KRD_20y","KRD_30y","OASD_Years"], fill_value=0.0).values
    S = get_compiled_solver(mu, pnl_matrix, tags, X)
    fc = params.get("fund_caps", FUND_CONSTRAINTS[fund]); fb = params.get("factor_budgets", FACTOR_BUDGETS_DEFAULT)
    # corridors
    S.p_max_non_ig.value=float(fc.get("max_non_ig",1.0)); S.p_max_em.value=float(fc.get("max_em",1.0))
    S.p_max_hybrid.value=float(fc.get("max_hybrid",0.0)); S.p_max_cash.value=float(fc.get("max_cash",1.0))
    S.p_max_at1.value=float(fc.get("max_at1",1.0))
    S.p_min_non_ig.value=float(params.get("min_non_ig",0.0)); S.p_min_em.value=float(params.get("min_em",0.0))
    S.p_min_hybrid.value=float(params.get("min_hybrid",0.0)); S.p_min_cash.value=float(params.get("min_cash",0.0))
    S.p_min_at1.value=float(params.get("min_at1",0.0))
    # budgets
    S.p_lim_krd10.value=float(fb.get("limit_krd10y",0.75)); S.p_lim_twist.value=float(fb.get("limit_twist",0.40))
    S.p_lim_sdv_ig.value=float(fb.get("limit_sdv01_ig",3.0)); S.p_lim_sdv_hy.value=float(fb.get("limit_sdv01_hy",1.5))
    S.p_min_krd10.value=float(params.get("min_krd10y",0.0));  S.p_max_krd10.value=float(params.get("max_krd10y",fb.get("limit_krd10y",1.0)))
    S.p_min_twist.value=float(params.get("min_twist",0.0));  S.p_max_twist.value=float(params.get("max_twist",fb.get("limit_twist",0.5)))
    S.p_min_sdv_ig.value=float(params.get("min_sdv01_ig",0.0)); S.p_min_sdv_hy.value=float(params.get("min_sdv01_hy",0.0))
    # risk & turnover
    S.cvar_cap.value=float(params.get("cvar_cap",VAR99_CAP[fund])); S.p_cvar_alpha.value=float(params.get("cvar_alpha",0.99))
    S.p_var_floor.value=float(params.get("min_var",0.0))
    n=len(df); apply_turn= prev_w is not None and _np.sum(prev_w)>1e-8
    S.prev_w.value=(prev_w if apply_turn else _np.zeros(n)); S.max_turn.value=float(params.get("max_turnover",TURNOVER_DEFAULTS["max_turnover"]) if apply_turn else 1.0)
    S.turn_pen.value=float(params.get("turnover_penalty",TURNOVER_DEFAULTS["penalty_bps_per_100"])/10000.0 if apply_turn else 0.0)
    S.p_mu.value=_np.asarray(mu,float)
    
    # Wire per-sleeve bounds to solver
    w_min_vec = df.reindex(columns=["Weight_Min"], fill_value=0.0).values.reshape(-1)
    w_max_vec = df.reindex(columns=["Weight_Max"], fill_value=1.0).values.reshape(-1)

    # Safety: ensure numerical feasibility (tiny slack prevents accidental infeasibility from float issues)
    eps = 1e-9
    w_min_vec = _np.clip(w_min_vec, 0.0, 1.0 - eps)
    w_max_vec = _np.clip(w_max_vec, eps, 1.0)
    w_max_vec = _np.maximum(w_max_vec, w_min_vec + eps)

    S.p_w_min.value = w_min_vec.astype(float)
    S.p_w_max.value = w_max_vec.astype(float)
    
    # rf proxy
    rf=0.0
    if "is_tbill" in tags and tags["is_tbill"].any():
        rf=float(_np.mean(df.loc[tags["is_tbill"].astype(bool),"Yield_Hedged_Pct"]))/100.0
    # golden-section on return floor to maximise (ER-rf)/CVaR
    S.target_ret.value=float(_np.min(mu)-1.0); S.prob_max_ret.solve(solver=cp.ECOS, verbose=False, max_iters=60000, warm_start=True)
    w_ret=_np.array(S.w.value).ravel() if S.w.value is not None else _np.zeros(n); er_hi=float(mu@w_ret)
    lo=float(_np.percentile(mu,20)); hi=max(er_hi,lo+1e-6); phi=(1+5**0.5)/2; a,b=lo,hi
    def eval_ratio(tau):
        S.target_ret.value=float(tau); S.prob_min_cvar.solve(solver=cp.ECOS, verbose=False, max_iters=60000, warm_start=True)
        if S.w.value is None: return (-_np.inf,None,None,None)
        w=_np.array(S.w.value).ravel(); pnl=pnl_matrix@w
        var=float(_np.quantile(-(pnl),0.99)); tail=-(pnl)[-(pnl)>=var]; cvar=float(tail.mean()) if tail.size else var
        er=float(mu@w); return ((er-rf)/max(cvar,1e-8), w, er, cvar)
    c=b-(b-a)/phi; d=a+(b-a)/phi; rc,wc,ec,cc=eval_ratio(c); rd,wd,ed,cd=eval_ratio(d)
    for _ in range(int(iters)):
        if rc>=rd: b,rd,wd,ed,cd=d,rc,wc,ec,cc; d=c; c=b-(b-a)/phi; rc,wc,ec,cc=eval_ratio(c)
        else:      a,rc,wc,ec,cc=c,rd,wd,ed,cd; c=d; d=a+(b-a)/phi; rd,wd,ed,cd=eval_ratio(d)
    if rc>=rd: ratio,w_star,er_star,cvar_star=rc,wc,ec,cc
    else:      ratio,w_star,er_star,cvar_star=rd,wd,ed,cd
    metrics={"status":"OPTIMAL","obj":ratio,"ExpRet_pct":er_star*100.0,
             "Yield_pct": float(df["Yield_Hedged_Pct"].values@w_star),
             "OAD_years": float(df["OAD_Years"].values@w_star),
             "VaR99_1M": float(_np.quantile(-(pnl_matrix@w_star),0.99)),
             "CVaR99_1M": cvar_star, "weights": w_star}
    return w_star, metrics

def solve_portfolio(df, tags, mu, pnl_matrix, fund, params, prev_w=None):
    import numpy as _np, cvxpy as cp, time as _t
    X=df.reindex(columns=["KRD_2y","KRD_5y","KRD_10y","KRD_20y","KRD_30y","OASD_Years"], fill_value=0.0).values
    S=get_compiled_solver(mu, pnl_matrix, tags, X)
    fc=params.get("fund_caps",FUND_CONSTRAINTS[fund]); fb=params.get("factor_budgets",FACTOR_BUDGETS_DEFAULT)
    # corridors max/min
    S.p_max_non_ig.value=float(fc.get("max_non_ig",1.0)); S.p_max_em.value=float(fc.get("max_em",1.0))
    S.p_max_hybrid.value=float(fc.get("max_hybrid",0.0)); S.p_max_cash.value=float(fc.get("max_cash",1.0))
    S.p_max_at1.value=float(fc.get("max_at1",1.0))
    S.p_min_non_ig.value=float(params.get("min_non_ig",0.0)); S.p_min_em.value=float(params.get("min_em",0.0))
    S.p_min_hybrid.value=float(params.get("min_hybrid",0.0)); S.p_min_cash.value=float(params.get("min_cash",0.0))
    S.p_min_at1.value=float(params.get("min_at1",0.0))
    # factor budgets & mins/maxs
    S.p_lim_krd10.value=float(fb.get("limit_krd10y",0.75)); S.p_lim_twist.value=float(fb.get("limit_twist",0.40))
    S.p_lim_sdv_ig.value=float(fb.get("limit_sdv01_ig",3.0)); S.p_lim_sdv_hy.value=float(fb.get("limit_sdv01_hy",1.5))
    S.p_min_krd10.value=float(params.get("min_krd10y",0.0)); S.p_max_krd10.value=float(params.get("max_krd10y",fb.get("limit_krd10y",1.0)))
    S.p_min_twist.value=float(params.get("min_twist",0.0)); S.p_max_twist.value=float(params.get("max_twist",fb.get("limit_twist",0.5)))
    S.p_min_sdv_ig.value=float(params.get("min_sdv01_ig",0.0)); S.p_min_sdv_hy.value=float(params.get("min_sdv01_hy",0.0))
    # risk & turnover
    S.cvar_cap.value=float(params.get("cvar_cap",VAR99_CAP[fund])); S.p_cvar_alpha.value=float(params.get("cvar_alpha",0.99))
    S.p_var_floor.value=float(params.get("min_var",0.0))
    n=len(df); apply_turn= prev_w is not None and _np.sum(prev_w)>1e-8
    S.prev_w.value=(prev_w if apply_turn else _np.zeros(n))
    S.max_turn.value=float(params.get("max_turnover",TURNOVER_DEFAULTS["max_turnover"]) if apply_turn else 1.0)
    S.turn_pen.value=float(params.get("turnover_penalty",TURNOVER_DEFAULTS["penalty_bps_per_100"])/10000.0 if apply_turn else 0.0)
    # expected returns
    S.p_mu.value=_np.asarray(mu,float)

    # Wire per-sleeve bounds to solver
    w_min_vec = df.reindex(columns=["Weight_Min"], fill_value=0.0).values.reshape(-1)
    w_max_vec = df.reindex(columns=["Weight_Max"], fill_value=1.0).values.reshape(-1)

    # Safety: ensure numerical feasibility (tiny slack prevents accidental infeasibility from float issues)
    eps = 1e-9
    w_min_vec = _np.clip(w_min_vec, 0.0, 1.0 - eps)
    w_max_vec = _np.clip(w_max_vec, eps, 1.0)
    w_max_vec = _np.maximum(w_max_vec, w_min_vec + eps)

    S.p_w_min.value = w_min_vec.astype(float)
    S.p_w_max.value = w_max_vec.astype(float)

    objective_name=params.get("objective","Max Return")
    auto_relaxed_flag=False; relaxed_pct=None

    if objective_name=="Max Excess Return / CVaR":
        return solve_max_excess_over_cvar(df,tags,mu,pnl_matrix,fund,params,prev_w)

    elif objective_name=="Risk-adjusted Return (λ·CVaR)":
        S.target_ret.value=float(_np.min(mu)-1.0); S.sharpe_lambda.value=float(params.get("sharpe_lambda",1.0)); prob=S.prob_max_sharpe

    elif objective_name=="Min VaR for Target Return Percentile":
        S.target_ret.value=float(params.get("target_return", float(_np.percentile(mu,60))))
        prob=S.prob_min_cvar

        # Try solve; if infeasible, relax target percentile downwards
        solve_ok=False
        try:
            prob.solve(solver=cp.OSQP, verbose=False, max_iter=30000, eps_abs=1e-5, eps_rel=1e-5, warm_start=True)
            solve_ok = (S.w.value is not None)
        except Exception:
            pass
        if not solve_ok:
            try:
                prob.solve(solver=cp.ECOS, verbose=False, max_iters=100000)
                solve_ok = (S.w.value is not None)
            except Exception:
                solve_ok=False
        if not solve_ok:
            for pct in range(95, 0, -1):
                S.target_ret.value=float(_np.percentile(mu, pct))
                try:
                    prob.solve(solver=cp.ECOS, verbose=False, max_iters=40000, warm_start=True)
                except Exception:
                    try:
                        prob.solve(solver=cp.OSQP, verbose=False, max_iter=30000, eps_abs=1e-5, eps_rel=1e-5, warm_start=True)
                    except Exception:
                        pass
                if S.w.value is not None:
                    solve_ok=True; auto_relaxed_flag=True; relaxed_pct=pct; break
        if not solve_ok:
            return None, {"status":"INFEASIBLE","message":"No feasible solution after relaxing return floor."}

    else:
        S.target_ret.value=float(_np.min(mu)-1.0); prob=S.prob_max_ret

    solved=False; errs=[]
    for solver,kwargs in [(cp.OSQP,{"verbose":False,"max_iter":30000,"eps_abs":1e-5,"eps_rel":1e-5,"warm_start":True}),
                          (cp.ECOS,{"verbose":False,"max_iters":100000})]:
        try:
            prob.solve(solver=solver, **kwargs)
            solved = (S.w.value is not None)
        except Exception as e:
            errs.append(f"{getattr(solver,'__name__',solver)}: {e}")
            solved=False
        if solved: break
    if not solved:
        return None, {"status":"INFEASIBLE","message":" | ".join(errs) if errs else "Solver failed (OSQP/ECOS)."}

    w=np.array(S.w.value).ravel(); pnl=pnl_matrix@w
    var=float(np.quantile(-(pnl),0.99)); tail=-(pnl)[-(pnl)>=var]; cvar=float(tail.mean()) if tail.size else var
    er=float(mu@w)
    yld=float(df["Yield_Hedged_Pct"].values@w); oad=float(df["OAD_Years"].values@w)
    metrics={"status":"OPTIMAL","obj":float(prob.value) if prob.value is not None else None,
             "ExpRet_pct": er*100.0, "Yield_pct": yld, "OAD_years": oad,
             "VaR99_1M": var, "CVaR99_1M": cvar, "weights": w,
             "_diag": {}}
    if objective_name=="Min VaR for Target Return Percentile" and auto_relaxed_flag:
        metrics["_diag"]["auto_relaxed"]=True; metrics["_diag"]["relaxed_to_percentile"]=int(relaxed_pct)
    return w, metrics

# ----------------------------- Visuals -----------------------------
def kpi_number(value: float, kind: str = "pct"):
    val = value * 100 if kind == "pct" else value
    fig = go.Figure(go.Indicator(mode="number", value=val, number={"suffix": "%", "valueformat": ".2f"}))
    fig.update_layout(template="rubrics", margin=dict(l=5, r=5, t=6, b=6), height=110, showlegend=False, title={"text": ""})
    return fig

def bar_allocation(df, weights, title, min_weight_threshold=0.001):
    ser = pd.Series(weights, index=df["Name"]).sort_values(ascending=False)
    ser = ser[ser >= min_weight_threshold]
    if len(ser) == 0:
        fig = go.Figure(); fig.add_annotation(text="No allocations above the minimum display threshold", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    else:
        fig = go.Figure(go.Bar(x=ser.index, y=ser.values))
    fig.update_layout(xaxis_title="Segment", yaxis_title="Weight", height=380, margin=dict(l=10,r=10,t=40,b=80))
    return fig

def exposures_vs_budgets(df, weights, budgets: dict, title: str):
    is_ig_mask = build_tags_from_meta(df)["is_ig"]; oasd = df["OASD_Years"].values
    used_vals = [abs(float(df["KRD_10y"].values @ weights)),
                 abs(float((df["KRD_30y"].values - df["KRD_2y"].values) @ weights)),
                 float(np.sum(oasd * weights * is_ig_mask)),
                 float(np.sum(oasd * weights * (~is_ig_mask)))]
    cap_vals = [budgets.get("limit_krd10y",0.75), budgets.get("limit_twist",0.40), budgets.get("limit_sdv01_ig",3.0), budgets.get("limit_sdv01_hy",1.5)]
    labels = ["KRD 10y", "Twist (30y–2y)", "sDV01 IG", "sDV01 HY"]
    fig = go.Figure()
    fig.add_bar(name="Cap", y=labels, x=cap_vals, orientation="h", marker_color=RB_COLORS["grey"], showlegend=False, opacity=0.7)
    fig.add_bar(name="Used", y=labels, x=used_vals, orientation="h", marker_color=RB_COLORS["blue"], showlegend=False)
    fig.update_layout(barmode="overlay", height=280, margin=dict(l=10, r=10, t=20, b=40), xaxis_title="Years", yaxis_title="", showlegend=False,
                      xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=True, zerolinecolor="rgba(128,128,128,0.4)"),
                      yaxis=dict(showgrid=False, zeroline=False))
    return fig

def scenario_histogram(port_pnl, title="Scenario P&L (1M)"):
    fig = go.Figure(data=[go.Histogram(x=port_pnl * 100, nbinsx=40)])
    var99, cvar99 = var_cvar_from_pnl(port_pnl, 0.99)
    fig.add_vline(x=-var99 * 100, line_dash="dash", annotation_text="VaR99", annotation_position="top left")
    fig.add_vline(x=-cvar99 * 100, line_dash="dot", annotation_text="CVaR99", annotation_position="top left")
    fig.update_layout(xaxis_title="% P&L", yaxis_title="Count", height=300, margin=dict(l=10,r=10,t=40,b=20))
    return fig

def cap_usage_gauge(label: str, used_w: float, cap_w: float) -> go.Figure:
    used_pct = max(0.0, used_w * 100.0); cap_pct = max(0.0, cap_w * 100.0)
    axis_max = max(cap_pct if cap_pct > 0 else 1.0, used_pct * 1.10, 1.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=used_pct,
            number={"suffix": "%", "valueformat": ".2f"},
            delta={"reference": cap_pct, "increasing": {"color": RB_COLORS["orange"]}, "decreasing": {"color": RB_COLORS["blue"]}},
            gauge={
                "axis": {"range": [0, axis_max]},
                "bar": {"color": RB_COLORS["medblue"]},
                "threshold": {"line": {"color": RB_COLORS["orange"], "width": 3}, "value": cap_pct},
                "steps": [{"range": [0, cap_pct], "color": RB_COLORS["grey"]}]
            }
        )
    )
    fig.update_layout(template="rubrics", height=150, margin=dict(l=4, r=4, t=18, b=4))
    return fig

def render_cap_usage_section(fund: str, w: np.ndarray, tags: dict, fc_current: dict):
    def _w(mask: np.ndarray) -> float: return float(mask.astype(float) @ w)
    rows = []
    if "max_non_ig" in fc_current: rows.append(("Non‑IG", _w(tags["is_non_ig"]), float(fc_current["max_non_ig"])))
    if "max_em" in fc_current:     rows.append(("EM",     _w(tags["is_em"]),     float(fc_current["max_em"])))
    if "max_hybrid" in fc_current: rows.append(("Hybrid (Global Hybrid)", _w(tags["is_hybrid"]), float(fc_current["max_hybrid"])))
    if "max_at1" in fc_current:    rows.append(("AT1 (Bank Capital)", _w(tags["is_at1"]), float(fc_current["max_at1"])))
    if "max_cash" in fc_current:   rows.append(("Cash (T‑Bills)", _w(tags["is_tbill"]), float(fc_current["max_cash"])))
    if rows:
        cols = st.columns(len(rows))
        for col, (lbl, used, cap) in zip(cols, rows):
            with col:
                title_with_help(lbl, "Usage of risk corridor. Gauge shows proposed portfolio weight (needle) vs corridor cap (orange marker).")
                safe_lbl = lbl.split("(")[0].strip().replace(" ", "_").replace("–","-").replace("—","-")
                st.plotly_chart(
                    cap_usage_gauge(lbl, used, cap),
                    use_container_width=True,
                    key=f"{fund}_cap_usage_{safe_lbl}"
                )
        data = [{"Cap": lbl, "Used %": used*100.0, "Cap %": cap*100.0, "Usage of cap": (used/cap if cap>0 else np.nan), "Status": ("over cap" if cap>0 and used>cap else ("n/a" if cap==0 else "within cap"))} for lbl, used, cap in rows]
        df_caps = pd.DataFrame(data)
        sty = (df_caps.style
               .format({"Used %": "{:.2f}%", "Cap %": "{:.2f}%", "Usage of cap": "{:.0%}"})
               .apply(lambda s: ["background-color: #ffe6e0" if (v == "over cap") else "" for v in s], subset=["Status"]))
        st.dataframe(sty, use_container_width=True, height=200)

# ----------------------------- UI -----------------------------
st.markdown("""
<style>
  .rb-header { display:flex; align-items:flex-start; justify-content:space-between; }
  .rb-title h1 { font-size:3.5rem; color:var(--rb-blue); font-weight:700; margin:0 0 .5rem 0; }
  .rb-logo img { height:48px; margin-top:6px; }
  @media (max-width:1200px){ .rb-title h1{ font-size:2.6rem; } .rb-logo img{ height:42px; } }
</style>
<div class="rb-header">
  <div class="rb-title"><h1>Regime Optimised Allocation Model</h1></div>
  <div class="rb-logo"><img src="https://rubricsam.com/wp-content/uploads/2021/01/cropped-rubrics-logo-tight.png" alt="Rubrics Logo"/></div>
</div>
""", unsafe_allow_html=True)
spacer(1)

def _init_state_default(key: str, default: str):
    """Set a session_state default once; do not overwrite on reruns."""
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar
with st.sidebar:
    st.header("Global Settings")
    st.caption(f"Build: {APP_BUILD}")
    upload = st.file_uploader("Upload Excel file (Optimiser_Input + MetaData)", type=["xlsx"], key="xlsx_upload")
    if upload is None:
        st.info("Upload your workbook to continue."); st.stop()
    if st.button("Reset all controls to defaults", key="reset_all"):
        for k in ["seed","n_draws","rate_2y","rate_5y","rate_10y","rate_30y","spr_IG","spr_HY","spr_AT1","spr_EM",
                  "penalty_bps_ctrl","max_turn_ctrl","sharpe_lambda_ctrl","var99_band","roll_incl_pct","target_ret_pctile",
                  "cap_non_ig","cap_em","cap_hybrid","cap_cash","cap_at1","fund_krd10y","allow_neg_twist","fund_twist",
                  "fund_sdv_ig","fund_sdv_hy","prev_weights","overview_obj","risk_regime","_prev_regime",
                  "rate_2y_input","rate_5y_input","rate_10y_input","rate_30y_input","spr_IG_input","spr_HY_input","spr_AT1_input","spr_EM_input"]:
            st.session_state.pop(k, None)
        st.session_state.pop("fund_overrides", None); st.session_state.pop("_perf_events", None); st.session_state.pop("last_run", None)
        st.rerun()
    if st.button("Force clear cache"): st.cache_data.clear(); st.rerun()

    seed = st.number_input("Random seed", min_value=0, value=_get_global_default("seed", DEFAULT_SEED), step=1, key="seed")
    n_draws = st.number_input("Monte Carlo draws (monthly)", min_value=200, max_value=10000, value=_get_global_default("n_draws", DEFAULT_DRAWS), step=100, key="n_draws")

    # ---- Macro scenario preset (auto-populates rate/spread magnitudes @99%) ----
    st.subheader("Macro scenario preset")
    _preset_map = {
        # k is the fraction of 1M 99% shock used for the deterministic drift
        "Benign":   {"rates": {"2y": 20.0, "5y": 15.0, "10y": 10.0, "30y": 10.0}, "spreads": {"IG": 40.0, "HY": 100.0, "AT1": 150.0, "EM": 80.0},
                     "defaults": {"rate_sign": -1, "spr_sign": -1, "k": 0.03}},
        "Moderate": {"rates": {"2y": 50.0, "5y": 40.0, "10y": 30.0, "30y": 20.0}, "spreads": {"IG": 100.0, "HY": 300.0, "AT1": 300.0, "EM": 200.0},
                     "defaults": {"rate_sign": -1, "spr_sign": +1, "k": 0.08}},
        "Crisis":   {"rates": {"2y": 150.0, "5y": 120.0, "10y": 100.0, "30y": 80.0}, "spreads": {"IG": 300.0, "HY": 700.0, "AT1": 800.0, "EM": 400.0},
                     "defaults": {"rate_sign": -1, "spr_sign": +1, "k": 0.15}},
    }
    selected_preset = st.selectbox(
        "Select market regime for 1M 99% shocks",
        list(_preset_map.keys()),
        index=0,
        key="risk_regime"
    )
    # Apply defaults only when selection changes (so manual edits aren't clobbered)
    if st.session_state.get("_prev_regime", None) != selected_preset:
        _p = _preset_map[selected_preset]
        # keep numeric state separate from text inputs
        st.session_state["rate_2y"]  = float(_p["rates"]["2y"])
        st.session_state["rate_5y"]  = float(_p["rates"]["5y"])
        st.session_state["rate_10y"] = float(_p["rates"]["10y"])
        st.session_state["rate_30y"] = float(_p["rates"]["30y"])
        st.session_state["spr_IG"]   = float(_p["spreads"]["IG"])
        st.session_state["spr_HY"]   = float(_p["spreads"]["HY"])
        st.session_state["spr_AT1"]  = float(_p["spreads"]["AT1"])
        st.session_state["spr_EM"]   = float(_p["spreads"]["EM"])
        st.session_state["_prev_regime"] = selected_preset
        # Initialize macro sign & magnitude controls from preset defaults on change
        _d = _preset_map[selected_preset]["defaults"]
        st.session_state["macro_rate_sign"] = int(_d["rate_sign"])
        st.session_state["macro_spr_sign"]  = int(_d["spr_sign"])
        st.session_state["macro_k"]         = float(_d["k"])
        # also initialise the visible text boxes with signed strings
        st.session_state["rate_2y_input"]  = f"-{abs(st.session_state['rate_2y']):.1f}"
        st.session_state["rate_5y_input"]  = f"-{abs(st.session_state['rate_5y']):.1f}"
        st.session_state["rate_10y_input"] = f"-{abs(st.session_state['rate_10y']):.1f}"
        st.session_state["rate_30y_input"] = f"-{abs(st.session_state['rate_30y']):.1f}"
        sign = "-" if selected_preset == "Benign" else "+"
        st.session_state["spr_IG_input"]   = f"{sign}{abs(st.session_state['spr_IG']):.1f}"
        st.session_state["spr_HY_input"]   = f"{sign}{abs(st.session_state['spr_HY']):.1f}"
        st.session_state["spr_AT1_input"]  = f"{sign}{abs(st.session_state['spr_AT1']):.1f}"
        st.session_state["spr_EM_input"]   = f"{sign}{abs(st.session_state['spr_EM']):.1f}"
    st.caption("Forecast rate and spread changes by scenario. Rates fall in all scenarios; spreads tighten in benign, widen in stress.")

    st.write("**Forecast Rate Δ (bp):**")
    st.caption("Displayed with the correct sign (negative = rates fall).")
    c1,c2,c3,c4 = st.columns(4)

    def _clean_signed_number(s, default_abs_float):
        try:
            return abs(float(str(s).replace("+","").replace("−","-").replace("--","-").replace(" ", "").replace("%","")))
        except Exception:
            return float(default_abs_float)

    # Initialise defaults ONCE (no value= on widgets; let Streamlit use session_state)
    _init_state_default("rate_2y_input",  f"-{abs(float(st.session_state.get('rate_2y',  DEFAULT_RATES['2y']))):.1f}")
    _init_state_default("rate_5y_input",  f"-{abs(float(st.session_state.get('rate_5y',  DEFAULT_RATES['5y']))):.1f}")
    _init_state_default("rate_10y_input", f"-{abs(float(st.session_state.get('rate_10y', DEFAULT_RATES['10y']))):.1f}")
    _init_state_default("rate_30y_input", f"-{abs(float(st.session_state.get('rate_30y', DEFAULT_RATES['30y']))):.1f}")

    with c1:
        rate_2y_input  = st.text_input("2y", key="rate_2y_input")
        rate_2y_clean  = _clean_signed_number(rate_2y_input, st.session_state.get("rate_2y", DEFAULT_RATES["2y"]))
    with c2:
        rate_5y_input  = st.text_input("5y", key="rate_5y_input")
        rate_5y_clean  = _clean_signed_number(rate_5y_input, st.session_state.get("rate_5y", DEFAULT_RATES["5y"]))
    with c3:
        rate_10y_input = st.text_input("10y", key="rate_10y_input")
        rate_10y_clean = _clean_signed_number(rate_10y_input, st.session_state.get("rate_10y", DEFAULT_RATES["10y"]))
    with c4:
        rate_30y_input = st.text_input("30y", key="rate_30y_input")
        rate_30y_clean = _clean_signed_number(rate_30y_input, st.session_state.get("rate_30y", DEFAULT_RATES["30y"]))

    RATES_BP99["2y"]=float(rate_2y_clean); RATES_BP99["5y"]=float(rate_5y_clean)
    RATES_BP99["10y"]=float(rate_10y_clean); RATES_BP99["30y"]=float(rate_30y_clean)
    RATES_BP99["20y"]=float(rate_10y_clean)

    st.write("**Forecast Spread Δ (bp):**")
    st.caption("TIGHTEN in benign (negative), WIDEN in moderate/severe (positive).")
    c1,c2,c3,c4 = st.columns(4)
    default_sign = "-" if selected_preset == "Benign" else "+"

    # Initialise once; widgets will read from session_state
    _init_state_default("spr_IG_input",  f"{default_sign}{abs(float(st.session_state.get('spr_IG',  DEFAULT_SPREAD['IG']))):.1f}")
    _init_state_default("spr_HY_input",  f"{default_sign}{abs(float(st.session_state.get('spr_HY',  DEFAULT_SPREAD['HY']))):.1f}")
    _init_state_default("spr_AT1_input", f"{default_sign}{abs(float(st.session_state.get('spr_AT1', DEFAULT_SPREAD['AT1']))):.1f}")
    _init_state_default("spr_EM_input",  f"{default_sign}{abs(float(st.session_state.get('spr_EM',  DEFAULT_SPREAD['EM']))):.1f}")

    with c1:
        spr_IG_input  = st.text_input("IG",  key="spr_IG_input")
        spr_IG_clean  = _clean_signed_number(spr_IG_input,  st.session_state.get("spr_IG",  DEFAULT_SPREAD["IG"]))
    with c2:
        spr_HY_input  = st.text_input("HY",  key="spr_HY_input")
        spr_HY_clean  = _clean_signed_number(spr_HY_input,  st.session_state.get("spr_HY",  DEFAULT_SPREAD["HY"]))
    with c3:
        spr_AT1_input = st.text_input("AT1", key="spr_AT1_input")
        spr_AT1_clean = _clean_signed_number(spr_AT1_input, st.session_state.get("spr_AT1", DEFAULT_SPREAD["AT1"]))
    with c4:
        spr_EM_input  = st.text_input("EM",  key="spr_EM_input")
        spr_EM_clean  = _clean_signed_number(spr_EM_input,  st.session_state.get("spr_EM",  DEFAULT_SPREAD["EM"]))

    SPREAD_BP99["IG"]=float(spr_IG_clean); SPREAD_BP99["HY"]=float(spr_HY_clean)
    SPREAD_BP99["AT1"]=float(spr_AT1_clean); SPREAD_BP99["EM"]=float(spr_EM_clean)

    st.divider()
    st.subheader("Scenario realism")
    use_corr = st.checkbox("Use correlated scenarios", value=False, key="use_corr",
        help="If off: independent Gaussian shocks (current behavior). If on: curve/credit/rates-credit correlations are applied.")
    use_tails = st.checkbox("Use fat tails (Student-t)", value=False, key="use_tails",
        help="Heavier tails for spread/rate shocks; CVaR will generally rise.")
    nu_df = st.slider("Tail degrees of freedom (ν)", 3, 30, value=7, step=1, key="nu_df",
        help="Lower ν ⇒ heavier tails. Ignored if 'Use fat tails' is off.")
    # Correlation shape parameters (only enabled if use_corr)
    rho_curve = st.slider("Curve correlation (ρ_curve)", 0.0, 0.99, value=0.85, step=0.01, key="rho_curve",
        help="Base correlation for adjacent tenors; effective corr decays with tenor gap.")
    rho_credit = st.slider("Credit block correlation (ρ_credit)", 0.0, 0.99, value=0.60, step=0.01, key="rho_credit",
        help="Common clustering across credit sleeves (IG/HY/EM/AT1).")
    rho_rates_credit = st.slider("Rates↔Credit correlation (ρ_rc)", -0.9, 0.9, value=-0.30, step=0.01, key="rho_rc",
        help="Negative typical: rates down ⇔ credit tightens.")

    st.subheader("Turnover")
    penalty_bps = st.number_input(
        "Annual turnover penalty (bps per 100% turnover)",
        value=_get_global_default("penalty_bps", 15.0),
        step=1.0,
        key="penalty_bps_ctrl",
        help="Interpreted as ANNUAL. In 'Min CVaR' (monthly CVaR) the penalty is divided by 12 to match units."
    )
    max_turn = st.slider("Max turnover per rebalance", 0.0, 1.0, _get_global_default("max_turn", 0.25), 0.01, key="max_turn_ctrl")
    st.caption("Penalty is annual. Min-CVaR uses monthly CVaR, so the penalty is internally divided by 12.")

    st.subheader("Sharpe settings")
    sharpe_lambda = st.number_input("CVaR penalty λ (annualised)", min_value=0.0, value=_get_global_default("sharpe_lambda", 1.0), step=0.1, key="sharpe_lambda_ctrl")

    st.subheader("Previous Weights (optional)")
    prev_file = st.file_uploader("CSV with columns [Segment or Name, Weight]", type=["csv"], key="prev_weights")
    if prev_file is not None:
        st.session_state["_prev_weights_csv"] = prev_file.getvalue()
        st.success("Previous weights uploaded.")

    st.subheader("Display options")
    min_weight_display = st.slider("Hide weights below", 0.0, 0.01, 0.001, 0.0005, format="%.3f")

# Load data
_perf_reset()
with perf_step("read_excel + merge"):
    try:
        df = load_joined_input(upload.getvalue(), None)
    except Exception as e:
        st.error(f"Failed to load input: {e}"); st.stop()
if len(df) == 0: st.error("No rows found after applying Include==True. Please check the input file."); st.stop()
with perf_step("build_tags"): tags = build_tags_from_meta(df)

rf_rate_dec = extract_risk_free_rate(df)

# Monte-Carlo for risk + macro add for returns
with perf_step("simulate_mc_draws", draws=int(n_draws), seed=int(seed)):
    mc = simulate_mc_draws(int(n_draws), int(seed), dict(RATES_BP99), dict(SPREAD_BP99),
                          use_corr=use_corr, use_tails=use_tails, nu_df=nu_df,
                          rho_curve=rho_curve, rho_credit=rho_credit, rho_rates_credit=rho_rates_credit)
with perf_step("build_asset_pnl_matrix", S=len(mc["d2"]), N=len(df)):
    pnl_matrix_assets = build_asset_pnl_matrix(df, tags, mc)
with perf_step("macro_add_percent"):
    regime_name = st.session_state.get("risk_regime", "Moderate")
    macro_add_pct_vec = compute_macro_add_percent(
        df, tags, regime_name, RATES_BP99, SPREAD_BP99,
        rate_sign=st.session_state.get("macro_rate_sign", -1),
        spread_sign=st.session_state.get("macro_spr_sign", -1),
        k=st.session_state.get("macro_k", 0.08)
    )
with perf_step("expected_return_vector", assets=len(df)):
    mu_base_percent = df["ExpRet_pct"].values.astype(float)  # carry + roll (annual %)
    mu_base = mu_base_percent / 100.0

# ----------------------------- Compare Funds -----------------------------
# Toggle replaces expander (persists via session_state)
if "show_compare" not in st.session_state:
    st.session_state["show_compare"] = False

show_compare = st.toggle(
    "Compare Funds: positioning & risk (from defaults)",
    value=st.session_state["show_compare"],
    key="show_compare"
)

if show_compare:
    st.caption("Each fund uses its stored per‑fund defaults (set in Fund Detail). If none saved, base defaults are used.")

    fund_outputs = {}
    ud = _user_defaults(); fund_ud = ud.get("funds", {})
    for _f, _d in fund_ud.items():
        if _d.get("objective") == "Max Drawdown Proxy": _d["objective"] = "Min VaR for Target Return Percentile"

    def run_fund(fund: str, objective: str, var_cap_override=None, prev_w=None, mu_override=None, fb_override=None, fc_override=None, target_return_override=None, min_var_override=None):
        fb_local = fb_override or FACTOR_BUDGETS_DEFAULT
        fc_local = fc_override or FUND_CONSTRAINTS[fund]
        mu_local = mu_override if mu_override is not None else mu_base
        cvar_cap_eff = (var_cap_override if var_cap_override is not None else VAR99_CAP[fund])
        params = {"factor_budgets": fb_local, "fund_caps": fc_local,
                  "turnover_penalty": penalty_bps, "max_turnover": max_turn,
                  "objective": objective, "cvar_cap": cvar_cap_eff, "sharpe_lambda": sharpe_lambda, "cvar_alpha": 0.99}
        if target_return_override is not None: params["target_return"] = float(target_return_override)
        if min_var_override is not None:       params["min_var"] = float(min_var_override)
        with perf_step(f"{fund} solve (single)", objective=objective):
            w, metrics = solve_portfolio(df, tags, mu_local, pnl_matrix_assets, fund, params, prev_w)
        if w is None: return None, metrics, None
        return w, metrics, (pnl_matrix_assets @ w)

    # Produce overview using saved defaults (and including macro add)
    for f in ["GFI","GCF","EYF"]:
        fdef = fund_ud.get(f, {}).copy()
        if fdef:
            # Use saved fund defaults with macro overlay
            if "roll_incl_pct" in fdef:
                rs = float(fdef["roll_incl_pct"]) / 100.0
                roll_values = df["Roll_Down_bps_1Y"].values if "Roll_Down_bps_1Y" in df.columns else np.zeros(len(df))
                mu_percent_overview = (
                    df["Yield_Hedged_Pct"].values
                    + roll_values * rs
                    + macro_add_pct_vec                # << macro overlay
                )
                mu_ov = mu_percent_overview / 100.0
            else:
                mu_ov = (df["ExpRet_pct"].values + macro_add_pct_vec) / 100.0
            var_band = fdef.get("var_band", (0.0, VAR99_CAP[f]*100.0))
            var_cap_ov = var_band[1] / 100.0; min_var_ov = var_band[0] / 100.0
            obj_ov = fdef.get("objective", "Max Excess Return / CVaR")
            if obj_ov == "Max Drawdown Proxy": obj_ov = "Min VaR for Target Return Percentile"
            fc_ovr = {"max_non_ig": fdef.get("cap_nonig_rng", (0.0, FUND_CONSTRAINTS[f].get("max_non_ig",1.0)))[1],
                      "max_em": fdef.get("cap_em_rng", (0.0, FUND_CONSTRAINTS[f].get("max_em",1.0)))[1],
                      "max_hybrid": fdef.get("cap_hyb_rng", (0.0, FUND_CONSTRAINTS[f].get("max_hybrid",0.0)))[1],
                      "max_cash": fdef.get("cap_cash_rng", (0.0, FUND_CONSTRAINTS[f].get("max_cash",1.0)))[1],
                      "max_at1": fdef.get("cap_at1_rng", (0.0, FUND_CONSTRAINTS[f].get("max_at1",1.0)))[1]}
            fb_ovr = {"limit_krd10y": fdef.get("krd10_max", FACTOR_BUDGETS_DEFAULT["limit_krd10y"]),
                      "limit_twist": fdef.get("twist_cap_val", FACTOR_BUDGETS_DEFAULT["limit_twist"]),
                      "limit_sdv01_ig": fdef.get("sdv_ig_rng", (0.0, FACTOR_BUDGETS_DEFAULT["limit_sdv01_ig"]))[1],
                      "limit_sdv01_hy": fdef.get("sdv_hy_rng", (0.0, FACTOR_BUDGETS_DEFAULT["limit_sdv01_hy"]))[1]}
            target_ret_override = target_return_from_percentile(mu_ov, int(fdef.get("target_ret_pctile", 60))) if obj_ov == "Min VaR for Target Return Percentile" else None
            w, metrics, _ = run_fund(f, obj_ov, var_cap_override=var_cap_ov, prev_w=None, mu_override=mu_ov, fb_override=fb_ovr, fc_override=fc_ovr, target_return_override=target_ret_override, min_var_override=min_var_ov)
            if w is not None:
                metrics["weights"] = w
                fund_outputs[f] = (metrics, pnl_matrix_assets, None)
        else:
            # No saved defaults - use base defaults with macro overlay
            mu_ov = (df["ExpRet_pct"].values + macro_add_pct_vec) / 100.0
            obj_ov = "Max Excess Return / CVaR"  # default objective
            var_cap_ov = VAR99_CAP[f]; min_var_ov = 0.0
            fc_ovr = FUND_CONSTRAINTS[f]
            fb_ovr = FACTOR_BUDGETS_DEFAULT
            target_ret_override = target_return_from_percentile(mu_ov, 60) if obj_ov == "Min VaR for Target Return Percentile" else None
            w, metrics, _ = run_fund(f, obj_ov, var_cap_override=var_cap_ov, prev_w=None, mu_override=mu_ov, fb_override=fb_ovr, fc_override=fc_ovr, target_return_override=target_ret_override, min_var_override=min_var_ov)
            if w is not None:
                metrics["weights"] = w
                fund_outputs[f] = (metrics, pnl_matrix_assets, None)

    if fund_outputs:
        cols = st.columns(3)
        for idx, f in enumerate(["GFI","GCF","EYF"]):
            if f not in fund_outputs: continue
            metrics, _, _ = fund_outputs[f]
            with cols[idx]:
                title_with_help(f"{f} – Expected Return", "Annualised carry + 1‑year roll‑down + macro drift (%).")
                st.plotly_chart(kpi_number(metrics.get("ExpRet_pct", 0.0), kind="pp"),
                                use_container_width=True, config=plotly_default_config, key=f"cmp_{f}_er")
                # VaR KPI stays (informational)...
                title_with_help(f"{f} – VaR99 1M", "Monthly 99% Value at Risk (loss).")
                st.plotly_chart(kpi_number(metrics.get("VaR99_1M", 0.0), kind="pct"),
                                use_container_width=True, config=plotly_default_config, key=f"cmp_{f}_var")

                # ...but cap status is evaluated on CVaR
                title_with_help(f"{f} – CVaR99 1M", "Average loss in the worst 1% of scenarios (cap applies to this).")
                st.plotly_chart(kpi_number(metrics.get("CVaR99_1M", 0.0), kind="pct"),
                                use_container_width=True, config=plotly_default_config, key=f"cmp_{f}_cvar")

                fdef = fund_ud.get(f, {}).copy()
                var_band = fdef.get("var_band", (0.0, VAR99_CAP[f]*100.0))
                cvar_cap_eff = var_band[1] / 100.0
                min_var_eff = var_band[0] / 100.0

                status = "within cap" if metrics.get("CVaR99_1M", 0.0) <= cvar_cap_eff else "over cap"
                st.caption(f"CVaR cap {cvar_cap_eff*100:.2f}% — {status}")
                if min_var_eff > 0:
                    st.caption(f"VaR floor {min_var_eff*100:.2f}% (VaR KPI is informational; cap applies to CVaR).")

        # Chart only (table removed as requested)
        title_with_help("Segment allocation (weights)", "Weights per segment for each fund (from saved defaults).")
        alloc_df = pd.DataFrame(index=df["Name"])
        for f in ["GFI","GCF","EYF"]:
            if f in fund_outputs:
                w = fund_outputs[f][0].get("weights", [])
                if len(w) == len(df): alloc_df[f] = w
        if not alloc_df.empty:
            alloc_df = alloc_df.fillna(0.0)
            max_w = alloc_df.max(axis=1); mask = max_w >= float(min_weight_display)
            alloc_df_plot = alloc_df.loc[mask].copy()
            fig_alloc = go.Figure()
            for col in alloc_df_plot.columns:
                fig_alloc.add_bar(name=col, x=alloc_df_plot.index, y=alloc_df_plot[col].values, marker_color=FUND_COLOR.get(col))
            fig_alloc.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=40, b=80), xaxis_title="Segment", yaxis_title="Weight")
            st.plotly_chart(fig_alloc, use_container_width=True, key="compare_funds_allocation")
    else:
        st.info("No funds optimized yet. Set fund defaults in **Fund Detail** to populate this section.")

spacer(1)

# ----------------------------- Fund Detail -----------------------------
tab_fund, = st.tabs(["Fund Detail"])
with tab_fund:
    c0, c1 = st.columns([1,2])
    with c0:
        fund = st.selectbox("Fund", ["GFI","GCF","EYF"], index=0, key="fund_selector", help="Choose the fund to optimise.")
        OBJECTIVE_CHOICES = ["Max Excess Return / CVaR","Risk-adjusted Return (λ·CVaR)","Min VaR for Target Return Percentile","Max Return"]
        objective = st.selectbox("Objective", OBJECTIVE_CHOICES, index=OBJECTIVE_CHOICES.index(_get_fund_default(fund, "objective", "Max Excess Return / CVaR")), key="fund_objective")
        var_default_cap = float(VAR99_CAP[fund] * 100.0)
        var_band = st.slider(
            f"{fund} risk band: VaR99 floor (left) & CVaR99 cap (right) (%)",
            0.0, 30.0,
            value=_get_fund_default(fund, "var_band", (0.0, var_default_cap)),
            step=0.1,
            key="var99_band"
        )
        min_var_pct = float(var_band[0]); var_cap = float(var_band[1]) / 100.0
        roll_incl_pct = st.slider("Include roll‑down return (%)", 0, 100, value=_get_fund_default(fund, "roll_incl_pct", 100), step=5, key="roll_incl_pct")
        target_ret_pctile = st.slider("Target Return Percentile", 1, 100, value=_get_fund_default(fund, "target_ret_pctile", 60), step=1, key="target_ret_pctile")

        # Renamed block
        st.write("Risk Corridors (min–max):")
        st.caption("Non-IG = HY ratings ∪ EM hard-currency ∪ Bank Capital (AT1/T2). "
                   "HY spread shocks apply to HY ratings, AT1 and EM sleeves; IG shocks apply otherwise.")
        fc = FUND_CONSTRAINTS[fund]
        cap_nonig_rng = st.slider("Non‑IG range", 0.0, 1.0, value=_get_fund_default(fund, "cap_nonig_rng", (0.0, float(fc.get('max_non_ig',1.0)))), step=0.01, key="cap_non_ig", help="Includes HY ratings, EM hard‑currency, and Bank Capital (AT1/T2).")
        cap_em_rng    = st.slider("EM range",      0.0, 1.0, value=_get_fund_default(fund, "cap_em_rng",    (0.0, float(fc.get('max_em',1.0)))), step=0.01, key="cap_em", help="EM hard‑currency sleeve only.")
        cap_hyb_rng   = st.slider("Hybrid range",  0.0, 1.0, value=_get_fund_default(fund, "cap_hyb_rng",   (0.0, float(fc.get('max_hybrid',0.0)) if 'max_hybrid' in fc else 0.0)), step=0.01, key="cap_hybrid", help="Global Hybrid sleeve only.")
        cap_cash_rng  = st.slider("Cash range",    0.0, 1.0, value=_get_fund_default(fund, "cap_cash_rng",  (0.0, float(fc.get('max_cash',1.0)))), step=0.01, key="cap_cash", help="US T‑Bills sleeve; caps cash balance.")
        cap_at1_rng   = st.slider("AT1 range",     0.0, 1.0, value=_get_fund_default(fund, "cap_at1_rng",   (0.0, float(fc.get('max_at1',1.0)))), step=0.01, key="cap_at1", help="Bank Additional Tier‑1 sleeve.")

        st.write("Factor ranges (yrs):")
        krd10_max = st.slider("|KRD 10y| cap (yrs)", 0.0, 15.0, value=_get_fund_default(fund, "krd10_max", FACTOR_BUDGETS_DEFAULT["limit_krd10y"]), step=0.05, format="%.2f", key="fund_krd10y")
        allow_negative_twist = st.checkbox("Allow flattener exposure (±Twist)", value=_get_fund_default(fund, "allow_negative_twist", False), key="allow_neg_twist")
        twist_cap_val = st.slider("Twist cap (yrs)", 0.0, 15.0, value=_get_fund_default(fund, "twist_cap_val", FACTOR_BUDGETS_DEFAULT["limit_twist"]), step=0.05, format="%.2f", key="fund_twist")
        min_krd10y = 0.0; max_krd10y = float(krd10_max)
        if allow_negative_twist: min_twist, max_twist = -float(twist_cap_val), float(twist_cap_val)
        else:                     min_twist, max_twist = 0.0, float(twist_cap_val)
        sdv_ig_rng = st.slider("sDV01 IG range (yrs)", 0.0, 15.0, value=_get_fund_default(fund, "sdv_ig_rng", (0.0, FACTOR_BUDGETS_DEFAULT["limit_sdv01_ig"])), step=0.1, format="%.1f", key="fund_sdv_ig")
        sdv_hy_rng = st.slider("sDV01 HY range (yrs)", 0.0, 15.0, value=_get_fund_default(fund, "sdv_hy_rng", (0.0, FACTOR_BUDGETS_DEFAULT["limit_sdv01_hy"])), step=0.1, format="%.1f", key="fund_sdv_hy")

        # Save defaults
        if st.button("💾 Save Defaults for this Fund", key=f"save_defaults_{fund}"):
            _save_fund_defaults(fund, {
                "objective": objective,
                "var_band": var_band,
                "roll_incl_pct": roll_incl_pct,
                "target_ret_pctile": target_ret_pctile,
                "cap_nonig_rng": cap_nonig_rng,
                "cap_em_rng": cap_em_rng,
                "cap_hyb_rng": cap_hyb_rng,
                "cap_cash_rng": cap_cash_rng,
                "cap_at1_rng": cap_at1_rng,
                "krd10_max": krd10_max,
                "allow_negative_twist": allow_negative_twist,
                "twist_cap_val": twist_cap_val,
                "sdv_ig_rng": sdv_ig_rng,
                "sdv_hy_rng": sdv_hy_rng,
            })
            st.success(f"Defaults saved for {fund}. Compare Funds will use these on the next run.")

    with c1:
        # Build μ with roll inclusion + macro drift
        roll_values = df["Roll_Down_bps_1Y"].values if "Roll_Down_bps_1Y" in df.columns else np.zeros(len(df))
        mu_fund_percent = (
            df["Yield_Hedged_Pct"].values
            + roll_values * (roll_incl_pct/100.0)
            + macro_add_pct_vec                # << macro overlay
        )
        mu_fund = mu_fund_percent / 100.0

        params_fd = {
            "factor_budgets": {"limit_krd10y": float(max_krd10y), "limit_twist": float(max_twist), "limit_sdv01_ig": float(sdv_ig_rng[1]), "limit_sdv01_hy": float(sdv_hy_rng[1])},
            "min_krd10y": float(min_krd10y), "max_krd10y": float(max_krd10y),
            "min_twist": float(min_twist), "max_twist": float(max_twist),
            "min_sdv01_ig": float(sdv_ig_rng[0]), "min_sdv01_hy": float(sdv_hy_rng[0]),
            "fund_caps": {"max_non_ig": float(cap_nonig_rng[1]), "max_em": float(cap_em_rng[1]), "max_hybrid": float(cap_hyb_rng[1]), "max_cash": float(cap_cash_rng[1]), "max_at1": float(cap_at1_rng[1])},
            "min_non_ig": float(cap_nonig_rng[0]), "min_em": float(cap_em_rng[0]), "min_hybrid": float(cap_hyb_rng[0]), "min_cash": float(cap_cash_rng[0]), "min_at1": float(cap_at1_rng[0]),
            "turnover_penalty": penalty_bps, "max_turnover": max_turn, "objective": objective,
            "cvar_cap": (var_cap if var_cap is not None else VAR99_CAP[fund]), "min_var": float(min_var_pct/100.0), "cvar_alpha": 0.99,
            "sharpe_lambda": float(sharpe_lambda)
        }

        # Previous weights mapping (optional)
        prev_w_vec = None
        raw = st.session_state.get("_prev_weights_csv")
        if raw:
            try:
                _pw = pd.read_csv(io.BytesIO(raw))
                name_col = "Segment" if "Segment" in _pw.columns else ("Name" if "Name" in _pw.columns else None)
                if name_col and "Weight" in _pw.columns:
                    _pw[name_col] = _pw[name_col].astype(str).str.strip()
                    prev_w_vec = df["Name"].astype(str).str.strip().map(_pw.set_index(name_col)["Weight"]).fillna(0.0).values
                    s = float(prev_w_vec.sum()); prev_w_vec = (prev_w_vec / s) if s > 1e-12 else None
                else:
                    st.warning("Prev weights CSV must have columns [Segment or Name, Weight].")
            except Exception as e:
                st.warning(f"Failed to parse previous weights: {e}")

        target_ret_override = target_return_from_percentile(mu_fund, target_ret_pctile) if objective == "Min VaR for Target Return Percentile" else None
        if target_ret_override is not None: params_fd["target_return"] = float(target_ret_override)

        with perf_step(f"{fund} Fund Detail solve", objective=objective):
            w, metrics = solve_portfolio(df, tags, mu_fund, pnl_matrix_assets, fund, params_fd, prev_w_vec)
        if w is None:
            st.error(f"Optimisation failed: {metrics.get('status','')} – {metrics.get('message','')}"); st.stop()

        # --- Turnover status & diagnostics ---
        turn_active = prev_w_vec is not None and float(np.sum(prev_w_vec)) > 1e-8
        realized_turn = float(np.sum(np.abs((prev_w_vec if turn_active else np.zeros_like(w)) - w)))
        turn_cap = float(params_fd.get("max_turnover", 1.0)) if turn_active else 1.0
        if turn_active:
            st.info(f"Turnover active. Realised L1 turnover = {realized_turn:.2%} (cap {turn_cap:.2%}).")
        else:
            st.caption("Turnover inactive (no previous weights loaded): penalty=0, max_turn=100%.")

        # Portfolio-level roll-down contribution (shown for transparency)
        roll_used = (roll_incl_pct / 100.0)
        port_roll_add_pct = float((df["Roll_Down_bps_1Y"].values if "Roll_Down_bps_1Y" in df.columns else np.zeros(len(df))) @ w) * roll_used
        st.caption(f"Roll-down inclusion = {roll_incl_pct:.0f}% | Portfolio roll-down add = {port_roll_add_pct:.2f}%")

        port_pnl = pnl_matrix_assets @ w
        cols = st.columns(4)
        with cols[0]:
            title_with_help("Expected Return (ann.)", "Annualised carry + roll‑down + macro drift (%).")
            st.plotly_chart(kpi_number(metrics["ExpRet_pct"], kind="pp"), use_container_width=True, config=plotly_default_config, key=f"{fund}_er_kpi")
        with cols[1]:
            title_with_help("VaR99 1M", "One‑month, 99% Value at Risk (loss).")
            st.plotly_chart(kpi_number(metrics["VaR99_1M"], kind="pct"), use_container_width=True, config=plotly_default_config, key=f"{fund}_var_kpi")
        with cols[2]:
            title_with_help("CVaR99 1M", "Average loss in the worst 1% of scenarios.")
            st.plotly_chart(kpi_number(metrics["CVaR99_1M"], kind="pct"), use_container_width=True, config=plotly_default_config, key=f"{fund}_cvar_kpi")
        with cols[3]:
            title_with_help("Portfolio Yield", "Yield component of return (%)")
            st.plotly_chart(kpi_number(metrics["Yield_pct"], kind="pp"), use_container_width=True, config=plotly_default_config, key=f"{fund}_yield_kpi")

        # Auto‑relax warning (Min VaR for Target Return Percentile)
        if objective == "Min VaR for Target Return Percentile":
            _diag = metrics.get("_diag", {})
            if _diag.get("auto_relaxed"):
                st.warning(f"Requested return floor was infeasible. Used percentile = {_diag.get('relaxed_to_percentile')} instead.")

        st.caption(f"CVaR99 1M: {metrics['CVaR99_1M']*100:.2f}% (cap {var_cap*100:.2f}%)")
        if min_var_pct > 0:
            st.caption(f"VaR99 1M floor: ≥ {min_var_pct:.2f}% (VaR KPI shown separately).")

        # Permanent diagnostic table (change in weights)
        st.markdown("### 🧾 Diagnostic: Change in weights since last run")
        last_runs = st.session_state.setdefault("last_run", {})
        prev = last_runs.get(fund, None)
        prev_w = prev["weights"] if prev is not None else np.zeros_like(w)
        delta = w - prev_w
        df_diag = pd.DataFrame({"Segment": df["Name"], "Prev_w": prev_w, "Curr_w": w, "Delta_w": delta})
        df_diag = df_diag.reindex(df_diag["Delta_w"].abs().sort_values(ascending=False).index)
        st.dataframe(df_diag.style.format({"Prev_w":"{:.2%}","Curr_w":"{:.2%}","Delta_w":"{:+.2%}"}), use_container_width=True, height=320)
        last_runs[fund] = {"weights": w,
                           "shock_params": {"rate_2y": RATES_BP99["2y"], "rate_5y": RATES_BP99["5y"], "rate_10y": RATES_BP99["10y"], "rate_30y": RATES_BP99["30y"],
                                            "spr_IG": SPREAD_BP99["IG"], "spr_HY": SPREAD_BP99["HY"], "spr_AT1": SPREAD_BP99["AT1"], "spr_EM": SPREAD_BP99["EM"]}}

        # Allocation bar chart
        title_with_help("Segment allocation (weights)", "Weights per sleeve after optimisation under the current corridors and budgets.")
        st.plotly_chart(bar_allocation(df, w, "Segment allocation (weights)", min_weight_display), use_container_width=True, key=f"{fund}_allocation_chart")

        # Corridor usage
        render_cap_usage_section(fund, w, tags, params_fd["fund_caps"])

        title_with_help(f"{fund} – Factor Exposures vs Budgets", "KRD10y, Twist(30y–2y), and sDV01 IG/HY vs budgets.")
        st.plotly_chart(exposures_vs_budgets(df, w, params_fd["factor_budgets"], ""), use_container_width=True, key=f"{fund}_exposures_chart")

        spacer(1)
        title_with_help(f"{fund} – Scenario P&L Distribution", "Monthly %P&L distribution from Monte Carlo. Vertical lines show VaR99 and CVaR99.")
        st.plotly_chart(scenario_histogram(port_pnl), use_container_width=True, key=f"{fund}_scenario_chart")

        spacer(1)
        title_with_help("Segment contributions table", "Weights, expected return contribution (%), yield & roll‑down, and duration metrics per segment.")
        contr = pd.DataFrame({
            "Segment": df["Name"],
            "Weight": w,
            "ER_Contribution_pct": w * (mu_fund * 100.0),
            "Yield_pct": df["Yield_Hedged_Pct"].values,
            "RollDown_pct": df["Roll_Down_bps_1Y"].values if "Roll_Down_bps_1Y" in df.columns else np.zeros(len(df)),
            "OAD_Years": df["OAD_Years"].values,
            "OASD_Years": df["OASD_Years"].values
        }).sort_values("Weight", ascending=False)
        st.dataframe(contr, use_container_width=True, height=360)
        st.download_button("Download weights & ER contributions (CSV)", contr[["Segment","Weight","ER_Contribution_pct"]].to_csv(index=False).encode("utf-8"), file_name=f"{fund}_allocation.csv", mime="text/csv")

st.caption("© Rubrics – internal research tool. Forward-looking estimates; not investment advice.")
st.markdown("<br>" * 5, unsafe_allow_html=True)
