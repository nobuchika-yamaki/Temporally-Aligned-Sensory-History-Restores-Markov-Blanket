#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reviewer_submission_markov_blanket_reanalysis.py

Reviewer-submission analysis code for the delayed Markov-blanket reanalysis.

Aim
---
Preserve the first manuscript's comparison design while replacing the delayed dynamics
with the separated sensory/active boundary model proposed in the second manuscript.

Non-negotiable fairness constraints
-----------------------------------
1. All models are evaluated on the same generated trajectory.
2. All models share the same separated boundary base:
       X_base = {S_n, A_n}
3. Model differences are only additional sensory-side temporal information.
4. No model is selected because it gives the desired result.
5. Stability diagnostics are computed before interpretation.
6. No unstable trajectory is silently removed:
       summaries are reported for all trajectories and stable-only trajectories.
7. Prediction uses normalized MSE (NMSE), not raw-scale MSE.
8. Residual dependence uses cross-fitted residual Gaussian CMI:
       fit residualization models on fit data,
       compute residual dependence on held-out test data.
   This avoids in-sample overfitting by high-dimensional history models.
9. H-grid is evaluated explicitly for residual CMI.
10. H-independent models are computed once per trajectory and replicated across H to
    prevent computational blow-up without changing the comparison.

Generative model
----------------
Second-paper separated dissipative blanket model:

    E -> S -> I -> A -> E
    delay: S(t - tau) -> I(t)

Baseline equations:

    dE = [-mu E + chi A]dt + sigma_E dW_E
    dS = [-lambda_S S + kappa E]dt + sigma_S dW_S
    dI = [-gamma I + alpha f(S(t - tau))]dt + sigma_I dW_I
    dA = [-lambda_A A + rho f(I)]dt + sigma_A dW_A

where f(x)=x in the linear condition and f(x)=tanh(x) in the saturating robustness
condition.

Comparison models
-----------------
instantaneous:
    {S_n, A_n}

generalized:
    {S_n, A_n, D S_n, D^2 S_n, ..., D^K S_n}

history:
    {S_n, A_n} union {S_{n-w}: w in Omega_H,r}

oracle_delay:
    {S_n, A_n, S_{n-d}}

random_history:
    {S_n, A_n} union random past sensory lags, dimension-matched to history

shuffled_history:
    {S_n, A_n} union temporally misaligned sensory history, dimension-matched to history

Recommended commands
--------------------
Smoke:
    python3 -u reviewer_submission_markov_blanket_reanalysis.py --mode smoke

Balanced run:
    python3 -u reviewer_submission_markov_blanket_reanalysis.py \
      --mode balanced \
      --outdir ~/Desktop/first_paper_fair_balanced \
      --workers 4

Full run:
    python3 -u reviewer_submission_markov_blanket_reanalysis.py \
      --mode full \
      --outdir ~/Desktop/first_paper_fair_full \
      --workers 4

Optional bounded robustness:
    add --condition tanh
or:
    add --condition both

Optional H-grid key prediction:
    add --run-hgrid-key-prediction
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Logging
# =============================================================================

class Logger:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.t0 = time.time()
        self.path = outdir / "run.log"
        outdir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"Started: {datetime.now().isoformat()}\n")

    def msg(self, text: str) -> None:
        line = f"[{time.time() - self.t0:10.1f}s] {text}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def section(self, text: str) -> None:
        self.msg("")
        self.msg("=" * 94)
        self.msg(text)
        self.msg("=" * 94)


# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class Params:
    # second-manuscript baseline parameters
    mu: float = 0.20
    lambda_s: float = 0.60
    gamma: float = 0.40
    lambda_a: float = 0.60
    kappa: float = 1.00
    alpha: float = 1.00
    rho: float = 0.80
    chi: float = 0.05
    sigma_e: float = 0.50
    sigma_s: float = 0.10
    sigma_i: float = 0.10
    sigma_a: float = 0.10
    dt: float = 0.01
    total_time: float = 2000.0
    burn_frac: float = 0.20
    response: str = "linear"  # linear or tanh

    # second-manuscript history stride
    history_stride: float = 0.10

    # first-manuscript comparison extension
    generalized_order: int = 2

    # estimation
    cmi_ridge_alpha: float = 1e-3
    pred_alpha_grid: str = "1e-8,1e-6,1e-4,1e-3,1e-2,1e-1,1,10"


MODEL_ORDER = [
    "instantaneous",
    "generalized",
    "history",
    "oracle_delay",
    "random_history",
    "shuffled_history",
]

MODEL_LABEL = {
    "instantaneous": "Instantaneous",
    "generalized": "Generalized",
    "history": "History",
    "oracle_delay": "Oracle-delay",
    "random_history": "Random-history",
    "shuffled_history": "Shuffled-history",
}

H_DEPENDENT_MODELS = {"history", "random_history", "shuffled_history"}


# =============================================================================
# Utilities
# =============================================================================

def parse_float_list(text: str) -> List[float]:
    return sorted(set([float(x.strip()) for x in text.split(",") if x.strip()]))


def response_fn(x: np.ndarray | float, kind: str) -> np.ndarray | float:
    if kind == "linear":
        return x
    if kind == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unknown response: {kind}")


def make_outdir(path: Optional[str]) -> Path:
    if path:
        out = Path(path).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path.home() / "Desktop" / f"first_paper_fair_reanalysis_{stamp}"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def seed_hash(*items: Any) -> int:
    return abs(hash(items)) % (2**32)


def sample_evenly(idx: np.ndarray, max_n: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=int)
    if max_n <= 0 or len(idx) <= max_n:
        return idx
    take = np.linspace(0, len(idx) - 1, max_n).round().astype(int)
    return idx[take]


def history_offsets_steps(H: float, r: float, dt: float) -> List[int]:
    if H <= 0:
        return []
    offsets: List[int] = []
    x = r
    while x <= H + 1e-12:
        step = int(round(x / dt))
        if step > 0:
            offsets.append(step)
        x += r

    # second manuscript: include H itself when not exactly represented by stride
    h_step = int(round(H / dt))
    if h_step > 0:
        offsets.append(h_step)

    return sorted(set(offsets))


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(vals: np.ndarray, n_boot: int, seed: int, stat: str = "mean") -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1 or n_boot <= 0:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    sample = vals[idx]
    if stat == "median":
        b = np.median(sample, axis=1)
    else:
        b = sample.mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def markdown_table_no_tabulate(df: pd.DataFrame, max_rows: int = 40) -> str:
    """Minimal markdown table writer that does not require the optional tabulate package."""
    if df is None or df.empty:
        return "No rows."

    d = df.head(max_rows).copy()
    cols = list(d.columns)

    def fmt(x: Any) -> str:
        if pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.6g}"
        return str(x)

    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    if len(df) > max_rows:
        rows.append("| " + " | ".join(["..."] * len(cols)) + " |")
    return "\n".join(rows)


def split_indices(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contiguous blocked split:
    60% fit, 20% validation, 20% held-out test.
    """
    n = len(idx)
    n_fit = int(round(0.60 * n))
    n_val = int(round(0.20 * n))
    idx_fit = idx[:n_fit]
    idx_val = idx[n_fit:n_fit + n_val]
    idx_test = idx[n_fit + n_val:]
    if len(idx_fit) < 10 or len(idx_val) < 10 or len(idx_test) < 10:
        raise ValueError("Split too small.")
    return idx_fit, idx_val, idx_test


# =============================================================================
# Exact separated model
# =============================================================================

def simulate_model(params: Params, tau: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = params.dt
    n_steps = int(round(params.total_time / dt))
    delay_steps = int(round(tau / dt))
    sqrt_dt = math.sqrt(dt)

    E = np.zeros(n_steps + 1, dtype=float)
    S = np.zeros(n_steps + 1, dtype=float)
    I = np.zeros(n_steps + 1, dtype=float)
    A = np.zeros(n_steps + 1, dtype=float)

    E[0], S[0], I[0], A[0] = rng.normal(0.0, 0.01, size=4)

    for n in range(n_steps):
        delayed_s = S[n - delay_steps] if n - delay_steps >= 0 else S[0]
        eE, eS, eI, eA = rng.normal(size=4)

        E[n + 1] = E[n] + dt * (-params.mu * E[n] + params.chi * A[n]) + params.sigma_e * sqrt_dt * eE
        S[n + 1] = S[n] + dt * (-params.lambda_s * S[n] + params.kappa * E[n]) + params.sigma_s * sqrt_dt * eS
        I[n + 1] = I[n] + dt * (-params.gamma * I[n] + params.alpha * response_fn(delayed_s, params.response)) + params.sigma_i * sqrt_dt * eI
        A[n + 1] = A[n] + dt * (-params.lambda_a * A[n] + params.rho * response_fn(I[n], params.response)) + params.sigma_a * sqrt_dt * eA

        if not np.isfinite(E[n + 1] + S[n + 1] + I[n + 1] + A[n + 1]):
            # return partial arrays; job will be flagged non-finite
            E[n + 1:] = np.nan
            S[n + 1:] = np.nan
            I[n + 1:] = np.nan
            A[n + 1:] = np.nan
            break

    burn = int(round(params.burn_frac * (n_steps + 1)))
    return {
        "E": E,
        "S": S,
        "I": I,
        "A": A,
        "burn": np.array([burn], dtype=int),
        "delay_steps": np.array([delay_steps], dtype=int),
    }


def stability_diagnostics(data: Dict[str, np.ndarray], idx: np.ndarray, stable_threshold: float, growth_threshold: float) -> Dict[str, Any]:
    arr = np.column_stack([data["E"][idx], data["S"][idx], data["I"][idx], data["A"][idx]])
    finite = bool(np.all(np.isfinite(arr)))
    if not finite:
        return {
            "finite": False,
            "max_abs_state": np.inf,
            "growth_ratio": np.inf,
            "stable": False,
        }

    max_abs = float(np.max(np.abs(arr)))
    n = len(arr)
    q = max(10, n // 4)
    rms_first = float(np.sqrt(np.mean(arr[:q] ** 2)))
    rms_last = float(np.sqrt(np.mean(arr[-q:] ** 2)))
    growth = float(rms_last / max(rms_first, 1e-12))
    stable = bool((max_abs <= stable_threshold) and (growth <= growth_threshold))

    return {
        "finite": finite,
        "max_abs_state": max_abs,
        "growth_ratio": growth,
        "stable": stable,
    }


def aligned_indices(n: int, burn: int, max_lag: int, max_future: int) -> np.ndarray:
    start = max(burn, max_lag)
    stop = n - max_future
    if stop <= start:
        raise ValueError("Aligned segment is empty.")
    return np.arange(start, stop, dtype=int)


# =============================================================================
# Comparison matrices
# =============================================================================

def causal_smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    out = np.empty_like(x, dtype=float)
    cs = np.cumsum(np.insert(x, 0, 0.0))
    for n in range(len(x)):
        start = max(0, n - window + 1)
        out[n] = (cs[n + 1] - cs[start]) / (n - start + 1)
    return out


def sensory_derivatives(S: np.ndarray, dt: float, K: int) -> List[np.ndarray]:
    if K <= 0:
        return []
    window = max(3, int(round(0.10 / dt)))
    cur = causal_smooth(S, window)
    derivs: List[np.ndarray] = []
    for _ in range(K):
        d = np.zeros_like(cur)
        d[1:] = np.diff(cur) / dt
        d = causal_smooth(d, window)
        derivs.append(d)
        cur = d
    return derivs


def make_X(
    model: str,
    data: Dict[str, np.ndarray],
    idx: np.ndarray,
    tau: float,
    H: float,
    params: Params,
    seed: int,
    deriv_cache: Optional[List[np.ndarray]] = None,
    shuffle_cache: Optional[np.ndarray] = None,
) -> np.ndarray:
    S = data["S"]
    A = data["A"]
    delay_steps = int(round(tau / params.dt))
    offsets = history_offsets_steps(H, params.history_stride, params.dt)

    # identical separated base for all models
    cols: List[np.ndarray] = [S[idx], A[idx]]

    if model == "instantaneous":
        return np.column_stack(cols)

    if model == "generalized":
        if deriv_cache is None:
            deriv_cache = sensory_derivatives(S, params.dt, params.generalized_order)
        for der in deriv_cache:
            cols.append(der[idx])
        return np.column_stack(cols)

    if model == "history":
        for off in offsets:
            cols.append(S[idx - off])
        return np.column_stack(cols)

    if model == "oracle_delay":
        cols.append(S[idx - delay_steps])
        return np.column_stack(cols)

    if model == "random_history":
        if len(offsets) > 0:
            rng = np.random.default_rng(seed + int(round(H * 10000)) + 130001)
            max_step = max(offsets)
            pool = np.arange(1, max_step + 1, dtype=int)
            if len(pool) > 1:
                pool = pool[pool != delay_steps]
            if len(pool) == 0:
                pool = np.arange(1, max_step + 1, dtype=int)
            replace = len(pool) < len(offsets)
            rand_offsets = sorted(rng.choice(pool, size=len(offsets), replace=replace).astype(int).tolist())
            for off in rand_offsets:
                cols.append(S[idx - off])
        return np.column_stack(cols)

    if model == "shuffled_history":
        if shuffle_cache is None:
            rng = np.random.default_rng(seed + 230007)
            n = len(S)
            min_shift = max(1, int(round(0.10 * n)))
            max_shift = max(min_shift + 1, int(round(0.90 * n)))
            shift = int(rng.integers(min_shift, max_shift))
            shuffle_cache = np.roll(S, shift)
        for off in offsets:
            cols.append(shuffle_cache[idx - off])
        return np.column_stack(cols)

    raise ValueError(f"Unknown model: {model}")


def max_required_lag(tau: float, H_grid: List[float], params: Params) -> int:
    d = int(round(tau / params.dt))
    max_h = max([0] + [max(history_offsets_steps(H, params.history_stride, params.dt) or [0]) for H in H_grid])
    der_margin = max(3, int(round(0.10 / params.dt))) * max(1, params.generalized_order)
    return max(d, max_h, der_margin)


# =============================================================================
# Ridge and metrics
# =============================================================================

def standardize_fit_apply(X_fit: np.ndarray, *Xs: np.ndarray) -> Tuple[np.ndarray, ...]:
    mean = X_fit.mean(axis=0, keepdims=True)
    sd = X_fit.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return tuple((X - mean) / sd for X in Xs)


def standardize_y_fit_apply(y_fit: np.ndarray, *ys: np.ndarray) -> Tuple[np.ndarray, ...]:
    mean = float(np.mean(y_fit))
    sd = float(np.std(y_fit))
    if sd < 1e-12:
        sd = 1.0
    return tuple((y - mean) / sd for y in ys)


def ridge_beta(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(X)), X])
    P = np.eye(Xd.shape[1]) * alpha
    P[0, 0] = 0.0
    return np.linalg.solve(Xd.T @ Xd + P, Xd.T @ y)


def ridge_predict_with_beta(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(X)), X])
    return Xd @ beta


def select_alpha_nmse(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha_grid: List[float],
) -> float:
    X_fit_s, X_val_s = standardize_fit_apply(X_fit, X_fit, X_val)
    y_fit_s, y_val_s = standardize_y_fit_apply(y_fit, y_fit, y_val)

    best_alpha = alpha_grid[0]
    best_nmse = np.inf
    var_val = float(np.var(y_val_s))
    if var_val < 1e-12:
        var_val = 1.0

    for a in alpha_grid:
        beta = ridge_beta(X_fit_s, y_fit_s, a)
        pred = ridge_predict_with_beta(X_val_s, beta)
        nmse = float(np.mean((y_val_s - pred) ** 2) / var_val)
        if nmse < best_nmse:
            best_nmse = nmse
            best_alpha = a
    return float(best_alpha)


def predict_nmse_cv(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: List[float],
) -> Tuple[float, float, float]:
    """
    Blocked validation chooses ridge alpha. Final model is fit on fit+val and evaluated
    on held-out test.

    Output:
    nmse_test, r2_test, selected_alpha.
    """
    alpha = select_alpha_nmse(X_fit, y_fit, X_val, y_val, alpha_grid)

    X_train = np.vstack([X_fit, X_val])
    y_train = np.concatenate([y_fit, y_val])

    X_train_s, X_test_s = standardize_fit_apply(X_train, X_train, X_test)
    y_train_s, y_test_s = standardize_y_fit_apply(y_train, y_train, y_test)

    beta = ridge_beta(X_train_s, y_train_s, alpha)
    pred = ridge_predict_with_beta(X_test_s, beta)

    var_test = float(np.var(y_test_s))
    if var_test < 1e-12:
        return np.nan, np.nan, alpha

    nmse = float(np.mean((y_test_s - pred) ** 2) / var_test)
    r2 = float(1.0 - nmse)
    return nmse, r2, alpha


def residuals_fixed_ridge(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, bool]:
    """
    Fit y ~ X on fit data, return held-out standardized residuals on test data.
    """
    if not (np.all(np.isfinite(X_fit)) and np.all(np.isfinite(X_test)) and np.all(np.isfinite(y_fit)) and np.all(np.isfinite(y_test))):
        return np.full(len(y_test), np.nan), False

    X_fit_s, X_test_s = standardize_fit_apply(X_fit, X_fit, X_test)
    y_fit_s, y_test_s = standardize_y_fit_apply(y_fit, y_fit, y_test)

    try:
        beta = ridge_beta(X_fit_s, y_fit_s, alpha)
        pred = ridge_predict_with_beta(X_test_s, beta)
    except np.linalg.LinAlgError:
        return np.full(len(y_test), np.nan), False

    resid = y_test_s - pred
    if not np.all(np.isfinite(resid)) or np.std(resid) < 1e-12:
        return resid, False
    return resid, True


def gaussian_mi_from_residuals(r1: np.ndarray, r2: np.ndarray) -> Tuple[float, float, bool]:
    good = np.isfinite(r1) & np.isfinite(r2)
    r1 = r1[good]
    r2 = r2[good]
    if len(r1) < 10 or np.std(r1) < 1e-12 or np.std(r2) < 1e-12:
        return np.nan, np.nan, False
    rho = float(np.corrcoef(r1, r2)[0, 1])
    if not np.isfinite(rho):
        return np.nan, np.nan, False
    rho = float(np.clip(rho, -0.999999, 0.999999))
    cmi = float(-0.5 * math.log(max(1e-12, 1.0 - rho * rho)))
    return cmi, rho, True


def crossfit_residual_cmi(
    X_fit: np.ndarray,
    I_fit: np.ndarray,
    E_fit: np.ndarray,
    X_test: np.ndarray,
    I_test: np.ndarray,
    E_test: np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    rI, okI = residuals_fixed_ridge(X_fit, I_fit, X_test, I_test, alpha)
    rE, okE = residuals_fixed_ridge(X_fit, E_fit, X_test, E_test, alpha)
    cmi, rho, okMI = gaussian_mi_from_residuals(rI, rE)
    ok = bool(okI and okE and okMI)
    return {
        "cmi_xfit": cmi,
        "resid_corr": rho,
        "valid_cmi": ok,
        "rI_sd": float(np.nanstd(rI)),
        "rE_sd": float(np.nanstd(rE)),
        "rI": rI,
        "rE": rE,
    }


def surrogate_residual_mi95(rI: np.ndarray, rE: np.ndarray, n_sur: int, seed: int) -> Tuple[float, float, float]:
    good = np.isfinite(rI) & np.isfinite(rE)
    rI = rI[good]
    rE = rE[good]
    if len(rI) < 10:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(rI)
    min_shift = max(1, int(round(0.10 * n)))
    max_shift = max(min_shift + 1, n - min_shift)
    vals = np.empty(n_sur, dtype=float)
    for k in range(n_sur):
        sh = int(rng.integers(min_shift, max_shift))
        vals[k], _, _ = gaussian_mi_from_residuals(rI, np.roll(rE, sh))
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(vals.mean()), float(np.percentile(vals, 95)), float(vals.std())


# =============================================================================
# Per-job analysis
# =============================================================================

def run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    condition: str = job["condition"]
    params: Params = job["params"]
    tau: float = job["tau"]
    seed: int = job["seed"]
    H_grid: List[float] = job["H_grid"]
    primary_H: float = job["primary_H"]
    n_sur: int = job["n_sur"]
    future_horizons: List[float] = job["future_horizons"]
    max_fit: int = job["max_fit_samples"]
    max_val: int = job["max_val_samples"]
    max_test: int = job["max_test_samples"]
    stable_threshold: float = job["stable_threshold"]
    growth_threshold: float = job["growth_threshold"]
    alpha_grid: List[float] = job["alpha_grid"]
    run_hgrid_key_prediction: bool = job["run_hgrid_key_prediction"]

    data = simulate_model(params, tau, seed)
    d = int(data["delay_steps"][0])
    burn = int(data["burn"][0])
    max_future = max([int(round(q / params.dt)) for q in future_horizons] + [0])
    idx_all = aligned_indices(len(data["E"]), burn, max_required_lag(tau, H_grid, params), max_future)

    stab = stability_diagnostics(data, idx_all, stable_threshold, growth_threshold)

    idx_fit_full, idx_val_full, idx_test_full = split_indices(idx_all)
    idx_fit = sample_evenly(idx_fit_full, max_fit)
    idx_val = sample_evenly(idx_val_full, max_val)
    idx_test = sample_evenly(idx_test_full, max_test)

    E = data["E"]
    S = data["S"]
    I = data["I"]
    A = data["A"]

    deriv_cache = sensory_derivatives(S, params.dt, params.generalized_order)
    rng = np.random.default_rng(seed + 230007)
    min_shift = max(1, int(round(0.10 * len(S))))
    max_shift = max(min_shift + 1, int(round(0.90 * len(S))))
    shuffle_cache = np.roll(S, int(rng.integers(min_shift, max_shift)))

    diag_row = {
        "condition": condition,
        "tau": tau,
        "seed": seed,
        **stab,
        "n_all": len(idx_all),
        "n_fit": len(idx_fit),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
    }

    cmi_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    hgrid_pred_rows: List[Dict[str, Any]] = []

    # CMI H-grid. H-independent models are computed once per seed.
    cmi_cache: Dict[str, Dict[str, Any]] = {}
    inst_resid_for_sur: Optional[Tuple[np.ndarray, np.ndarray]] = None
    inst_sur = (np.nan, np.nan, np.nan)

    for H in H_grid:
        for model in MODEL_ORDER:
            cache_key = model if model not in H_DEPENDENT_MODELS else f"{model}_H_{H:g}"

            if cache_key in cmi_cache:
                res = cmi_cache[cache_key]
            else:
                X_fit = make_X(model, data, idx_fit, tau, H, params, seed, deriv_cache, shuffle_cache)
                X_test = make_X(model, data, idx_test, tau, H, params, seed, deriv_cache, shuffle_cache)

                res = crossfit_residual_cmi(
                    X_fit=X_fit,
                    I_fit=I[idx_fit],
                    E_fit=E[idx_fit],
                    X_test=X_test,
                    I_test=I[idx_test],
                    E_test=E[idx_test],
                    alpha=params.cmi_ridge_alpha,
                )
                res["feature_dim"] = X_fit.shape[1]

                # Use instantaneous residuals for surrogate baseline, once.
                if model == "instantaneous" and inst_resid_for_sur is None:
                    inst_resid_for_sur = (res["rI"], res["rE"])
                    inst_sur = surrogate_residual_mi95(res["rI"], res["rE"], n_sur, seed + 440011)

                # Remove residual arrays before caching table fields.
                rI = res.pop("rI")
                rE = res.pop("rE")
                cmi_cache[cache_key] = res

            cmi_rows.append({
                "condition": condition,
                "tau": tau,
                "H": H,
                "seed": seed,
                "model": model,
                "cmi_xfit": res["cmi_xfit"],
                "resid_corr": res["resid_corr"],
                "valid_cmi": res["valid_cmi"],
                "feature_dim": res["feature_dim"],
                "rI_sd": res["rI_sd"],
                "rE_sd": res["rE_sd"],
                "inst_sur_mean": inst_sur[0],
                "inst_sur95": inst_sur[1],
                "inst_sur_sd": inst_sur[2],
                "inst_screening_failure": bool(np.isfinite(inst_sur[1]) and cmi_cache.get("instantaneous", {}).get("cmi_xfit", np.nan) > inst_sur[1]),
                **stab,
            })

    # Primary-H target prediction: all targets.
    target_specs: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("E_current", E[idx_fit], E[idx_val], E[idx_test]),
        ("I_current", I[idx_fit], I[idx_val], I[idx_test]),
        ("S_delay", S[idx_fit - d], S[idx_val - d], S[idx_test - d]),
    ]
    for q in future_horizons:
        qs = int(round(q / params.dt))
        target_specs.append((f"E_future_{q:g}", E[idx_fit + qs], E[idx_val + qs], E[idx_test + qs]))
        target_specs.append((f"A_future_{q:g}", A[idx_fit + qs], A[idx_val + qs], A[idx_test + qs]))

    for model in MODEL_ORDER:
        X_fit = make_X(model, data, idx_fit, tau, primary_H, params, seed, deriv_cache, shuffle_cache)
        X_val = make_X(model, data, idx_val, tau, primary_H, params, seed, deriv_cache, shuffle_cache)
        X_test = make_X(model, data, idx_test, tau, primary_H, params, seed, deriv_cache, shuffle_cache)

        for target, y_fit, y_val, y_test in target_specs:
            if not (np.all(np.isfinite(X_fit)) and np.all(np.isfinite(X_val)) and np.all(np.isfinite(X_test)) and
                    np.all(np.isfinite(y_fit)) and np.all(np.isfinite(y_val)) and np.all(np.isfinite(y_test))):
                nmse, r2, alpha = np.nan, np.nan, np.nan
                valid_pred = False
            else:
                try:
                    nmse, r2, alpha = predict_nmse_cv(X_fit, y_fit, X_val, y_val, X_test, y_test, alpha_grid)
                    valid_pred = bool(np.isfinite(nmse))
                except Exception:
                    nmse, r2, alpha = np.nan, np.nan, np.nan
                    valid_pred = False

            pred_rows.append({
                "condition": condition,
                "tau": tau,
                "H": primary_H,
                "seed": seed,
                "model": model,
                "target": target,
                "nmse": nmse,
                "r2": r2,
                "selected_alpha": alpha,
                "valid_pred": valid_pred,
                "feature_dim": X_fit.shape[1],
                **stab,
            })

    # Optional H-grid key prediction: only key targets to control runtime.
    if run_hgrid_key_prediction:
        key_targets = [
            ("I_current", lambda idx: I[idx]),
            ("S_delay", lambda idx: S[idx - d]),
        ]
        q16 = int(round(1.6 / params.dt))
        if q16 <= max_future:
            key_targets.append(("A_future_1.6", lambda idx: A[idx + q16]))

        for H in H_grid:
            for model in MODEL_ORDER:
                X_fit = make_X(model, data, idx_fit, tau, H, params, seed, deriv_cache, shuffle_cache)
                X_val = make_X(model, data, idx_val, tau, H, params, seed, deriv_cache, shuffle_cache)
                X_test = make_X(model, data, idx_test, tau, H, params, seed, deriv_cache, shuffle_cache)

                for target, getter in key_targets:
                    y_fit = getter(idx_fit)
                    y_val = getter(idx_val)
                    y_test = getter(idx_test)
                    if not (np.all(np.isfinite(X_fit)) and np.all(np.isfinite(X_val)) and np.all(np.isfinite(X_test)) and
                            np.all(np.isfinite(y_fit)) and np.all(np.isfinite(y_val)) and np.all(np.isfinite(y_test))):
                        nmse, r2, alpha = np.nan, np.nan, np.nan
                        valid_pred = False
                    else:
                        try:
                            nmse, r2, alpha = predict_nmse_cv(X_fit, y_fit, X_val, y_val, X_test, y_test, alpha_grid)
                            valid_pred = bool(np.isfinite(nmse))
                        except Exception:
                            nmse, r2, alpha = np.nan, np.nan, np.nan
                            valid_pred = False

                    hgrid_pred_rows.append({
                        "condition": condition,
                        "tau": tau,
                        "H": H,
                        "seed": seed,
                        "model": model,
                        "target": target,
                        "nmse": nmse,
                        "r2": r2,
                        "selected_alpha": alpha,
                        "valid_pred": valid_pred,
                        "feature_dim": X_fit.shape[1],
                        **stab,
                    })

    return {
        "diagnostics": diag_row,
        "cmi": cmi_rows,
        "prediction_primary": pred_rows,
        "prediction_hgrid_key": hgrid_pred_rows,
    }


# =============================================================================
# Summaries
# =============================================================================

def make_analysis_sets(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    sets = [("all", df)]
    if "stable" in df.columns:
        stable_df = df[df["stable"] == True].copy()
        sets.append(("stable_only", stable_df))
    return sets


def summarize_metric(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    n_boot: int,
    outpath: Path,
    valid_col: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for analysis_set, dset in make_analysis_sets(df):
        if valid_col is not None and valid_col in dset.columns:
            dset_valid = dset[dset[valid_col] == True].copy()
        else:
            dset_valid = dset.copy()

        if dset_valid.empty:
            continue

        for keys, g in dset_valid.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = dict(zip(group_cols, keys))
            vals = g[value_col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            lo_mean, hi_mean = bootstrap_ci(vals, n_boot, seed_hash(outpath.name, analysis_set, *keys, "mean"), "mean")
            lo_med, hi_med = bootstrap_ci(vals, n_boot, seed_hash(outpath.name, analysis_set, *keys, "median"), "median")
            rec.update({
                "analysis_set": analysis_set,
                "n": int(len(vals)),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals)),
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "ci95_mean_low": lo_mean,
                "ci95_mean_high": hi_mean,
                "ci95_median_low": lo_med,
                "ci95_median_high": hi_med,
            })
            if "feature_dim" in g.columns:
                rec["feature_dim_mean"] = float(g["feature_dim"].mean())
            if "stable" in g.columns:
                rec["stable_rate_in_group"] = float(g["stable"].astype(float).mean())
            rows.append(rec)

    out = pd.DataFrame(rows)
    if "model" in out.columns:
        out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    sort_cols = ["analysis_set"] + [c for c in group_cols if c in out.columns]
    if not out.empty:
        out = out.sort_values(sort_cols)
    out.to_csv(outpath, index=False)
    return out


def summarize_cmi_history_response(cmi_df: pd.DataFrame, n_boot: int, outdir: Path) -> pd.DataFrame:
    rows = []
    valid = cmi_df[cmi_df["valid_cmi"] == True].copy()
    for (condition, tau, seed, model), g in valid.groupby(["condition", "tau", "seed", "model"]):
        gg = g.sort_values("H")
        if len(gg) < 3:
            continue
        rho = spearman(gg["H"].to_numpy(), gg["cmi_xfit"].to_numpy())
        h0 = gg[np.isclose(gg["H"], 0.0)]
        d0 = float(h0["cmi_xfit"].iloc[0]) if not h0.empty else np.nan
        dmin = float(gg["cmi_xfit"].min())
        hmin = float(gg.loc[gg["cmi_xfit"].idxmin(), "H"])
        RH = float((d0 - dmin) / d0) if np.isfinite(d0) and d0 > 0 else np.nan
        rows.append({
            "condition": condition,
            "tau": tau,
            "seed": seed,
            "model": model,
            "rho_H": rho,
            "delta_H0": d0,
            "delta_min": dmin,
            "H_min": hmin,
            "R_H": RH,
            "monotone_decrease": bool(np.isfinite(rho) and rho < 0),
            "stable": bool(gg["stable"].iloc[0]) if "stable" in gg.columns else True,
        })

    resp = pd.DataFrame(rows)
    resp.to_csv(outdir / "tables" / "cmi_history_response_seed_level.csv", index=False)

    if resp.empty:
        out = pd.DataFrame()
        out.to_csv(outdir / "tables" / "cmi_history_response_summary.csv", index=False)
        return out

    # Summarize rho_H and R_H separately
    parts = []
    for metric in ["rho_H", "R_H"]:
        s = summarize_metric(
            resp,
            value_col=metric,
            group_cols=["condition", "model"],
            n_boot=n_boot,
            outpath=outdir / "tables" / f"_tmp_{metric}.csv",
            valid_col=None,
        )
        s["metric"] = metric
        parts.append(s)
        try:
            (outdir / "tables" / f"_tmp_{metric}.csv").unlink()
        except Exception:
            pass
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Add monotone rate
    mono_rows = []
    for analysis_set, dset in make_analysis_sets(resp):
        if dset.empty:
            continue
        for (condition, model), g in dset.groupby(["condition", "model"]):
            mono_rows.append({
                "analysis_set": analysis_set,
                "condition": condition,
                "model": model,
                "monotone_decrease_rate": float(g["monotone_decrease"].astype(float).mean()),
                "n_mono": len(g),
            })
    mono = pd.DataFrame(mono_rows)
    if not out.empty and not mono.empty:
        out = out.merge(mono, on=["analysis_set", "condition", "model"], how="left")
    out.to_csv(outdir / "tables" / "cmi_history_response_summary.csv", index=False)
    return out


def summarize_diagnostics(diag_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for condition, g in diag_df.groupby("condition"):
        rows.append({
            "condition": condition,
            "n": len(g),
            "finite_rate": float(g["finite"].astype(float).mean()),
            "stable_rate": float(g["stable"].astype(float).mean()),
            "max_abs_state_median": float(np.median(g["max_abs_state"].replace([np.inf], np.nan).dropna())) if np.any(np.isfinite(g["max_abs_state"])) else np.inf,
            "growth_ratio_median": float(np.median(g["growth_ratio"].replace([np.inf], np.nan).dropna())) if np.any(np.isfinite(g["growth_ratio"])) else np.inf,
            "max_abs_state_max": float(np.nanmax(g["max_abs_state"].replace([np.inf], np.nan))) if np.any(np.isfinite(g["max_abs_state"])) else np.inf,
            "growth_ratio_max": float(np.nanmax(g["growth_ratio"].replace([np.inf], np.nan))) if np.any(np.isfinite(g["growth_ratio"])) else np.inf,
        })
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "tables" / "stability_summary.csv", index=False)
    return out


# =============================================================================
# Plotting
# =============================================================================

def select_summary_for_plot(summary: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    if summary.empty:
        return summary
    d = summary[summary["analysis_set"] == analysis_set].copy()
    if d.empty and analysis_set == "stable_only":
        d = summary[summary["analysis_set"] == "all"].copy()
    return d


def plot_bar(
    summary: pd.DataFrame,
    outpath: Path,
    title: str,
    ylabel: str,
    analysis_set: str,
    target: Optional[str] = None,
    H: Optional[float] = None,
    condition: Optional[str] = None,
) -> None:
    df = select_summary_for_plot(summary, analysis_set)
    if df.empty:
        return
    if target is not None and "target" in df.columns:
        df = df[df["target"] == target]
    if H is not None and "H" in df.columns:
        df = df[np.isclose(df["H"], H)]
    if condition is not None and "condition" in df.columns:
        df = df[df["condition"] == condition]
    if df.empty:
        return

    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values("model")
    x = np.arange(len(df))
    mean = df["mean"].to_numpy(dtype=float)
    lo = df["ci95_mean_low"].to_numpy(dtype=float)
    hi = df["ci95_mean_high"].to_numpy(dtype=float)
    yerr = np.vstack([mean - lo, hi - mean])
    yerr[~np.isfinite(yerr)] = 0

    fig, ax = plt.subplots(figsize=(8.5, 5.1))
    ax.bar(x, mean)
    ax.errorbar(x, mean, yerr=yerr, fmt="none", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[str(m)] for m in df["model"].astype(str)], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({analysis_set})")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_hgrid_cmi(
    cmi_summary: pd.DataFrame,
    outpath: Path,
    analysis_set: str,
    condition: Optional[str] = None,
) -> None:
    df = select_summary_for_plot(cmi_summary, analysis_set)
    if condition is not None and "condition" in df.columns:
        df = df[df["condition"] == condition]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for model in MODEL_ORDER:
        g = df[df["model"] == model].sort_values("H")
        if g.empty:
            continue
        mean = g["mean"].to_numpy(dtype=float)
        lo = g["ci95_mean_low"].to_numpy(dtype=float)
        hi = g["ci95_mean_high"].to_numpy(dtype=float)
        yerr = np.vstack([mean - lo, hi - mean])
        yerr[~np.isfinite(yerr)] = 0
        ax.errorbar(g["H"], mean, yerr=yerr, marker="o", capsize=3, label=MODEL_LABEL[model])
    ax.set_xlabel("History depth H")
    ax.set_ylabel("Cross-fitted residual Gaussian CMI")
    ax.set_title(f"H-grid residual dependence ({analysis_set})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_future_active(
    pred_summary: pd.DataFrame,
    outpath: Path,
    analysis_set: str,
    condition: Optional[str] = None,
) -> None:
    df = select_summary_for_plot(pred_summary, analysis_set)
    if condition is not None and "condition" in df.columns:
        df = df[df["condition"] == condition]
    if df.empty:
        return
    rows = []
    for target in sorted(df["target"].unique()):
        if str(target).startswith("A_future_"):
            q = float(str(target).replace("A_future_", ""))
            for _, r in df[df["target"] == target].iterrows():
                rows.append({
                    "q": q,
                    "model": r["model"],
                    "mean": r["mean"],
                    "ci95_mean_low": r["ci95_mean_low"],
                    "ci95_mean_high": r["ci95_mean_high"],
                })
    d = pd.DataFrame(rows)
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    for model in MODEL_ORDER:
        g = d[d["model"] == model].sort_values("q")
        if g.empty:
            continue
        mean = g["mean"].to_numpy(dtype=float)
        lo = g["ci95_mean_low"].to_numpy(dtype=float)
        hi = g["ci95_mean_high"].to_numpy(dtype=float)
        yerr = np.vstack([mean - lo, hi - mean])
        yerr[~np.isfinite(yerr)] = 0
        ax.errorbar(g["q"], mean, yerr=yerr, marker="o", capsize=3, label=MODEL_LABEL[model])
    ax.set_xlabel("Prediction horizon q")
    ax.set_ylabel("NMSE")
    ax.set_title(f"Future active-boundary prediction ({analysis_set})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# =============================================================================
# Report
# =============================================================================

def write_report(
    outdir: Path,
    args: argparse.Namespace,
    params_by_condition: Dict[str, Params],
    stability: pd.DataFrame,
    cmi_primary: pd.DataFrame,
    cmi_response: pd.DataFrame,
    pred_primary: pd.DataFrame,
) -> None:
    def lines_for_cmi(analysis_set: str, condition: str) -> str:
        df = cmi_primary[(cmi_primary["analysis_set"] == analysis_set) & (cmi_primary["condition"] == condition)].copy()
        if df.empty:
            return "No rows."
        out = []
        for model in MODEL_ORDER:
            g = df[df["model"] == model]
            if not g.empty:
                r = g.iloc[0]
                out.append(
                    f"- {MODEL_LABEL[model]}: {r['mean']:.6g} "
                    f"[{r['ci95_mean_low']:.6g}, {r['ci95_mean_high']:.6g}], n={int(r['n'])}"
                )
        return "\n".join(out)

    def lines_for_pred(analysis_set: str, condition: str) -> str:
        df = pred_primary[(pred_primary["analysis_set"] == analysis_set) & (pred_primary["condition"] == condition)].copy()
        if df.empty:
            return "No rows."
        out = []
        for target in ["E_current", "I_current", "S_delay", "A_future_1.6"]:
            g = df[df["target"] == target].sort_values("mean")
            if not g.empty:
                best = g.iloc[0]
                inst = g[g["model"] == "instantaneous"]
                inst_val = float(inst["mean"].iloc[0]) if not inst.empty else np.nan
                out.append(
                    f"- {target}: best={MODEL_LABEL[str(best['model'])]} "
                    f"(NMSE={best['mean']:.6g}); instantaneous={inst_val:.6g}"
                )
        return "\n".join(out)

    report = f"""# RESULTS REPORT

Generated: {datetime.now().isoformat()}

## Analysis principle

This run is designed to avoid three reviewer-critical problems:

1. Unequal comparison models.
2. In-sample residual-dependence reduction by high-dimensional conditioning.
3. Choosing a model because it gives the desired result.

The comparison structure is fixed before analysis. All six models share the same separated boundary base {{S_n, A_n}}. Prediction uses held-out NMSE. Residual dependence uses cross-fitted residual Gaussian CMI on held-out data.

## Generative model

The dynamics are the separated sensory/active model proposed in the second manuscript:

    E -> S -> I -> A -> E
    delay: S(t - tau) -> I(t)

No mixed boundary variable is used.

## Run settings

```json
{json.dumps(vars(args), indent=2)}
```

## Parameters by condition

```json
{json.dumps({k: asdict(v) for k, v in params_by_condition.items()}, indent=2)}
```

## Stability summary

{markdown_table_no_tabulate(stability, max_rows=30) if not stability.empty else "No stability summary."}

## Primary-H CMI summary, all trajectories

{lines_for_cmi("all", list(params_by_condition.keys())[0])}

## Primary-H CMI summary, stable-only trajectories

{lines_for_cmi("stable_only", list(params_by_condition.keys())[0])}

## Primary-H prediction summary, all trajectories

{lines_for_pred("all", list(params_by_condition.keys())[0])}

## Primary-H prediction summary, stable-only trajectories

{lines_for_pred("stable_only", list(params_by_condition.keys())[0])}

## Main output tables

- stability_seed_level.csv
- stability_summary.csv
- cmi_hgrid_seed_level.csv
- cmi_hgrid_summary.csv
- cmi_primaryH_summary.csv
- cmi_history_response_seed_level.csv
- cmi_history_response_summary.csv
- prediction_primaryH_seed_level.csv
- prediction_primaryH_summary.csv
- prediction_primaryH_paired_vs_instantaneous_seed_level.csv
- prediction_hgrid_key_seed_level.csv
- prediction_hgrid_key_summary.csv

## Main output figures

Both all-trajectory and stable-only versions are generated when possible.

- fig1_current_external_inference_*.
- fig2A_residual_cmi_primaryH_*.
- fig2B_residual_cmi_hgrid_*.
- fig3A_internal_reconstruction_*.
- fig3B_delay_aligned_sensory_*.
- fig4_future_active_prediction_*.

## Interpretation rule

If the linear condition is unstable, the instability is a result, not a reason to hide the condition. Stable-only summaries are provided only as a pre-specified diagnostic subset. If the tanh condition is run, it is a robustness condition, not a replacement chosen post hoc.
"""
    with open(outdir / "RESULTS_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)


# =============================================================================
# Main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "balanced", "full"], default="smoke")
    p.add_argument("--outdir", default=None)
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))

    p.add_argument("--condition", choices=["linear", "tanh", "both"], default="linear")
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-sur", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--taus", default="0,0.05,0.1,0.2,0.4,0.8,1.2,1.6,2.0")
    p.add_argument("--H-grid", default="0,0.05,0.1,0.2,0.4,0.8,1.2,1.6,2.0,3.2,4.0")
    p.add_argument("--primary-H", type=float, default=4.0)
    p.add_argument("--future-horizons", default="0.2,0.8,1.6")
    p.add_argument("--seed-base", type=int, default=881000)

    # sample caps control runtime; 0 uses all samples
    p.add_argument("--max-fit-samples", type=int, default=30000)
    p.add_argument("--max-val-samples", type=int, default=15000)
    p.add_argument("--max-test-samples", type=int, default=30000)

    # stability criteria are pre-specified and reported, not silently used
    p.add_argument("--stable-threshold", type=float, default=1e6)
    p.add_argument("--growth-threshold", type=float, default=1e4)

    p.add_argument("--run-hgrid-key-prediction", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "smoke":
        args.n_seeds = min(args.n_seeds, 2)
        args.n_sur = min(args.n_sur, 20)
        args.n_boot = min(args.n_boot, 100)
        args.taus = "0,0.4"
        args.H_grid = "0,0.1,0.4"
        args.primary_H = 0.4
        args.max_fit_samples = min(args.max_fit_samples, 3000)
        args.max_val_samples = min(args.max_val_samples, 1000)
        args.max_test_samples = min(args.max_test_samples, 3000)
    elif args.mode == "balanced":
        args.n_seeds = min(args.n_seeds, 30)
        args.n_sur = min(args.n_sur, 300)
        args.n_boot = min(args.n_boot, 1000)
        args.max_fit_samples = min(args.max_fit_samples, 20000)
        args.max_val_samples = min(args.max_val_samples, 10000)
        args.max_test_samples = min(args.max_test_samples, 20000)

    outdir = make_outdir(args.outdir)
    logger = Logger(outdir)
    logger.section("Reviewer-safe fair reanalysis")

    condition_names = ["linear", "tanh"] if args.condition == "both" else [args.condition]
    params_by_condition = {name: Params(response=name) for name in condition_names}
    if args.mode == "smoke":
        params_by_condition = {name: replace(p, total_time=80.0) for name, p in params_by_condition.items()}

    taus = parse_float_list(args.taus)
    H_grid = parse_float_list(args.H_grid)
    if 0.0 not in H_grid:
        H_grid = sorted([0.0] + H_grid)
    if not any(np.isclose(args.primary_H, h) for h in H_grid):
        H_grid = sorted(set(H_grid + [args.primary_H]))
    future_horizons = parse_float_list(args.future_horizons)
    alpha_grid = parse_float_list(next(iter(params_by_condition.values())).pred_alpha_grid)

    config = {
        "args": vars(args),
        "params_by_condition": {k: asdict(v) for k, v in params_by_condition.items()},
        "taus": taus,
        "H_grid": H_grid,
        "future_horizons": future_horizons,
        "alpha_grid": alpha_grid,
        "models": MODEL_ORDER,
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.msg(f"Output directory: {outdir}")
    logger.msg(f"Mode: {args.mode}")
    logger.msg(f"Conditions: {condition_names}")
    logger.msg(f"Workers: {args.workers}")
    logger.msg(f"Seeds per tau-condition: {args.n_seeds}")
    logger.msg(f"Surrogates: {args.n_sur}")
    logger.msg(f"H_grid: {H_grid}")
    logger.msg(f"Primary H: {args.primary_H}")
    logger.msg(f"Sample caps fit/val/test: {args.max_fit_samples}/{args.max_val_samples}/{args.max_test_samples}")
    logger.msg(f"Stability threshold: {args.stable_threshold}; growth threshold: {args.growth_threshold}")

    jobs = []
    for ci, (condition, params) in enumerate(params_by_condition.items()):
        for tau in taus:
            for si in range(args.n_seeds):
                seed = args.seed_base + ci * 100_000_000 + int(round(tau * 1000)) * 1000 + si
                jobs.append({
                    "condition": condition,
                    "params": params,
                    "tau": tau,
                    "seed": seed,
                    "H_grid": H_grid,
                    "primary_H": args.primary_H,
                    "n_sur": args.n_sur,
                    "future_horizons": future_horizons,
                    "max_fit_samples": args.max_fit_samples,
                    "max_val_samples": args.max_val_samples,
                    "max_test_samples": args.max_test_samples,
                    "stable_threshold": args.stable_threshold,
                    "growth_threshold": args.growth_threshold,
                    "alpha_grid": alpha_grid,
                    "run_hgrid_key_prediction": args.run_hgrid_key_prediction,
                })

    logger.section(f"Running {len(jobs)} trajectory jobs")
    diag_rows: List[Dict[str, Any]] = []
    cmi_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    hpred_rows: List[Dict[str, Any]] = []

    done = 0
    if args.workers <= 1:
        for job in jobs:
            res = run_one(job)
            diag_rows.append(res["diagnostics"])
            cmi_rows.extend(res["cmi"])
            pred_rows.extend(res["prediction_primary"])
            hpred_rows.extend(res["prediction_hgrid_key"])
            done += 1
            if done == 1 or done % max(1, len(jobs)//20) == 0 or done == len(jobs):
                logger.msg(f"Progress {done}/{len(jobs)}")
    else:
        with futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, job) for job in jobs]
            for fut in futures.as_completed(futs):
                res = fut.result()
                diag_rows.append(res["diagnostics"])
                cmi_rows.extend(res["cmi"])
                pred_rows.extend(res["prediction_primary"])
                hpred_rows.extend(res["prediction_hgrid_key"])
                done += 1
                if done == 1 or done % max(1, len(jobs)//20) == 0 or done == len(jobs):
                    logger.msg(f"Progress {done}/{len(jobs)}")

    logger.section("Saving seed-level tables")
    diag_df = pd.DataFrame(diag_rows)
    cmi_df = pd.DataFrame(cmi_rows)
    pred_df = pd.DataFrame(pred_rows)
    hpred_df = pd.DataFrame(hpred_rows)

    diag_df.to_csv(outdir / "tables" / "stability_seed_level.csv", index=False)
    cmi_df.to_csv(outdir / "tables" / "cmi_hgrid_seed_level.csv", index=False)
    pred_df.to_csv(outdir / "tables" / "prediction_primaryH_seed_level.csv", index=False)
    hpred_df.to_csv(outdir / "tables" / "prediction_hgrid_key_seed_level.csv", index=False)

    logger.msg(f"diagnostics rows: {len(diag_df)}")
    logger.msg(f"CMI rows: {len(cmi_df)}")
    logger.msg(f"primary prediction rows: {len(pred_df)}")
    logger.msg(f"H-grid prediction rows: {len(hpred_df)}")

    logger.section("Summaries")
    stability_summary = summarize_diagnostics(diag_df, outdir)

    cmi_hgrid_summary = summarize_metric(
        cmi_df, "cmi_xfit",
        group_cols=["condition", "H", "model"],
        n_boot=args.n_boot,
        outpath=outdir / "tables" / "cmi_hgrid_summary.csv",
        valid_col="valid_cmi",
    )
    cmi_primary_df = cmi_df[np.isclose(cmi_df["H"], args.primary_H)].copy()
    cmi_primary_summary = summarize_metric(
        cmi_primary_df, "cmi_xfit",
        group_cols=["condition", "H", "model"],
        n_boot=args.n_boot,
        outpath=outdir / "tables" / "cmi_primaryH_summary.csv",
        valid_col="valid_cmi",
    )
    cmi_response_summary = summarize_cmi_history_response(cmi_df, args.n_boot, outdir)

    pred_primary_summary = summarize_metric(
        pred_df, "nmse",
        group_cols=["condition", "H", "target", "model"],
        n_boot=args.n_boot,
        outpath=outdir / "tables" / "prediction_primaryH_summary.csv",
        valid_col="valid_pred",
    )
    hpred_summary = summarize_metric(
        hpred_df, "nmse",
        group_cols=["condition", "H", "target", "model"],
        n_boot=args.n_boot,
        outpath=outdir / "tables" / "prediction_hgrid_key_summary.csv",
        valid_col="valid_pred",
    )

    # Paired differences for prediction
    key_cols = ["condition", "tau", "H", "seed", "target"]
    if not pred_df.empty:
        inst = pred_df[pred_df["model"] == "instantaneous"][key_cols + ["nmse"]].rename(columns={"nmse": "nmse_inst"})
        m = pred_df.merge(inst, on=key_cols, how="left")
        m["diff_vs_instantaneous"] = m["nmse"] - m["nmse_inst"]
        m[m["model"] != "instantaneous"].to_csv(outdir / "tables" / "prediction_primaryH_paired_vs_instantaneous_seed_level.csv", index=False)
    if not hpred_df.empty:
        inst = hpred_df[hpred_df["model"] == "instantaneous"][key_cols + ["nmse"]].rename(columns={"nmse": "nmse_inst"})
        m = hpred_df.merge(inst, on=key_cols, how="left")
        m["diff_vs_instantaneous"] = m["nmse"] - m["nmse_inst"]
        m[m["model"] != "instantaneous"].to_csv(outdir / "tables" / "prediction_hgrid_key_paired_vs_instantaneous_seed_level.csv", index=False)

    logger.msg("Saved summary tables")

    logger.section("Figures")
    for primary_condition in condition_names:
        for analysis_set in ["all", "stable_only"]:
            suffix = f"{primary_condition}_{analysis_set}"
            plot_bar(
                pred_primary_summary,
                outdir / "figures" / f"fig1_current_external_inference_{suffix}.png",
                "Current external-state inference",
                "NMSE",
                analysis_set,
                target="E_current",
                H=args.primary_H,
                condition=primary_condition,
            )
            plot_bar(
                cmi_primary_summary,
                outdir / "figures" / f"fig2A_residual_cmi_primaryH_{suffix}.png",
                "Cross-fitted residual internal-external CMI at primary H",
                "Residual Gaussian CMI",
                analysis_set,
                H=args.primary_H,
                condition=primary_condition,
            )
            plot_hgrid_cmi(
                cmi_hgrid_summary,
                outdir / "figures" / f"fig2B_residual_cmi_hgrid_{suffix}.png",
                analysis_set,
                condition=primary_condition,
            )
            plot_bar(
                pred_primary_summary,
                outdir / "figures" / f"fig3A_internal_reconstruction_{suffix}.png",
                "Current internal-state reconstruction",
                "NMSE",
                analysis_set,
                target="I_current",
                H=args.primary_H,
                condition=primary_condition,
            )
            plot_bar(
                pred_primary_summary,
                outdir / "figures" / f"fig3B_delay_aligned_sensory_{suffix}.png",
                "Delay-aligned sensory reconstruction",
                "NMSE",
                analysis_set,
                target="S_delay",
                H=args.primary_H,
                condition=primary_condition,
            )
            plot_future_active(
                pred_primary_summary,
                outdir / "figures" / f"fig4_future_active_prediction_{suffix}.png",
                analysis_set,
                condition=primary_condition,
            )
    logger.msg("Saved figures")

    logger.section("Report")
    write_report(outdir, args, params_by_condition, stability_summary, cmi_primary_summary, cmi_response_summary, pred_primary_summary)
    logger.msg(f"Saved report: {outdir / 'RESULTS_REPORT.md'}")

    logger.section("Finished")
    logger.msg(f"Results directory: {outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FATAL ERROR:", str(exc), file=sys.stderr)
        traceback.print_exc()
        raise
