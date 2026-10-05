# -*- coding: utf-8 -*-
"""
gaf_encoding.py
===============

Core functions to encode a time series as a Gramian Angular (Summation)
Field (GAF), following Wang & Oates (2015):

    x  ->  rescale to [-1, 1]  ->  φ = arccos(x̃)  ->  G_ij = cos(φ_i + φ_j)

plus the sliding-window pipeline used for the Dow Jones analysis and the
two scalar descriptors Ḡ (mean) and Var(G) (variance) of each GAF matrix.

No classes are used; every step is a documented function.

Reference
---------
Wang, Z. & Oates, T. (2015). Encoding time series as images for visual
inspection and classification using tiled convolutional neural networks.
AAAI Workshop on Neural Networks for Time Series Analysis.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. RESCALING  (eq. 1 of Wang & Oates 2015)
# ─────────────────────────────────────────────────────────────────────────────

def rescale_to_minus1_plus1(x: np.ndarray) -> np.ndarray:
    """
    Rescale a 1-D time series to the interval [-1, 1] using eq. (1) of
    Wang & Oates (2015):

        x̃_i = [ (x_i - max(X)) + (x_i - min(X)) ] / (max(X) - min(X))

    This is algebraically equivalent to x̃_i = 2 (x_i - min) / (max - min) - 1.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Raw time series.

    Returns
    -------
    x_scaled : np.ndarray, shape (n,)
        Rescaled series with values in [-1, 1].

    Raises
    ------
    ValueError
        If the series is constant (max == min); rescaling is undefined.
    """
    x_min = np.min(x)
    x_max = np.max(x)
    if np.isclose(x_min, x_max):
        raise ValueError("Cannot rescale a constant time series (max == min).")
    return ((x - x_max) + (x - x_min)) / (x_max - x_min)


# ─────────────────────────────────────────────────────────────────────────────
# 2. POLAR COORDINATE ENCODING  (eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

def encode_polar(x_scaled: np.ndarray) -> tuple:
    """
    Map a rescaled time series x̃ ∈ [-1, 1] to polar coordinates:

        φ_i = arccos(x̃_i)  ∈ [0, π]
        r_i = t_i / N       ∈ (0, 1]

    where t_i is the 1-based time stamp and N = len(x_scaled). The encoding
    is bijective because cos is monotone on [0, π], and the radius keeps the
    temporal ordering.

    Parameters
    ----------
    x_scaled : np.ndarray, shape (n,)
        Rescaled time series with values in [-1, 1].

    Returns
    -------
    phi : np.ndarray, shape (n,)
        Angular values in [0, π].
    r : np.ndarray, shape (n,)
        Radial values in (0, 1].
    """
    x_clipped = np.clip(x_scaled, -1.0, 1.0)  # guard against round-off
    phi = np.arccos(x_clipped)
    N = len(x_scaled)
    r = np.arange(1, N + 1, dtype=float) / N
    return phi, r


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAMIAN ANGULAR FIELD  (eqs. 3-4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gaf(phi: np.ndarray) -> np.ndarray:
    """
    Compute the GAF matrix from the angles φ:

        G_ij = cos(φ_i + φ_j) = x̃_i x̃_j - √(1 - x̃_i²) √(1 - x̃_j²)

    The main diagonal G_ii = cos(2φ_i) encodes the (squared) rescaled
    values and allows an approximate, sign-less reconstruction of x̃.

    Parameters
    ----------
    phi : np.ndarray, shape (n,)
        Angles produced by encode_polar().

    Returns
    -------
    G : np.ndarray, shape (n, n)
        Symmetric GAF matrix with values in [-1, 1].
    """
    return np.cos(phi[:, np.newaxis] + phi[np.newaxis, :])


def gaf_descriptors(gaf: np.ndarray) -> tuple:
    """
    Scalar descriptors of a GAF matrix:

        Ḡ      = (1/n²) Σ_ij G_ij
        Var(G) = (1/n²) Σ_ij (G_ij - Ḡ)²

    Parameters
    ----------
    gaf : np.ndarray, shape (n, n)
        GAF matrix.

    Returns
    -------
    g_avg : float
        Mean of all entries.
    g_var : float
        Variance of all entries.
    """
    return float(np.mean(gaf)), float(np.var(gaf))


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTIONAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def subsample_series(x: np.ndarray, target_length: int) -> np.ndarray:
    """
    Reduce a series to `target_length` points by uniform subsampling.

    This is NOT PAA (Piecewise Aggregate Approximation): it does not average
    within windows. It is a simple way to compare GAF resolutions without
    introducing smoothing.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Original series.
    target_length : int
        Desired length (must be ≤ len(x)).

    Returns
    -------
    np.ndarray, shape (target_length,)
        Subsampled series.

    Raises
    ------
    ValueError
        If target_length > len(x).
    """
    if target_length > len(x):
        raise ValueError(f"target_length ({target_length}) cannot exceed "
                         f"series length ({len(x)}).")
    indices = np.linspace(0, len(x) - 1, target_length, dtype=int)
    return x[indices]


def time_series_to_gaf(x: np.ndarray, target_length: int | None = None) -> dict:
    """
    Full pipeline for one series: (optional subsampling) → rescale → polar
    → GAF.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Raw input series.
    target_length : int or None
        If given, subsample x to this length first.

    Returns
    -------
    dict with keys 'x_raw', 'x_sub', 'x_scaled', 'phi', 'r', 'gaf',
    'g_avg', 'g_var'.
    """
    x_sub = subsample_series(x, target_length) if target_length else x.copy()
    x_scaled = rescale_to_minus1_plus1(x_sub)
    phi, r = encode_polar(x_scaled)
    gaf = compute_gaf(phi)
    g_avg, g_var = gaf_descriptors(gaf)
    return {"x_raw": x, "x_sub": x_sub, "x_scaled": x_scaled, "phi": phi,
            "r": r, "gaf": gaf, "g_avg": g_avg, "g_var": g_var}


def reconstruct_from_gaf_diagonal(gaf: np.ndarray) -> np.ndarray:
    """
    Approximate reconstruction from the main diagonal, eq. (8):

        |x̃_i| = √( (G_ii + 1) / 2 )

    Only the magnitude is recovered (sign ambiguity of arccos).

    Parameters
    ----------
    gaf : np.ndarray, shape (n, n)
        GAF matrix.

    Returns
    -------
    np.ndarray, shape (n,)
        Reconstructed |x̃| in [0, 1].
    """
    diag = np.clip(np.diag(gaf), -1.0, 1.0)
    return np.sqrt((diag + 1.0) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SLIDING-WINDOW PIPELINE  (used for the DJI)
# ─────────────────────────────────────────────────────────────────────────────

def sliding_window_gafs(df: pd.DataFrame,
                        value_col: str = "DJI_log_return",
                        date_col: str = "Date",
                        n: int = 64,
                        m: int = 32,
                        verbose: bool = True) -> dict:
    """
    Build one GAF per sliding window of length `n` with stride `m`
    (windows overlap by n - m points). Each window is rescaled to [-1, 1]
    independently (window-level rescaling).

    Window k covers rows [k*m, k*m + n) of `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `date_col` (Timestamps, ascending) and `value_col`
        (float, no NaN).
    value_col : str
        Column to encode (e.g. log-returns).
    date_col : str
        Date column.
    n : int
        Window size (GAF is n x n).
    m : int
        Stride between consecutive windows.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with parallel entries (one per window):
        'gafs'                 : np.ndarray (num_windows, n, n)
        'phis', 'rs'           : np.ndarray (num_windows, n)
        'ventanas_originales'  : np.ndarray (num_windows, n), raw values
        'fechas_inicio'        : list of Timestamps, first date of window
        'fechas_fin'           : list of Timestamps, last date of window
        'G_avg', 'G_var'       : list of float descriptors
    """
    values = df[value_col].to_numpy()
    dates = df[date_col].to_list()
    num_windows = ((len(df) - n) // m) + 1
    if verbose:
        print(f"Total windows: {num_windows}  (n = {n}, stride m = {m})")

    res = {k: [] for k in ("gafs", "phis", "rs", "ventanas_originales",
                           "fechas_inicio", "fechas_fin", "G_avg", "G_var")}

    for i in range(num_windows):
        start, end = i * m, i * m + n
        window = values[start:end]

        x_scaled = rescale_to_minus1_plus1(window)
        phi, r = encode_polar(x_scaled)
        gaf = compute_gaf(phi)
        g_avg, g_var = gaf_descriptors(gaf)

        res["gafs"].append(gaf)
        res["phis"].append(phi)
        res["rs"].append(r)
        res["ventanas_originales"].append(window)
        res["fechas_inicio"].append(dates[start])
        res["fechas_fin"].append(dates[end - 1])
        res["G_avg"].append(g_avg)
        res["G_var"].append(g_var)

        if verbose and (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{num_windows} windows...")

    for k in ("gafs", "phis", "rs", "ventanas_originales"):
        res[k] = np.array(res[k])
    return res


def find_window_index(resultados: dict, date: str,
                      key: str = "fechas_inicio") -> int:
    """
    Index of the window whose `key` date ('fechas_inicio' or 'fechas_fin')
    equals `date` (day precision).

    Parameters
    ----------
    resultados : dict
        Output of sliding_window_gafs().
    date : str
        Target date, 'YYYY-MM-DD'.
    key : str
        'fechas_inicio' (start dates) or 'fechas_fin' (end dates).

    Returns
    -------
    int
        Window index.

    Raises
    ------
    ValueError
        If no window matches.
    """
    target = pd.Timestamp(date).normalize()
    for idx, ts in enumerate(resultados[key]):
        if pd.Timestamp(ts).normalize() == target:
            return idx
    raise ValueError(
        f"No window with {key} = '{date}'. Available range: "
        f"{pd.Timestamp(resultados[key][0]).date()} to "
        f"{pd.Timestamp(resultados[key][-1]).date()}.")


def nearest_window_index(resultados: dict, date: str,
                         key: str = "fechas_fin") -> int:
    """
    Index of the window whose `key` date is closest to `date`.

    Used to locate the critical date t_c among window END dates, as in the
    original notebook.

    Parameters
    ----------
    resultados : dict
        Output of sliding_window_gafs().
    date : str
        Target date, 'YYYY-MM-DD'.
    key : str
        'fechas_fin' (default) or 'fechas_inicio'.

    Returns
    -------
    int
        Index of the nearest window.
    """
    target = pd.Timestamp(date).value
    stamps = np.array([pd.Timestamp(f).value for f in resultados[key]])
    return int(np.argmin(np.abs(stamps - target)))
