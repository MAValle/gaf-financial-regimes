# -*- coding: utf-8 -*-
"""
dji_gaf_analysis.py
===================

Reproduces the figures of the report

    "Gramian Angular Fields as Financial Regime Detectors:
     Evidence from the 1929 Wall Street Crash"  (M. A. Valle)

from daily Dow Jones Industrial Average (DJI) closing values:

    Fig. 1  DJI level, 1921-1932
    Fig. 1b DJI log-returns, 1921-1932
    Fig. 2  30 GAFs evenly spaced over 1922-1932
    Fig. 3  12 GAFs before the critical date t_c = 1929-09-03
    Fig. 4  12 GAFs from t_c onwards
    Fig. 5  Ḡ and Var(G) over the whole sample and zoomed on 1920-1940
    Fig. 6  Selected GAFs next to the log-returns of the same window

Pipeline: log-returns → sliding windows (n = 64, stride m = 32) →
window-level rescaling to [-1, 1] → GAF → descriptors Ḡ and Var(G).

Date conventions (important when reading the figures)
-----------------------------------------------------
* Figs. 3 and 4 are titled with the window END date. t_c is located as the
  window whose END date is closest to 1929-09-03; Fig. 3 shows the 12
  windows before it, Fig. 4 that window and the 11 after it.
* Figs. 2, 5 and 6 use the window START date.

Usage
-----
    python dji_gaf_analysis.py --data data/dji_sp_1900_2026.csv
    python dji_gaf_analysis.py --data data/dji_sp_1900_2026.csv --show

Expected CSV format: see README.md (columns `Date` as MM/DD/YYYY and
`DJI` closing value; any row order).

No classes are used; everything is a documented function.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gaf_encoding import (sliding_window_gafs, find_window_index,
                          nearest_window_index)

TC_DEFAULT = "1929-09-03"   # pre-crash peak of the DJI (381.17)
CMAP = "RdBu_r"             # blue = -1, white = 0, red = +1


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_dji(path: str, date_format: str = "%m/%d/%Y",
             price_col: str = "DJI") -> pd.DataFrame:
    """
    Load the DJI CSV, sort it chronologically and add log-returns.

        r_t = log(P_t / P_{t-1})

    Parameters
    ----------
    path : str
        Path to the CSV file.
    date_format : str
        strftime format of the `Date` column.
    price_col : str
        Name of the closing-value column.

    Returns
    -------
    pd.DataFrame
        Columns 'Date', price_col, 'DJI_log_return' (first row dropped,
        since it has no return), ascending in time.
    """
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()          # stray spaces in headers
    data["Date"] = pd.to_datetime(data["Date"], format=date_format)
    data = data.sort_values("Date").reset_index(drop=True)
    data["DJI_log_return"] = np.log(data[price_col]).diff()
    data = data.dropna(subset=["DJI_log_return"]).reset_index(drop=True)
    print(f"Loaded {len(data)} daily returns from "
          f"{data['Date'].iloc[0].date()} to {data['Date'].iloc[-1].date()}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, outdir: str, name: str) -> None:
    """Save `fig` as outdir/name at 200 dpi and report the path."""
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")


def plot_dji_series(data: pd.DataFrame, start: str = "1921-02-01",
                    end: str = "1932-07-31", tc: str = TC_DEFAULT,
                    price_col: str = "DJI") -> tuple:
    """
    Fig. 1: DJI level and log-returns over [start, end], with t_c marked.

    Parameters
    ----------
    data : pd.DataFrame
        Output of load_dji().
    start, end : str
        Date range 'YYYY-MM-DD'.
    tc : str
        Critical date drawn as a dashed vertical line.
    price_col : str
        Closing-value column.

    Returns
    -------
    (fig_level, fig_returns) : tuple of matplotlib Figures
    """
    sub = data[(data["Date"] >= start) & (data["Date"] <= end)]
    tc_ts = pd.Timestamp(tc)

    fig1, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sub["Date"], sub[price_col], lw=1, color="#1f4e8c")
    ax.axvline(tc_ts, color="purple", ls="--", lw=1, alpha=0.7)
    ax.set_title("Dow Jones Industrial Average", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("DJI closing value")
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()

    fig2, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sub["Date"], sub["DJI_log_return"], lw=0.8, color="#2e7d32")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(tc_ts, color="purple", ls="--", lw=1, alpha=0.7)
    ax.set_title(f"DJI log-returns ({start[:4]}-{end[:4]})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-return")
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    return fig1, fig2


def plot_gaf_grid(resultados: dict, start: str = "1922-01-01",
                  end: str = "1932-12-31", num_gafs: int = 30,
                  n_cols: int = 6) -> plt.Figure:
    """
    Fig. 2: `num_gafs` GAFs evenly spaced among windows whose START date
    falls in [start, end]. Titles are start dates.

    Note: colours are fixed to [-1, 1] so every panel shares one scale
    (the original notebook let each panel auto-scale).

    Parameters
    ----------
    resultados : dict
        Output of sliding_window_gafs().
    start, end : str
        Range for window start dates.
    num_gafs : int
        Number of panels.
    n_cols : int
        Panels per row.

    Returns
    -------
    matplotlib.figure.Figure
    """
    idx_period = [i for i, f in enumerate(resultados["fechas_inicio"])
                  if pd.Timestamp(start) <= f <= pd.Timestamp(end)]
    if len(idx_period) >= num_gafs:
        pick = np.linspace(0, len(idx_period) - 1, num_gafs, dtype=int)
        idx_sel = [idx_period[i] for i in pick]
    else:
        idx_sel = idx_period
        print(f"  warning: only {len(idx_period)} windows in period")

    n_rows = int(np.ceil(len(idx_sel) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, idx_sel):
        ax.imshow(resultados["gafs"][k], cmap=CMAP, vmin=-1, vmax=1,
                  aspect="equal")
        ax.set_title(resultados["fechas_inicio"][k].strftime("%Y-%m-%d"),
                     fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    for ax in axes[len(idx_sel):]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.15, left=0.02, right=0.98,
                        top=0.98, bottom=0.02)
    return fig


def plot_gafs_around_tc(resultados: dict, tc: str = TC_DEFAULT,
                        num_snapshots: int = 12, side: str = "before",
                        n_cols: int = 4) -> plt.Figure:
    """
    Figs. 3-4: GAFs just before, or from, the critical date.

    The critical window is the one whose END date is closest to `tc`.
    side = "before": the `num_snapshots` windows preceding it.
    side = "after" : the critical window and the following ones.
    Titles show the window END date, Ḡ and Var(G).

    Parameters
    ----------
    resultados : dict
        Output of sliding_window_gafs().
    tc : str
        Critical date 'YYYY-MM-DD'.
    num_snapshots : int
        Number of panels.
    side : {"before", "after"}
        Which side of t_c.
    n_cols : int
        Panels per row.

    Returns
    -------
    matplotlib.figure.Figure
    """
    i_c = nearest_window_index(resultados, tc, key="fechas_fin")
    total = len(resultados["fechas_fin"])
    if side == "before":
        idx = list(range(max(0, i_c - num_snapshots), i_c))
    elif side == "after":
        idx = list(range(i_c, min(total, i_c + num_snapshots)))
    else:
        raise ValueError("side must be 'before' or 'after'.")
    print(f"  {side} t_c: critical window ends "
          f"{resultados['fechas_fin'][i_c].date()}, {len(idx)} panels")

    n_rows = int(np.ceil(len(idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, idx):
        ax.imshow(resultados["gafs"][k], cmap=CMAP, vmin=-1, vmax=1,
                  aspect="equal")
        ax.set_title(f"end {resultados['fechas_fin'][k]:%Y-%m-%d}\n"
                     f"Ḡ={resultados['G_avg'][k]:.4f}, "
                     f"Var={resultados['G_var'][k]:.4f}",
                     fontsize=9, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("darkgray")
            spine.set_linewidth(0.5)
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.08, hspace=0.25)
    return fig


def plot_descriptor_series(resultados: dict, tc: str = TC_DEFAULT,
                           zoom: tuple = ("1920-01-01", "1940-12-31")
                           ) -> plt.Figure:
    """
    Fig. 5: Ḡ and Var(G) against window START date, for the full sample
    (left) and a zoom around the crash (right). Each descriptor gets its
    own panel and y-axis (no shared dual axis).

    Parameters
    ----------
    resultados : dict
        Output of sliding_window_gafs().
    tc : str
        Critical date drawn as a dashed vertical line.
    zoom : tuple of str
        (start, end) of the zoomed panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dates = np.array(resultados["fechas_inicio"])
    g_avg = np.array(resultados["G_avg"])
    g_var = np.array(resultados["G_var"])
    tc_ts = pd.Timestamp(tc)
    mask = (dates >= pd.Timestamp(zoom[0])) & (dates <= pd.Timestamp(zoom[1]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col",
                             gridspec_kw={"width_ratios": [1.6, 1]})
    panels = [(axes[0, 0], dates, g_avg, "Ḡ", "#1f4e8c"),
              (axes[1, 0], dates, g_var, "Var(G)", "#b23a3a"),
              (axes[0, 1], dates[mask], g_avg[mask], "Ḡ", "#1f4e8c"),
              (axes[1, 1], dates[mask], g_var[mask], "Var(G)", "#b23a3a")]
    for ax, x, y, label, color in panels:
        ax.plot(x, y, color=color, lw=1)
        ax.axvline(tc_ts, color="purple", ls="--", lw=1.2, alpha=0.7)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title("Full sample", fontsize=12, fontweight="bold")
    axes[0, 1].set_title(f"Zoom {zoom[0][:4]}-{zoom[1][:4]} "
                         f"(dashed line: t_c = {tc})",
                         fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Window start date")
    axes[1, 1].set_xlabel("Window start date")
    fig.tight_layout()
    return fig


def plot_gaf_with_returns(start_dates: list, resultados: dict,
                          df: pd.DataFrame, tc: str = TC_DEFAULT,
                          date_col: str = "Date",
                          return_col: str = "DJI_log_return") -> plt.Figure:
    """
    Fig. 6: for each window START date, the GAF (left) next to the
    log-returns of the same window (right; blue ≥ 0, red < 0, with a
    5-day centred rolling mean). t_c is marked when inside the window.

    Parameters
    ----------
    start_dates : list of str
        Window start dates 'YYYY-MM-DD' (must exist in
        resultados['fechas_inicio']).
    resultados : dict
        Output of sliding_window_gafs().
    df : pd.DataFrame
        Data with `date_col` and `return_col`.
    tc : str
        Critical date.
    date_col, return_col : str
        Column names.

    Returns
    -------
    matplotlib.figure.Figure or None
        One row per date found; None if no date matched.
    """
    tc_ts = pd.Timestamp(tc)
    rows = []
    for sd in start_dates:
        try:
            rows.append(find_window_index(resultados, sd, key="fechas_inicio"))
        except ValueError as e:
            print(f"  warning: {e}")
    if not rows:
        print("  warning: none of the requested start dates were found.")
        return None

    fig = plt.figure(figsize=(11, 3.2 * len(rows)))
    gs = gridspec.GridSpec(len(rows), 2, figure=fig, width_ratios=[1, 1.8],
                           wspace=0.25, hspace=0.75)
    for r, k in enumerate(rows):
        gaf = resultados["gafs"][k]
        n = gaf.shape[0]
        s_ts = pd.Timestamp(resultados["fechas_inicio"][k])
        e_ts = pd.Timestamp(resultados["fechas_fin"][k])
        win = df[(df[date_col] >= s_ts) & (df[date_col] <= e_ts)]
        dates, rets = win[date_col].values, win[return_col].values

        ax_g = fig.add_subplot(gs[r, 0])
        ax_g.imshow(gaf, origin="upper", cmap=CMAP, vmin=-1, vmax=1,
                    aspect="equal")
        ax_g.plot([0, n - 1], [0, n - 1], color="white", lw=0.7, ls="--",
                  alpha=0.45)
        ax_g.set_title(f"{s_ts:%Y-%m-%d} → {e_ts:%Y-%m-%d}\n"
                       f"Ḡ = {resultados['G_avg'][k]:.4f}, "
                       f"Var(G) = {resultados['G_var'][k]:.4f}", fontsize=9)
        ax_g.set_xlabel("time step j", fontsize=8)
        ax_g.set_ylabel("time step i", fontsize=8)

        ax_r = fig.add_subplot(gs[r, 1])
        ax_r.bar(dates, rets, color=np.where(rets >= 0, "steelblue", "tomato"),
                 width=2.0, alpha=0.85)
        ax_r.axhline(0, color="black", lw=0.7)
        if len(rets) >= 5:
            roll = pd.Series(rets).rolling(5, center=True, min_periods=1).mean()
            ax_r.plot(dates, roll.values, color="navy", lw=1.4, alpha=0.8,
                      label="5-day rolling mean")
        if s_ts <= tc_ts <= e_ts:
            ax_r.axvline(tc_ts, color="purple", lw=1.3, ls="--",
                         label=f"t_c = {tc}")
        ax_r.legend(fontsize=7, loc="upper left")
        ax_r.set_ylabel("Log-return", fontsize=8)
        ax_r.tick_params(axis="x", rotation=30, labelsize=7)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────────────────────────────────────

FIG6_START_DATES = ["1929-05-02", "1929-06-18", "1929-08-02",
                    "1929-09-18", "1929-11-04", "1929-12-23"]


def main(data_path: str, outdir: str = "figures", n: int = 64, m: int = 32,
         tc: str = TC_DEFAULT, show: bool = False) -> dict:
    """
    Run the full DJI analysis and save every figure in `outdir`.

    Parameters
    ----------
    data_path : str
        CSV with columns Date (MM/DD/YYYY) and DJI.
    outdir : str
        Output folder for PNG files.
    n, m : int
        Window size and stride.
    tc : str
        Critical date.
    show : bool
        Display figures interactively at the end.

    Returns
    -------
    dict
        The sliding-window results (GAFs, dates, descriptors).
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_dji(data_path)
    resultados = sliding_window_gafs(df, value_col="DJI_log_return", n=n, m=m)

    print("Building figures...")
    f1, f1b = plot_dji_series(df, tc=tc)
    _save(f1, outdir, "fig1_dji_1921_1932.png")
    _save(f1b, outdir, "fig1b_dji_log_returns_1921_1932.png")
    _save(plot_gaf_grid(resultados), outdir, "fig2_gaf_grid_1922_1932.png")
    _save(plot_gafs_around_tc(resultados, tc, side="before"),
          outdir, "fig3_gafs_before_tc.png")
    _save(plot_gafs_around_tc(resultados, tc, side="after"),
          outdir, "fig4_gafs_after_tc.png")
    _save(plot_descriptor_series(resultados, tc), outdir,
          "fig5_gavg_gvar_series.png")
    if n == 64 and m == 32:
        fig6 = plot_gaf_with_returns(FIG6_START_DATES, resultados, df, tc)
        if fig6 is not None:
            _save(fig6, outdir, "fig6_gaf_with_returns.png")
    else:
        print("  Fig. 6 skipped: its start dates assume n = 64, m = 32.")

    if show:
        plt.show()
    plt.close("all")
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/dji_sp_1900_2026.csv",
                        help="CSV with Date (MM/DD/YYYY) and DJI columns")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--n", type=int, default=64, help="window size")
    parser.add_argument("--m", type=int, default=32, help="stride")
    parser.add_argument("--tc", default=TC_DEFAULT, help="critical date")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.data, args.outdir, args.n, args.m, args.tc, args.show)
