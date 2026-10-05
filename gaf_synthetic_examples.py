# -*- coding: utf-8 -*-
"""
gaf_synthetic_examples.py
=========================

Three synthetic 64 x 64 Gramian Angular Fields (GAF) that illustrate how to
read the colours of a GAF heat map, for the digest
"Gramian Angular Fields as Financial Regime Detectors (1929)".

Every case is built as an artificial *price* path. The GAF is computed on
its log-returns, rescaled to [-1, 1] within the window, which is the same
pipeline used for the DJI (dji_gaf_analysis.py):

    price  ->  log-returns  ->  window rescaling [-1, 1]  ->  phi = arccos  ->  G_ij = cos(phi_i + phi_j)

Cases
-----
A. Sinusoid + noise        : a regular, periodic market.
B. White noise             : a market with no structure at all (baseline).
C. Trend, then reversal    : a calm rise followed by a short, violent crash.

Colour convention (RdBu_r, vmin = -1, vmax = +1, same as the DJI figures):
    blue  (G ~ -1) : phi_i + phi_j ~ pi
                     -> both days ordinary (x~0, x~0)
                        OR two extremes of opposite sign (x~+1, x~-1)
    white (G ~  0) : phi_i + phi_j ~ pi/2 or 3pi/2
                     -> one extreme day paired with an ordinary day
    red   (G ~ +1) : phi_i + phi_j ~ 0 or 2pi
                     -> two extremes of the SAME sign

Usage
-----
    python gaf_synthetic_examples.py                 # encode log-returns (default)
    python gaf_synthetic_examples.py --encode levels # encode the price itself

Outputs  figures/gaf_synthetic_<encode>.png  and prints
the mean and variance of each GAF (the same descriptors used for the DJI).

No classes are used; everything is a documented function.
Requirements: numpy, matplotlib.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from gaf_encoding import (rescale_to_minus1_plus1, encode_polar,
                          compute_gaf, gaf_descriptors)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC PRICE PATHS
#    Each returns n + 1 prices, so that the log-returns have exactly n points.
# ─────────────────────────────────────────────────────────────────────────────

def make_sinusoid_price(n: int = 64,
                        n_cycles: float = 2.0,
                        amplitude: float = 0.08,
                        noise_sd: float = 0.004,
                        p0: float = 100.0,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Price that oscillates periodically around p0, plus small noise.

        log P_t = log p0 + amplitude * sin(2π n_cycles t / n) + ε_t,
        ε_t ~ N(0, noise_sd²)

    Its log-returns are approximately a (noisy) cosine wave, so many days sit
    near the window maximum or minimum. This produces a regular, periodic
    GAF with red, white and blue all present.

    Parameters
    ----------
    n : int
        Number of returns (GAF size).
    n_cycles : float
        Number of full oscillations inside the window.
    amplitude : float
        Amplitude of the log-price oscillation.
    noise_sd : float
        Standard deviation of the additive noise on log-price.
    p0 : float
        Reference price level.
    rng : np.random.Generator or None
        Random generator (for reproducibility).

    Returns
    -------
    np.ndarray, shape (n + 1,)
        Price path.
    """
    rng = rng or np.random.default_rng()
    t = np.arange(n + 1)
    log_p = (np.log(p0)
             + amplitude * np.sin(2 * np.pi * n_cycles * t / n)
             + rng.normal(0.0, noise_sd, size=n + 1))
    return np.exp(log_p)


def make_white_noise_price(n: int = 64,
                           sigma: float = 0.01,
                           p0: float = 100.0,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Random-walk price whose log-returns are Gaussian white noise.

        r_t ~ N(0, sigma²),   P_t = p0 * exp(Σ r)

    This is the "no information" baseline: most rescaled returns are close
    to 0, so the GAF is dominated by blue, with thin white crosses at the
    few largest days.

    Parameters
    ----------
    n : int
        Number of returns.
    sigma : float
        Daily volatility of log-returns.
    p0 : float
        Initial price.
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    np.ndarray, shape (n + 1,)
        Price path.
    """
    rng = rng or np.random.default_rng()
    r = rng.normal(0.0, sigma, size=n)
    return p0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def make_trend_reversal_price(n: int = 64,
                              t_reversal: int = 40,
                              mu_up: float = 0.004,
                              sigma_up: float = 0.006,
                              crash_returns: tuple = (-0.11, -0.13, -0.09),
                              rebound: float = 0.10,
                              sigma_after: float = 0.02,
                              p0: float = 100.0,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Calm uptrend, then a short cluster of crash days, then a volatile market.

    Stylised after October 1929:
      * days [0, t_reversal)            : small positive drift (the bubble);
      * days t_reversal ... + len(crash_returns) - 1
                                        : consecutive large losses
                                          (Black Thursday / Monday / Tuesday);
      * the next day                    : one large rebound
                                          (DJI rose ~+12% on 30 Oct 1929);
      * remaining days                  : slightly negative drift,
                                          higher volatility.

    Because the window contains both very negative and one very positive day,
    ordinary days land near x̃ ≈ 0, as in the real DJI windows. The crash
    cluster then shows up as a red block (same-sign extremes) with white
    stripes (extreme vs ordinary days), on a blue background.

    Parameters
    ----------
    n : int
        Number of returns.
    t_reversal : int
        Index of the first crash day.
    mu_up, sigma_up : float
        Mean and volatility of returns during the uptrend.
    crash_returns : tuple of float
        Log-returns of the consecutive crash days.
    rebound : float
        Log-return of the rebound day right after the crash cluster.
    sigma_after : float
        Volatility of the (slightly negative-drift) returns after the rebound.
    p0 : float
        Initial price.
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    np.ndarray, shape (n + 1,)
        Price path.
    """
    rng = rng or np.random.default_rng()
    k = len(crash_returns)
    if t_reversal + k + 1 > n:
        raise ValueError("t_reversal + crash cluster + rebound must fit in n.")

    r = np.empty(n)
    r[:t_reversal] = rng.normal(mu_up, sigma_up, size=t_reversal)
    r[t_reversal:t_reversal + k] = crash_returns
    r[t_reversal + k] = rebound
    n_after = n - (t_reversal + k + 1)
    # Post-crash: negative drift, higher volatility (continued decline)
    r[t_reversal + k + 1:] = rng.normal(-0.004, sigma_after, size=n_after)
    return p0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def price_to_gaf(price: np.ndarray, encode: str = "returns") -> dict:
    """
    Price path -> encoded series -> window-rescaled -> GAF.

    Parameters
    ----------
    price : np.ndarray, shape (n + 1,)
        Price path.
    encode : {"returns", "levels"}
        "returns": GAF of log-returns (the DJI pipeline, n points).
        "levels" : GAF of log-price (last n points), for comparison.

    Returns
    -------
    dict with keys
        'price'    : the input price path
        'signal'   : series fed to the GAF (log-returns or log-price)
        'x_scaled' : signal rescaled to [-1, 1]
        'gaf'      : GAF matrix (n x n)
        'g_avg', 'g_var' : scalar descriptors
    """
    if encode == "returns":
        signal = np.diff(np.log(price))
    elif encode == "levels":
        signal = np.log(price)[1:]
    else:
        raise ValueError("encode must be 'returns' or 'levels'.")

    x_scaled = rescale_to_minus1_plus1(signal)
    phi, _ = encode_polar(x_scaled)
    gaf = compute_gaf(phi)
    g_avg, g_var = gaf_descriptors(gaf)
    return {"price": price, "signal": signal, "x_scaled": x_scaled,
            "gaf": gaf, "g_avg": g_avg, "g_var": g_var}


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_gaf_examples(cases: list,
                      encode: str = "returns",
                      cmap: str = "RdBu_r",
                      outfile: str | None = None) -> plt.Figure:
    """
    Three-column figure; one column per synthetic case.

      Row 1 — the price path (what a reader recognises);
      Row 2 — the rescaled series actually encoded (bars; red = negative);
      Row 3 — its 64 x 64 GAF (RdBu_r, fixed range [-1, 1]).

    Parameters
    ----------
    cases : list of (title, result_dict)
        result_dict as returned by price_to_gaf().
    encode : str
        Only used for labels ("returns" or "levels").
    cmap : str
        Colormap for the GAF (RdBu_r matches the DJI figures).
    outfile : str or None
        If given, the figure is saved there (dpi = 200).

    Returns
    -------
    matplotlib.figure.Figure
    """
    ink, muted = "#222222", "#6b6b6b"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted})

    fig = plt.figure(figsize=(12, 9.2))
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           height_ratios=[0.8, 0.8, 2.2],
                           hspace=0.38, wspace=0.28)
    signal_label = ("rescaled log-return  x̃" if encode == "returns"
                    else "rescaled log-price  x̃")

    im = None
    for col, (title, res) in enumerate(cases):
        n = len(res["x_scaled"])

        # Row 1: price path
        ax_p = fig.add_subplot(gs[0, col])
        ax_p.plot(res["price"], color="#1f4e8c", lw=1.8)
        ax_p.set_title(title, fontsize=12, color=ink, loc="left",
                       fontweight="bold")
        ax_p.set_xticks([])
        ax_p.set_ylabel("price" if col == 0 else "")
        ax_p.spines[["top", "right"]].set_visible(False)
        ax_p.grid(axis="y", alpha=0.25)

        # Row 2: encoded, rescaled series
        ax_s = fig.add_subplot(gs[1, col])
        x = res["x_scaled"]
        colors = np.where(x >= 0, "#4a7fb5", "#c0504d")
        ax_s.bar(np.arange(n), x, color=colors, width=0.8)
        ax_s.axhline(0, color=ink, lw=0.6)
        ax_s.set_ylim(-1.1, 1.1)
        ax_s.set_yticks([-1, 0, 1])
        ax_s.set_xlim(-0.5, n - 0.5)
        ax_s.set_xlabel("day", color=muted)
        ax_s.set_ylabel(signal_label if col == 0 else "")
        ax_s.spines[["top", "right"]].set_visible(False)

        # Row 3: GAF
        ax_g = fig.add_subplot(gs[2, col])
        im = ax_g.imshow(res["gaf"], origin="upper", cmap=cmap,
                         vmin=-1, vmax=1, aspect="equal",
                         interpolation="nearest")
        ax_g.set_title(f"$\\bar{{G}}$ = {res['g_avg']:.3f}    "
                       f"Var($G$) = {res['g_var']:.3f}",
                       fontsize=10, color=muted)
        ax_g.set_xlabel("day $j$", color=muted)
        ax_g.set_ylabel("day $i$" if col == 0 else "", color=muted)
        ax_g.set_xticks([0, n // 2, n - 1])
        ax_g.set_yticks([0, n // 2, n - 1])

    # One shared colour bar with the reading key
    cax = fig.add_axes([0.93, 0.11, 0.015, 0.38])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["−1  blue", "0  white", "+1  red"])
    cb.outline.set_visible(False)

    fig.text(0.07, 0.015,
             "Blue: both days ordinary, or extremes of opposite sign.   "
             "White: one extreme day with an ordinary day.   "
             "Red: two extremes of the same sign.",
             fontsize=9.5, color=muted)

    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(encode: str = "returns", n: int = 64, seed: int = 7,
         show: bool = True, outdir: str = "figures") -> list:
    """
    Build the three synthetic cases, print their descriptors, draw the figure.

    Parameters
    ----------
    encode : {"returns", "levels"}
        What the GAF encodes (see price_to_gaf).
    n : int
        GAF size (number of returns per window).
    seed : int
        Random seed; change it to see other realisations.
    show : bool
        Call plt.show() at the end.
    outdir : str
        Folder where the PNG is saved.

    Returns
    -------
    list of (title, result_dict)
    """
    rng = np.random.default_rng(seed)

    cases = [
        ("A. Sinusoid + noise",
         price_to_gaf(make_sinusoid_price(n=n, rng=rng), encode)),
        ("B. White noise",
         price_to_gaf(make_white_noise_price(n=n, rng=rng), encode)),
        ("C. Uptrend, then crash",
         price_to_gaf(make_trend_reversal_price(n=n, rng=rng), encode)),
    ]

    print(f"Encoding: {encode}   n = {n}   seed = {seed}")
    for title, res in cases:
        print(f"  {title:<26s}  G_avg = {res['g_avg']:+.4f}   "
              f"G_var = {res['g_var']:.4f}")

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"gaf_synthetic_{encode}.png")
    plot_gaf_examples(cases, encode=encode, outfile=outfile)
    print(f"  saved {outfile}")
    if show:
        plt.show()
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encode", choices=["returns", "levels"],
                        default="returns")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(encode=args.encode, n=args.n, seed=args.seed,
         show=not args.no_show, outdir=args.outdir)
