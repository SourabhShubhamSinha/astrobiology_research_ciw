"""
Process 2: incremental multivariate KDE densities.

For each stage (a growing set of sample categories):
  1. z-score the features using only the samples in that stage
  2. fit a Gaussian KDE (full bandwidth matrix, Scott's rule) to those samples
  3. leave-one-out density for every sample:
         d_i = 1/(n-1) * sum_{j != i} K_H(x_i - x_j)
  4. probability of a row:   P_i = d_i / sum_j d_j          (sums to 1 within the stage)
     stage-comparable form:  nP_i = n * P_i = d_i / mean(d)  (1 = average for that stage)
  5. where the density comes from: the share of each row's density contributed
     by neighbours of its own category vs by the stage-1 (abiotic) samples.
     A new group can be dense simply because its members sit close together;
     if almost none of that density comes from abiotic neighbours, the group
     is distinct even though its P is not low.

Memory: the pairwise step is done in blocks, so memory stays ~ block x n floats
(a few MB) whatever n is. For a few thousand rows it runs in well under a second;
parallelisation is not needed at this size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def scott_factor(n: int, d: int) -> float:
    """Scott's rule bandwidth factor (same as scipy.stats.gaussian_kde)."""
    return n ** (-1.0 / (d + 4))


def loo_kde_density(Z: np.ndarray, block: int = 2048, groups=None):
    """Leave-one-out Gaussian KDE density at every row of Z (n x d).

    Kernel covariance H = factor^2 * cov(Z) (scipy's convention). Returns
    (density, factor), or (density, factor, contributions) when `groups`
    (one integer code per row) is given: contributions[i, g] = share of row
    i's density that comes from neighbours in group g (rows sum to 1).
    """
    Z = np.asarray(Z, float)
    n, d = Z.shape
    f = scott_factor(n, d)
    H = np.atleast_2d(np.cov(Z, rowvar=False)) * f ** 2
    L = np.linalg.cholesky(H)
    W = np.linalg.solve(L, Z.T).T                    # whitened coordinates: kernel becomes N(0, I)
    log_norm = -0.5 * d * np.log(2 * np.pi) - np.log(np.diag(L)).sum()
    sq = (W ** 2).sum(axis=1)
    dens = np.empty(n)
    if groups is not None:
        groups = np.asarray(groups)
        G = groups.max() + 1
        contrib = np.zeros((n, G))
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * W[s:e] @ W.T
        np.maximum(d2, 0, out=d2)
        k = np.exp(-0.5 * d2)
        k[np.arange(e - s), np.arange(s, e)] = 0.0   # leave the point itself out
        dens[s:e] = k.sum(axis=1)
        if groups is not None:
            for g in range(G):
                contrib[s:e, g] = k[:, groups == g].sum(axis=1)
    if groups is not None:
        contrib = contrib / np.where(dens[:, None] > 0, dens[:, None], np.nan)
    dens = np.exp(log_norm) * dens / (n - 1)
    return (dens, f, contrib) if groups is not None else (dens, f)


def run_stages(F: pd.DataFrame, category: pd.Series, stages: list[list[str]],
               zscore: str = "per_stage") -> pd.DataFrame:
    """Densities for a sequence of stages.

    F        : complete feature table (rows = samples, already in the units you want, e.g. log10 ppm)
    category : category label per row (same index as F)
    stages   : e.g. [["A"], ["A", "B"], ["A", "B", "C"]] - categories included at each stage
    zscore   : "per_stage" (recompute mean/sd on each stage's samples) or "first_stage"
    Returns a long table: stage, stage_label, category, density, P, nP, n, bandwidth_factor.
    """
    out = []
    first = F[category.isin(stages[0])]
    mu0, sd0 = first.mean(), first.std()
    for i, cats in enumerate(stages, start=1):
        sub = F[category.isin(cats)]
        mu, sd = (sub.mean(), sub.std()) if zscore == "per_stage" else (mu0, sd0)
        Z = ((sub - mu) / sd).to_numpy()
        sub_cat = category.loc[sub.index].to_numpy()
        codes = np.array([cats.index(c) for c in sub_cat])
        dens, f, contrib = loo_kde_density(Z, groups=codes)
        P = dens / dens.sum()
        out.append(pd.DataFrame({
            "stage": i,
            "stage_label": f"{i}: + {cats[-1]}" if i > 1 else f"1: {cats[0]}",
            "category": category.loc[sub.index].to_numpy(),
            "density": dens, "P": P, "nP": P * len(sub),
            "n": len(sub), "bandwidth_factor": f,
            # where each row's density comes from:
            "from_own_category": contrib[np.arange(len(sub)), codes],
            "from_first_stage": contrib[:, 0],      # share contributed by the stage-1 (abiotic) samples
        }, index=sub.index))
    return pd.concat(out)


def stage_summary(res: pd.DataFrame) -> pd.DataFrame:
    """Median P and nP per category per stage (wide table)."""
    g = res.groupby(["stage", "stage_label", "category"])
    s = g.agg(count=("P", "size"), median_P=("P", "median"), median_nP=("nP", "median"),
              frac_nP_below_0_1=("nP", lambda x: (x < 0.1).mean()),
              median_from_own=("from_own_category", "median"),
              median_from_first_stage=("from_first_stage", "median"))
    return s.round(4)
