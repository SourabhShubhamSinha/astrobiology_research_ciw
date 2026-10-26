"""
Diagnose and remove study (lab / method) offsets while protecting geology.

Model, fitted separately for each feature (in log10 ppm):

    y = class mean (rock_class, fixed)  +  study offset (random)  +  noise

The study offset is a random intercept, estimated with a linear mixed model
(fitted by EM, maximum likelihood). Its estimate (BLUP) is *shrunk towards zero*: a study with few samples,
or whose samples look like everyone else's in the same rock class, gets almost
no correction. Only offsets that are consistent within a study AND unusual for
its rock class are removed.

Limits worth stating in any write-up:
- If a rock class is sampled by a single study, class and study cannot be told
  apart; the model gives the difference to the class (no correction). Those
  classes are listed by ``confounded_classes``.
- Corrections are additive in log space, i.e. a multiplicative factor per study
  and element (e.g. "study X reads La 25% high"). Differences in precision or
  detection limits between labs are not corrected.
- A new study (e.g. O'Sullivan's phosphorites) has no estimated offset, so it is
  left uncorrected; its offset simply adds to its scatter.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _fit_one(y: pd.Series, labels: pd.DataFrame, protect: str, group: str, n_iter=500, tol=1e-8):
    """Random-intercept model fitted by EM (no external dependencies, never singular).

    y_i = beta[class(i)] + u[study(i)] + e_i,  u ~ N(0, tau2),  e ~ N(0, sigma2)
    Returns (tau2, sigma2, blup_u as Series indexed by study).
    """
    yv = np.asarray(y, dtype=float)
    cls = pd.Categorical(labels[protect].astype(str)).codes
    grp_cat = pd.Categorical(labels[group].astype(str))
    grp = grp_cat.codes
    n_c, n_g = cls.max() + 1, grp.max() + 1
    n_s = np.bincount(grp, minlength=n_g).astype(float)
    u = np.zeros(n_g)
    sigma2 = np.var(yv) / 2 + 1e-12
    tau2 = np.var(yv) / 2 + 1e-12
    for _ in range(n_iter):
        beta = np.bincount(cls, weights=yv - u[grp], minlength=n_c) / np.bincount(cls, minlength=n_c)
        r = yv - beta[cls]
        sum_r = np.bincount(grp, weights=r, minlength=n_g)
        denom = sigma2 + n_s * tau2
        u_new = tau2 * sum_r / denom
        v = sigma2 * tau2 / denom
        tau2_new = np.mean(u_new ** 2 + v)
        sigma2_new = np.mean((r - u_new[grp]) ** 2 + v[grp])
        done = (abs(tau2_new - tau2) < tol and abs(sigma2_new - sigma2) < tol
                and np.max(np.abs(u_new - u)) < tol)
        u, tau2, sigma2 = u_new, max(tau2_new, 1e-12), max(sigma2_new, 1e-12)
        if done:
            break
    return tau2, sigma2, pd.Series(u, index=grp_cat.categories)


def confounded_classes(labels: pd.DataFrame, protect="rock_class", group="study") -> pd.Series:
    """Rock classes sampled by only one study (their study offset can't be estimated)."""
    n = labels.groupby(protect)[group].nunique()
    return n[n == 1]


def study_variance(Y: pd.DataFrame, labels: pd.DataFrame, protect="rock_class", group="study") -> pd.DataFrame:
    """Per feature: how much of the within-class variance is between studies.

    icc_study = var_study / (var_study + var_resid): the share of the variance
    left after rock class that is explained by which study measured the sample.
    """
    rows = {}
    for c in Y.columns:
        ok = Y[c].notna()
        vs, vr, _ = _fit_one(Y.loc[ok, c], labels.loc[ok], protect, group)
        rows[c] = {"var_study": vs, "var_resid": vr, "icc_study": vs / (vs + vr) if vs + vr > 0 else np.nan,
                   "typical_study_factor": 10 ** np.sqrt(vs)}   # 1-sigma multiplicative offset
    return pd.DataFrame(rows).T


def remove_study_effect(Y: pd.DataFrame, labels: pd.DataFrame, protect="rock_class", group="study"):
    """Subtract each study's estimated (shrunk) offset from every feature.

    Y      : features in log10 ppm (NOT z-scored yet)
    Returns (Y_corrected, offsets) where offsets is a study x feature table in
    log10 units (0.1 = that study reads ~26% high for that element).
    """
    Yc = Y.copy()
    offsets = {}
    for c in Y.columns:
        ok = Y[c].notna()
        _, _, blup = _fit_one(Y.loc[ok, c], labels.loc[ok], protect, group)
        offsets[c] = blup
        Yc.loc[ok, c] = Y.loc[ok, c] - labels.loc[ok, group].astype(str).map(blup).fillna(0).values
    return Yc, pd.DataFrame(offsets)


def neighbour_mixing(Z, labels: pd.DataFrame, cols=("study", "rock_class"), k=10) -> pd.DataFrame:
    """Fraction of each point's k nearest neighbours that share its label.

    High 'study' fraction = points cluster by who measured them (bad).
    High 'rock_class' fraction = points cluster by geology (good).
    'chance' is the value expected if labels were scattered at random.
    """
    from sklearn.neighbors import NearestNeighbors

    Z = np.asarray(Z)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    out = {}
    for c in cols:
        lab = labels[c].astype(str).to_numpy()
        same = (lab[idx] == lab[:, None]).mean()
        p = pd.Series(lab).value_counts(normalize=True)
        out[c] = {"same_label_fraction": same, "chance": float((p ** 2).sum())}
    return pd.DataFrame(out).T


def cap_per_study(labels: pd.DataFrame, n_max=10, seed=0, group="study") -> pd.Index:
    """Index of a subsample with at most n_max rows per study (random, reproducible).

    Stops a few large studies from defining the baseline.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in labels.groupby(group).groups.items():
        idx = list(idx)
        keep += idx if len(idx) <= n_max else list(rng.choice(idx, n_max, replace=False))
    return pd.Index(keep)


def leave_one_study_out(Z, labels: pd.DataFrame, cluster_fn, group="study", min_rows=1):
    """How much does the clustering depend on any single study?

    cluster_fn(Z_subset) -> array of cluster labels. For each study, re-cluster
    without it and compare (adjusted Rand index) with the full clustering on the
    rows that remain. Returns a Series of ARI by left-out study (1 = no change).
    """
    from sklearn.metrics import adjusted_rand_score

    Z = np.asarray(Z)
    full = np.asarray(cluster_fn(Z))
    g = labels[group].astype(str).to_numpy()
    out = {}
    for s in pd.unique(g):
        m = g != s
        if (~m).sum() < min_rows:
            continue
        out[s] = adjusted_rand_score(full[m], np.asarray(cluster_fn(Z[m])))
    return pd.Series(out, name="ARI_vs_full").sort_values()
