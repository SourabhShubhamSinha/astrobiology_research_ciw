"""
Deviation vectors: how far, and in which direction, a sample departs from its
closest abiotic context.

For each sample x:
  1. find the abiotic baseline component it most resembles (GMM component k
     with the smallest Mahalanobis distance) - its "abiotic expectation" mu_k;
  2. deviation vector       d = x - mu_k                    (feature units)
     standardised version   z = d / sd_k                    (per feature, readable)
     whitened version       w = L_k^-1 d, Sigma_k = L L^T   (accounts for correlations)
  3. magnitude = |w| (Mahalanobis distance = how unusual)
     direction = z / |z| (unit vector = which features move, and which way)

A process that is genuinely different from the abiotic one should push its
samples in a CONSISTENT direction (high coherence), and that direction should
differ from the direction of other non-abiotic processes (low cosine similarity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def fit_baseline_gmm(F_base: pd.DataFrame, k_range=range(1, 13), random_state=42) -> GaussianMixture:
    fits = {k: GaussianMixture(k, covariance_type="full", n_init=3, random_state=random_state).fit(F_base)
            for k in k_range}
    k = min(fits, key=lambda k: fits[k].bic(F_base.to_numpy()))
    return fits[k]


def deviation_vectors(F: pd.DataFrame, gmm: GaussianMixture):
    """Returns (z, info): z = per-feature standardised deviations (rows = samples),
    info = nearest component, Mahalanobis distance, and the |z| norm."""
    X = F.to_numpy()
    n, p = X.shape
    best_d2 = np.full(n, np.inf)
    best_k = np.zeros(n, int)
    chols = [np.linalg.cholesky(S) for S in gmm.covariances_]
    for k, (mu, L) in enumerate(zip(gmm.means_, chols)):
        w = np.linalg.solve(L, (X - mu).T).T
        d2 = (w ** 2).sum(axis=1)
        better = d2 < best_d2
        best_d2[better], best_k[better] = d2[better], k
    mu = gmm.means_[best_k]
    sd = np.sqrt(np.stack([np.diag(gmm.covariances_[k]) for k in best_k]))
    z = pd.DataFrame((X - mu) / sd, index=F.index, columns=F.columns)
    info = pd.DataFrame({"component": best_k, "mahalanobis": np.sqrt(best_d2),
                         "z_norm": np.linalg.norm(z.to_numpy(), axis=1)}, index=F.index)
    return z, info


def group_directions(z: pd.DataFrame, groups: pd.Series, min_norm=0.0):
    """Mean unit direction per group and its coherence.

    coherence = length of the mean of unit vectors: 1 = every sample deviates the
    same way, ~0 = deviations point in random directions (just noise).
    Only samples with |z| > min_norm are used (ignore samples that barely deviate).
    """
    norm = np.linalg.norm(z.to_numpy(), axis=1)
    u = z.div(np.where(norm > 0, norm, np.nan), axis=0)
    use = norm > min_norm
    mean_u = u[use].groupby(groups[use]).mean()
    coherence = np.linalg.norm(mean_u.to_numpy(), axis=1)
    direction = mean_u.div(coherence, axis=0)            # unit-length group direction
    stats = pd.DataFrame({"n": groups[use].value_counts(), "coherence": coherence,
                          "median_mahalanobis_proxy": pd.Series(norm, index=z.index)[use].groupby(groups[use]).median()},
                         index=mean_u.index)
    return direction, stats


def cosine_matrix(direction: pd.DataFrame) -> pd.DataFrame:
    D = direction.to_numpy()
    return pd.DataFrame(D @ D.T, index=direction.index, columns=direction.index)


def fisher_direction(F: pd.DataFrame, is_target: pd.Series, is_base: pd.Series, ridge=1e-3) -> pd.Series:
    """Single axis that best separates target from abiotic (Fisher / LDA):
    w = (S_pooled)^-1 (mean_target - mean_base), normalised. Projection = F @ w."""
    A, B = F[is_target].to_numpy(), F[is_base].to_numpy()
    S = (np.cov(A.T) * (len(A) - 1) + np.cov(B.T) * (len(B) - 1)) / (len(A) + len(B) - 2)
    S += ridge * np.eye(S.shape[0]) * np.trace(S) / S.shape[0]
    w = np.linalg.solve(S, A.mean(0) - B.mean(0))
    return pd.Series(w / np.linalg.norm(w), index=F.columns)
