"""
Score any apatite analysis against an abiotic baseline, using whatever
elements it has.

A Gaussian mixture is fitted to the baseline in log10(ppm). The marginal of a
Gaussian mixture on any subset of elements is again a Gaussian mixture (same
weights, sub-vector of each mean, sub-block of each covariance), so a sample
that measured only Ce, Sm, Eu, Tb, Yb and Lu can be scored on exactly those six.

Because log-likelihoods are not comparable between subsets of different size,
each score is turned into a percentile: the fraction of baseline samples that
are *more typical* than this one on the same subset. 0.99 = more unusual than
99% of the abiotic baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class Baseline:
    def __init__(self, Xlog: pd.DataFrame, k_range=range(1, 16), random_state=42):
        """Xlog: complete baseline rows, log10(ppm). k chosen by minimum BIC."""
        self.cols = list(Xlog.columns)
        self.ref = Xlog.to_numpy()
        fits = {k: GaussianMixture(k, covariance_type="full", n_init=3, random_state=random_state).fit(self.ref)
                for k in k_range}
        self.bic = pd.Series({k: m.bic(self.ref) for k, m in fits.items()})
        self.k = int(self.bic.idxmin())
        self.gmm = fits[self.k]
        self._ref_cache = {}

    def _loglik(self, A: np.ndarray, idx: list[int]) -> np.ndarray:
        g = self.gmm
        comps = [np.log(w) + multivariate_normal(mean=m[idx], cov=c[np.ix_(idx, idx)], allow_singular=True).logpdf(A)
                 for w, m, c in zip(g.weights_, g.means_, g.covariances_)]
        return logsumexp(np.vstack(comps).reshape(len(comps), -1), axis=0)

    def score(self, Xlog: pd.DataFrame, min_elements=4) -> pd.DataFrame:
        """Percentile of unusualness for each row, using its non-missing elements."""
        Xlog = Xlog.reindex(columns=self.cols)
        mask = Xlog.notna()
        out = pd.DataFrame(index=Xlog.index, columns=["n_elements", "loglik", "percentile_unusual", "elements_used"],
                           dtype=object)
        for pattern, rows in mask.groupby(list(mask.columns)).groups.items():
            idx = [i for i, ok in enumerate(pattern) if ok]
            used = [self.cols[i] for i in idx]
            if len(idx) < min_elements:
                out.loc[rows, ["n_elements", "elements_used"]] = [len(idx), ",".join(used)]
                continue
            key = tuple(idx)
            if key not in self._ref_cache:
                self._ref_cache[key] = np.sort(self._loglik(self.ref[:, idx], idx))
            ref = self._ref_cache[key]
            ll = self._loglik(Xlog.loc[rows].to_numpy()[:, idx], idx)
            pct = 1.0 - np.searchsorted(ref, ll, side="right") / len(ref)
            out.loc[rows, "n_elements"] = len(idx)
            out.loc[rows, "loglik"] = ll
            out.loc[rows, "percentile_unusual"] = pct
            out.loc[rows, "elements_used"] = ",".join(used)
        out["n_elements"] = pd.to_numeric(out["n_elements"])
        out["loglik"] = pd.to_numeric(out["loglik"])
        out["percentile_unusual"] = pd.to_numeric(out["percentile_unusual"])
        return out


def to_apatite_basis(X: pd.DataFrame, p2o5_wt: pd.Series, apatite_p2o5=41.0) -> pd.DataFrame:
    """Scale bulk-rock concentrations to a pure-apatite basis (x apatite_p2o5 / P2O5).

    For bulk phosphorites: removes dilution by non-apatite material, assuming
    the REE sit in the apatite. Ratios between elements are unchanged.
    """
    f = (apatite_p2o5 / p2o5_wt).where(p2o5_wt > 0)
    return X.mul(f, axis=0)
