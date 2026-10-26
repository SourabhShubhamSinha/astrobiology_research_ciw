"""
REE pattern shape (lambda) coefficients and anomalies relative to the fitted shape.

After O'Neill (2016, J. Petrology 57:1463): ln(REE / CI chondrite) is fitted as a
polynomial in ionic radius using orthogonal polynomials, so the coefficients are
independent of each other:

    lambda0  overall level (mean ln abundance)
    lambda1  slope        (LREE vs HREE)
    lambda2  curvature    (MREE hump or trough)

Ce, Eu and Y are NOT used in the fit; their departures from the fitted smooth
curve are the anomalies:

    Ce_dev = ln(Ce_N) - fit(r_Ce)        (0 = no anomaly; -0.36 ~ Ce/Ce* 0.7)
    Eu_dev = ln(Eu_N) - fit(r_Eu)
    Y_dev  = ln(Y_N)  - fit(r_Y)         (Y behaves like Ho in melts, not in seawater)

Because each sample is fitted to whichever REE it has (>= min_points), samples
with different element lists (O'Sullivan without Tb/Tm, Tambiev's INAA subset)
end up in the same feature space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import CI

# Shannon (1976) ionic radii for 8-fold coordinated REE3+ and Y3+ (angstrom)
RADIUS = {"La": 1.160, "Ce": 1.143, "Pr": 1.126, "Nd": 1.109, "Sm": 1.079, "Eu": 1.066,
          "Gd": 1.053, "Tb": 1.040, "Dy": 1.027, "Ho": 1.015, "Y": 1.019, "Er": 1.004,
          "Tm": 0.994, "Yb": 0.985, "Lu": 0.977}

FIT_ELEMENTS = ["La", "Pr", "Nd", "Sm", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


def _orthogonal_basis(radii: np.ndarray, degree: int):
    """Polynomials in r, orthogonal over the given radii (Gram-Schmidt), as callables."""
    r = np.asarray(radii, float)
    polys = [np.poly1d([1.0])]
    for d in range(1, degree + 1):
        p = np.poly1d([1.0] + [0.0] * d)      # r**d
        for q in polys:
            p = p - q * (np.sum(p(r) * q(r)) / np.sum(q(r) ** 2))
        polys.append(p)
    return polys


_BASIS = _orthogonal_basis([RADIUS[e] for e in FIT_ELEMENTS], degree=3)


def lambda_features(X: pd.DataFrame, degree=2, min_points=4) -> pd.DataFrame:
    """Lambda coefficients + anomalies for every row of X (ppm, NaN = not measured / below detection)."""
    basis = _BASIS[: degree + 1]
    fit_cols = [e for e in FIT_ELEMENTS if e in X.columns]
    L = np.log(X.reindex(columns=fit_cols).where(lambda d: d > 0) / pd.Series({e: CI[e] for e in fit_cols}))
    out = pd.DataFrame(index=X.index,
                       columns=[f"lambda{i}" for i in range(degree + 1)] + ["n_fit", "fit_rmse",
                                                                             "Ce_dev", "Eu_dev", "Y_dev"],
                       dtype=float)
    A_all = np.column_stack([np.array([b(RADIUS[e]) for e in fit_cols]) for b in basis])
    vals = L.to_numpy()
    for i, row in enumerate(vals):
        ok = np.isfinite(row)
        n = int(ok.sum())
        out.iat[i, degree + 1] = n
        if n < max(min_points, degree + 2):
            continue
        coef, *_ = np.linalg.lstsq(A_all[ok], row[ok], rcond=None)
        out.iloc[i, : degree + 1] = coef
        resid = row[ok] - A_all[ok] @ coef
        out.iat[i, degree + 2] = np.sqrt(np.mean(resid ** 2))
    # anomalies relative to the fitted curve
    lam = out[[f"lambda{i}" for i in range(degree + 1)]].to_numpy()
    for el, col in [("Ce", "Ce_dev"), ("Eu", "Eu_dev"), ("Y", "Y_dev")]:
        if el not in X.columns:
            continue
        b = np.array([f(RADIUS[el]) for f in basis])
        predicted = lam @ b
        obs = np.log(X[el].where(X[el] > 0) / CI[el]).to_numpy()
        out[col] = obs - predicted
    return out
