"""
Reference normalisations and the anomaly / shape features built on them.

Dividing by a reference (chondrite, PAAS, ...) is a fixed number per element.
It changes how patterns LOOK and how anomalies are DEFINED, but on its own it
does not change any distance-based analysis after log + z-score (it is a
constant shift per column, which z-scoring removes). The process information
comes from the features below (anomalies, slopes, curvature), which compare
each element with its neighbours *within the same sample*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Post-Archean Australian Shale, Taylor & McLennan (1985) / McLennan (1989), ppm
PAAS = {
    "La": 38.2, "Ce": 79.6, "Pr": 8.83, "Nd": 33.9, "Sm": 5.55, "Eu": 1.08,
    "Gd": 4.66, "Tb": 0.774, "Dy": 4.68, "Ho": 0.991, "Er": 2.85, "Tm": 0.405,
    "Yb": 2.82, "Lu": 0.433, "Y": 27.0, "Sr": 200.0, "Th": 14.6, "U": 3.1,
}

# CI chondrite, McDonough & Sun (1995), ppm
CI = {
    "La": 0.237, "Ce": 0.613, "Pr": 0.0928, "Nd": 0.457, "Sm": 0.148,
    "Eu": 0.0563, "Gd": 0.199, "Tb": 0.0361, "Dy": 0.246, "Ho": 0.0546,
    "Er": 0.160, "Tm": 0.0247, "Yb": 0.161, "Lu": 0.0246, "Y": 1.57,
}

REFERENCES = {"PAAS": PAAS, "CI": CI}


def normalise(X: pd.DataFrame, ref="PAAS") -> pd.DataFrame:
    """X (ppm) divided by the reference, for the columns the reference has."""
    r = REFERENCES[ref] if isinstance(ref, str) else ref
    cols = [c for c in X.columns if c in r]
    return X[cols] / pd.Series({c: r[c] for c in cols})


def shale_features(X: pd.DataFrame) -> pd.DataFrame:
    """Process-oriented features from PAAS-normalised REE+Y (SN = shale-normalised).

    Uses only La, Ce, Pr, Nd, Sm, Eu, Gd, Dy, Ho, Y, Er, Yb, so it also works
    for O'Sullivan's phosphorites (no Tb, Tm).

    Ce_anom      Ce_SN / (0.5 La_SN + 0.5 Pr_SN)          redox (Bau & Dulski 1996)
    Pr_anom      Pr_SN / (0.5 Ce_SN + 0.5 Nd_SN)          >1 = true negative Ce anomaly;
                                                           with Ce_anom it separates real Ce
                                                           anomalies from La excesses
    Ce_anom_LK   Ce_SN / (Pr_SN^2 / Nd_SN)                La-independent version (Lawrence et al. 2006)
    Eu_anom      Eu_SN / sqrt(Sm_SN * Gd_SN)              plagioclase / high-T fluids
    Y_Ho         Y / Ho (mass ratio, reference-free)      seawater ~44-74, igneous ~26-28
    Pr_Yb_SN     (Pr/Yb)_SN                               LREE vs HREE slope
    MREE_bulge   mean(Sm,Gd,Dy)_SN / mean of (La,Pr)_SN and (Er,Yb)_SN
                                                           bell shape typical of diagenetic /
                                                           recrystallised apatite
    sum_REE      sum of the 12 REE used, ppm              overall uptake
    """
    n = normalise(X, "PAAS")
    f = pd.DataFrame(index=X.index)
    f["Ce_anom"] = n["Ce"] / (0.5 * n["La"] + 0.5 * n["Pr"])
    f["Pr_anom"] = n["Pr"] / (0.5 * n["Ce"] + 0.5 * n["Nd"])
    f["Ce_anom_LK"] = n["Ce"] / (n["Pr"] ** 2 / n["Nd"])
    f["Eu_anom"] = n["Eu"] / np.sqrt(n["Sm"] * n["Gd"])
    f["Y_Ho"] = X["Y"] / X["Ho"]
    f["Pr_Yb_SN"] = n["Pr"] / n["Yb"]
    mree = n[["Sm", "Gd", "Dy"]].mean(axis=1)
    f["MREE_bulge"] = mree / ((n[["La", "Pr"]].mean(axis=1) + n[["Er", "Yb"]].mean(axis=1)) / 2)
    ree = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Dy", "Ho", "Er", "Yb", "Lu"]
    f["sum_REE"] = X[[c for c in ree if c in X]].sum(axis=1)
    return f
