"""
Analysis-ready tables: features (X), labels, and metadata.

    from utils.tables import apatite_table
    X, labels, meta = apatite_table(processed_dir)

X      : element concentrations in ppm (one row per analysis, or per sample)
labels : columns you might colour / group / classify by
         dataset, study, lithology (as reported), rock_class, origin_class,
         temp_regime, setting, data_type, also_in_georoc
meta   : sample_id, location, lat, lon, age_ma, mineral, spot, rim_core,
         primary_secondary, alteration, n_spots
All three share the same index, so X.loc[labels.rock_class == "carbonatite"]
etc. just works.

The rock_class / origin_class rules below are plain keyword lists - edit them
as your judgement changes. They are first-pass groupings, not interpretations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import Dataset, REE_Y

# --------------------------------------------------------------------------
# 1. Rock class from the reported rock / lithology name (first match wins)
# --------------------------------------------------------------------------
ROCK_CLASS_RULES = [
    ("phosphorite",              ["PHOSPHORITE"]),
    ("fossil_bioapatite",        ["FOSSIL", "BONE", "TOOTH", "TEETH", "ENAMEL", "DENTINE", "CONODONT", "FISH"]),
    ("bif_chert",                ["BIF", "BANDED IRON", "METACHERT", "CHERT"]),
    ("carbonatite",              ["CARBONATITE", "SOVITE", "SÖVITE", "SÃ¶VITE", "PHOSCORITE", "BEFORSITE", "ALVIKITE"]),
    ("kimberlite_lamprophyre",   ["KIMBERLITE", "LAMPROITE", "LAMPROPHYRE", "AILLIKITE", "MINETTE",
                                  "KERSANTITE", "SPESSARTITE", "ORANGEITE"]),
    ("metamorphic",              ["AMPHIBOLITE", "GRANULITE", "GNEISS", "MIGMATITE", "HORNFELS", "ECLOGITE",
                                  "SCHIST", "MARBLE", "SKARN", "METASEDIMENT", "METABASITE", "BLUESCHIST",
                                  "GREENSCHIST", "METAMORPH", "QUARTZITE", "LOWER CRUSTAL", "FACIES"]),
    ("ultramafic",               ["PYROXENITE", "HORNBLENDITE", "PERIDOTITE", "LHERZOLITE", "HARZBURGITE",
                                  "WEHRLITE", "DUNITE", "CHROMITITE", "GLIMMERITE", "MICAITE", "ULTRAMAFIC"]),
    ("ore_magmatic_hydrothermal",["IRON ORE", "MAGNETITITE", "NELSONITE", "GOLD ORE", " ORE", "KIRUNA"]),
    ("alkaline_silicate",        ["SYENITE", "IJOLITE", "PHONOLITE", "NEPHELIN", "MELILITITE", "OKAITE",
                                  "SHONKINITE", "JACUPIRANGITE", "FENITE", "TEPHRITE", "BASANITE", "LEUCITITE",
                                  "FOIDITE", "AGPAITE", "LARVIKITE", "MELTEIGITE", "URTITE", "AUGITESYENITE"]),
    ("sedimentary_clastic",      ["SANDSTONE", "ARKOSE", "MUDSTONE", "SILTSTONE", "SHALE", "SAND", "CONGLOMERATE",
                                  "DOLOSTONE", "LIMESTONE", "SEDIMENTARY ROCK", "DETRITAL"]),
    ("volcaniclastic",           ["PYROCLASTIC", "TUFF", "ASH", "PUMICE", "SCORIA", "VOLCANICLASTIC", "IGNIMBRITE",
                                  "TEPHRA"]),
    ("felsic_volcanic",          ["RHYOLITE", "DACITE", "TRACHYTE", "OBSIDIAN"]),
    ("mafic_interm_volcanic",    ["BASALT", "ANDESITE", "HAWAIITE", "MUGEARITE", "BENMOREITE", "PICRITE"]),
    ("mafic_plutonic",           ["GABBRO", "NORITE", "TROCTOLITE", "ANORTHOSITE", "DOLERITE", "DIABASE",
                                  "CUMULATE", "MAFIC IGNEOUS"]),
    ("felsic_plutonic",          ["GRANITE", "GRANODIORITE", "GRANITOID", "TONALITE", "TRONDHJEMITE", "APLITE",
                                  "PEGMATITE", "PORPHYRY", "LEUCOGRANITE", "I-TYPE", "S-TYPE", "A-TYPE",
                                  "CHARNOCKITE", "ADAKITE"]),
    ("intermediate_plutonic",    ["DIORITE", "MONZONITE", "MONZODIORITE"]),
]

# Fallback for O'Sullivan et al. lithology codes when the name doesn't match
OSULLIVAN_CODE_CLASS = {
    "ALK": "alkaline_silicate",        # alkali-rich igneous
    "IM":  "mafic_plutonic",           # mafic I-type granitoids and mafic igneous
    "S":   "felsic_plutonic",          # S-type and felsic I-type granitoids
    "UM":  "ultramafic",               # ultramafic rocks incl. carbonatites
    "LM":  "metamorphic",              # low/medium-grade metamorphic
    "HM":  "metamorphic",              # high-grade metamorphic / partial melts
    "AUT": "phosphorite",              # authigenic
}

# --------------------------------------------------------------------------
# 2. Origin and temperature regime per rock class
#    (the label you will eventually contrast with biogenic data)
# --------------------------------------------------------------------------
ORIGIN_BY_CLASS = {
    # rock_class                 : (origin_class,                     temp_regime)
    "carbonatite":                 ("abiotic_igneous",                "high_T"),
    "kimberlite_lamprophyre":      ("abiotic_igneous",                "high_T"),
    "ultramafic":                  ("abiotic_igneous",                "high_T"),
    "alkaline_silicate":           ("abiotic_igneous",                "high_T"),
    "felsic_volcanic":             ("abiotic_igneous",                "high_T"),
    "mafic_interm_volcanic":       ("abiotic_igneous",                "high_T"),
    "volcaniclastic":              ("abiotic_igneous",                "high_T"),
    "mafic_plutonic":              ("abiotic_igneous",                "high_T"),
    "intermediate_plutonic":       ("abiotic_igneous",                "high_T"),
    "felsic_plutonic":             ("abiotic_igneous",                "high_T"),
    "metamorphic":                 ("abiotic_metamorphic",            "high_T"),
    "ore_magmatic_hydrothermal":   ("abiotic_magmatic_hydrothermal",  "high_T"),
    "sedimentary_clastic":         ("detrital_unknown_source",        "mixed"),
    "phosphorite":                 ("ambiguous_authigenic",           "low_T"),
    "bif_chert":                   ("ambiguous_authigenic",           "low_T"),
    "fossil_bioapatite":           ("biogenic_skeletal_diagenetic",   "low_T"),
    "modern_bioapatite":           ("biogenic_skeletal",              "low_T"),
    "recent_authigenic":           ("biologically_induced_authigenic", "low_T"),
    "seamount_phosphorite":        ("ambiguous_authigenic",           "low_T"),
    "unknown":                     ("unknown",                        "unknown"),
}


def _as_bool(s: pd.Series) -> pd.Series:
    """True/False column that may have been saved as text ("True"/"False"/NA)."""
    return s.astype("string").str.strip().str.lower().eq("true").fillna(False).astype(bool)


def rock_class(name, code=None) -> str:
    """Map a reported rock name (and optional O'Sullivan code) to a rock_class."""
    s = "" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name).upper()
    for cls, keys in ROCK_CLASS_RULES:
        if any(k in s for k in keys):
            return cls
    if code is not None and code in OSULLIVAN_CODE_CLASS:
        return OSULLIVAN_CODE_CLASS[code]
    return "unknown"


def add_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Return a labels table (same index as meta)."""
    is_os = meta["dataset"].eq("OSullivan2019")
    codes = meta["context"].where(is_os)
    rc = pd.Series([rock_class(n, c if isinstance(c, str) else None)
                    for n, c in zip(meta["lithology"], codes)], index=meta.index)
    if "rock_class" in meta:                     # datasets that state their class explicitly
        given = meta["rock_class"].astype("string")
        rc = given.where(given.notna() & given.ne(""), rc)
    lab = pd.DataFrame(index=meta.index)
    lab["dataset"] = meta["dataset"]
    lab["study"] = meta["study"]
    lab["lithology"] = meta["lithology"]
    lab["rock_class"] = rc.astype(str)
    lab["origin_class"] = lab["rock_class"].map(lambda r: ORIGIN_BY_CLASS.get(r, ("unknown", "unknown"))[0])
    lab["temp_regime"] = lab["rock_class"].map(lambda r: ORIGIN_BY_CLASS.get(r, ("unknown", "unknown"))[1])
    # GEOROC: tectonic setting; O'Sullivan: lithology code (ALK, IM, S, UM, LM, HM, AUT)
    lab["setting"] = meta["context"]
    lab["data_type"] = meta["data_type"]
    # same locality AND rock class: the unit within which different studies
    # should agree if there were no lab / method offsets
    lab["site"] = meta["location"].astype("string").str.upper().fillna("?") + " || " + lab["rock_class"]
    lab["also_in_georoc"] = _as_bool(meta["also_in_georoc"]) if "also_in_georoc" in meta else False
    return lab


# --------------------------------------------------------------------------
# 3. The one function you call from the notebook
# --------------------------------------------------------------------------
DEFAULT_APATITE_ELEMENTS = REE_Y + ["Sr", "Th", "U", "Mn"]

# Feature presets
# FULL   : best GEOROC-internal set (all REE + Y + Sr, Th, U, Mn)
# SHARED : what O'Sullivan's authigenic (AUT) phosphorites also have -> use this
#          when a GEOROC-trained model is tested on the phosphorites
#          (AUT rows have no Tb, Tm, Sr, Th, U or Mn)
FEATURES_FULL = DEFAULT_APATITE_ELEMENTS
FEATURES_SHARED = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Dy", "Ho", "Y", "Er", "Yb", "Lu"]

# Values entered in GEOROC's ppm columns that are really wt% (found by study-effect
# checks: Ce/La ~ 0.0003 and P = 18 "ppm" are impossible; x 1e4 restores
# Ce/La ~ 2.6, Nd/Pr ~ 4.4 and P2O5 ~ 42 wt%, consistent with the rest of the study).
KNOWN_UNIT_ERRORS = {
    "[25837] WANG CHANGHONG (2022)": {"Ce": 1e4, "Nd": 1e4, "Sr": 1e4, "P": 1e4},
}

META_COLS = ["sample_id", "location", "lat", "lon", "age_ma", "mineral", "spot", "rim_core",
             "primary_secondary", "alteration"]


def apatite_table(processed_dir, elements=None, require=None, level="spot",
                  spots_only=True, drop_dup_studies=True, drop_unknown=False,
                  datasets=None, fix_known_errors=True):
    """Return (X, labels, meta) for apatite from GEOROC + O'Sullivan.

    elements   : which elements go in X (default: REE + Y + Sr, Th, U, Mn)
    require    : elements that must be detected for a row to be kept
                 (default: the 14 REE + Y). Use [] to keep everything.
    level      : "spot" (each analysis) or "sample" (median of spots per
                 dataset + study + sample_id; adds n_spots to meta)
    spots_only : drop compiled summary rows (O'Sullivan averages, means,
                 medians, percentiles, mineral concentrates). Whole-specimen
                 analyses (dissolved teeth, bulk phosphorites) are kept;
                 filter labels.data_type if you want spots only.
    drop_dup_studies : drop O'Sullivan studies that also exist in GEOROC
    drop_unknown     : drop rows whose rock_class is "unknown" (NOT GIVEN)
    datasets         : e.g. ["GEOROC_apatite"] to use GEOROC only (default: all)
    fix_known_errors : apply KNOWN_UNIT_ERRORS (wt% entered as ppm)
    """
    elements = list(elements or DEFAULT_APATITE_ELEMENTS)
    require = list(REE_Y if require is None else require)
    ds = Dataset.load(Path(processed_dir), "apatite_all")
    meta = ds.meta.copy()
    if "also_in_georoc" not in meta:
        meta["also_in_georoc"] = False
    X = ds.wide(variables=sorted(set(elements) | set(require)))
    X = X.reindex(columns=sorted(set(elements) | set(require)))
    if fix_known_errors:
        for study, fixes in KNOWN_UNIT_ERRORS.items():
            rows = meta["study"].eq(study)
            for el, f in fixes.items():
                if el in X.columns:
                    X.loc[rows, el] = X.loc[rows, el] * f

    keep = pd.Series(True, index=meta.index)
    if datasets is not None:
        keep &= meta["dataset"].isin(list(datasets))
    if spots_only:
        keep &= ~meta["data_type"].fillna("spot").isin(["average", "mean", "median", "percentile", "concentrate"])
    if drop_dup_studies:
        keep &= ~_as_bool(meta["also_in_georoc"])
    if require:
        keep &= X[require].notna().all(axis=1)
    labels = add_labels(meta)
    if drop_unknown:
        keep &= labels["rock_class"].ne("unknown")

    X, labels, meta = X[keep], labels[keep], meta[keep]

    if level == "sample":
        key = meta["dataset"].astype(str) + "|" + meta["study"].astype(str) + "|" + meta["sample_id"].astype(str)
        Xs = X.groupby(key).median()
        labels = labels.groupby(key).first().reindex(Xs.index)
        m = meta.groupby(key).first().reindex(Xs.index)
        m["n_spots"] = key.value_counts().reindex(Xs.index)
        X, meta = Xs, m
        X.index.name = labels.index.name = meta.index.name = "sample_key"
    else:
        meta = meta.assign(n_spots=1)

    X = X[[e for e in elements if e in X.columns]]
    meta = meta[[c for c in META_COLS + ["n_spots"] if c in meta.columns]]
    return X, labels, meta


# --------------------------------------------------------------------------
# 4. Small feature helpers
# --------------------------------------------------------------------------
def log_features(X: pd.DataFrame) -> pd.DataFrame:
    """log10 of ppm (concentrations span 5+ orders of magnitude)."""
    return np.log10(X)


def clr(X: pd.DataFrame) -> pd.DataFrame:
    """Centred log-ratio on the rows (only meaningful for complete rows)."""
    L = np.log(X)
    return L.sub(L.mean(axis=1), axis=0)
