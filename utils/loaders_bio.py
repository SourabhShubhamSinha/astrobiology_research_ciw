"""
Loaders for biogenic and low-temperature apatite datasets.

Each returns (Dataset, report) in the same format as utils.loaders, so they
combine with GEOROC / O'Sullivan into apatite_all. Each meta table carries an
explicit ``rock_class`` (see ORIGIN_BY_CLASS in utils.tables for what each
class means) because these files have no rock-name column to classify.

data_type values used here:
    spot              in-situ LA-ICP-MS spot
    whole_specimen    dissolved whole tooth / nodule / rock (solution ICP-MS, INAA)
    profile_interior  de Winter et al.: modelled unaltered enamel value (C0)
    profile_surface   de Winter et al.: modelled diagenetic surface value (C1)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import Dataset, _melt, _mark_preferred


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_pangaea_tab(path) -> tuple[pd.DataFrame, str]:
    """Read a PANGAEA .tab file -> (table, header text)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    head, body = text.split("*/", 1) if text.lstrip().startswith("/*") else ("", text)
    return pd.read_csv(io.StringIO(body.lstrip("\n")), sep="\t"), head


def _element_columns(df: pd.DataFrame, units=("mg/kg", "µg/g")) -> dict:
    """{'La [mg/kg]': 'La', ...} for single-element concentration columns."""
    out = {}
    for c in df.columns:
        for u in units:
            if c.endswith(f"[{u}]"):
                el = c.split(" [")[0].strip()
                if el.isalpha() and len(el) <= 2:
                    out[c] = el
    return out


def _dataset(name, raw, el_map, meta, scale=None, positive=None):
    raw = raw.copy()
    raw.index = [f"{name}:{i:06d}" for i in range(len(raw))]
    raw.index.name = "analysis_id"
    meta = meta.copy()
    meta.index = raw.index
    series = {(el, c): raw[c] for c, el in el_map.items()}
    sc = {(el, c): scale[c] for c, el in el_map.items() if scale and c in scale}
    long, unparsed = _melt(series, {k: "ppm" for k in series}, {k: "ppm" for k in series},
                           positive or {}, scale=sc)
    long = _mark_preferred(long, {"ppm": 0})
    meta.insert(0, "dataset", name)
    meta["mineral_group"] = "apatite"
    meta["mineral"] = meta.get("mineral", "apatite")
    return Dataset(name, meta, long), {"analyses_kept": len(meta), "unparsed_cells_by_column": unparsed.to_dict()}


# --------------------------------------------------------------------------
# Huck et al. 2017 - Eocene fossil fish teeth, REE + Y (PANGAEA.875928)
# --------------------------------------------------------------------------
def load_huck2017(path, name="Huck2017_fishteeth"):
    df, _ = read_pangaea_tab(path)
    meta = pd.DataFrame({
        "study": "Huck et al. 2017 Paleoceanography (PANGAEA.875928)",
        "sample_id": df["Sample label"].astype(str),
        "location": df["Event"].astype(str) + " / " + df["Sample comment"].astype(str),
        "lat": df["Latitude"], "lon": df["Longitude"],
        "lithology": "fossil fish teeth (Eocene, marine sediment)",
        "context": df["Sample comment"].astype(str),
        "rock_class": "fossil_bioapatite",
        "data_type": "whole_specimen",
        "age_ma": np.nan,
    })
    return _dataset(name, df, _element_columns(df), meta)


# --------------------------------------------------------------------------
# Huck et al. 2016 - fossil fish teeth majors + traces, IODP U1356A (PANGAEA.858098)
# --------------------------------------------------------------------------
HUCK2016_WT_PERCENT = ["Fe", "Mn", "Al", "Ca", "Na", "Mg", "Si", "K", "Ti"]


def load_huck2016(path, name="Huck2016_fishteeth"):
    """Fe, Mn, Al, Ca, Na, Mg, Si, K, Ti are labelled mg/kg but the PANGAEA
    parameter comments say wt % -> converted to ppm (x 1e4).
    Caution: Ca is 15-50 'wt %', above stoichiometric Ca in apatite (~40 wt%)
    for some rows; it may be reported as CaO. Treat Sr/Ca etc. as uncertain
    by up to ~1.4x until checked against the paper.
    "/1" = uncleaned, "/2" = fully cleaned sample (kept as `cleaning`)."""
    df, _ = read_pangaea_tab(path)
    el_map = _element_columns(df)
    scale = {c: 1e4 for c, el in el_map.items() if el in HUCK2016_WT_PERCENT}
    lab = df["Sample label"].astype(str)
    meta = pd.DataFrame({
        "study": "Huck et al. 2016 G-cubed (PANGAEA.858098)",
        "sample_id": lab,
        "location": "IODP 318-U1356A, Wilkes Land margin",
        "lat": -63.31023, "lon": 135.99896,
        "lithology": "fossil fish teeth (Eocene, shallow marine)",
        "context": "sediment colour " + df["Color descr"].astype(str),
        "rock_class": "fossil_bioapatite",
        "data_type": "whole_specimen",
        "cleaning": np.where(lab.str.endswith("/2"), "full oxidative-reductive",
                             np.where(lab.str.endswith("/1"), "water + methanol only", "standard")),
    })
    return _dataset(name, df, el_map, meta, scale=scale)


# --------------------------------------------------------------------------
# Basak et al. 2011 - Pb, U, Th in fossil and MODERN fish teeth (PANGAEA.780147)
# --------------------------------------------------------------------------
def load_basak2011(path, name="Basak2011_fishteeth"):
    df, _ = read_pangaea_tab(path)
    modern = df["Age [ka BP]"].fillna(0).eq(0)
    meta = pd.DataFrame({
        "study": "Basak et al. 2011 Chem Geol (PANGAEA.780147)",
        "sample_id": df["Sample label"].astype(str).str.replace("_Replicate", "", regex=False),
        "location": np.where(modern, "modern fish (market specimens)", "ODP 177-1090, South Atlantic"),
        "lat": np.where(modern, np.nan, -42.913617), "lon": np.where(modern, np.nan, 8.899817),
        "lithology": np.where(modern, "modern fish teeth", "fossil fish teeth (Miocene, deep sea)"),
        "context": np.where(modern, "modern", "fossil"),
        "rock_class": np.where(modern, "modern_bioapatite", "fossil_bioapatite"),
        "data_type": "whole_specimen",
        "age_ma": df["Age [ka BP]"] / 1000.0,
    })
    return _dataset(name, df, _element_columns(df), meta)


# --------------------------------------------------------------------------
# Lumiste et al. 2019 Chem Geol - Recent sedimentary apatite grains, LA-ICP-MS
# (tables extracted from supplementary PDFs mmc1 = grain centres, mmc2 = exteriors)
# --------------------------------------------------------------------------
def load_lumiste2019(csv_path, name="Lumiste2019_recent"):
    df = pd.read_csv(csv_path, dtype=str)
    el_cols = [c for c in df.columns if c not in ("Sample", "Core", "position", "spot_size_um")]
    meta = pd.DataFrame({
        "study": "Lumiste et al. 2019 Chem Geol (ScienceDirect PII S0009254119303602)",
        "sample_id": df["Core"] + "-" + df["position"] + "-" + df["Sample"],
        "location": "core " + df["Core"],
        "lithology": "Recent sedimentary (authigenic) apatite grain, " + df["position"].str.replace("_", " "),
        "context": df["position"],
        "rock_class": "recent_authigenic",
        "data_type": "spot",
        "spot_size_um": pd.to_numeric(df["spot_size_um"]),
    })
    return _dataset(name, df, {c: c for c in el_cols}, meta)


# --------------------------------------------------------------------------
# Tambiev & Baturin 1982 - bulk phosphorites (INAA, partial REE)
# --------------------------------------------------------------------------
def load_tambiev1982(folder, name="Tambiev1982_phosphorite", min_p2o5=15.0):
    """Tables 1, 2 and 4 (Table 3 has only total REE). Bulk rock, so
    data_type = whole_specimen. Rows with P2O5 < min_p2o5 % (oozes, muds,
    insoluble residues) are dropped. Zeros = below detection."""
    folder = Path(folder)
    parts = []
    for tab, cls, setting in [(1, "recent_authigenic", "Namibian shelf, Recent"),
                              (2, "recent_authigenic", "Chile/Peru shelf, Late Quaternary"),
                              (4, "seamount_phosphorite", "Pacific seamounts, Cretaceous-Neogene")]:
        f = next(folder.rglob(f"*Tab{tab}.tab"))
        d, _ = read_pangaea_tab(f)
        d["_table"], d["_class"], d["_setting"] = tab, cls, setting
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    p = pd.to_numeric(df["P2O5 [%]"], errors="coerce")
    df = df[p >= min_p2o5].reset_index(drop=True)
    el_map = _element_columns(df)
    meta = pd.DataFrame({
        "study": "Tambiev & Baturin 1982 (PANGAEA.778166)",
        "sample_id": df["Event"].astype(str) + " | " + df["Samp type"].astype(str),
        "location": df["_setting"] + " / " + df["Event"].astype(str),
        "lat": df["Latitude"], "lon": df["Longitude"],
        "lithology": df["Samp type"].astype(str),
        "context": df["_setting"],
        "rock_class": df["_class"],
        "data_type": "whole_specimen",
        "p2o5_wt": pd.to_numeric(df["P2O5 [%]"], errors="coerce"),
    })
    return _dataset(name, df, el_map, meta)


# --------------------------------------------------------------------------
# de Winter et al. - archaeological human enamel, micro-XRF line scans with a
# diffusion-adsorption model: C0 = unaltered interior, C1 = diagenetic surface
# (failed fits are removed - see below)
# --------------------------------------------------------------------------
def load_dewinter(model_xlsx, name="deWinter_enamel"):
    d = pd.read_excel(model_xlsx, header=0, skiprows=[1])
    d.columns = ["Locality", "Locality_type", "Specimen", "Tooth", "Profile", "From_side", "leach_from",
                 "element", "element_no", "C0", "C0_se", "C1", "C1_se", "C2", "C2_se", "x0", "x0_se",
                 "t", "t_se", "C1_C0"] + list(d.columns[20:])
    d = d[d["leach_from"].eq("out")]                       # profiles from the outer enamel surface
    # Drop failed model fits: non-positive values, relative standard error > 50 %,
    # or values above 50 wt% (impossible in apatite). ~10-30 % of fits per element fail.
    for col, se in [("C0", "C0_se"), ("C1", "C1_se")]:
        v, e = pd.to_numeric(d[col], errors="coerce"), pd.to_numeric(d[se], errors="coerce")
        bad = (v <= 0) | (e.abs() / v.abs() > 0.5) | (v > 5e5)
        d[col] = v.where(~bad)
    key = ["Locality", "Locality_type", "Specimen", "Tooth", "Profile", "From_side"]
    rows = []
    for endmember, col, cls in [("profile_interior", "C0", "modern_bioapatite"),
                                ("profile_surface", "C1", "fossil_bioapatite")]:
        w = d.pivot_table(index=key, columns="element", values=col, aggfunc="median").reset_index()
        w["data_type"], w["rock_class"] = endmember, cls
        rows.append(w)
    w = pd.concat(rows, ignore_index=True)
    els = [c for c in w.columns if c not in key + ["data_type", "rock_class"]]
    meta = pd.DataFrame({
        "study": "de Winter et al. (Mendeley rjw79x25j5), Late Neolithic/Early Chalcolithic molars",
        "sample_id": (w["Locality"].astype(str) + "-" + w["Specimen"].astype(str) + "-" + w["Tooth"].astype(str)
                      + "-" + w["data_type"]),
        "location": "Rioja Alavesa, Spain / " + w["Locality"].astype(str) + " (" + w["Locality_type"].astype(str) + ")",
        "lat": 42.55, "lon": -2.6,
        "lithology": np.where(w["data_type"].eq("profile_interior"),
                              "human tooth enamel, modelled unaltered interior",
                              "human tooth enamel, modelled diagenetic surface"),
        "context": "archaeological (~5 ka)",
        "rock_class": w["rock_class"],
        "data_type": w["data_type"],
        "age_ma": 0.005,
        "profile": w["Profile"].astype(str) + "-" + w["From_side"].astype(str),
    })
    positive = {(el, el): True for el in els}
    return _dataset(name, w, {e: e for e in els}, meta, positive=positive)


BIO_FILES = {
    "huck2017": "SO_Eocene_RRE.tab",
    "huck2016": "318-U1356A_major_trace_fish.tab",
    "basak2011": "177-1090_Pb_U_Th_fish.tab",
    "lumiste2019": "Lumiste2019_ChemGeol_LAICPMS_apatite.csv",
    "tambiev1982": "Tambiev-Baturin-1982",
    "dewinter": ("Data for Trace element distributions in Late NeolithicEarly Chalcolithic human molars "
                 "from the Rioja Alavesa region (north-central Spain) Biogenic or Diagenetic/SI/SI_model_data.xlsx"),
}


def load_all_bio(raw_dir, verbose=True):
    raw_dir = Path(raw_dir)
    fns = {"huck2017": load_huck2017, "huck2016": load_huck2016, "basak2011": load_basak2011,
           "lumiste2019": load_lumiste2019, "tambiev1982": load_tambiev1982, "dewinter": load_dewinter}
    out, reports = {}, {}
    for k, fn in fns.items():
        p = raw_dir / BIO_FILES[k]
        if not p.exists():
            if verbose:
                print(f"  (skipped {k}: {p.name} not found)")
            continue
        out[k], reports[k] = fn(p)
    return out, reports
