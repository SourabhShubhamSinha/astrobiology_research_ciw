"""
Loaders that turn the raw apatite and pyrite compilations into one common format.

Every loader returns a ``Dataset`` with two tables:

meta  : one row per analysis (index = ``analysis_id``) with sample / study /
        location / lithology / context information.
long  : one row per (analysis_id, variable) measurement, with columns
        ``variable, value, unit, censored, lod, unc, source, preferred``.
        - censored = True  -> reported below detection ("<0.5", "<LOD", "n.d.",
          or a concentration <= 0). ``value`` is NaN, ``lod`` holds the reported
          limit when one was given.
        - source    -> where the number came from (e.g. "ppm", "wt%", "oxide:CaO").
        - preferred -> for elements measured more than one way in the same
          analysis (e.g. Mn(PPM) and MnO(WT%)), the single value to use.

``Dataset.wide()`` pivots the long table into a familiar analysis x variable
matrix (preferred values only; censored cells are NaN unless you ask for a
substitution).

Nothing in the raw files is modified. Typical use (from a notebook)::

    from utils.loaders import build_all
    datasets = build_all(raw_dir, processed_dir)   # loads + saves everything
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ATOMIC_WEIGHT = {
    "H": 1.008, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
    "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098,
    "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971,
    "Br": 79.904, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Ag": 107.87, "Cd": 112.41, "In": 114.82,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50,
    "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97,
    "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21, "Pt": 195.08,
    "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
    "Th": 232.04, "U": 238.03, "He": 4.0026, "Ne": 20.180, "Ar": 39.948,
    "Kr": 83.798, "Xe": 131.29,
}
ELEMENTS = set(ATOMIC_WEIGHT)

REE = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
REE_Y = REE[:10] + ["Y"] + REE[10:]  # Y placed next to Ho (similar ionic radius)

# Chondrite (CI) values, McDonough & Sun (1995), ppm
CI_CHONDRITE = {
    "La": 0.237, "Ce": 0.613, "Pr": 0.0928, "Nd": 0.457, "Sm": 0.148,
    "Eu": 0.0563, "Gd": 0.199, "Tb": 0.0361, "Dy": 0.246, "Ho": 0.0546,
    "Er": 0.160, "Tm": 0.0247, "Yb": 0.161, "Lu": 0.0246, "Y": 1.57,
}

# Oxide-type GEOROC columns that are NOT cation oxides (no element derivation)
_NON_CATION_OXIDES = {"H2O", "H2OP", "H2OM", "CO2", "SO2", "SO3", "OH", "O", "O2", "LOI"}

# Text that means "below detection / not detected"
_CENSOR_WORDS = {"<lod", "lod", "bdl", "b.d.l.", "b.d.l", "<dl", "n.d.", "nd", "n.d", "<mdl", "bd", "b.d."}
# Text that means "no value"
_MISSING_WORDS = {"", "-", "--", "na", "n.a.", "nan", "none", "n/a", "not analysed", "not analyzed"}


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    name: str
    meta: pd.DataFrame
    long: pd.DataFrame

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"Dataset({self.name!r}: {len(self.meta):,} analyses, "
                f"{self.long['variable'].nunique()} variables, {len(self.long):,} measurements)")

    def wide(self, variables=None, unit=None, censored="nan", with_meta=False) -> pd.DataFrame:
        """Analysis x variable matrix of preferred values.

        variables : list of variable names to keep (default: all)
        unit      : keep only variables in this unit, e.g. "ppm"
        censored  : "nan" (default) | "half_lod" | "lod" -> what to put in
                    below-detection cells (falls back to NaN if no LOD reported)
        with_meta : prepend the meta columns
        """
        L = self.long[self.long["preferred"]]
        if unit is not None:
            L = L[L["unit"] == unit]
        if variables is not None:
            L = L[L["variable"].isin(variables)]
        v = L["value"].copy()
        if censored in ("half_lod", "lod"):
            fill = L["lod"] * (0.5 if censored == "half_lod" else 1.0)
            v = v.where(~L["censored"], fill)
        W = L.assign(v=v).pivot(index="analysis_id", columns="variable", values="v")
        W = W.reindex(self.meta.index)
        if variables is not None:
            W = W.reindex(columns=[c for c in variables if c in W.columns])
        W.columns.name = None
        if with_meta:  # variables already present in meta (e.g. age_ma) stay in meta
            return self.meta.join(W.drop(columns=[c for c in W.columns if c in self.meta.columns]))
        return W

    def completeness(self, variables) -> pd.Series:
        """Fraction of analyses with a detected (non-censored) value for each variable."""
        W = self.wide(variables=variables)
        return W.notna().mean().reindex(variables).fillna(0).sort_values(ascending=False)

    def save(self, out_dir: Path, stem: str | None = None, csv_wide=True) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or self.name
        meta = self.meta.copy()
        for c in meta.columns:                   # parquet needs one type per column
            if meta[c].dtype == object:
                meta[c] = meta[c].astype("string")
        meta.to_parquet(out_dir / f"{stem}_meta.parquet")
        self.long.to_parquet(out_dir / f"{stem}_long.parquet", index=False)
        if csv_wide:
            self.wide(with_meta=True).to_csv(out_dir / f"{stem}_wide.csv")

    @classmethod
    def load(cls, out_dir: Path, stem: str) -> "Dataset":
        out_dir = Path(out_dir)
        return cls(stem, pd.read_parquet(out_dir / f"{stem}_meta.parquet"),
                   pd.read_parquet(out_dir / f"{stem}_long.parquet"))


def combine(datasets, name: str) -> Dataset:
    """Stack several Datasets (e.g. GEOROC + O'Sullivan apatite) into one."""
    meta = pd.concat([d.meta for d in datasets], axis=0)
    long = pd.concat([d.long for d in datasets], axis=0, ignore_index=True)
    assert meta.index.is_unique, "analysis_id collision between datasets"
    return Dataset(name, meta, long)


# --------------------------------------------------------------------------
# Value parsing
# --------------------------------------------------------------------------

def _parse_text(x: str):
    """Parse one non-numeric cell -> (value, censored, lod, status)."""
    s = str(x).strip().replace(",", ".").replace("−", "-")
    low = s.lower()
    if low in _MISSING_WORDS:
        return np.nan, False, np.nan, "missing"
    if low in _CENSOR_WORDS or low.startswith("<lod") or low.startswith("bdl"):
        return np.nan, True, np.nan, "censored"
    m = re.fullmatch(r"<+\s*=?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", s)
    if m:
        return np.nan, True, float(m.group(1)), "censored"
    m = re.fullmatch(r"=\s*0?\.5\s*\*\s*([0-9]*\.?[0-9]+)", s)
    if m:  # spreadsheet formula "=0.5*LOD" used by some authors for below-detection cells
        return np.nan, True, float(m.group(1)), "censored"
    m = re.fullmatch(r">\s*([0-9]*\.?[0-9]+)", s)
    if m:  # above range: keep the bound as a value, it is rare
        return float(m.group(1)), False, np.nan, "ok"
    try:
        return float(s), False, np.nan, "ok"
    except ValueError:
        return np.nan, False, np.nan, "unparsed"


def parse_series(s: pd.Series, positive_only: bool = True) -> pd.DataFrame:
    """Vectorised parse of a raw column.

    Returns DataFrame(value, censored, lod, status) aligned with ``s``.
    status is one of: ok, missing, censored, unparsed.
    positive_only=True treats values <= 0 as below detection (right for
    concentrations, wrong for delta values / epsilon values).
    """
    num = pd.to_numeric(s, errors="coerce")
    out = pd.DataFrame({"value": num.astype(float), "censored": False,
                        "lod": np.nan, "status": np.where(num.notna(), "ok", "missing")},
                       index=s.index)
    txt_mask = num.isna() & s.notna()
    if txt_mask.any():
        parsed = s[txt_mask].map(_parse_text)
        out.loc[txt_mask, "value"] = [p[0] for p in parsed]
        out.loc[txt_mask, "censored"] = [p[1] for p in parsed]
        out.loc[txt_mask, "lod"] = [p[2] for p in parsed]
        out.loc[txt_mask, "status"] = [p[3] for p in parsed]
    if positive_only:
        le0 = out["value"] <= 0
        out.loc[le0, "value"] = np.nan
        out.loc[le0, "censored"] = True
        out.loc[le0, "status"] = "censored"
    return out


def _melt(frame_by_var: dict, units: dict, sources: dict, positive: dict,
          unc_by_var: dict | None = None, scale: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Build the long table from {variable_key: raw Series indexed by analysis_id}.

    scale[key] multiplies value, lod and unc after parsing (unit conversion).
    """
    parts, unparsed = [], {}
    scale = scale or {}
    for key, s in frame_by_var.items():
        p = parse_series(s, positive_only=positive.get(key, True))
        f = scale.get(key, 1.0)
        if f != 1.0:
            p["value"] *= f
            p["lod"] *= f
        n_bad = int((p["status"] == "unparsed").sum())
        if n_bad:
            unparsed[key] = n_bad
        p = p[p["status"].isin(["ok", "censored"])]
        if p.empty:
            continue
        var = key[0] if isinstance(key, tuple) else key
        p = p.drop(columns="status").assign(variable=var, unit=units[key], source=sources[key])
        if unc_by_var and key in unc_by_var:
            p["unc"] = pd.to_numeric(unc_by_var[key], errors="coerce").reindex(p.index) * f
        else:
            p["unc"] = np.nan
        parts.append(p.rename_axis("analysis_id").reset_index())
    long = pd.concat(parts, ignore_index=True)
    long = long[["analysis_id", "variable", "value", "unit", "censored", "lod", "unc", "source"]]
    long["censored"] = long["censored"].astype(bool)
    return long, pd.Series(unparsed, dtype=int)


def _mark_preferred(long: pd.DataFrame, source_rank: dict) -> pd.DataFrame:
    """For duplicate (analysis, variable) pairs keep the best source.

    A detected value always beats a censored one; then lower rank wins.
    """
    rank = long["source"].map(lambda s: source_rank.get(s.split(":")[0], 9))
    order = (long.assign(_cens=long["censored"].astype(int), _rank=rank)
                 .sort_values(["analysis_id", "variable", "_cens", "_rank"], kind="mergesort"))
    first = ~order.duplicated(["analysis_id", "variable"])
    long = long.copy()
    long["preferred"] = False
    long.loc[order.index[first], "preferred"] = True
    return long


def _snake(c: str) -> str:
    c = re.sub(r"[^0-9a-zA-Z]+", "_", str(c)).strip("_").lower()
    return c


# --------------------------------------------------------------------------
# GEOROC precompiled mineral files (e.g. 2024-12-SGFTFN_APATITES.csv)
# --------------------------------------------------------------------------

_GEOROC_META = ["CITATION", "SAMPLE NAME", "TECTONIC SETTING", "LOCATION", "LOCATION COMMENT",
                "LATITUDE (MIN.)", "LATITUDE (MAX.)", "LONGITUDE (MIN.)", "LONGITUDE (MAX.)",
                "LAND/SEA (SAMPLING)", "ELEVATION (MIN.)", "ELEVATION (MAX.)", "ROCK NAME",
                "ROCK TEXTURE", "DRILLING DEPTH (MIN.)", "DRILLING DEPTH (MAX.)", "ALTERATION",
                "MINERAL", "SPOT", "CRYSTAL", "RIM/CORE (MINERAL GRAINS)", "GRAIN SIZE",
                "PRIMARY/SECONDARY"]


def _split_formula(f: str):
    """'FE2O3' -> [('Fe', 2), ('O', 3)] using greedy two-letter symbols."""
    out, i = [], 0
    while i < len(f):
        two = f[i:i + 2].capitalize()
        if len(two) == 2 and two in ELEMENTS:
            sym, i = two, i + 2
        elif f[i].upper() in ELEMENTS:
            sym, i = f[i].upper(), i + 1
        else:
            raise ValueError(f)
        m = re.match(r"\d+", f[i:])
        n = int(m.group()) if m else 1
        i += len(m.group()) if m else 0
        out.append((sym, n))
    return out


def _oxide_info(name: str):
    """'FE2O3T' -> ('Fe2O3T', 'Fe', factor element/oxide) or None if not a cation oxide."""
    base = name[:-1] if name.endswith("T") and name not in ("PT",) else name
    if base in _NON_CATION_OXIDES or "O" not in base:
        return None
    try:
        parts = _split_formula(base)
    except ValueError:
        return None
    cations = [(s, n) for s, n in parts if s != "O"]
    if len(cations) != 1 or not any(s == "O" for s, _ in parts):
        return None
    el, n = cations[0]
    mass = sum(ATOMIC_WEIGHT[s] * k for s, k in parts)
    pretty = "".join(f"{s}{k if k > 1 else ''}" for s, k in parts) + ("T" if base != name else "")
    return pretty, el, ATOMIC_WEIGHT[el] * n / mass


def _georoc_variable(col: str):
    """Map a GEOROC column name to (variable, unit, source, positive_only, element_info).

    element_info is (element, factor_to_ppm) when the column can supply an
    element concentration in ppm, else None.
    """
    m = re.fullmatch(r"([A-Z0-9_]+)\((.+)\)", col)
    if m:
        what, unit = m.group(1), m.group(2)
        if unit in ("PPM", "PPB", "WT%"):
            # element or isotope-of-element concentration
            sym = re.fullmatch(r"([A-Z]{1,2})(\d*)", what)
            if sym and sym.group(1).capitalize() in ELEMENTS:
                el, iso = sym.group(1).capitalize(), sym.group(2)
                to_ppm = {"PPM": 1.0, "PPB": 1e-3, "WT%": 1e4}[unit]
                if iso:
                    # Mono-isotopic or near-total: treat as the element; otherwise keep isotope
                    if (el, iso) in {("Mn", "55"), ("Th", "232")}:
                        return el, "ppm", f"{unit.lower()}:{el}{iso}", True, (el, to_ppm)
                    return f"{el}{iso}", "ppm", unit.lower(), True, None
                return el, "ppm", unit.lower(), True, (el, to_ppm)
            if what == "FET" and unit == "WT%":
                return "Fe", "ppm", "wt%", True, ("Fe", 1e4)
            if unit == "WT%":
                ox = _oxide_info(what)
                if ox:
                    pretty, el, f = ox
                    return pretty, "wt%", "wt%", True, (el, f * 1e4)
                return what.replace("H2OP", "H2O+").replace("H2OM", "H2O-"), "wt%", "wt%", True, None
        if what.startswith("D") and unit.startswith("VS") or unit == "PER MIL":
            ref = unit.replace("VS ", "").replace(" ", "")
            iso = what[1:]
            pretty = "d" + re.sub(r"(\d+)([A-Z]+)", lambda k: k.group(1) + k.group(2).capitalize(), iso)
            if what == "DD":
                pretty = "dD"
            return f"{pretty}_{ref}" if ref != "PERMIL" else pretty, "permil", "raw", False, None
        if what.startswith("AGE"):
            return what.lower().replace("age_", "age_") + f"_{unit.lower()}", unit.lower(), "raw", True, None
        return f"{what}_{unit}", unit, "raw", True, None
    # ratios and epsilon values have no unit in the header
    pos = not col.startswith("EPSILON")
    return col, "ratio" if pos else "epsilon", "raw", pos, None


def load_georoc_mineral(path, dataset="GEOROC", mineral_filter=None) -> tuple[Dataset, dict]:
    """Load a GEOROC precompiled mineral CSV.

    Returns (Dataset, report). Element concentrations from ppm, ppb, element
    wt% and oxide wt% columns are all converted to ppm and stored under the
    element symbol (e.g. 'Mn'), with ``source`` telling them apart; the best
    one per analysis is marked ``preferred`` (ppm > ppb > wt% > oxide).
    Oxides are also kept in wt% under their own name (e.g. 'MnO').
    """
    path = Path(path)
    raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
    report = {"rows_in_file": len(raw)}
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    raw = raw[raw["SAMPLE NAME"].notna()]                  # footer = reference list
    report["rows_after_dropping_reference_footer"] = len(raw)
    n0 = len(raw)
    raw = raw.drop_duplicates()
    report["exact_duplicate_rows_dropped"] = n0 - len(raw)
    if mineral_filter is not None:
        raw = raw[raw["MINERAL"].astype(str).str.upper().str.contains(mineral_filter.upper())]
    raw = raw.reset_index(drop=True)
    raw.index = [f"{dataset}:{i:06d}" for i in range(len(raw))]
    raw.index.name = "analysis_id"

    num_cols = [c for c in raw.columns if c not in _GEOROC_META]
    series, units, sources, positive, scale = {}, {}, {}, {}, {}
    for c in num_cols:
        var, unit, src, pos, elinfo = _georoc_variable(c)
        key = (var, c)
        series[key], units[key], sources[key], positive[key] = raw[c], unit, src, pos
        if elinfo and unit == "wt%":              # oxide wt% -> also element ppm
            el, f = elinfo
            ekey = (el, c + "->ppm")
            series[ekey], units[ekey], positive[ekey], scale[ekey] = raw[c], "ppm", True, f
            sources[ekey] = f"oxide:{var}"
        elif elinfo and elinfo[1] != 1.0:         # element ppb or wt% -> ppm
            scale[key] = elinfo[1]

    long, unparsed = _melt(series, units, sources, positive, scale=scale)
    long = _mark_preferred(long, {"ppm": 0, "ppb": 1, "wt%": 2, "oxide": 3, "raw": 0})
    has_data = long.loc[long["value"].notna(), "analysis_id"].unique()
    report["rows_without_any_detected_value_dropped"] = len(raw) - len(has_data)
    report["unparsed_cells_by_column"] = unparsed.to_dict()

    lat = raw[["LATITUDE (MIN.)", "LATITUDE (MAX.)"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    lon = raw[["LONGITUDE (MIN.)", "LONGITUDE (MAX.)"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    meta = pd.DataFrame({
        "dataset": dataset,
        "mineral_group": "apatite" if "APATITE" in path.name.upper() else path.stem.split("_")[-1].lower(),
        "study": raw["CITATION"],
        "sample_id": raw["SAMPLE NAME"].astype(str),
        "lithology": raw["ROCK NAME"],
        "context": raw["TECTONIC SETTING"],
        "location": raw["LOCATION"],
        "lat": lat, "lon": lon,
        "mineral": raw["MINERAL"],
        "spot": raw["SPOT"],
        "rim_core": raw["RIM/CORE (MINERAL GRAINS)"],
        "primary_secondary": raw["PRIMARY/SECONDARY"],
        "alteration": raw["ALTERATION"],
        "data_type": "spot",
    }, index=raw.index)
    age = long[(long["variable"] == "age_ma") & long["preferred"]].set_index("analysis_id")["value"]
    meta["age_ma"] = age.reindex(meta.index)
    meta = meta.loc[has_data]
    long = long[long["analysis_id"].isin(has_data)].reset_index(drop=True)
    report["analyses_kept"] = len(meta)
    return Dataset(dataset, meta, long), report


# --------------------------------------------------------------------------
# O'Sullivan et al. (2019/2020) PANGAEA apatite compilation (.tab)
# --------------------------------------------------------------------------

def load_osullivan(path, dataset="OSullivan2019") -> tuple[Dataset, dict]:
    """PANGAEA doi:10.1594/PANGAEA.906570 (CC-BY-NC-SA-4.0).

    Keeps the measured elements (mg/kg = ppm); the pre-computed normalised
    values and anomalies are dropped so they can be recomputed consistently.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    body = text.split("*/", 1)[1].lstrip("\n") if text.lstrip().startswith("/*") else text
    raw = pd.read_csv(io.StringIO(body), sep="\t")
    report = {"rows_in_file": len(raw)}
    raw = raw.reset_index(drop=True)
    raw.index = [f"{dataset}:{i:06d}" for i in range(len(raw))]
    raw.index.name = "analysis_id"

    el_cols = {c: c.split(" [")[0] for c in raw.columns if c.endswith("[mg/kg]")}
    series = {(el, c): raw[c] for c, el in el_cols.items()}
    units = {k: "ppm" for k in series}
    sources = {k: "ppm" for k in series}
    long, unparsed = _melt(series, units, sources, {})
    long = _mark_preferred(long, {"ppm": 0})

    meta = pd.DataFrame({
        "dataset": dataset,
        "mineral_group": "apatite",
        "study": raw["Reference"],
        "sample_id": raw["Analysis"].astype(str),
        "lithology": raw["Lithology (Name given by cited author)"],
        "context": raw["Lithology (Lithology Code, see explanati...)"],   # ALK, IM, LM, HM, S, UM, AUT, ...
        "data_type": raw["Description (Data Type)"].str.lower(),
        "mineral": "apatite",
    }, index=raw.index)
    report["analyses_kept"] = len(meta)
    report["unparsed_cells_by_column"] = unparsed.to_dict()
    return Dataset(dataset, meta, long), report


# --------------------------------------------------------------------------
# Pyrite: Mukherjee & Large (2020, Geology) Table S1 - marine sedimentary pyrite
# --------------------------------------------------------------------------

def load_mukherjee_large_2020(path, dataset="MukherjeeLarge2020") -> tuple[Dataset, dict]:
    """G47890_SuppTab1.xlsx: LA-ICP-MS of sedimentary pyrite, 0.1-3515 Ma.

    NOTE: columns to the right of Bi_Py (Geochronology..., Location, Country,
    Formation, Description, ...) are a *separate sample table pasted beside the
    data* and are NOT row-aligned with the analyses, so they are ignored here.
    """
    raw = pd.read_excel(path, sheet_name=0, header=1)
    report = {"rows_in_file": len(raw)}
    raw = raw.iloc[:, :23].copy()                # analysis block only (sample .. Bi_Py)
    raw = raw.rename(columns={raw.columns[0]: "sample_id"})
    raw = raw[raw["Location"].astype(str).str.strip().ne("All averages")]
    el_cols = [c for c in raw.columns if str(c).endswith("_Py")]
    has_any = raw[el_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    raw = raw[has_any].reset_index(drop=True)
    report["rows_with_data"] = len(raw)
    raw.index = [f"{dataset}:{i:06d}" for i in range(len(raw))]
    raw.index.name = "analysis_id"

    def var_name(c):
        c = c.replace("_Py", "")
        if c == "208Pb":
            return "Pb"
        if c in ("206Pb", "207Pb"):
            return f"Pb_{c[:3]}est"   # Pb calculated from the 206 / 207 isotope
        return c

    series = {(var_name(c), c): raw[c] for c in el_cols}
    long, unparsed = _melt(series, {k: "ppm" for k in series}, {k: "ppm" for k in series}, {})
    long = _mark_preferred(long, {"ppm": 0})
    meta = pd.DataFrame({
        "dataset": dataset,
        "mineral_group": "pyrite",
        "study": "Mukherjee & Large 2020 (compiling Large et al. 2014, 2015a,b)",
        "sample_id": raw["sample_id"].astype("string"),
        "location": raw["Location"].astype("string").str.strip(),
        "age_ma": pd.to_numeric(raw["Age Ma"], errors="coerce").fillna(pd.to_numeric(raw["Age Ma.1"], errors="coerce")),
        "context": "marine sedimentary pyrite (black shale)",
        "mineral": "pyrite",
        "data_type": "spot",
    }, index=raw.index)
    report["analyses_kept"] = len(meta)
    report["unparsed_cells_by_column"] = unparsed.to_dict()
    return Dataset(dataset, meta, long), report


# --------------------------------------------------------------------------
# Pyrite: Gregory et al. sedimentary pyrite nodules (Appendix 3, transposed)
# --------------------------------------------------------------------------

def load_gregory_nodules(appendix_path, sample_table_path=None, dataset="GregoryNodules") -> tuple[Dataset, dict]:
    """'Appendix 3 LA-icpms data': LA-ICP-MS spots across sedimentary pyrite nodules.

    Columns: spot size, depth, analysis name, (unnamed) sample, 'median (Ni)'
    (a single summary cell - ignored), spot number, S3_Py ... Bi_Py (ppm; Pb is
    given three times, calculated from 206, 207 and 208), then 1-sigma
    uncertainties 'Py±Mn55' ... 'Py±Bi209'. Joined to 'Table 1 sample
    information' for age / location / formation when given.
    """
    g = pd.read_excel(appendix_path, header=0)
    cols = [str(c).strip() for c in g.columns]
    g.columns = cols
    g = g.rename(columns={cols[2]: "analysis_name", cols[3]: "sample_id", cols[5]: "spot_no"})
    g = g[g["analysis_name"].notna()].reset_index(drop=True)
    g.index = [f"{dataset}:{i:06d}" for i in range(len(g))]
    g.index.name = "analysis_id"

    el_cols = [c for c in g.columns if re.fullmatch(r"[A-Z][a-z]?\d?_Py(\.\d+)?", c)]
    series, unc, pb_seen = {}, {}, 0
    for c in el_cols:
        el = re.sub(r"\d?_Py(\.\d+)?$", "", c)
        if el == "Pb":                             # Pb_Py, Pb_Py.1, Pb_Py.2 = 206, 207, 208 estimates
            el = ["Pb_206est", "Pb_207est", "Pb"][pb_seen]
            pb_seen += 1
        series[(el, c)] = g[c]
    for c in g.columns:
        if c.startswith("Py±"):
            el = re.sub(r"\d+$", "", c[3:])
            key = next((k for k in series if k[0] == el), None)
            if key:
                unc[key] = g[c]
    long, unparsed = _melt(series, {k: "ppm" for k in series}, {k: "ppm" for k in series}, {}, unc)
    long = _mark_preferred(long, {"ppm": 0})

    meta = pd.DataFrame({
        "dataset": dataset,
        "mineral_group": "pyrite",
        "study": "Gregory et al. (sedimentary pyrite nodules)",
        "analysis_name": g["analysis_name"].astype("string"),
        "sample_id": g["sample_id"].astype("string").str.strip(),
        "spot_no": g["spot_no"],
        "spot_size": g[cols[0]].astype("string"),
        "context": "sedimentary pyrite nodule (black shale)",
        "mineral": "pyrite",
        "data_type": "spot",
    }, index=g.index)
    if sample_table_path is not None:
        t = pd.read_excel(sample_table_path).dropna(how="all")
        t = t.rename(columns=lambda c: _snake(c))
        t = t.rename(columns={t.columns[1]: "age_ma"})
        t["sample_no"] = t["sample_no"].astype("string").str.strip()
        t = t[["sample_no", "age_ma", "period", "location", "country", "formation", "description"]]
        meta = (meta.reset_index().merge(t, left_on="sample_id", right_on="sample_no", how="left")
                    .drop(columns="sample_no").set_index("analysis_id"))
    report = {"analyses_kept": len(meta),
              "analyses_without_sample_info": int(meta.get("age_ma", pd.Series(index=meta.index)).isna().sum()),
              "unparsed_cells_by_column": unparsed.to_dict()}
    return Dataset(dataset, meta, long), report


# --------------------------------------------------------------------------
# Pyrite: Falkenberg et al. - submarine hydrothermal vent pyrite (SIMS d34S + LA)
# --------------------------------------------------------------------------

_FALK_LA = ["Co", "Ni", "Cu", "Zn", "Ge", "As", "Se", "Mo", "Ag", "Cd", "Sb", "Te", "Au", "Tl", "Pb", "Bi"]


def load_falkenberg(path, dataset="FalkenbergVents") -> tuple[Dataset, dict]:
    """Arc (IA) and back-arc (BAB) vent-field pyrite.

    Sheet 1 'Coupled LA-ICP-MS and SIMS data': d34S + EPMA (wt%) + LA (ppm)
    on the same spots. Sheet 'Additional LA-ICP-MS data': LA only.
    """
    c = pd.read_excel(path, sheet_name=0, header=None, skiprows=2)
    c = c.dropna(how="all")
    c.columns = (["sample", "setting", "vent_field", "sample_type", "pyrite_type", "spot", "grain",
                  "grainspot", "d34S", "d34S_unc", "S_epma", "Cu_epma", "Fe_epma", "Zn_epma",
                  "total_epma", "comment_epma", "_blank", "spot_la"] + _FALK_LA)
    c = c[c["sample"].notna()].reset_index(drop=True)
    a = pd.read_excel(path, sheet_name="Additional LA-ICP-MS data", header=0)
    a.columns = ["sample", "setting", "vent_field", "sample_type", "pyrite_type", "spot"] + _FALK_LA
    a = a[a["sample"].notna()].reset_index(drop=True)
    c["sheet"], a["sheet"] = "coupled", "additional"
    raw = pd.concat([c, a], ignore_index=True)
    raw.index = [f"{dataset}:{i:06d}" for i in range(len(raw))]
    raw.index.name = "analysis_id"

    series, units, sources, positive, unc = {}, {}, {}, {}, {}
    for el in _FALK_LA:
        k = (el, el)
        series[k], units[k], sources[k] = raw[el], "ppm", "ppm"
    for el in ["S", "Cu", "Fe", "Zn"]:
        k = (el, f"{el}_epma")
        series[k] = pd.to_numeric(raw[f"{el}_epma"], errors="coerce") * 1e4
        units[k], sources[k] = "ppm", "wt%"
    k = ("d34S_VCDT", "d34S")
    series[k], units[k], sources[k], positive[k] = raw["d34S"], "permil", "SIMS", False
    unc[k] = raw["d34S_unc"]
    long, unparsed = _melt(series, units, sources, positive, unc)
    long = _mark_preferred(long, {"ppm": 0, "wt%": 1, "SIMS": 0})

    meta = pd.DataFrame({
        "dataset": dataset,
        "mineral_group": "pyrite",
        "study": "Falkenberg et al. (coupled SIMS d34S + LA-ICP-MS, arc/back-arc vents)",
        "sample_id": raw["sample"].astype("string"),
        "location": raw["vent_field"],
        "context": "submarine hydrothermal vent (" + raw["setting"].astype("string") + ")",
        "lithology": raw["sample_type"],
        "pyrite_type": raw["pyrite_type"],
        "spot": raw["spot"].astype("string"),
        "grainspot": raw["grainspot"],
        "sheet": raw["sheet"],
        "mineral": "pyrite",
        "data_type": "spot",
    }, index=raw.index)
    report = {"analyses_kept": len(meta), "unparsed_cells_by_column": unparsed.to_dict()}
    return Dataset(dataset, meta, long), report


# --------------------------------------------------------------------------
# One call to build everything
# --------------------------------------------------------------------------

# Found by matching author + year between the two reference lists (Oct 2026 versions)
OSULLIVAN_STUDIES_ALSO_IN_GEOROC = {"Chakhmouradian et al 2017.", "Tang et al. 2012"}

RAW_FILES = {
    "georoc_apatite": "2024-12-SGFTFN_APATITES.csv",
    "osullivan": "OSullivan-etal_2019.tab",
    "mukherjee_large": "12456332/G47890_SuppTab1.xlsx",
    "gregory_appendix": "Appendix 3 LA-icpms data_gregory et al.xlsx",
    "gregory_samples": "Table 1 sample information_gregory et al.xlsx",
    "falkenberg": ("Coupled in-situ SIMS d34S and LA-ICP-MS trace element data for pyrite from "
                   "subduction zone related submarine hydrothermal vent fields/"
                   "Dataset_new coupled SIMS and LA-ICP-MS data_Falkenberg.xlsx"),
}


def build_all(raw_dir, out_dir, verbose=True) -> dict:
    """Load every raw file, save processed tables, return {name: Dataset}."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    f = {k: raw_dir / v for k, v in RAW_FILES.items()}
    results, reports = {}, {}

    results["georoc_apatite"], reports["georoc_apatite"] = load_georoc_mineral(f["georoc_apatite"], "GEOROC_apatite")
    results["osullivan_apatite"], reports["osullivan_apatite"] = load_osullivan(f["osullivan"])
    results["ml2020_pyrite"], reports["ml2020_pyrite"] = load_mukherjee_large_2020(f["mukherjee_large"])
    results["gregory_pyrite"], reports["gregory_pyrite"] = load_gregory_nodules(f["gregory_appendix"], f["gregory_samples"])
    results["falkenberg_pyrite"], reports["falkenberg_pyrite"] = load_falkenberg(f["falkenberg"])

    # O'Sullivan studies that GEOROC also contains -> exclude from any GEOROC-vs-O'Sullivan test
    osm = results["osullivan_apatite"].meta
    osm["also_in_georoc"] = osm["study"].isin(OSULLIVAN_STUDIES_ALSO_IN_GEOROC)

    results["apatite_all"] = combine([results["georoc_apatite"], results["osullivan_apatite"]], "apatite_all")
    results["pyrite_all"] = combine([results["ml2020_pyrite"], results["gregory_pyrite"],
                                     results["falkenberg_pyrite"]], "pyrite_all")
    for name, ds in results.items():
        ds.save(out_dir, name, csv_wide=name.endswith("_all"))
    if verbose:
        for name, r in reports.items():
            print(f"{name}: {r.get('analyses_kept')} analyses kept",
                  {k: v for k, v in r.items() if k not in ('analyses_kept', 'unparsed_cells_by_column')})
            if r.get("unparsed_cells_by_column"):
                print("   unparsed cells:", r["unparsed_cells_by_column"])
    results["_reports"] = reports
    return results


# --------------------------------------------------------------------------
# Small helpers for the first look
# --------------------------------------------------------------------------

def chondrite_normalise(wide: pd.DataFrame, elements=REE_Y) -> pd.DataFrame:
    cols = [e for e in elements if e in wide.columns]
    return wide[cols] / pd.Series({e: CI_CHONDRITE[e] for e in cols})


def ree_anomalies(wide: pd.DataFrame) -> pd.DataFrame:
    """Eu/Eu*, Ce/Ce* (geometric, chondrite-normalised), Y/Ho, ΣREE, (La/Yb)N."""
    n = chondrite_normalise(wide)
    out = pd.DataFrame(index=wide.index)
    if {"Eu", "Sm", "Gd"} <= set(n):
        out["Eu_anom"] = n["Eu"] / np.sqrt(n["Sm"] * n["Gd"])
    if {"Ce", "La", "Pr"} <= set(n):
        out["Ce_anom"] = n["Ce"] / np.sqrt(n["La"] * n["Pr"])
    if {"Y", "Ho"} <= set(wide):
        out["Y_Ho"] = wide["Y"] / wide["Ho"]
    present = [e for e in REE if e in wide]
    out["sum_REE"] = wide[present].sum(axis=1, min_count=len(present))
    if {"La", "Yb"} <= set(n):
        out["La_Yb_N"] = n["La"] / n["Yb"]
    return out


def apatite_qc_flags(wide: pd.DataFrame, f_max=6.0e4, cl_max=7.0e4,
                     ca_range=(3.2e5, 4.3e5), p_range=(1.5e5, 2.0e5)) -> pd.DataFrame:
    """Flag apatite analyses with physically implausible major/halogen values (ppm).

    Defaults: F <= 6 wt% (end-member fluorapatite is 3.8 wt%, but EPMA F in
    apatite is often over-estimated by beam-induced F migration, so ~4-5 wt%
    values are common and not necessarily wrong), Cl <= 7 wt%
    (chlorapatite 6.8 wt%), Ca 32-43 wt% (CaO ~45-60 wt%), P 15-20 wt%
    (P2O5 ~34-46 wt%). Only checked where the value exists. Thresholds are
    deliberately loose - tighten after looking at the distributions.
    """
    flags = pd.DataFrame(index=wide.index)
    if "F" in wide:
        flags["F_too_high"] = wide["F"] > f_max
    if "Cl" in wide:
        flags["Cl_too_high"] = wide["Cl"] > cl_max
    if "Ca" in wide:
        flags["Ca_out_of_range"] = (wide["Ca"] < ca_range[0]) | (wide["Ca"] > ca_range[1])
    if "P" in wide:
        flags["P_out_of_range"] = (wide["P"] < p_range[0]) | (wide["P"] > p_range[1])
    flags["any_flag"] = flags.any(axis=1)
    return flags
