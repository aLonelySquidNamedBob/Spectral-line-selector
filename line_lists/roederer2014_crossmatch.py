"""Cross-match the Roederer et al. (2014, AJ 147, 136) line list against the Turbospectrum
Gaia-ESO NLTE line lists, and write a CSV that review_yy_lines.py reads as its candidate list.

Inputs (paths are arguments, nothing is downloaded here):
  --table8   VizieR J/AJ/147/136 table8.dat (atomic data, 479 lines)
  --table5   VizieR J/AJ/147/136 table5.dat[.gz] (EWs per star, optional: usage counts)
  --ges      nlte_ges_linelist_jmg6may2025_I_II (4200-9200 A, the "master" list in Internship_files)
  --blue     optional extra Turbospectrum lists (e.g. vald-3800-4200-for-grid-nlte-6may2025),
             reported separately because the current synthesis config does not use them
  --star     star name in table5 whose EWs are copied into the output (e.g. "HD 196944")

Matching: same element and ion (any isotope), |dlambda| <= --dwl (default 0.05 A) and
|dchi| <= --dchi (default 0.06 eV); Roederer lists lambda, chi and log gf to 0.01. The
best candidate is the strongest such line. All components with the same transition label, the
same chi (+-0.005 eV) and within +-0.2 A of it (hyperfine and isotopic components) are summed:
log gf_GES = log10(sum gf_i * f_iso), where f_iso is the solar isotopic fraction that
Turbospectrum applies at run time to species with an isotope number (e.g. 63.151). The
wavelength written for the selector is the gf-weighted centroid of those components.
`ges_loggf_listed_sum` is the same sum without f_iso (what you get if you add up the file).
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROMAN = {"I": 1, "II": 2, "III": 3}
SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se "
    "Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb "
    "Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U"
).split()
Z_OF = {symbol: index + 1 for index, symbol in enumerate(SYMBOLS)}
# Solar isotopic (atom) fractions, Asplund et al. (2009) Table 3. Turbospectrum multiplies the gf
# of a line whose species carries an isotope number (e.g. 56.138) by this fraction at run time
# unless an 'ISOTOPES' block overrides it, so the effective gf of such a line is gf * fraction.
SOLAR_ISOTOPE_FRACTION = {
    (3, 6): 0.0759, (3, 7): 0.9241,
    (20, 40): 0.96941, (20, 42): 0.00647, (20, 43): 0.00135, (20, 44): 0.02086, (20, 46): 0.00004,
    (20, 48): 0.00187,
    (29, 63): 0.6915, (29, 65): 0.3085,
    (56, 130): 0.00106, (56, 132): 0.00101, (56, 134): 0.02417, (56, 135): 0.06592, (56, 136): 0.07854,
    (56, 137): 0.11232, (56, 138): 0.71698,
    (57, 139): 0.99911, (59, 141): 1.0,
    (60, 142): 0.27152, (60, 143): 0.12174, (60, 144): 0.23798, (60, 145): 0.08293, (60, 146): 0.17189,
    (60, 148): 0.05756, (60, 150): 0.05638,
    (62, 144): 0.0307, (62, 147): 0.1499, (62, 148): 0.1124, (62, 149): 0.1382, (62, 150): 0.0738,
    (62, 152): 0.2675, (62, 154): 0.2275,
    (63, 151): 0.4781, (63, 153): 0.5219,
}
HEADER = re.compile(r"^'\s*(\d+)\.(\d+)\s*'\s+(\d+)\s+(\d+)")


@dataclass
class TSLine:
    z: int
    isotope: int
    ion: int
    wavelength: float
    chi: float
    loggf: float
    label: str
    nlte: bool
    source: str


def read_turbospectrum(path: Path, source: str) -> pd.DataFrame:
    """Atomic Turbospectrum (bsyn) line list -> one row per component."""
    rows = []
    lines = path.read_text(encoding="latin-1").splitlines()
    i = 0
    while i < len(lines):
        match = HEADER.match(lines[i])
        if not match:
            i += 1
            continue
        z, isotope = int(match.group(1)), int(match.group(2))
        ion, count = int(match.group(3)), int(match.group(4))
        nlte = "NLTE" in lines[i + 1]
        for record in lines[i + 2 : i + 2 + count]:
            parts = record.split()
            quoted = re.findall(r"'([^']*)'", record)
            # quoted fields: two term letters, the transition label, then (NLTE lists) level ids/flags
            label = next((q for q in quoted if len(q) > 3), "")
            levels = quoted[quoted.index(label) + 1 : quoted.index(label) + 3] if label in quoted else []
            has_levels = nlte and any(level.strip() not in ("none", "") for level in levels)
            rows.append(
                TSLine(z, isotope, ion, float(parts[0]), float(parts[1]), float(parts[2]), label, has_levels, source)
            )
        i += 2 + count
    return pd.DataFrame([row.__dict__ for row in rows])


def read_table8(path: Path) -> pd.DataFrame:
    rows = []
    for record in path.read_text().splitlines():
        if not record.strip():
            continue
        species = record[8:13].strip()
        forbidden = species.startswith("[")
        element, stage = species.strip("[]").split()
        rows.append(
            {
                "r14_species": species,
                "z": Z_OF[element],
                "ion": ROMAN[stage],
                "species": f"{element} {ROMAN[stage]}",
                "r14_wavelength": float(record[0:7]),
                "r14_chi": float(record[14:18]),
                "r14_loggf": float(record[19:24]),
                "r14_ref": int(record[25:27]),
                "forbidden": forbidden,
            }
        )
    return pd.DataFrame(rows)


def read_table5(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as handle:
        for record in handle:
            ew = record[38:43].strip()
            rows.append(
                {
                    "star": record[0:12].strip(),
                    "r14_wavelength": float(record[13:20]),
                    "r14_species": record[21:26].strip(),
                    "ew": float(ew) if ew else math.nan,
                    "flag": record[44:45].strip() if len(record) > 44 else "",
                }
            )
    return pd.DataFrame(rows)


# Rough extent of the carbon-molecule bands that matter in CEMP stars (A). Swan and CN violet bands
# are degraded to the violet, so each range runs blueward from the head. A line inside a range is
# not necessarily blended; it means "check the synthesis with CH/CN/C2 switched on".
CARBON_BANDS = [
    ("CN 3883", 3820.0, 3884.0),
    ("CH B-X 3900", 3870.0, 3910.0),
    ("CN 4216", 4120.0, 4216.0),
    ("CH G band", 4200.0, 4400.0),
    ("C2 Swan 4737", 4650.0, 4737.0),
    ("C2 Swan 5165", 5030.0, 5165.0),
    ("C2 Swan 5636", 5400.0, 5636.0),
    ("CN red (weak)", 6900.0, 9200.0),
]


def carbon_band(wavelength: float) -> str:
    return "; ".join(name for name, lo, hi in CARBON_BANDS if lo <= wavelength <= hi)


# Elements whose lines Gaia-ESO lists with hyperfine components (closely spaced entries are expected).
HFS_ELEMENTS = {21, 23, 25, 27, 29, 56, 57, 59, 63, 67}


def possible_duplicate(group: pd.DataFrame, z: int) -> bool:
    """Two entries of one transition within 0.01 A in a non-HFS, non-isotope group: likely the same
    line listed twice (Turbospectrum then synthesises it twice)."""
    if z in HFS_ELEMENTS or group.isotope.any() or len(group) < 2:
        return False
    wavelengths = np.sort(group.wavelength.to_numpy())
    return bool((np.diff(wavelengths) < 0.01).any())


def match_line(line: pd.Series, components: pd.DataFrame, dwl: float, dchi: float) -> dict:
    same = components[(components.z == line.z) & (components.ion == line.ion)]
    window = same[(same.wavelength - line.r14_wavelength).abs() <= 0.5]
    result = {"match": "missing", "ges_source": "", "n_components": 0}
    if window.empty:
        return result
    close = window[
        ((window.wavelength - line.r14_wavelength).abs() <= dwl) & ((window.chi - line.r14_chi).abs() <= dchi)
    ]
    if close.empty:
        nearest = window.iloc[(window.wavelength - line.r14_wavelength).abs().argsort()[:1]].iloc[0]
        result.update(
            match="no_match_within_tolerance",
            nearest_ges_wavelength=nearest.wavelength,
            nearest_ges_chi=nearest.chi,
            nearest_ges_loggf=nearest.loggf,
        )
        return result
    effective = close.loggf + np.log10(
        [SOLAR_ISOTOPE_FRACTION.get((line.z, iso), 1.0) if iso else 1.0 for iso in close.isotope]
    )
    best = close.loc[effective.idxmax()]
    group = window[
        (window.label == best.label)
        & ((window.chi - best.chi).abs() <= 0.005)
        & ((window.wavelength - best.wavelength).abs() <= 0.2)
    ]
    fraction = np.array([SOLAR_ISOTOPE_FRACTION.get((line.z, iso), 1.0) if iso else 1.0 for iso in group.isotope])
    listed = np.power(10.0, group.loggf)
    gf = listed * fraction
    loggf_total = math.log10(gf.sum())
    centroid = float((group.wavelength * gf).sum() / gf.sum())
    result.update(
        match="matched",
        ges_source=best.source,
        ges_wavelength=round(centroid, 4),
        ges_strongest_wavelength=best.wavelength,
        ges_chi=best.chi,
        ges_loggf=round(loggf_total, 3),
        ges_loggf_listed_sum=round(math.log10(listed.sum()), 3),
        isotopes=" ".join(str(i) for i in sorted(set(group.isotope)) if i),
        n_components=len(group),
        possible_duplicate=possible_duplicate(group, line.z),
        nlte_levels=bool(best.nlte),
        ges_label=best.label,
    )
    result["dwl"] = round(centroid - line.r14_wavelength, 4)
    result["dchi"] = round(best.chi - line.r14_chi, 3)
    result["dloggf"] = round(loggf_total - line.r14_loggf, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table8", type=Path, required=True)
    parser.add_argument("--table5", type=Path)
    parser.add_argument("--ges", type=Path, required=True)
    parser.add_argument("--blue", type=Path, nargs="*", default=[])
    parser.add_argument("--star", default="HD 196944")
    parser.add_argument("--dwl", type=float, default=0.05)
    parser.add_argument("--dchi", type=float, default=0.06)
    parser.add_argument("--out-all", type=Path, default=Path("roederer2014_crossmatch.csv"))
    parser.add_argument("--out-selector", type=Path, default=Path("roederer2014_selector_lines.csv"))
    args = parser.parse_args()

    lines = read_table8(args.table8)
    components = pd.concat(
        [read_turbospectrum(args.ges, "ges_master")]
        + [read_turbospectrum(path, "extra:" + path.name) for path in args.blue],
        ignore_index=True,
    )
    matched = pd.DataFrame([match_line(row, components, args.dwl, args.dchi) for _, row in lines.iterrows()])
    table = pd.concat([lines.reset_index(drop=True), matched], axis=1)
    table["in_current_synthesis"] = table.ges_source == "ges_master"
    # Closely spaced copies count as a duplicate only when they make the line clearly stronger
    # than R14's gf (Mg I 3832/3838 have genuine J components within 0.01 A).
    table["possible_duplicate"] = table.possible_duplicate.fillna(False).astype(bool) & (table.dloggf > 0.1)
    table["carbon_band"] = table.r14_wavelength.map(carbon_band)

    if args.table5 is not None:
        ews = read_table5(args.table5)
        # table5 wavelengths can differ from table8 by a few 0.01 A (e.g. Ba II 4934.10 vs 4934.08):
        # attach each table5 line to the nearest table8 line of the same species.
        lookup = lines[["r14_species", "r14_wavelength"]].rename(columns={"r14_wavelength": "t8"})
        ews["t8"] = np.nan
        for species, group in ews.groupby("r14_species"):
            ref = lookup[lookup.r14_species == species].t8.to_numpy()
            if ref.size:
                nearest = ref[np.abs(group.r14_wavelength.to_numpy()[:, None] - ref[None, :]).argmin(axis=1)]
                keep = np.abs(nearest - group.r14_wavelength.to_numpy()) <= 0.05
                ews.loc[group.index[keep], "t8"] = nearest[keep]
        ews = ews.dropna(subset=["t8"]).drop(columns="r14_wavelength").rename(columns={"t8": "r14_wavelength"})
        ews = ews[ews.ew.notna() | (ews.flag != "")]
        key = ["r14_wavelength", "r14_species"]
        usage = ews.groupby(key).agg(
            r14_n_stars=("star", "size"),
            r14_n_synth=("flag", lambda f: int((f == "s").sum())),
            r14_n_limit=("flag", lambda f: int((f == "l").sum())),
            r14_median_ew=("ew", "median"),
        )
        table = table.merge(usage.reset_index(), on=key, how="left")
        star = ews[ews.star == args.star].set_index(key)
        if not star.empty:
            tag = args.star.replace(" ", "")
            star = star.rename(columns={"ew": f"r14_ew_{tag}", "flag": f"r14_flag_{tag}"})
            table = table.merge(star[[f"r14_ew_{tag}", f"r14_flag_{tag}"]].reset_index(), on=key, how="left")

    # The selector reads species/wavelength/excitation_potential; other columns are ignored there.
    table["wavelength"] = table.ges_wavelength.fillna(table.r14_wavelength)
    table["excitation_potential"] = table.ges_chi.fillna(table.r14_chi)
    first = ["species", "wavelength", "excitation_potential", "match", "in_current_synthesis"]
    table = table[first + [c for c in table.columns if c not in first]]
    table = table.sort_values(["species", "wavelength"]).reset_index(drop=True)
    table.to_csv(args.out_all, index=False)
    table[table.in_current_synthesis].to_csv(args.out_selector, index=False)

    print(f"{len(table)} Roederer lines")
    print(table.groupby(["match", "ges_source"], dropna=False).size().to_string())
    ok = table[table.match == "matched"]
    print(f"dloggf (GES - R14): median {ok.dloggf.median():+.3f}, MAD-std "
          f"{1.4826 * (ok.dloggf - ok.dloggf.median()).abs().median():.3f}, "
          f"|d|>0.10: {(ok.dloggf.abs() > 0.10).sum()}, |d|>0.30: {(ok.dloggf.abs() > 0.30).sum()}")


if __name__ == "__main__":
    main()
