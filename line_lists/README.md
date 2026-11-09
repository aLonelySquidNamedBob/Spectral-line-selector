# Candidate line lists for the selector

## `roederer2014_selector_lines.csv`

The Roederer et al. (2014, AJ 147, 136; VizieR J/AJ/147/136, table 8) line list, restricted to the
**291 of 479** lines that are in `nlte_ges_linelist_jmg6may2025_I_II`, the atomic list the
Internship_files synthesis uses (`master` in `config/line_lists.yaml`). Load it as the "YY line list"
in `review_yy_lines.py`. The selector reads `species` ("Fe 1" style), `wavelength` (the gf-weighted
centroid of the matched GES components) and `excitation_potential`; the other columns are ignored there.
CSV lists are not filtered by gf/syn flag, and this file has no GES quality flags (the Turbospectrum
format does not carry them).

`roederer2014_crossmatch.csv` has all 479 lines with the same columns, so you can see what was lost.

Rebuild with `roederer2014_crossmatch.py` (paths as arguments; see its docstring).

## How the match was done

Same element and ion, |Δλ| ≤ 0.05 Å, |Δχ| ≤ 0.06 eV (Roederer gives λ, χ, log gf to 0.01).
Hyperfine and isotopic components (same transition, same χ, within ±0.2 Å) are summed, with each
isotope's gf multiplied by the solar isotopic fraction, because Turbospectrum applies that fraction at
run time to species such as `63.151`. `ges_loggf` is therefore the effective gf that gets synthesised;
`ges_loggf_listed_sum` is the plain sum of the file.

## Caveats (read before trusting a line)

| Column / issue | What it means |
|---|---|
| `match`, `in_current_synthesis` | 411 lines matched: 291 in the master list (4200–9200 Å), 120 only in the VALD lists `vald-3700-3800`/`vald-3800-4200-for-grid-nlte-6may2025`, which the current config does **not** load. 64 lines are below 3700 Å (no list covers them), Tc I 4262 is absent, and Cr II 4558.59/4588.14/4591.99 sit 0.06 Å from the GES lines with log gf 0.2 dex higher. |
| `dloggf` | GES − Roederer effective log gf. Median 0.000; 30 lines differ by > 0.1 dex. Fe II 4515/4522/4555 are +0.15 to +0.24. |
| Eu II | All four Eu II lines are −0.30 dex in the 6may2025 list: its Eu gf already includes the isotopic fraction and Turbospectrum applies it again (fixed upstream on 15 Jan 2026). Eu would come out ~0.3 dex too high. |
| Ba II 4554 | −0.24 dex effective. The other Ba II lines agree. The list author notes Ba isotope scaling as not yet checked. |
| `possible_duplicate` | Ti I 4533.24 and Ti II 4636.32 are listed twice in GES within 0.01 Å (one copy with default damping), so they are synthesised ~0.2–0.27 dex too strong. |
| `carbon_band` | The line is inside a CH, CN or C₂ band range. In a CEMP star, check it with the molecular lists on. This is a wavelength range only, not a blend calculation. |
| `r14_n_synth` | How many stars Roederer fitted by synthesis (`s`) rather than EW: these are the HFS/blended lines (Ba, Mn, Sc, V, Y, Zr, Li, Sr). |
| `r14_ew_HD196944` | Roederer's EW for HD 196944 (in his sample: Teff 5310, log g 1.75, vt 1.65, [M/H] −2.39). |
