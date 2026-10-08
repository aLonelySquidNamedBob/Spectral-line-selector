# Spectral line selector

A Tkinter GUI for reviewing candidate absorption lines (e.g. GES "YY" Fe I/II lines) in one or
more normalised spectra, measuring each line, and keeping the good ones in a CSV.

## Requirements

Python 3.10 or newer with Tk, plus the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Running

```
cp review_yy_lines_config.example.json review_yy_lines_config.json   # then edit the paths
python review_yy_lines.py
```

The configuration file next to the script is loaded at start-up; **Save configuration** writes it
back. Everything in it can also be set in the GUI under *Show input configuration*.

### Inputs

- **YY line list**: a CSV with `species` and `wavelength` columns (optional `excitation_potential`,
  `lower_energy`, `upper_energy`, `gfflag`, `synflag`), or a GES fixed-width `.dat` line list.
  For a `.dat` file only lines with `gfflag = Y` and `synflag` in `{Y, U}` are used.
- **Non-yy line list** (optional, drawn for reference only): same formats. Given a `.dat` file,
  it shows the complement of the YY selection, so the same file can be used for both.
- **Spectra**: CSV/TSV/whitespace tables with a wavelength column (`wavelength` or `wave`) and a
  normalised flux column (`normed_flux` or `flux`). Wavelengths in Å, at rest.

### Measurement modes

- **EW**: trapezoidal integral of $1 - F$ between two draggable bounds per spectrum.
- **Manual Voigt**: a Voigt profile set with the sliders; its area is reported.
- **Fit Voigt**: least-squares Voigt fit inside a draggable fit region (refits when the region
  is released).

In both Voigt modes the equivalent width $W$ is the profile integrated over the fitted centre
$\pm 2\,\mathrm{FWHM}$, with the Voigt FWHM from Olivero & Longbothum (1977). The full
(infinite-wing) area is still reported and saved for comparison; it is larger whenever the
profile has Lorentzian wings (a pure Lorentzian keeps only 84 % of its area within $\pm 2\,\mathrm{FWHM}$,
a pure Gaussian all of it). Widths are in mÅ; the fitted mode also gives $\log_{10}(W/\lambda)$
and the local RV, $c\,\Delta\lambda/\lambda$.

### Per-line review

- **Continuum**: a manual continuum level per line (0.9–1.1, default 1). The spectrum is divided
  by it before fitting and EW integration; the level is drawn as a dashed line. No continuum is
  fitted automatically.
- **Blended**: a manual flag, saved with the line.
- **Comment**: free text, saved with the line. Enter in this field keeps the line.
- **Warnings** (Fit Voigt) appear under these controls:
  - the local RV exceeds the *RV warning* threshold (km/s, in the Measurement panel);
  - *possible blend*: the fit residuals within $\pm 2\,\mathrm{FWHM}$ differ between the blue
    and red side by more than 3 mÅ. This is a hint only: on HD 2454 Fe 1 it caught 14 of 19
    lines noted as blended, with 24 false alarms among 110 others.

These controls take effect immediately, without **Apply**. For a line that is already kept,
the output CSV is rewritten as soon as you change them (a comment once you stop typing), so
there is no need to press Keep again.

The navigation row shows how many lines are kept per species. Continuum, blend flag
and comment are restored from the output CSV when you revisit a kept line.

### Shortcuts

| Key | Action |
|---|---|
| Left / Right | previous / next line |
| Enter | keep the current line (in a text field: apply the settings) |
| Delete | remove the current line from the output |
| B | toggle the blended flag |

Kept lines are written to the output CSV immediately.
