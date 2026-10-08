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
  is released). The area is the analytic integral of the profile.

Areas are reported in mÅ; the fitted mode also gives $\log_{10}(W/\lambda)$.

### Shortcuts

| Key | Action |
|---|---|
| Left / Right | previous / next line |
| Enter | keep the current line (in a text field: apply the settings) |
| Delete | remove the current line from the output |

Kept lines are written to the output CSV immediately.
