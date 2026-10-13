"""Tests for the measurement and I/O parts of review_yy_lines (no window is opened).

Run with:  python -m pytest test_review_yy_lines.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import voigt_profile

import review_yy_lines as r

# Three records copied verbatim from the GES line list (geslines.dat): two YY lines and one
# non-YY line whose 6-character energy fields fill their columns completely.
GES_RECORDS = [
    "Cu 2   0 4750.0743 K11       -4.187 0.000 K11                                      - - '  LS                                                              3d8.(3F).4s.4p.(1P*) 3D*'  3.0 16.829 K11      '  JK                                                               3d9.(2D<3/2>).7g 2[9/2]'  4.0 19.438 K11       9.12 K11      -3.30 K11        -7.130 K11     ",
    "Fe 1   0 4779.4391 K07       -2.020 0.086 BWL                                      Y U '  LS                                                                   3p6.3d7.(2P).4s a1P'  1.0  3.415 K07      '  LS                                                                  3p6.3d7.(4P).4p x3P*'  0.0  6.008 K07       8.00 K07      -6.07 K07       317.261 BA-J,BPM",
    "Fe 2   0 5169.0282 K13       -1.000 0.000 2009A&A...497..611M                      Y U '  LS                                                                           3d5.4s2 a6S'  2.5  2.891 K13      '  LS                                                                      3d6.(5D).4p z6P*'  3.5  5.289 K13       8.49 K13      -6.53 K13       174.206 BA-J,BPM",
]


def make_app(n_spectra: int = 1) -> r.LineReviewerApp:
    """An app object without the Tk window, for testing methods that only need plain state."""
    app = object.__new__(r.LineReviewerApp)
    app.spectrum_cache = {}
    app.active_view_width = 1.0
    app.spectra = [
        r.Spectrum(wavelength=np.array([0.0, 1.0]), flux=np.ones(2), label=f"Star {i}") for i in range(n_spectra)
    ]
    app.spectrum_flux_columns = [f"min_flux_star_{i}" for i in range(n_spectra)]
    app.line_continuum, app.line_blended, app.line_comment = {}, {}, {}
    return app


def gaussian_spectrum(center: float, depth: float, sigma: float, step: float = 0.01) -> r.Spectrum:
    wave = np.arange(center - 3.0, center + 3.0, step)
    flux = 1.0 - depth * np.exp(-0.5 * ((wave - center) / sigma) ** 2)
    return r.Spectrum(wavelength=wave, flux=flux, label="synthetic")


def test_ges_parser_reads_fixed_columns_and_splits_yy(tmp_path):
    path = tmp_path / "lines.dat"
    path.write_text("\n".join(GES_RECORDS) + "\n", encoding="latin-1")
    app = make_app()

    yy = app._load_ges_line_list(path, yy_selection=True)
    assert yy["species"].tolist() == ["Fe 1", "Fe 2"]
    assert yy["wavelength"].tolist() == [4779.4391, 5169.0282]
    assert yy["excitation_potential"].tolist() == [3.415, 2.891]
    assert yy["upper_energy"].tolist() == [6.008, 5.289]

    other = app._load_ges_line_list(path, yy_selection=False)
    assert other["species"].tolist() == ["Cu 2"]
    assert other[["lower_energy", "upper_energy"]].values.tolist() == [[16.829, 19.438]]


@pytest.mark.parametrize("separator", [",", ";", "\t", "   "])
def test_read_table_detects_the_delimiter(tmp_path, separator):
    path = tmp_path / "spectrum.txt"
    path.write_text(separator.join(["wave", "flux"]) + "\n" + separator.join(["5000.0", "0.9"]) + "\n")
    table = make_app()._read_table(path)
    assert table.columns.tolist() == ["wave", "flux"]
    assert table.iloc[0].tolist() == [5000.0, 0.9]


def test_load_spectrum_sorts_and_drops_bad_rows(tmp_path):
    path = tmp_path / "spectrum.csv"
    path.write_text("wave,unnormed_flux,flux\n5000.02,1,0.8\n5000.00,1,0.9\n5000.01,1,nan\n")
    spectrum = make_app()._load_spectrum(path, "test")
    assert spectrum.wavelength.tolist() == [5000.00, 5000.02]
    assert spectrum.flux.tolist() == [0.9, 0.8]


@pytest.mark.parametrize("sigma, gamma", [(0.05, 1e-9), (1e-9, 0.05), (0.05, 0.03)])
def test_voigt_fwhm_matches_numerical_width(sigma, gamma):
    x = np.linspace(-1.0, 1.0, 400001)
    profile = voigt_profile(x, sigma, gamma)
    above_half = x[profile >= profile.max() / 2]
    assert r.voigt_fwhm(sigma, gamma) == pytest.approx(above_half[-1] - above_half[0], rel=5e-4)


def test_equivalent_width_window_keeps_gaussian_and_cuts_lorentzian_wings():
    depth = 0.3
    sigma, gamma = 0.05, 0.05
    gaussian = r.voigt_equivalent_width(depth, sigma, 1e-9, 2 * r.voigt_fwhm(sigma, 1e-9))
    assert gaussian == pytest.approx(depth * sigma * np.sqrt(2 * np.pi), rel=1e-4)
    # A Lorentzian of HWHM gamma has area depth * pi * gamma; +-2 FWHM keeps (2/pi) arctan(4) of it.
    lorentzian = r.voigt_equivalent_width(depth, 1e-9, gamma, 2 * r.voigt_fwhm(1e-9, gamma))
    assert lorentzian == pytest.approx(depth * np.pi * gamma * 2 / np.pi * np.arctan(4), rel=1e-3)


@pytest.mark.parametrize("continuum", [1.0, 0.98])
def test_integrated_ew_of_a_gaussian_line(continuum):
    depth, sigma = 0.2, 0.06
    spectrum = gaussian_spectrum(5000.0, depth, sigma)
    scaled = r.Spectrum(spectrum.wavelength, spectrum.flux * continuum, spectrum.label)
    ew = make_app()._integrate_equivalent_width(scaled, 4999.0, 5001.0, continuum)
    assert ew == pytest.approx(depth * sigma * np.sqrt(2 * np.pi), rel=1e-4)


def test_log_reduced_ew_rejects_non_positive_widths():
    app = make_app()
    assert app._log_reduced_ew(0.02, 5000.0) == pytest.approx(np.log10(0.02 / 5000.0))
    assert np.isnan(app._log_reduced_ew(0.0, 5000.0))
    assert np.isnan(app._log_reduced_ew(float("nan"), 5000.0))


def test_residual_asymmetry_flags_a_line_in_one_wing():
    app = make_app()
    depth, sigma = 0.2, 0.06
    clean = gaussian_spectrum(5000.0, depth, sigma)
    blend_flux = clean.flux - 0.08 * np.exp(-0.5 * ((clean.wavelength - 5000.12) / 0.05) ** 2)
    blended = r.Spectrum(clean.wavelength, blend_flux, "blended")
    args = (5000.0, 1.0, depth, sigma, 1e-9, 0.0)
    threshold = r.BLEND_ASYMMETRY_MILLIANGSTROM / r.ANGSTROM_TO_MILLIANGSTROM
    assert app._residual_asymmetry(clean, *args) < 1e-6
    assert app._residual_asymmetry(blended, *args) > threshold


def test_output_columns_are_suffixed_only_for_several_spectra():
    assert make_app(1)._measurement_column("voigt_fit_area", 0) == "voigt_fit_area"
    assert make_app(2)._measurement_column("voigt_fit_area", 1) == "voigt_fit_area_star_1"


def test_review_state_survives_a_csv_round_trip(tmp_path):
    path = tmp_path / "kept.csv"
    pd.DataFrame(
        {
            "species": ["Fe 1", "Fe 1"],
            "wavelength": [4779.4391, 4787.8266],
            "continuum": [0.995, 1.0],
            "blended": [True, False],
            "comment": ["blue wing, mask", ""],
            "fit_left": [4779.24, 4787.63],
            "fit_right": [4779.64, 4788.03],
        }
    ).to_csv(path, index=False)
    app = make_app()
    app.kept_lines = pd.read_csv(path)
    app._refresh_saved_integration_bounds()

    first, second = ("Fe 1", 4779.4391), ("Fe 1", 4787.8266)
    assert app.line_continuum == {first: 0.995, second: 1.0}
    assert app.line_blended == {first: True, second: False}
    assert app.line_comment == {first: "blue wing, mask"}
    assert app.saved_fit_bounds[first] == (4779.24, 4779.64)
