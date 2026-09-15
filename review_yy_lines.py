from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from scipy.special import voigt_profile


# DEFAULT VALUES
DEFAULT_VIEW_WIDTH = 1.0
DEFAULT_PEAK_WINDOW = 0.05
# DEFAULT_PEAK_THRESHOLD = 0.4 
# DEFAULT_YMIN = 0.35
DEFAULT_PEAK_THRESHOLD = 0.0
DEFAULT_YMIN = -0.05
DEFAULT_YMAX = 1.1
DEFAULT_OUTPUT_NAME = "kept_lines_new_yy.csv"
CONFIGURATION_NAME = "review_yy_lines_config.json"
DEFAULT_VOIGT_DEPTH = 0.5
DEFAULT_VOIGT_SIGMA = 0.05
DEFAULT_VOIGT_GAMMA = 0.05
DEFAULT_VOIGT_SHIFT = 0.0
DEFAULT_VOIGT_FIT_WIDTH = 0.2
SPECTRUM_COLOR_CYCLE = [
    "steelblue",
    "tomato",
    "seagreen",
    "darkorchid",
    "sienna",
    "royalblue",
    "goldenrod",
]

# CLASSES

@dataclass(frozen=True)
class Spectrum:
    wavelength: np.ndarray
    flux: np.ndarray
    label: str

    def bounds(self, center: float, half_width: float) -> tuple[int, int]:
        left = np.searchsorted(self.wavelength, center - half_width, side="left")
        right = np.searchsorted(self.wavelength, center + half_width, side="right")
        return int(left), int(right)

    def region(self, center: float, half_width: float) -> tuple[np.ndarray, np.ndarray]:
        left, right = self.bounds(center, half_width)
        return self.wavelength[left:right], self.flux[left:right]

    def min_flux(self, center: float, half_width: float) -> float:
        left, right = self.bounds(center, half_width)
        if right <= left:
            return float("nan")
        return float(np.min(self.flux[left:right]))


@dataclass
class SpectrumConfigRow:
    frame: ttk.Frame
    label_var: tk.StringVar
    path_var: tk.StringVar


class LineReviewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YY Line Reviewer")

        self.configuration_path = Path(__file__).with_name(CONFIGURATION_NAME)
        configuration = self._load_configuration(self.configuration_path)

        self.project_root = Path(__file__).resolve().parent.parent
        self.results_dir = self.project_root / "6_python stuff" / "results"
        self.output_path = Path(configuration.get("output", ""))
        self.default_spectra = [
            (str(item.get("label", "")), Path(str(item.get("path", ""))))
            for item in configuration.get("spectra", [])
            if isinstance(item, dict)
        ]

        self.spectrum_rows: list[SpectrumConfigRow] = []
        self.spectra: list[Spectrum] = []
        self.spectrum_flux_columns: list[str] = []
        self.yy_lines = pd.DataFrame(columns=["species", "wavelength"])
        self.non_yy_lines = pd.DataFrame(columns=["species", "wavelength"])

        self.yy_lines_path_var = tk.StringVar(value=str(configuration.get("yy_lines", "")))
        self.non_yy_lines_path_var = tk.StringVar(value=str(configuration.get("non_yy_lines", "")))

        self.species_options: list[str] = []

        self.current_species = tk.StringVar(value="")
        self.peak_threshold_var = tk.StringVar(value=str(configuration.get("peak_threshold", DEFAULT_PEAK_THRESHOLD)))
        self.peak_window_var = tk.StringVar(value=str(configuration.get("peak_window", DEFAULT_PEAK_WINDOW)))
        self.view_width_var = tk.StringVar(value=str(configuration.get("view_width", DEFAULT_VIEW_WIDTH)))
        self.y_min_var = tk.StringVar(value=str(configuration.get("y_min", DEFAULT_YMIN)))
        self.y_max_var = tk.StringVar(value=str(configuration.get("y_max", DEFAULT_YMAX)))
        self.output_var = tk.StringVar(value=str(configuration.get("output", "")))
        self.show_other_yy_var = tk.BooleanVar(value=bool(configuration.get("show_other_yy", True)))
        self.show_non_yy_var = tk.BooleanVar(value=bool(configuration.get("show_non_yy", True)))
        self.measurement_mode_var = tk.StringVar(value=str(configuration.get("measurement_mode", "EW")))
        if self.measurement_mode_var.get() not in {"EW", "Manual Voigt", "Fit Voigt"}:
            self.measurement_mode_var.set("EW")
        self.voigt_depth_var = tk.DoubleVar(value=self._configuration_float(configuration, "voigt_depth", DEFAULT_VOIGT_DEPTH))
        self.voigt_sigma_var = tk.DoubleVar(value=self._configuration_float(configuration, "voigt_sigma", DEFAULT_VOIGT_SIGMA))
        self.voigt_gamma_var = tk.DoubleVar(value=self._configuration_float(configuration, "voigt_gamma", DEFAULT_VOIGT_GAMMA))
        self.voigt_shift_var = tk.DoubleVar(value=self._configuration_float(configuration, "voigt_shift", DEFAULT_VOIGT_SHIFT))
        self.voigt_fit_width_var = tk.DoubleVar(value=self._configuration_float(configuration, "voigt_fit_width", DEFAULT_VOIGT_FIT_WIDTH))
        self.voigt_depth_label_var = tk.StringVar()
        self.voigt_sigma_label_var = tk.StringVar()
        self.voigt_gamma_label_var = tk.StringVar()
        self.voigt_shift_label_var = tk.StringVar()
        self.voigt_fit_width_label_var = tk.StringVar()
        self.status_var = tk.StringVar(value="")
        self.active_peak_threshold = DEFAULT_PEAK_THRESHOLD
        self.active_peak_window = DEFAULT_PEAK_WINDOW
        self.active_view_width = DEFAULT_VIEW_WIDTH
        self.active_y_min: float | None = None
        self.active_y_max: float | None = None
        self.loaded_output_path = Path(self.output_var.get())

        self.filtered_lines = pd.DataFrame(columns=["species", "wavelength", "min_flux"])
        self.current_index = 0
        self.peak_cache: dict[tuple[float, float], dict[str, float]] = {}
        self.kept_lines = self._load_existing_output()
        self.saved_integration_bounds: dict[tuple[str, float], dict[str, tuple[float, float]]] = {}
        self.fit_bounds: dict[tuple[str, float], tuple[float, float]] = {}
        self.config_visible = False
        self.integration_bounds: dict[str, tuple[float, float]] = {}
        self.dragging_handle: tuple[str, str] | None = None
        self.last_line_key: tuple[str, float] | None = None
        self.last_ew_values: dict[str, float] = {}
        self.last_voigt_area = float("nan")
        self.fitted_voigt_parameters: dict[str, tuple[float, float, float, float]] = {}
        self.fitted_voigt_areas: dict[str, float] = {}
        self.fitted_voigt_line_key: tuple[str, float] | None = None

        self._build_ui()

        for default_label, default_path in self.default_spectra:
            self.add_spectrum_row(default_label, str(default_path))
        if not self.default_spectra:
            self.add_spectrum_row()

        self.set_config_visibility(False)

        if self.yy_lines_path_var.get().strip() and any(row.path_var.get().strip() for row in self.spectrum_rows):
            self.apply_filters(reset_index=True)
        else:
            self.status_var.set("Configure input files and click Apply")

    def _configuration_float(self, configuration: dict[str, Any], key: str, default: float) -> float:
        try:
            return float(configuration.get(key, default))
        except (TypeError, ValueError):
            return default

    def _load_configuration(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as configuration_file:
                loaded = json.load(configuration_file)
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _configuration_values(self) -> dict[str, Any]:
        return {
            "yy_lines": self.yy_lines_path_var.get().strip(),
            "non_yy_lines": self.non_yy_lines_path_var.get().strip(),
            "output": self.output_var.get().strip(),
            "peak_threshold": self.peak_threshold_var.get().strip(),
            "peak_window": self.peak_window_var.get().strip(),
            "view_width": self.view_width_var.get().strip(),
            "y_min": self.y_min_var.get().strip(),
            "y_max": self.y_max_var.get().strip(),
            "show_other_yy": self.show_other_yy_var.get(),
            "show_non_yy": self.show_non_yy_var.get(),
            "spectra": [
                {"label": row.label_var.get().strip(), "path": row.path_var.get().strip()}
                for row in self.spectrum_rows
            ],
            "measurement_mode": self.measurement_mode_var.get(),
            "voigt_depth": self.voigt_depth_var.get(),
            "voigt_sigma": self.voigt_sigma_var.get(),
            "voigt_gamma": self.voigt_gamma_var.get(),
            "voigt_shift": self.voigt_shift_var.get(),
            "voigt_fit_width": self.voigt_fit_width_var.get(),
        }

    def save_configuration(self, path: Path | None = None) -> None:
        if path is None:
            path = self.configuration_path
        configuration = self._configuration_values()
        try:
            with path.open("w", encoding="utf-8") as configuration_file:
                json.dump(configuration, configuration_file, indent=2)
        except OSError as exc:
            messagebox.showerror("Configuration error", f"Could not save configuration: {exc}")
            return
        self.configuration_path = path
        self.status_var.set(f"Configuration saved to {path.name}")

    def save_configuration_as(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Save reviewer configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=self.configuration_path.name,
        )
        if selected:
            self.save_configuration(Path(selected))

    def load_configuration_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Load reviewer configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not selected:
            return

        configuration = self._load_configuration(Path(selected))
        if not configuration:
            messagebox.showerror("Configuration error", "The selected file is missing or is not valid JSON.")
            return
        self._apply_configuration(configuration)
        self.configuration_path = Path(selected)

    def _apply_configuration(self, configuration: dict[str, Any]) -> None:
        self.yy_lines_path_var.set(str(configuration.get("yy_lines", "")))
        self.non_yy_lines_path_var.set(str(configuration.get("non_yy_lines", "")))
        self.output_var.set(str(configuration.get("output", "")))
        for variable, key, default in [
            (self.peak_threshold_var, "peak_threshold", DEFAULT_PEAK_THRESHOLD),
            (self.peak_window_var, "peak_window", DEFAULT_PEAK_WINDOW),
            (self.view_width_var, "view_width", DEFAULT_VIEW_WIDTH),
            (self.y_min_var, "y_min", DEFAULT_YMIN),
            (self.y_max_var, "y_max", DEFAULT_YMAX),
        ]:
            variable.set(str(configuration.get(key, default)))
        self.show_other_yy_var.set(bool(configuration.get("show_other_yy", True)))
        self.show_non_yy_var.set(bool(configuration.get("show_non_yy", True)))
        self.measurement_mode_var.set(str(configuration.get("measurement_mode", "EW")))
        if self.measurement_mode_var.get() not in {"EW", "Manual Voigt", "Fit Voigt"}:
            self.measurement_mode_var.set("EW")
        self.voigt_depth_var.set(self._configuration_float(configuration, "voigt_depth", DEFAULT_VOIGT_DEPTH))
        self.voigt_sigma_var.set(self._configuration_float(configuration, "voigt_sigma", DEFAULT_VOIGT_SIGMA))
        self.voigt_gamma_var.set(self._configuration_float(configuration, "voigt_gamma", DEFAULT_VOIGT_GAMMA))
        self.voigt_shift_var.set(self._configuration_float(configuration, "voigt_shift", DEFAULT_VOIGT_SHIFT))
        self.voigt_fit_width_var.set(self._configuration_float(configuration, "voigt_fit_width", DEFAULT_VOIGT_FIT_WIDTH))

        for row in self.spectrum_rows:
            row.frame.destroy()
        self.spectrum_rows.clear()
        spectra = configuration.get("spectra", [])
        for item in spectra if isinstance(spectra, list) else []:
            if isinstance(item, dict):
                self.add_spectrum_row(str(item.get("label", "")), str(item.get("path", "")))
        if not self.spectrum_rows:
            self.add_spectrum_row()
        self._update_voigt_labels()
        self._update_mode_controls()
        self.apply_filters(reset_index=True)

    def _find_column(self, columns: list[str], candidates: list[str]) -> str | None:
        lower_to_original = {column.lower(): column for column in columns}
        for candidate in candidates:
            match = lower_to_original.get(candidate.lower())
            if match is not None:
                return match
        return None

    def _safe_column_name(self, label: str, used: set[str]) -> str:
        base = "".join(ch.lower() if ch.isalnum() else "_" for ch in label.strip())
        base = "_".join(part for part in base.split("_") if part)
        if not base:
            base = "spectrum"

        candidate = f"min_flux_{base}"
        suffix = 2
        while candidate in used:
            candidate = f"min_flux_{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    def _load_spectrum(self, path: Path, label: str) -> Spectrum:
        if not path.exists():
            raise ValueError(f"Spectrum file not found: {path}")

        frame = pd.read_csv(path)
        wavelength_col = self._find_column(frame.columns.tolist(), ["Wavelength", "wavelength"])
        flux_col = self._find_column(frame.columns.tolist(), ["Flux", "flux"])

        if wavelength_col is None or flux_col is None:
            raise ValueError(
                f"Spectrum file {path} must contain Wavelength and Flux columns."
            )

        wavelength = pd.to_numeric(frame[wavelength_col], errors="coerce")
        flux = pd.to_numeric(frame[flux_col], errors="coerce")
        valid = ~(wavelength.isna() | flux.isna())
        wavelength = wavelength[valid].to_numpy(dtype=float)
        flux = flux[valid].to_numpy(dtype=float)

        if wavelength.size == 0:
            raise ValueError(f"Spectrum file {path} contains no valid numeric rows.")

        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        flux = flux[order]

        return Spectrum(
            wavelength=wavelength,
            flux=flux,
            label=label,
        )

    def _load_line_list(self, path_text: str, name: str, required: bool) -> pd.DataFrame:
        stripped = path_text.strip()
        if not stripped:
            if required:
                raise ValueError(f"{name} path is empty.")
            return pd.DataFrame(columns=["species", "wavelength"])

        path = Path(stripped)

        if not path.exists():
            if required:
                raise ValueError(f"{name} not found: {path}")
            return pd.DataFrame(columns=["species", "wavelength"])

        frame = pd.read_csv(path)
        species_col = self._find_column(frame.columns.tolist(), ["species", "Species"])
        wavelength_col = self._find_column(frame.columns.tolist(), ["wavelength", "Wavelength"])
        if species_col is None or wavelength_col is None:
            raise ValueError(
                f"{name} must contain species and wavelength columns."
            )

        loaded = pd.DataFrame(
            {
                "species": frame[species_col].astype(str),
                "wavelength": pd.to_numeric(frame[wavelength_col], errors="coerce"),
            }
        )
        loaded = loaded.replace({"species": {"nan": ""}})
        loaded = loaded[(loaded["species"].str.strip() != "") & loaded["wavelength"].notna()]

        if required and loaded.empty:
            raise ValueError(f"{name} has no valid lines after parsing.")

        if loaded.empty:
            return pd.DataFrame(columns=["species", "wavelength"])

        return loaded.sort_values(["species", "wavelength"]).reset_index(drop=True)

    def _reload_inputs(self) -> None:
        yy_path_text = self.yy_lines_path_var.get().strip()
        non_yy_path_text = self.non_yy_lines_path_var.get().strip()

        yy_lines = self._load_line_list(yy_path_text, "YY line list", required=True)
        non_yy_lines = self._load_line_list(non_yy_path_text, "Non-yy line list", required=False)

        spectra: list[Spectrum] = []
        spectrum_flux_columns: list[str] = []
        used_columns: set[str] = set()

        for index, row in enumerate(self.spectrum_rows, start=1):
            path_text = row.path_var.get().strip()
            if not path_text:
                continue

            label = row.label_var.get().strip() or f"Spectrum {index}"
            spectrum = self._load_spectrum(Path(path_text), label)
            spectra.append(spectrum)
            spectrum_flux_columns.append(self._safe_column_name(label, used_columns))

        if not spectra:
            raise ValueError("At least one valid spectrum must be configured.")

        species_options = sorted(yy_lines["species"].dropna().unique().tolist())
        if not species_options:
            raise ValueError("No species found in the YY line list.")

        self.yy_lines = yy_lines
        self.non_yy_lines = non_yy_lines
        self.spectra = spectra
        self.spectrum_flux_columns = spectrum_flux_columns
        self.species_options = species_options
        self.peak_cache.clear()
        self._refresh_saved_integration_bounds()
        self.integration_bounds.clear()

        current = self.current_species.get()
        if current not in self.species_options:
            self.current_species.set(self.species_options[0])
        if hasattr(self, "species_box"):
            self.species_box.configure(values=self.species_options)

    def _load_existing_output(self) -> pd.DataFrame:
        path = Path(self.output_var.get())
        if path.is_file() and path.stat().st_size > 0:
            frame = pd.read_csv(path, on_bad_lines='warn')
            if "species" in frame.columns and "wavelength" in frame.columns:
                self.loaded_output_path = path
                return frame
        self.loaded_output_path = path
        return pd.DataFrame(columns=["species", "wavelength", "min_flux", "source"])

    def _refresh_saved_integration_bounds(self) -> None:
        saved_bounds: dict[tuple[str, float], dict[str, tuple[float, float]]] = {}

        if self.kept_lines.empty or not self.spectrum_flux_columns:
            self.saved_integration_bounds = saved_bounds
            return

        for _, row in self.kept_lines.iterrows():
            species = str(row.get("species", "")).strip()
            wavelength = row.get("wavelength")
            if not species or pd.isna(wavelength):
                continue

            line_key = (species, round(float(wavelength), 6))
            bounds_for_line: dict[str, tuple[float, float]] = {}

            for column_name in self.spectrum_flux_columns:
                suffix = self._ew_name_suffix(column_name)
                left_key = f"ew_left_{suffix}"
                right_key = f"ew_right_{suffix}"
                if left_key not in row or right_key not in row:
                    continue

                left_value = row[left_key]
                right_value = row[right_key]
                if pd.isna(left_value) or pd.isna(right_value):
                    continue

                bounds_for_line[column_name] = (float(left_value), float(right_value))

            if bounds_for_line:
                saved_bounds[line_key] = bounds_for_line

        self.saved_integration_bounds = saved_bounds

    def browse_yy_lines(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select YY line list",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.yy_lines_path_var.set(selected)

    def browse_non_yy_lines(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select non-yy line list",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.non_yy_lines_path_var.set(selected)

    def browse_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Select output CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=Path(self.output_var.get()).name,
        )
        if selected:
            self.output_var.set(selected)

    def add_spectrum_row(self, label: str = "", path: str = "") -> None:
        frame = ttk.Frame(self.spectra_rows_frame)
        label_var = tk.StringVar(value=label)
        path_var = tk.StringVar(value=path)

        ttk.Label(frame, text="Label").pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=label_var, width=16).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(frame, text="CSV").pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=path_var, width=58).pack(side=tk.LEFT, padx=(4, 4), fill=tk.X, expand=True)
        ttk.Button(frame, text="Browse", command=lambda var=path_var: self._browse_spectrum_path(var)).pack(side=tk.LEFT)

        row = SpectrumConfigRow(frame=frame, label_var=label_var, path_var=path_var)
        ttk.Button(frame, text="Remove", command=lambda target=row: self.remove_spectrum_row(target)).pack(
            side=tk.LEFT,
            padx=(6, 0),
        )

        frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.spectrum_rows.append(row)

    def _browse_spectrum_path(self, target: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            title="Select spectrum CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            target.set(selected)

    def remove_spectrum_row(self, target: SpectrumConfigRow) -> None:
        if len(self.spectrum_rows) <= 1:
            messagebox.showerror("Invalid configuration", "At least one spectrum row is required.")
            return
        target.frame.destroy()
        self.spectrum_rows = [row for row in self.spectrum_rows if row is not target]

    def set_config_visibility(self, visible: bool) -> None:
        self.config_visible = visible
        if visible:
            if not self.config_container.winfo_manager():
                self.config_container.pack(side=tk.TOP, fill=tk.X, before=self.controls_frame)
            self.config_toggle_button.configure(text="Hide input configuration")
        else:
            if self.config_container.winfo_manager():
                self.config_container.pack_forget()
            self.config_toggle_button.configure(text="Show input configuration")

    def toggle_config_section(self) -> None:
        self.set_config_visibility(not self.config_visible)

    def _build_ui(self) -> None:
        config_shell = ttk.Frame(self.root, padding=(10, 10, 10, 0))
        config_shell.pack(side=tk.TOP, fill=tk.X)
        config_buttons = ttk.Frame(config_shell)
        config_buttons.pack(side=tk.TOP, anchor="w")
        self.config_toggle_button = ttk.Button(
            config_buttons,
            text="Hide input configuration",
            command=self.toggle_config_section,
        )
        self.config_toggle_button.pack(side=tk.LEFT)
        ttk.Button(config_buttons, text="Save configuration", command=self.save_configuration).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(config_buttons, text="Save configuration as...", command=self.save_configuration_as).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(config_buttons, text="Load configuration...", command=self.load_configuration_file).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        self.config_container = ttk.Frame(self.root, padding=(10, 2, 10, 4))
        self.config_container.pack(side=tk.TOP, fill=tk.X)
        self.config_frame = ttk.LabelFrame(self.config_container, text="Input configuration", padding=10)
        self.config_frame.pack(side=tk.TOP, fill=tk.X)

        linelist_frame = ttk.Frame(self.config_frame)
        linelist_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(linelist_frame, text="YY line list").grid(row=0, column=0, sticky="w")
        ttk.Entry(linelist_frame, textvariable=self.yy_lines_path_var, width=86).grid(
            row=0,
            column=1,
            padx=(6, 6),
            sticky="ew",
        )
        ttk.Button(linelist_frame, text="Browse", command=self.browse_yy_lines).grid(row=0, column=2, sticky="w")

        ttk.Label(linelist_frame, text="Non-yy line list").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(linelist_frame, textvariable=self.non_yy_lines_path_var, width=86).grid(
            row=1,
            column=1,
            padx=(6, 6),
            pady=(6, 0),
            sticky="ew",
        )
        ttk.Button(linelist_frame, text="Browse", command=self.browse_non_yy_lines).grid(
            row=1,
            column=2,
            sticky="w",
            pady=(6, 0),
        )
        linelist_frame.grid_columnconfigure(1, weight=1)

        spectrum_header = ttk.Frame(self.config_frame)
        spectrum_header.pack(side=tk.TOP, fill=tk.X, pady=(10, 2))
        ttk.Label(spectrum_header, text="Spectra to compare").pack(side=tk.LEFT)
        ttk.Button(spectrum_header, text="Add spectrum", command=self.add_spectrum_row).pack(side=tk.LEFT, padx=(10, 0))

        self.spectra_rows_frame = ttk.Frame(self.config_frame)
        self.spectra_rows_frame.pack(side=tk.TOP, fill=tk.X)

        self.controls_frame = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        self.controls_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(self.controls_frame, text="YY species").grid(row=0, column=0, sticky="w")
        self.species_box = ttk.Combobox(
            self.controls_frame,
            textvariable=self.current_species,
            values=self.species_options,
            state="readonly",
            width=18,
        )
        self.species_box.grid(row=0, column=1, padx=(6, 12), sticky="w")
        self.species_box.bind("<<ComboboxSelected>>", lambda _event: self.apply_filters(reset_index=True))

        ttk.Label(self.controls_frame, text="Max min flux (dip)").grid(row=0, column=2, sticky="w")
        ttk.Entry(self.controls_frame, textvariable=self.peak_threshold_var, width=8).grid(row=0, column=3, padx=(6, 12))

        ttk.Label(self.controls_frame, text="Peak window (A)").grid(row=0, column=4, sticky="w")
        ttk.Entry(self.controls_frame, textvariable=self.peak_window_var, width=8).grid(row=0, column=5, padx=(6, 12))

        ttk.Label(self.controls_frame, text="View half-width (A)").grid(row=0, column=6, sticky="w")
        ttk.Entry(self.controls_frame, textvariable=self.view_width_var, width=8).grid(row=0, column=7, padx=(6, 12))

        ttk.Label(self.controls_frame, text="Lower y-limit").grid(row=0, column=8, sticky="w")
        ttk.Entry(self.controls_frame, textvariable=self.y_min_var, width=8).grid(row=0, column=9, padx=(6, 12))

        ttk.Label(self.controls_frame, text="Upper y-limit").grid(row=0, column=10, sticky="w")
        ttk.Entry(self.controls_frame, textvariable=self.y_max_var, width=8).grid(row=0, column=11, padx=(6, 12))

        ttk.Button(self.controls_frame, text="Apply", command=self.apply_filters).grid(
            row=0, column=12, padx=(0, 12)
        )

        overlay_frame = ttk.Frame(self.controls_frame)
        overlay_frame.grid(row=1, column=0, columnspan=9, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            overlay_frame,
            text="Show other yy lines",
            variable=self.show_other_yy_var,
            command=self.refresh_plot,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            overlay_frame,
            text="Show non-yy lines",
            variable=self.show_non_yy_var,
            command=self.refresh_plot,
        ).pack(side=tk.LEFT, padx=(12, 0))

        voigt_frame = ttk.Frame(self.controls_frame)
        voigt_frame.grid(row=2, column=0, columnspan=13, sticky="w", pady=(8, 0))
        ttk.Label(voigt_frame, text="Measurement").pack(side=tk.LEFT)
        self.measurement_mode_box = ttk.Combobox(
            voigt_frame,
            textvariable=self.measurement_mode_var,
            values=["EW", "Manual Voigt", "Fit Voigt"],
            state="readonly",
            width=14,
        )
        self.measurement_mode_box.pack(side=tk.LEFT, padx=(6, 8))
        self.measurement_mode_box.bind("<<ComboboxSelected>>", lambda _event: self.measurement_mode_changed())
        self.manual_voigt_controls = ttk.Frame(voigt_frame)
        self.manual_voigt_controls.pack(side=tk.LEFT)
        self._add_voigt_scale(self.manual_voigt_controls, "Depth", self.voigt_depth_var, 0.0, 1.0, self.voigt_depth_label_var)
        self._add_voigt_scale(self.manual_voigt_controls, "Sigma (A)", self.voigt_sigma_var, 0.001, 0.3, self.voigt_sigma_label_var)
        self._add_voigt_scale(self.manual_voigt_controls, "Gamma (A)", self.voigt_gamma_var, 0.001, 0.3, self.voigt_gamma_label_var)
        self._add_voigt_scale(self.manual_voigt_controls, "Shift (A)", self.voigt_shift_var, -0.1, 0.1, self.voigt_shift_label_var)
        ttk.Button(voigt_frame, text="Fit Voigt", command=self.fit_voigt).pack(side=tk.LEFT, padx=(10, 0))
        self._update_voigt_labels()

        ttk.Label(overlay_frame, text="Output CSV").pack(side=tk.LEFT, padx=(18, 6))
        ttk.Entry(overlay_frame, textvariable=self.output_var, width=55).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(overlay_frame, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=(6, 0))

        nav = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        nav.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(nav, text="Previous", command=self.previous_line).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next", command=self.next_line).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav, text="Keep", command=self.keep_current_line).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(nav, text="Remove kept", command=self.remove_current_line).pack(side=tk.LEFT, padx=(6, 0))
        self.reset_bounds_button = ttk.Button(nav, text="Reset active bounds", command=self.reset_active_bounds)
        self.reset_bounds_button.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(nav, text="Show output path", command=self.show_output_path_message).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(nav, textvariable=self.status_var).pack(side=tk.LEFT, padx=(18, 0))
        self._update_mode_controls()

        figure = Figure(figsize=(13, 7), dpi=100)
        self.ax = figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        self.root.bind("<Left>", lambda _event: self.previous_line())
        self.root.bind("<Right>", lambda _event: self.next_line())
        self.root.bind("<Return>", lambda _event: self.keep_current_line())

    def parse_float(self, value: str, field_name: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a number.") from exc

    def _add_voigt_scale(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        minimum: float,
        maximum: float,
        value_label: tk.StringVar,
    ) -> None:
        ttk.Label(parent, text=label).pack(side=tk.LEFT, padx=(10, 3))
        ttk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            variable=variable,
            orient=tk.HORIZONTAL,
            length=90,
            command=lambda _value: self._voigt_slider_changed(),
        ).pack(side=tk.LEFT)
        ttk.Label(parent, textvariable=value_label, width=7).pack(side=tk.LEFT, padx=(2, 0))

    def _update_voigt_labels(self) -> None:
        self.voigt_depth_label_var.set(f"{self.voigt_depth_var.get():.3f}")
        self.voigt_sigma_label_var.set(f"{self.voigt_sigma_var.get():.4f}")
        self.voigt_gamma_label_var.set(f"{self.voigt_gamma_var.get():.4f}")
        self.voigt_shift_label_var.set(f"{self.voigt_shift_var.get():+.4f}")
        self.voigt_fit_width_label_var.set(f"{self.voigt_fit_width_var.get():.3f}")

    def _voigt_slider_changed(self) -> None:
        self._update_voigt_labels()
        if self.measurement_mode_var.get() == "Manual Voigt" and hasattr(self, "canvas"):
            self.refresh_plot()

    def measurement_mode_changed(self) -> None:
        self.dragging_handle = None
        self._update_mode_controls()
        self.refresh_plot()

    def _update_mode_controls(self) -> None:
        if self.measurement_mode_var.get() == "Manual Voigt":
            if not self.manual_voigt_controls.winfo_manager():
                self.manual_voigt_controls.pack(side=tk.LEFT)
        elif self.manual_voigt_controls.winfo_manager():
            self.manual_voigt_controls.pack_forget()
        if self.measurement_mode_var.get() == "Manual Voigt":
            self.reset_bounds_button.configure(state=tk.DISABLED)
        else:
            self.reset_bounds_button.configure(state=tk.NORMAL)

    def parse_optional_float(self, value: str, field_name: str) -> float | None:
        stripped = value.strip()
        if not stripped:
            return None
        return self.parse_float(stripped, field_name)

    def voigt_parameters(self) -> tuple[float, float, float]:
        depth = self.parse_float(self.voigt_depth_var.get(), "Voigt depth")
        sigma = self.parse_float(self.voigt_sigma_var.get(), "Voigt sigma")
        gamma = self.parse_float(self.voigt_gamma_var.get(), "Voigt gamma")
        if depth < 0 or sigma <= 0 or gamma <= 0:
            raise ValueError("Voigt depth must be non-negative; sigma and gamma must be positive.")
        return depth, sigma, gamma

    def voigt_values(self, wavelength: float, x: np.ndarray) -> tuple[np.ndarray, float]:
        depth, sigma, gamma = self.voigt_parameters()
        shift = float(self.voigt_shift_var.get())
        profile = voigt_profile(x - wavelength - shift, sigma, gamma)
        peak = float(voigt_profile(0.0, sigma, gamma))
        normalized_profile = profile / peak
        area = depth / peak
        return 1.0 - depth * normalized_profile, area

    def _voigt_model(
        self,
        wavelength: np.ndarray,
        center: float,
        depth: float,
        sigma: float,
        gamma: float,
        shift: float,
    ) -> np.ndarray:
        peak = float(voigt_profile(0.0, sigma, gamma))
        profile = voigt_profile(wavelength - center - shift, sigma, gamma) / peak
        return 1.0 - depth * profile

    def fit_voigt(self) -> None:
        row = self.current_line_row()
        if row is None or pd.isna(row["wavelength"]):
            return

        try:
            depth, sigma, gamma = self.voigt_parameters()
            fit_width = float(self.voigt_fit_width_var.get())
            if fit_width <= 0:
                raise ValueError("Voigt fit region must be positive.")
        except ValueError as exc:
            messagebox.showerror("Invalid Voigt parameters", str(exc))
            return

        center = float(row["wavelength"])
        line_key = (str(row["species"]), round(center, 6))
        fit_left, fit_right = self._get_fit_bounds(str(row["species"]), center)
        fit_width = max(center - fit_left, fit_right - center)
        fitted_parameters: dict[str, tuple[float, float, float, float]] = {}
        fitted_areas: dict[str, float] = {}
        failures: list[str] = []
        initial = [
            depth,
            min(sigma, fit_width * 0.8),
            min(gamma, fit_width * 0.8),
            float(self.voigt_shift_var.get()),
        ]
        bounds = ([0.0, 1e-5, 1e-5, -0.1], [2.0, fit_width, fit_width, 0.1])

        for spectrum in self.spectra:
            left_index, right_index = spectrum.bounds(center, fit_width)
            wave = spectrum.wavelength[left_index:right_index]
            flux = spectrum.flux[left_index:right_index]
            region_mask = (wave >= fit_left) & (wave <= fit_right)
            wave = wave[region_mask]
            flux = flux[region_mask]
            valid = np.isfinite(wave) & np.isfinite(flux)
            wave = wave[valid]
            flux = flux[valid]
            if wave.size < 5:
                failures.append(spectrum.label)
                continue
            try:
                parameters, _ = curve_fit(
                    lambda values, fit_depth, fit_sigma, fit_gamma, fit_shift: self._voigt_model(
                        values, center, fit_depth, fit_sigma, fit_gamma, fit_shift
                    ),
                    wave,
                    flux,
                    p0=initial,
                    bounds=bounds,
                    maxfev=10000,
                )
            except (RuntimeError, ValueError):
                failures.append(spectrum.label)
                continue

            fit_depth, fit_sigma, fit_gamma, fit_shift = map(float, parameters)
            fitted_parameters[spectrum.label] = (fit_depth, fit_sigma, fit_gamma, fit_shift)
            fitted_areas[spectrum.label] = fit_depth / float(voigt_profile(0.0, fit_sigma, fit_gamma))

        self.fitted_voigt_parameters = fitted_parameters
        self.fitted_voigt_areas = fitted_areas
        self.fitted_voigt_line_key = line_key
        self.measurement_mode_var.set("Fit Voigt")
        self.refresh_plot()
        if failures:
            messagebox.showwarning("Voigt fit incomplete", f"No fit was found for: {', '.join(failures)}")

    def min_flux_values(self, wavelength: float, half_window: float) -> dict[str, float]:
        key = (round(float(wavelength), 6), round(float(half_window), 6))
        if key not in self.peak_cache:
            minima: dict[str, float] = {}
            values: list[float] = []
            for spectrum, column_name in zip(self.spectra, self.spectrum_flux_columns):
                value = spectrum.min_flux(wavelength, half_window)
                minima[column_name] = value
                values.append(value)

            if values and not np.all(np.isnan(values)):
                minima["min_flux"] = float(np.nanmin(values))
            else:
                minima["min_flux"] = float("nan")
            self.peak_cache[key] = minima
        return self.peak_cache[key]

    def apply_filters(self, reset_index: bool = False) -> None:
        previous_row = None if reset_index else self.current_line_row()

        try:
            peak_threshold = self.parse_float(self.peak_threshold_var.get(), "Min peak flux")
            peak_window = self.parse_float(self.peak_window_var.get(), "Peak window")
            view_width = self.parse_float(self.view_width_var.get(), "View half-width")
            y_min = self.parse_optional_float(self.y_min_var.get(), "Lower y-limit")
            y_max = self.parse_optional_float(self.y_max_var.get(), "Upper y-limit")
            self.voigt_parameters()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        if peak_window <= 0 or view_width <= 0:
            messagebox.showerror("Invalid input", "Peak window and view half-width must be positive.")
            return

        self.active_peak_threshold = peak_threshold
        self.active_peak_window = peak_window
        self.active_view_width = view_width
        self.active_y_min = y_min
        self.active_y_max = y_max

        try:
            self._reload_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid input configuration", str(exc))
            self.status_var.set("Fix input paths/labels and apply again.")
            return

        self.ensure_output_loaded()

        species = self.current_species.get()
        species_lines = self.yy_lines[self.yy_lines["species"] == species].copy()
        if species_lines.empty:
            self.filtered_lines = species_lines
            self.current_index = 0
            self.refresh_plot()
            return

        peaks = species_lines["wavelength"].apply(lambda wl: self.min_flux_values(wl, self.active_peak_window))
        peak_columns = self.spectrum_flux_columns + ["min_flux"]
        species_lines[peak_columns] = pd.DataFrame(
            peaks.tolist(), index=species_lines.index
        )
        species_lines = species_lines[species_lines["min_flux"] >= self.active_peak_threshold].reset_index(drop=True)

        self.filtered_lines = species_lines
        if reset_index:
            self.current_index = 0
        elif previous_row is not None and not self.filtered_lines.empty:
            matching_rows = self.filtered_lines[
                (self.filtered_lines["species"] == previous_row["species"])
                & (np.isclose(self.filtered_lines["wavelength"], previous_row["wavelength"], atol=1e-6))
            ]
            if matching_rows.empty:
                self.current_index = 0
            else:
                self.current_index = int(matching_rows.index[0])
        elif self.current_index >= len(self.filtered_lines):
            self.current_index = max(len(self.filtered_lines) - 1, 0)
        self.refresh_plot()

    def current_line_row(self) -> pd.Series | None:
        if self.filtered_lines.empty:
            return None
        return self.filtered_lines.iloc[self.current_index]

    def previous_line(self) -> None:
        if self.filtered_lines.empty:
            return
        self.current_index = max(self.current_index - 1, 0)
        self.refresh_plot()

    def next_line(self) -> None:
        if self.filtered_lines.empty:
            return
        self.current_index = min(self.current_index + 1, len(self.filtered_lines) - 1)
        self.refresh_plot()

    def is_kept(self, species: str, wavelength: float) -> bool:
        if self.kept_lines.empty:
            return False
        matches = self.kept_lines[
            (self.kept_lines["species"] == species)
            & (np.isclose(self.kept_lines["wavelength"], wavelength, atol=1e-6))
        ]
        return not matches.empty

    def keep_current_line(self) -> None:
        row = self.current_line_row()
        if row is None:
            return

        self.ensure_output_loaded()
        output_text = self.output_var.get().strip()
        if not output_text:
            messagebox.showerror("Output required", "Choose an output CSV before keeping a line.")
            return
        output_path = Path(output_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        record = pd.DataFrame(
            [
                {
                    "species": row["species"],
                    "wavelength": row["wavelength"],
                    "min_flux": row["min_flux"],
                    "source": Path(self.yy_lines_path_var.get()).name,
                }
            ]
        )

        for column_name in self.spectrum_flux_columns:
            if column_name in row.index:
                record[column_name] = row[column_name]

        mode = self.measurement_mode_var.get()
        ew_values = self.current_ew_values() if mode == "EW" else {}
        for key, value in ew_values.items():
            record[key] = value

        if mode == "Fit Voigt" and self.fitted_voigt_line_key == (str(row["species"]), round(float(row["wavelength"]), 6)):
            for index, spectrum in enumerate(self.spectra):
                fitted = self.fitted_voigt_parameters.get(spectrum.label)
                if fitted is None:
                    continue
                suffix = self._ew_name_suffix(self.spectrum_flux_columns[index])
                record[f"voigt_fit_depth_{suffix}"] = fitted[0]
                record[f"voigt_fit_sigma_{suffix}"] = fitted[1]
                record[f"voigt_fit_gamma_{suffix}"] = fitted[2]
                record[f"voigt_fit_shift_{suffix}"] = fitted[3]
                record[f"voigt_fit_area_{suffix}"] = self.fitted_voigt_areas[spectrum.label]

        center = float(row["wavelength"])
        if mode == "EW":
            for column_name in self.spectrum_flux_columns:
                suffix = self._ew_name_suffix(column_name)
                left, right = self._get_ew_bounds(column_name, center)
                record[f"ew_left_{suffix}"] = left
                record[f"ew_right_{suffix}"] = right
        elif mode == "Manual Voigt":
            _, manual_area = self.voigt_values(center, np.array([center]))
            record["voigt_manual_area"] = manual_area
            record["voigt_manual_shift"] = float(self.voigt_shift_var.get())

        if self.kept_lines.empty:
            self.kept_lines = record
        else:
            remaining = self.kept_lines[
                ~(
                    (self.kept_lines["species"] == row["species"])
                    & (np.isclose(self.kept_lines["wavelength"], row["wavelength"], atol=1e-6))
                )
            ]
            self.kept_lines = pd.concat([remaining, record], ignore_index=True)

        self._refresh_saved_integration_bounds()

        self.kept_lines = self.kept_lines.sort_values(["species", "wavelength"]).reset_index(drop=True)
        self.kept_lines.to_csv(output_path, index=False)
        self.refresh_plot()

    def remove_current_line(self) -> None:
        row = self.current_line_row()
        self.ensure_output_loaded()
        if row is None or self.kept_lines.empty:
            return

        output_text = self.output_var.get().strip()
        if not output_text:
            messagebox.showerror("Output required", "Choose an output CSV before removing a line.")
            return
        output_path = Path(output_text)
        self.kept_lines = self.kept_lines[
            ~(
                (self.kept_lines["species"] == row["species"])
                & (np.isclose(self.kept_lines["wavelength"], row["wavelength"], atol=1e-6))
            )
        ].reset_index(drop=True)
        self._refresh_saved_integration_bounds()
        self.kept_lines.to_csv(output_path, index=False)
        self.refresh_plot()

    def show_output_path_message(self) -> None:
        output_path = Path(self.output_var.get())
        messagebox.showinfo("Output path", str(output_path.resolve()))

    def ensure_output_loaded(self) -> None:
        output_path = Path(self.output_var.get())
        if output_path != self.loaded_output_path:
            self.kept_lines = self._load_existing_output()
            self._refresh_saved_integration_bounds()

    def _ew_name_suffix(self, column_name: str) -> str:
        return column_name.replace("min_flux_", "", 1)

    def reset_active_bounds(self) -> None:
        row = self.current_line_row()
        if row is None:
            return
        center = float(row["wavelength"])
        if self.measurement_mode_var.get() == "Fit Voigt":
            self._initialize_fit_bounds(str(row["species"]), center)
        elif self.measurement_mode_var.get() == "EW":
            self._initialize_ew_bounds(center)
        self.refresh_plot()

    def reset_ew_bounds(self) -> None:
        self.reset_active_bounds()

    def _initialize_ew_bounds(self, center: float) -> None:
        self.integration_bounds = {
            column_name: (center - 3 * self.active_peak_window, center + 3 * self.active_peak_window)
            for column_name in self.spectrum_flux_columns
        }

    def _restore_ew_bounds(self, species: str, center: float) -> None:
        self._initialize_ew_bounds(center)
        saved_bounds = self.saved_integration_bounds.get((species, round(center, 6)), {})
        for column_name, bounds in saved_bounds.items():
            if column_name in self.integration_bounds:
                self.integration_bounds[column_name] = self._clamp_bounds(bounds[0], bounds[1], center)

    def _initialize_fit_bounds(self, species: str, center: float) -> None:
        width = min(float(self.voigt_fit_width_var.get()), self.active_view_width)
        self.fit_bounds[(species, round(center, 6))] = self._clamp_bounds(
            center - width,
            center + width,
            center,
        )

    def _get_fit_bounds(self, species: str, center: float) -> tuple[float, float]:
        key = (species, round(center, 6))
        if key not in self.fit_bounds:
            self._initialize_fit_bounds(species, center)
        left, right = self._clamp_bounds(*self.fit_bounds[key], center)
        self.fit_bounds[key] = (left, right)
        return left, right

    def _clamp_bounds(self, left: float, right: float, center: float) -> tuple[float, float]:
        min_x = center - self.active_view_width
        max_x = center + self.active_view_width
        min_width = max(self.active_view_width * 1e-4, 1e-5)

        left = float(np.clip(left, min_x, max_x))
        right = float(np.clip(right, min_x, max_x))

        if right - left < min_width:
            midpoint = 0.5 * (left + right)
            left = max(min_x, midpoint - 0.5 * min_width)
            right = min(max_x, midpoint + 0.5 * min_width)
            if right - left < min_width:
                right = min(max_x, left + min_width)
                left = max(min_x, right - min_width)

        return left, right

    def _get_ew_bounds(self, column_name: str, center: float) -> tuple[float, float]:
        default_left = center - 3 * self.active_peak_window
        default_right = center + 3 * self.active_peak_window
        left, right = self.integration_bounds.get(column_name, (default_left, default_right))
        left, right = self._clamp_bounds(left, right, center)
        self.integration_bounds[column_name] = (left, right)
        return left, right

    def _integrate_equivalent_width(self, spectrum: Spectrum, left: float, right: float) -> float:
        if left >= right:
            return float("nan")

        wave = spectrum.wavelength
        flux = spectrum.flux
        if wave.size < 2 or right < wave[0] or left > wave[-1]:
            return float("nan")

        lo = max(left, float(wave[0]))
        hi = min(right, float(wave[-1]))
        if lo >= hi:
            return float("nan")

        mask = (wave > lo) & (wave < hi)
        segment_wave = wave[mask]
        sample_wave = np.concatenate(([lo], segment_wave, [hi]))
        sample_flux = np.interp(sample_wave, wave, flux)
        integrand = 1.0 - sample_flux
        return float(np.trapezoid(integrand, sample_wave))

    def current_ew_values(self) -> dict[str, float]:
        row = self.current_line_row()
        if row is None:
            return {}

        center = float(row["wavelength"])
        ew_values: dict[str, float] = {}
        ew_scalar: list[float] = []

        for spectrum, column_name in zip(self.spectra, self.spectrum_flux_columns):
            left, right = self._get_ew_bounds(column_name, center)
            ew = self._integrate_equivalent_width(spectrum, left, right)
            suffix = self._ew_name_suffix(column_name)

            ew_values[f"ew_{suffix}"] = ew

            if not np.isnan(ew):
                ew_scalar.append(ew)

        ew_values["ew_mean"] = float(np.mean(ew_scalar)) if ew_scalar else float("nan")
        return ew_values

    def on_mouse_press(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        row = self.current_line_row()
        if row is None:
            return

        center = float(row["wavelength"])
        x = float(event.xdata)
        tolerance = max(0.02, self.active_view_width * 0.02)
        candidates: list[tuple[float, str, str]] = []

        if self.measurement_mode_var.get() == "Fit Voigt":
            left, right = self._get_fit_bounds(str(row["species"]), center)
            candidates.extend([(abs(x - left), "__fit__", "left"), (abs(x - right), "__fit__", "right")])
        elif self.measurement_mode_var.get() == "EW":
            for column_name in self.spectrum_flux_columns:
                left, right = self._get_ew_bounds(column_name, center)
                candidates.append((abs(x - left), column_name, "left"))
                candidates.append((abs(x - right), column_name, "right"))

        if not candidates:
            return

        distance, column_name, side = min(candidates, key=lambda item: item[0])
        if distance <= tolerance:
            self.dragging_handle = (column_name, side)

    def on_mouse_move(self, event: Any) -> None:
        if self.dragging_handle is None or event.inaxes != self.ax or event.xdata is None:
            return
        row = self.current_line_row()
        if row is None:
            return

        column_name, side = self.dragging_handle
        center = float(row["wavelength"])
        min_x = center - self.active_view_width
        max_x = center + self.active_view_width
        min_width = max(self.active_view_width * 1e-4, 1e-5)
        x = float(np.clip(event.xdata, min_x, max_x))

        if column_name == "__fit__":
            left, right = self._get_fit_bounds(str(row["species"]), center)
            if side == "left":
                left = min(x, right - min_width)
            else:
                right = max(x, left + min_width)
            self.fit_bounds[(str(row["species"]), round(center, 6))] = self._clamp_bounds(left, right, center)
        else:
            left, right = self._get_ew_bounds(column_name, center)
            if side == "left":
                left = min(x, right - min_width)
            else:
                right = max(x, left + min_width)
            self.integration_bounds[column_name] = self._clamp_bounds(left, right, center)
        self.refresh_plot()

    def on_mouse_release(self, _event: Any) -> None:
        self.dragging_handle = None

    def refresh_plot(self) -> None:
        self.ax.clear()
        row = self.current_line_row()
        if row is None:
            self.ax.set_title("No yy lines match the current species and peak-flux filter")
            self.ax.set_xlabel("Wavelength (A)")
            self.ax.set_ylabel("Flux")
            self.status_var.set("0 lines available")
            self.canvas.draw_idle()
            return

        wavelength = float(row["wavelength"])
        line_key = (str(row["species"]), round(wavelength, 6))
        if self.last_line_key != line_key:
            self._restore_ew_bounds(str(row["species"]), wavelength)
            fit_key = (str(row["species"]), round(wavelength, 6))
            if fit_key not in self.fit_bounds:
                self._initialize_fit_bounds(str(row["species"]), wavelength)
            self.last_line_key = line_key
            if self.fitted_voigt_line_key != line_key:
                self.fitted_voigt_parameters = {}
                self.fitted_voigt_areas = {}

        view_width = self.active_view_width
        peak_window = self.active_peak_window

        flux_segments: list[np.ndarray] = []
        for index, spectrum in enumerate(self.spectra):
            wave, flux = spectrum.region(wavelength, view_width)
            color = SPECTRUM_COLOR_CYCLE[index % len(SPECTRUM_COLOR_CYCLE)]
            self.ax.plot(wave, flux, color=color, linewidth=1.0, label=spectrum.label)
            if flux.size:
                flux_segments.append(flux)

            if self.measurement_mode_var.get() == "EW":
                column_name = self.spectrum_flux_columns[index]
                left, right = self._get_ew_bounds(column_name, wavelength)
                self.ax.axvspan(left, right, color=color, alpha=0.06)
                self.ax.axvline(left, color=color, linewidth=1.1, alpha=0.9)
                self.ax.axvline(right, color=color, linewidth=1.1, linestyle="--", alpha=0.9)

        self.last_voigt_area = float("nan")
        mode = self.measurement_mode_var.get()
        if mode == "Manual Voigt":
            voigt_wave = np.linspace(wavelength - view_width, wavelength + view_width, 1200)
            try:
                voigt_flux, self.last_voigt_area = self.voigt_values(wavelength, voigt_wave)
            except ValueError:
                voigt_flux = None
            if voigt_flux is not None:
                self.ax.plot(
                    voigt_wave,
                    voigt_flux,
                    color="black",
                    linewidth=1.4,
                    linestyle="-.",
                    label="Manual Voigt",
                )
        elif mode == "Fit Voigt":
            fit_left, fit_right = self._get_fit_bounds(str(row["species"]), wavelength)
            self.ax.axvspan(fit_left, fit_right, color="black", alpha=0.06, label="Fit region")
            self.ax.axvline(fit_left, color="black", linewidth=1.1, alpha=0.9)
            self.ax.axvline(fit_right, color="black", linewidth=1.1, linestyle="--", alpha=0.9)
            voigt_wave = np.linspace(wavelength - view_width, wavelength + view_width, 1200)
            for index, spectrum in enumerate(self.spectra):
                fitted = self.fitted_voigt_parameters.get(spectrum.label)
                if fitted is None:
                    continue
                fitted_flux = self._voigt_model(
                    voigt_wave,
                    wavelength,
                    fitted[0],
                    fitted[1],
                    fitted[2],
                    fitted[3],
                )
                self.ax.plot(
                    voigt_wave,
                    fitted_flux,
                    color=SPECTRUM_COLOR_CYCLE[index % len(SPECTRUM_COLOR_CYCLE)],
                    linewidth=1.5,
                    linestyle="--",
                    label=f"Fitted Voigt: {spectrum.label}",
                )
        self.ax.axvline(wavelength, color="forestgreen", linewidth=1.8, label=f"Current yy line: {row['species']}")

        if self.show_other_yy_var.get():
            yy_visible = self.yy_lines[
                (self.yy_lines["wavelength"] >= wavelength - view_width)
                & (self.yy_lines["wavelength"] <= wavelength + view_width)
            ]
            yy_visible = yy_visible[
                ~(
                    (yy_visible["species"] == row["species"])
                    & (np.isclose(yy_visible["wavelength"], wavelength, atol=1e-6))
                )
            ]
            self._draw_reference_lines(yy_visible, "darkorange", "--", 0.82, "Other yy")

        if self.show_non_yy_var.get():
            non_yy_visible = self.non_yy_lines[
                (self.non_yy_lines["wavelength"] >= wavelength - view_width)
                & (self.non_yy_lines["wavelength"] <= wavelength + view_width)
            ]
            self._draw_reference_lines(non_yy_visible, "dimgray", ":", 0.68, "Non-yy")

        combined_flux = np.concatenate(flux_segments) if flux_segments else np.array([0.0, 1.0])
        lower = float(np.nanmin(combined_flux))
        upper = float(np.nanmax(combined_flux))
        padding = max((upper - lower) * 0.12, 0.03)
        y_min = self.active_y_min if self.active_y_min is not None else self.active_peak_threshold + padding
        y_max = self.active_y_max if self.active_y_max is not None else upper + padding
        if y_min >= y_max:
            y_min = min(lower - padding, y_max - 0.01)

        self.ax.set_xlim(wavelength - view_width, wavelength + view_width)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel("Wavelength (A)")
        self.ax.set_ylabel("Flux")
        self.ax.set_title(f"{row['species']} at {wavelength:.4f} A")
        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.grid(alpha=0.18)

        kept_marker = "kept" if self.is_kept(row["species"], wavelength) else "not kept"
        minima_text_parts: list[str] = []
        for spectrum, column_name in zip(self.spectra, self.spectrum_flux_columns):
            if column_name in row.index:
                minima_text_parts.append(f"{spectrum.label}={row[column_name]:.3f}")
        minima_text = "  ".join(minima_text_parts)

        ew_values = self.current_ew_values() if mode == "EW" else {}
        self.last_ew_values = ew_values
        ew_text_parts: list[str] = []
        for spectrum, column_name in zip(self.spectra, self.spectrum_flux_columns):
            suffix = self._ew_name_suffix(column_name)
            ew_value = ew_values.get(f"ew_{suffix}", float("nan"))
            ew_text_parts.append(f"{spectrum.label}={ew_value:.5f}A")
        ew_text = "  ".join(ew_text_parts)
        ew_mean = ew_values.get("ew_mean", float("nan"))
        voigt_text = f"  Voigt area={self.last_voigt_area:.5f}A" if mode == "Manual Voigt" else ""
        fitted_text = (
            "  Fit area=" + ", ".join(
                f"{label}={area:.5f}A" for label, area in self.fitted_voigt_areas.items()
            )
            if mode == "Fit Voigt"
            else ""
        )

        measurement_text = ""
        if mode == "EW":
            measurement_text = f"EW: {ew_text}  mean={ew_mean:.5f}A"
        elif mode == "Manual Voigt":
            measurement_text = voigt_text.strip()
        else:
            measurement_text = fitted_text.strip() or "Fit area=not fitted"
        self.status_var.set(
            f"{self.current_index + 1}/{len(self.filtered_lines)}  |  {minima_text}  dip={row['min_flux']:.3f}  |  {measurement_text}  |  {kept_marker}  |  saved={len(self.kept_lines)}"
        )

        if mode == "EW":
            self.ax.axvspan(wavelength - peak_window, wavelength + peak_window, color="forestgreen", alpha=0.06)
        self.canvas.draw_idle()

    def _draw_reference_lines(
        self,
        lines: pd.DataFrame,
        color: str,
        linestyle: str,
        text_y: float,
        label_prefix: str,
    ) -> None:
        if lines.empty:
            return
        label_used = False
        for offset, (_, line) in enumerate(lines.iterrows()):
            label = label_prefix if not label_used else None
            self.ax.axvline(line["wavelength"], color=color, linestyle=linestyle, linewidth=0.8, alpha=0.7, label=label)
            text_level = text_y - 0.06 * (offset % 3)
            self.ax.text(
                line["wavelength"],
                text_level,
                line["species"],
                transform=self.ax.get_xaxis_transform(),
                rotation=90,
                fontsize=6,
                ha="center",
                va="top",
                color=color,
            )
            label_used = True


def main() -> None:
    root = tk.Tk()
    app = LineReviewerApp(root)
    root.minsize(1150, 600)
    root.mainloop()


if __name__ == "__main__":
    main()