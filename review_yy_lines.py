from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# DEFAULT VALUES
DEFAULT_VIEW_WIDTH = 3.0
DEFAULT_PEAK_WINDOW = 0.05
DEFAULT_PEAK_THRESHOLD = 0.4 
DEFAULT_YMIN = 0.35
DEFAULT_YMAX = 1.1
DEFAULT_OUTPUT_NAME = "kept_lines_yy.csv"
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

        self.project_root = Path(__file__).resolve().parent.parent
        self.results_dir = self.project_root / "python stuff" / "results"
        self.output_path = self.results_dir / DEFAULT_OUTPUT_NAME
        self.default_spectra = [
            ("Star A", self.project_root / "data" / "Al_Phe_A_sorted.csv"),
            ("Star B", self.project_root / "data" / "Al_Phe_B_sorted.csv"),
        ]

        self.spectrum_rows: list[SpectrumConfigRow] = []
        self.spectra: list[Spectrum] = []
        self.spectrum_flux_columns: list[str] = []
        self.yy_lines = pd.DataFrame(columns=["species", "wavelength"])
        self.non_yy_lines = pd.DataFrame(columns=["species", "wavelength"])

        self.yy_lines_path_var = tk.StringVar(value=str(self.results_dir / "lines_yy.csv"))
        self.non_yy_lines_path_var = tk.StringVar(value=str(self.results_dir / "lines.csv"))

        self.species_options: list[str] = []

        self.current_species = tk.StringVar(value="")
        self.peak_threshold_var = tk.StringVar(value=f"{DEFAULT_PEAK_THRESHOLD:.2f}")
        self.peak_window_var = tk.StringVar(value=f"{DEFAULT_PEAK_WINDOW:.2f}")
        self.view_width_var = tk.StringVar(value=f"{DEFAULT_VIEW_WIDTH:.2f}")
        self.y_min_var = tk.StringVar(value=str(DEFAULT_YMIN))
        self.y_max_var = tk.StringVar(value=str(DEFAULT_YMAX))
        self.output_var = tk.StringVar(value=str(self.output_path))
        self.show_other_yy_var = tk.BooleanVar(value=True)
        self.show_non_yy_var = tk.BooleanVar(value=True)
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

        self._build_ui()

        for default_label, default_path in self.default_spectra:
            self.add_spectrum_row(default_label, str(default_path))

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

        current = self.current_species.get()
        if current not in self.species_options:
            self.current_species.set(self.species_options[0])
        if hasattr(self, "species_box"):
            self.species_box.configure(values=self.species_options)

    def _load_existing_output(self) -> pd.DataFrame:
        path = Path(self.output_var.get())
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            if "species" in frame.columns and "wavelength" in frame.columns:
                self.loaded_output_path = path
                return frame
        self.loaded_output_path = path
        return pd.DataFrame(columns=["species", "wavelength", "min_flux", "source"])

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

    def _build_ui(self) -> None:
        config = ttk.LabelFrame(self.root, text="Input configuration", padding=10)
        config.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 4))

        linelist_frame = ttk.Frame(config)
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

        spectrum_header = ttk.Frame(config)
        spectrum_header.pack(side=tk.TOP, fill=tk.X, pady=(10, 2))
        ttk.Label(spectrum_header, text="Spectra to compare").pack(side=tk.LEFT)
        ttk.Button(spectrum_header, text="Add spectrum", command=self.add_spectrum_row).pack(side=tk.LEFT, padx=(10, 0))

        self.spectra_rows_frame = ttk.Frame(config)
        self.spectra_rows_frame.pack(side=tk.TOP, fill=tk.X)

        controls = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="YY species").grid(row=0, column=0, sticky="w")
        self.species_box = ttk.Combobox(
            controls,
            textvariable=self.current_species,
            values=self.species_options,
            state="readonly",
            width=18,
        )
        self.species_box.grid(row=0, column=1, padx=(6, 12), sticky="w")
        self.species_box.bind("<<ComboboxSelected>>", lambda _event: self.apply_filters(reset_index=True))

        ttk.Label(controls, text="Max min flux (dip)").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.peak_threshold_var, width=8).grid(row=0, column=3, padx=(6, 12))

        ttk.Label(controls, text="Peak window (A)").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self.peak_window_var, width=8).grid(row=0, column=5, padx=(6, 12))

        ttk.Label(controls, text="View half-width (A)").grid(row=0, column=6, sticky="w")
        ttk.Entry(controls, textvariable=self.view_width_var, width=8).grid(row=0, column=7, padx=(6, 12))

        ttk.Label(controls, text="Lower y-limit").grid(row=0, column=8, sticky="w")
        ttk.Entry(controls, textvariable=self.y_min_var, width=8).grid(row=0, column=9, padx=(6, 12))

        ttk.Label(controls, text="Upper y-limit").grid(row=0, column=10, sticky="w")
        ttk.Entry(controls, textvariable=self.y_max_var, width=8).grid(row=0, column=11, padx=(6, 12))

        ttk.Button(controls, text="Apply", command=self.apply_filters).grid(
            row=0, column=12, padx=(0, 12)
        )

        overlay_frame = ttk.Frame(controls)
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

        ttk.Label(overlay_frame, text="Output CSV").pack(side=tk.LEFT, padx=(18, 6))
        ttk.Entry(overlay_frame, textvariable=self.output_var, width=55).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(overlay_frame, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=(6, 0))

        nav = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        nav.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(nav, text="Previous", command=self.previous_line).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next", command=self.next_line).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav, text="Keep", command=self.keep_current_line).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(nav, text="Remove kept", command=self.remove_current_line).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav, text="Show output path", command=self.show_output_path_message).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(nav, textvariable=self.status_var).pack(side=tk.LEFT, padx=(18, 0))

        figure = Figure(figsize=(13, 7), dpi=100)
        self.ax = figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.root.bind("<Left>", lambda _event: self.previous_line())
        self.root.bind("<Right>", lambda _event: self.next_line())
        self.root.bind("<Return>", lambda _event: self.keep_current_line())

    def parse_float(self, value: str, field_name: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a number.") from exc

    def parse_optional_float(self, value: str, field_name: str) -> float | None:
        stripped = value.strip()
        if not stripped:
            return None
        return self.parse_float(stripped, field_name)

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
        output_path = Path(self.output_var.get())
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

        self.kept_lines = self.kept_lines.sort_values(["species", "wavelength"]).reset_index(drop=True)
        self.kept_lines.to_csv(output_path, index=False)
        self.refresh_plot()

    def remove_current_line(self) -> None:
        row = self.current_line_row()
        self.ensure_output_loaded()
        if row is None or self.kept_lines.empty:
            return

        output_path = Path(self.output_var.get())
        self.kept_lines = self.kept_lines[
            ~(
                (self.kept_lines["species"] == row["species"])
                & (np.isclose(self.kept_lines["wavelength"], row["wavelength"], atol=1e-6))
            )
        ].reset_index(drop=True)
        self.kept_lines.to_csv(output_path, index=False)
        self.refresh_plot()

    def show_output_path_message(self) -> None:
        output_path = Path(self.output_var.get())
        messagebox.showinfo("Output path", str(output_path.resolve()))

    def ensure_output_loaded(self) -> None:
        output_path = Path(self.output_var.get())
        if output_path != self.loaded_output_path:
            self.kept_lines = self._load_existing_output()

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
        view_width = self.active_view_width
        peak_window = self.active_peak_window

        flux_segments: list[np.ndarray] = []
        for index, spectrum in enumerate(self.spectra):
            wave, flux = spectrum.region(wavelength, view_width)
            color = SPECTRUM_COLOR_CYCLE[index % len(SPECTRUM_COLOR_CYCLE)]
            self.ax.plot(wave, flux, color=color, linewidth=1.0, label=spectrum.label)
            if flux.size:
                flux_segments.append(flux)
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
        self.status_var.set(
            f"{self.current_index + 1}/{len(self.filtered_lines)}  |  {minima_text}  dip={row['min_flux']:.3f}  |  {kept_marker}  |  saved={len(self.kept_lines)}"
        )

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
    root.minsize(1150, 760)
    root.mainloop()


if __name__ == "__main__":
    main()