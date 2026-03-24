from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


DEFAULT_VIEW_WIDTH = 3.0
DEFAULT_PEAK_WINDOW = 0.05
DEFAULT_PEAK_THRESHOLD = 0.4  # keep lines whose min flux is at or above this value
DEFAULT_YMIN_OVERRIDE = 0.35
DEFAULT_YMAX_OVERRIDE = 1.1
DEFAULT_OUTPUT_NAME = "kept_lines_yy.csv"


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


class LineReviewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YY Line Reviewer")

        self.project_root = Path(__file__).resolve().parent.parent
        self.results_dir = self.project_root / "python stuff" / "results"
        self.output_path = self.results_dir / DEFAULT_OUTPUT_NAME

        self.yy_lines = pd.read_csv(self.results_dir / "lines_yy.csv")
        self.non_yy_lines = pd.read_csv(self.results_dir / "lines.csv")
        self.star_a = self._load_spectrum(self.project_root / "data" / "Al_Phe_A_sorted.csv", "Star A")
        self.star_b = self._load_spectrum(self.project_root / "data" / "Al_Phe_B_sorted.csv", "Star B")

        self.yy_lines = self.yy_lines.sort_values(["species", "wavelength"]).reset_index(drop=True)
        self.non_yy_lines = self.non_yy_lines.sort_values("wavelength").reset_index(drop=True)

        self.species_options = sorted(self.yy_lines["species"].dropna().unique().tolist())
        if not self.species_options:
            raise ValueError("No species found in lines_yy.csv")

        self.current_species = tk.StringVar(value=self.species_options[0])
        self.peak_threshold_var = tk.StringVar(value=f"{DEFAULT_PEAK_THRESHOLD:.2f}")
        self.peak_window_var = tk.StringVar(value=f"{DEFAULT_PEAK_WINDOW:.2f}")
        self.view_width_var = tk.StringVar(value=f"{DEFAULT_VIEW_WIDTH:.2f}")
        self.y_min_var = tk.StringVar(value=DEFAULT_YMIN_OVERRIDE)
        self.y_max_var = tk.StringVar(value=DEFAULT_YMAX_OVERRIDE)
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

        self.filtered_lines = pd.DataFrame(columns=self.yy_lines.columns)
        self.current_index = 0
        self.peak_cache: dict[tuple[float, float], tuple[float, float, float]] = {}
        self.kept_lines = self._load_existing_output()

        self._build_ui()
        self.apply_filters(reset_index=True)

    def _load_spectrum(self, path: Path, label: str) -> Spectrum:
        frame = pd.read_csv(path)
        return Spectrum(
            wavelength=frame["Wavelength"].to_numpy(dtype=float),
            flux=frame["Flux"].to_numpy(dtype=float),
            label=label,
        )

    def _load_existing_output(self) -> pd.DataFrame:
        path = Path(self.output_var.get())
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            if "species" in frame.columns and "wavelength" in frame.columns:
                self.loaded_output_path = path
                return frame
        self.loaded_output_path = path
        return pd.DataFrame(
            columns=["species", "wavelength", "min_flux_a", "min_flux_b", "min_flux", "source"]
        )

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="YY species").grid(row=0, column=0, sticky="w")
        species_box = ttk.Combobox(
            controls,
            textvariable=self.current_species,
            values=self.species_options,
            state="readonly",
            width=18,
        )
        species_box.grid(row=0, column=1, padx=(6, 12), sticky="w")
        species_box.bind("<<ComboboxSelected>>", lambda _event: self.apply_filters(reset_index=True))

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

    def min_flux_values(self, wavelength: float, half_window: float) -> tuple[float, float, float]:
        key = (round(float(wavelength), 6), round(float(half_window), 6))
        if key not in self.peak_cache:
            min_a = self.star_a.min_flux(wavelength, half_window)
            min_b = self.star_b.min_flux(wavelength, half_window)
            combined = float(np.nanmin([min_a, min_b])) if not (np.isnan(min_a) and np.isnan(min_b)) else float("nan")
            self.peak_cache[key] = (min_a, min_b, combined)
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
        self.ensure_output_loaded()

        species = self.current_species.get()
        species_lines = self.yy_lines[self.yy_lines["species"] == species].copy()
        if species_lines.empty:
            self.filtered_lines = species_lines
            self.current_index = 0
            self.refresh_plot()
            return

        peaks = species_lines["wavelength"].apply(lambda wl: self.min_flux_values(wl, self.active_peak_window))
        species_lines[["min_flux_a", "min_flux_b", "min_flux"]] = pd.DataFrame(
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
                    "min_flux_a": row["min_flux_a"],
                    "min_flux_b": row["min_flux_b"],
                    "min_flux": row["min_flux"],
                    "source": "lines_yy.csv",
                }
            ]
        )

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

        wave_a, flux_a = self.star_a.region(wavelength, view_width)
        wave_b, flux_b = self.star_b.region(wavelength, view_width)

        self.ax.plot(wave_a, flux_a, color="steelblue", linewidth=1.0, label=self.star_a.label)
        self.ax.plot(wave_b, flux_b, color="tomato", linewidth=1.0, label=self.star_b.label)
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

        combined_flux = np.concatenate([flux_a, flux_b]) if len(flux_a) or len(flux_b) else np.array([0.0, 1.0])
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
        self.status_var.set(
            f"{self.current_index + 1}/{len(self.filtered_lines)}  |  min A={row['min_flux_a']:.3f}  min B={row['min_flux_b']:.3f}  dip={row['min_flux']:.3f}  |  {kept_marker}  |  saved={len(self.kept_lines)}"
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