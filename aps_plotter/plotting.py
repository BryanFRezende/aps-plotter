"""Publication-quality Matplotlib plots for APS journals.

The main function, :func:`aps_plot`, accepts a flat sequence of x/y arrays::

    arrays = [x1, y1, x2, y2, ...]

and one plot type per dataset::

    plot_types = ["scatter", "plot", ...]  # short forms "s" and "p" work

The defaults are designed around APS's general figure guidance:

* single-column figures are 8.5 cm (3 3/8 in) wide;
* type remains legible after reduction (9 pt default);
* data markers are at least 1 mm across (4 pt default);
* lines are at least 0.5 pt wide (1.25 pt default);
* colors use an accessible palette, with distinct markers and line styles
  providing redundant, grayscale-readable encoding;
* units can be supplied separately and are rendered in parentheses; and
* vector output is supported, while raster output defaults to 600 dpi.

These are APS-oriented defaults rather than a guarantee of journal compliance.
Always inspect the finished figure at its intended publication size.

The function changes no global Matplotlib settings. It returns ``(fig, ax)``
so the result can be refined with ordinary Matplotlib commands.

Requires
--------
matplotlib >= 3.5 and numpy
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Literal
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter


# Okabe-Ito-inspired palette, reordered so yellow is not used early on white.
# Color should never be the only way datasets are distinguished; default line
# styles and markers also cycle.
APS_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow (use cautiously on white)
)

APS_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
APS_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))

# APS specifies 8.5 cm for one column and permits 1.5- or 2-column figures.
# The double-column value is the approximately 7-inch REVTeX text width.
APS_WIDTHS_IN = {
    "single": 8.5 / 2.54,
    "onehalf": 1.5 * 8.5 / 2.54,
    "double": 7.0,
}

APS_RCPARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "font.size": 9.0,
    "mathtext.fontset": "stix",
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "axes.linewidth": 0.8,
    "axes.unicode_minus": True,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4,
    "lines.linewidth": 1.25,
    "lines.markersize": 4.0,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType fonts; keeps text editable
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # preserve text rather than converting to paths
}

APS_MIN_LINEWIDTH_PT = 0.5
APS_MIN_MARKER_DIAMETER_PT = 72.0 / 25.4  # 1 mm in points
APS_PREFERRED_RASTER_DPI = 600

_RASTER_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff", "webp"}


def _datasets_from_flat_arrays(
    arrays: Sequence[Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validate ``[x1, y1, x2, y2, ...]`` and return paired arrays."""
    if isinstance(arrays, (str, bytes)) or not isinstance(arrays, Sequence):
        raise TypeError("arrays must be a sequence: [x1, y1, x2, y2, ...].")
    if len(arrays) == 0 or len(arrays) % 2:
        raise ValueError("arrays must contain a nonzero, even number of arrays.")

    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, len(arrays), 2):
        x = np.asarray(arrays[i])
        y = np.asarray(arrays[i + 1])
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"Dataset {i // 2}: x and y must both be one-dimensional.")
        if x.size != y.size:
            raise ValueError(
                f"Dataset {i // 2}: x has {x.size} points but y has {y.size}."
            )
        if x.size == 0:
            raise ValueError(f"Dataset {i // 2}: x and y cannot be empty.")
        datasets.append((x, y))
    return datasets


def _per_series(
    value: Any,
    n: int,
    name: str,
    *,
    default: Sequence[Any] | Any,
) -> list[Any]:
    """Expand a scalar/default or validate a per-series sequence."""
    if value is None:
        if isinstance(default, Sequence) and not isinstance(default, (str, bytes)):
            return list(islice(cycle(default), n))
        return [default] * n
    if isinstance(value, Mapping):
        return [value] * n
    if isinstance(value, (str, bytes)) or np.isscalar(value):
        return [value] * n
    result = list(value)
    if len(result) != n:
        raise ValueError(f"{name} must have length {n}; received {len(result)}.")
    return result


def _errors_per_series(value: Any, n: int, name: str) -> list[Any]:
    """Normalize error arrays without mistaking a single ndarray for n arrays."""
    if value is None:
        return [None] * n
    if np.isscalar(value):
        return [value] * n
    if n == 1:
        # Accept a direct N-vector or (2, N) array. Also accept the explicitly
        # nested one-dataset form [error_array].
        if (
            isinstance(value, (list, tuple))
            and len(value) == 1
            and not np.isscalar(value[0])
        ):
            return [value[0]]
        return [value]
    result = list(value)
    if len(result) != n:
        raise ValueError(
            f"{name} must be None or contain one scalar/array per dataset ({n})."
        )
    return result


def _axis_label(quantity: str | None, unit: str | None) -> str:
    """Build the APS form 'quantity (unit)' while allowing prebuilt labels."""
    quantity = "" if quantity is None else quantity
    return f"{quantity} ({unit})".strip() if unit else quantity


def _normalize_plot_type(value: str) -> Literal["scatter", "plot", "both"]:
    aliases = {
        "s": "scatter",
        "scatter": "scatter",
        "p": "plot",
        "plot": "plot",
        "line": "plot",
        "b": "both",
        "both": "both",
        "line+markers": "both",
        "plot+scatter": "both",
    }
    key = value.lower().strip()
    if key not in aliases:
        raise ValueError(
            f"Unknown plot type {value!r}; use 'scatter'/'s', 'plot'/'p', "
            "or 'both'/'b'."
        )
    return aliases[key]  # type: ignore[return-value]


def _validate_log_data(
    datasets: Sequence[tuple[np.ndarray, np.ndarray]], xscale: str, yscale: str
) -> None:
    """Give a useful error before Matplotlib silently drops nonpositive log data."""
    for i, (x, y) in enumerate(datasets):
        if xscale == "log" and np.any(np.asarray(x, dtype=float) <= 0):
            raise ValueError(f"Dataset {i}: x contains nonpositive values on a log axis.")
        if yscale == "log" and np.any(np.asarray(y, dtype=float) <= 0):
            raise ValueError(f"Dataset {i}: y contains nonpositive values on a log axis.")


def _apply_scientific_formatter(
    ax: Axes,
    axis: Literal["x", "y"],
    powerlimits: tuple[int, int] | None,
) -> None:
    if powerlimits is None:
        return
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits(powerlimits)
    formatter.set_useOffset(False)
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(formatter)


def _save_figure(
    fig: Figure,
    save: str | Path,
    formats: Sequence[str],
    dpi: int,
    transparent: bool,
    savefig_kwargs: Mapping[str, Any] | None,
) -> list[Path]:
    """Save one named file or a basename in each requested format."""
    path = Path(save).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    format_list = [formats] if isinstance(formats, str) else list(formats)
    targets = (
        [path]
        if path.suffix
        else [path.with_suffix(f".{f.lstrip('.')}") for f in format_list]
    )

    raster_targets = [
        target for target in targets if target.suffix.lower().lstrip(".") in _RASTER_FORMATS
    ]
    if raster_targets and dpi < APS_PREFERRED_RASTER_DPI:
        warnings.warn(
            "APS recommends high-resolution raster output; 600 dpi or higher is "
            f"preferred. Requested dpi={dpi}.",
            UserWarning,
            stacklevel=2,
        )

    kwargs = dict(savefig_kwargs or {})
    saved: list[Path] = []
    for target in targets:
        fig.savefig(target, dpi=dpi, transparent=transparent, **kwargs)
        saved.append(target)
    return saved


def aps_plot(
    arrays: Sequence[Any],
    *,
    plot_types: str | Sequence[str] = "scatter",
    labels: Sequence[str | None] | None = None,
    colors: str | Sequence[Any] | None = None,
    markers: str | Sequence[str | None] | None = None,
    linestyles: str | Sequence[Any] | None = None,
    linewidths: float | Sequence[float] = 1.25,
    markersizes: float | Sequence[float] = 4.0,
    markerfacecolors: str | Sequence[Any] | None = None,
    markeredgecolors: str | Sequence[Any] | None = None,
    alphas: float | Sequence[float] = 1.0,
    xerr: Sequence[Any] | np.ndarray | None = None,
    yerr: Sequence[Any] | np.ndarray | None = None,
    capsize: float = 2.0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    xscale: Literal["linear", "log", "symlog", "logit"] = "linear",
    yscale: Literal["linear", "log", "symlog", "logit"] = "linear",
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xticks: Sequence[float] | None = None,
    yticks: Sequence[float] | None = None,
    x_tick_decimals: int | None = None,
    y_tick_decimals: int | None = None,
    x_sci_powerlimits: tuple[int, int] | None = (-3, 3),
    y_sci_powerlimits: tuple[int, int] | None = (-3, 3),
    title: str | None = None,
    panel_label: str | None = None,
    column: Literal["single", "onehalf", "double"] | float = "single",
    height: float | None = None,
    aspect: float = 0.75,
    legend: bool = True,
    legend_loc: str = "best",
    legend_ncol: int = 1,
    legend_kwargs: Mapping[str, Any] | None = None,
    minor_ticks: bool = True,
    ticks_top: bool = True,
    ticks_right: bool = True,
    grid: bool = False,
    grid_which: Literal["major", "minor", "both"] = "major",
    hlines: Sequence[float] | None = None,
    vlines: Sequence[float] | None = None,
    reference_line_kwargs: Mapping[str, Any] | None = None,
    series_kwargs: Sequence[Mapping[str, Any] | None] | None = None,
    rc_overrides: Mapping[str, Any] | None = None,
    ax: Axes | None = None,
    save: str | Path | None = None,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = APS_PREFERRED_RASTER_DPI,
    transparent: bool = False,
    savefig_kwargs: Mapping[str, Any] | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot any number of x/y datasets using APS-oriented defaults.

    Parameters
    ----------
    arrays
        Flat sequence ``[x1, y1, x2, y2, ...]``. Each x and y must be a
        nonempty one-dimensional array of equal length.
    plot_types
        One value per dataset, or one value broadcast to every dataset.
        Accepted values are ``"scatter"``/``"s"``, ``"plot"``/``"p"``, and
        ``"both"``/``"b"`` (line plus markers).
    labels, colors, markers, linestyles, linewidths, markersizes,
    markerfacecolors, markeredgecolors, alphas
        Per-dataset styling. A scalar is broadcast. A sequence must have one
        item per dataset. Defaults cycle through accessible colors and through
        distinct markers/line styles. Marker sizes and line widths are in
        points. APS guidance calls for data points at least 1 mm across and
        curve widths of at least 0.5 pt at final publication size.
    xerr, yerr
        ``None`` or one scalar/array per dataset. For one dataset, a NumPy
        array can be passed directly. Asymmetric errors may have shape (2, N),
        as accepted by ``Axes.errorbar``.
    capsize
        Error-bar cap size in points.
    xlabel, ylabel, xunit, yunit
        Axis quantities and optional units. Supplying ``xlabel="Time"`` and
        ``xunit="s"`` produces ``"Time (s)"``.
    xscale, yscale
        ``"linear"``, ``"log"``, ``"symlog"``, or ``"logit"``.
    xlim, ylim, xticks, yticks
        Optional explicit limits and major-tick positions.
    x_tick_decimals, y_tick_decimals
        Fix the number of decimal places on a linear axis.
    x_sci_powerlimits, y_sci_powerlimits
        Thresholds passed to ``ScalarFormatter.set_powerlimits``. The default
        uses scientific notation outside 10^-3 through 10^3. Set to ``None``
        to retain Matplotlib's formatter. Ignored when fixed decimals are set.
    title
        Optional axes title. Figure captions normally carry this information
        in journal articles, so ``None`` is the publication-friendly default.
    panel_label
        Optional panel tag such as ``"(a)"``, placed at the upper-left inside
        the axes.
    column
        ``"single"`` (8.5 cm), ``"onehalf"``, ``"double"`` (7 in), or an
        explicit figure width in inches.
    height, aspect
        Figure height in inches, or width times ``aspect`` when omitted.
    legend, legend_loc, legend_ncol, legend_kwargs
        Legend controls. A legend is created only when at least one label is
        not ``None``. ``legend_kwargs`` is forwarded to ``Axes.legend``.
    minor_ticks, ticks_top, ticks_right
        Tick controls. Inward ticks on all four sides are the package defaults.
    grid, grid_which
        Optional subtle grid. Grids are off by default to keep data prominent.
    hlines, vlines, reference_line_kwargs
        Optional reference-line locations and common ``axhline``/``axvline``
        styling. Defaults are thin, gray, and dashed.
    series_kwargs
        Optional list of dictionaries, one per dataset, applied last to the
        corresponding Matplotlib call. Use this escape hatch for options such
        as ``zorder``, ``rasterized``, ``dash_capstyle``, or custom error-bar
        styling. Explicit values here override the named style parameters.
    rc_overrides
        Temporary Matplotlib rcParams overrides. Global state is not changed.
    ax
        Existing axes to draw into. When supplied, ``column``, ``height``, and
        ``aspect`` do not resize its figure.
    save
        Output path. If it has an extension, save that single file. If it has
        no extension, save one file for every item in ``formats``. Vector
        formats such as PDF/SVG/EPS preserve vector artwork; raster formats
        use ``dpi``.
    formats
        Output formats used when ``save`` has no extension. Defaults to PDF and
        PNG.
    dpi, transparent, savefig_kwargs
        Save controls forwarded to ``Figure.savefig``. The package defaults to
        600 dpi for raster output.
    show
        Call ``plt.show()`` after plotting. Keep false when additional edits
        are planned; set true for immediate display in notebook cells.

    Returns
    -------
    fig, ax
        The Matplotlib figure and axes. The output remains fully editable, e.g.
        ``ax.annotate(...)`` followed by ``fig.savefig(...)``.

    Examples
    --------
    Two datasets, scatter for the measurements and a line for the model::

        fig, ax = aps_plot(
            arrays=[x_data, y_data, x_fit, y_fit],
            plot_types=["s", "p"],
            labels=["Experiment", "Model"],
            colors=["#0072B2", "#D55E00"],
            xlabel=r"Mean photon number $\\mu$",
            ylabel="Signal-to-noise ratio",
            xscale="log",
            yscale="log",
            xlim=(1e-4, 1e1),
            save="figure_2",       # writes figure_2.pdf and figure_2.png
        )

    Error bars and a double-column figure::

        fig, ax = aps_plot(
            [x1, y1, x2, y2],
            plot_types=["scatter", "both"],
            yerr=[dy1, dy2],
            labels=[r"$\\chi=0.2$", r"$\\chi=0.8$"],
            xlabel="Photon number", xunit="counts per mode",
            ylabel="Fisher information", yunit=r"rad$^{-2}$",
            column="double",
            panel_label="(a)",
        )

    Notes
    -----
    Journal compliance depends on the complete figure and caption. Check
    readability at final manuscript size, define every symbol/curve, and avoid
    relying on color alone. Vector PDF is generally preferable for line art;
    use high-resolution raster output when rasterization is appropriate.
    """
    datasets = _datasets_from_flat_arrays(arrays)
    n = len(datasets)
    types = [
        _normalize_plot_type(v)
        for v in _per_series(plot_types, n, "plot_types", default="scatter")
    ]
    labels_i = _per_series(labels, n, "labels", default=None)
    colors_i = _per_series(colors, n, "colors", default=APS_COLORS)
    markers_i = _per_series(markers, n, "markers", default=APS_MARKERS)
    linestyles_i = _per_series(
        linestyles, n, "linestyles", default=APS_LINESTYLES
    )
    linewidths_i = _per_series(linewidths, n, "linewidths", default=1.25)
    markersizes_i = _per_series(markersizes, n, "markersizes", default=4.0)
    facecolors_i = _per_series(
        markerfacecolors, n, "markerfacecolors", default=None
    )
    edgecolors_i = _per_series(
        markeredgecolors, n, "markeredgecolors", default=None
    )
    alphas_i = _per_series(alphas, n, "alphas", default=1.0)
    xerr_i = _errors_per_series(xerr, n, "xerr")
    yerr_i = _errors_per_series(yerr, n, "yerr")
    series_kwargs_i = _per_series(series_kwargs, n, "series_kwargs", default=None)

    if any(float(w) < APS_MIN_LINEWIDTH_PT for w in linewidths_i):
        raise ValueError(
            f"APS guidance calls for curve linewidths of at least "
            f"{APS_MIN_LINEWIDTH_PT:g} pt at final publication size."
        )
    if any(float(s) < APS_MIN_MARKER_DIAMETER_PT for s in markersizes_i):
        raise ValueError(
            "APS guidance calls for data-point diameters of at least 1 mm "
            f"({APS_MIN_MARKER_DIAMETER_PT:.2f} pt) at final publication size."
        )
    if x_tick_decimals is not None and x_tick_decimals < 0:
        raise ValueError("x_tick_decimals must be nonnegative.")
    if y_tick_decimals is not None and y_tick_decimals < 0:
        raise ValueError("y_tick_decimals must be nonnegative.")

    _validate_log_data(datasets, xscale, yscale)

    rc = dict(APS_RCPARAMS)
    if rc_overrides:
        rc.update(rc_overrides)

    with mpl.rc_context(rc):
        if ax is None:
            if isinstance(column, str):
                if column not in APS_WIDTHS_IN:
                    raise ValueError(f"column must be one of {tuple(APS_WIDTHS_IN)}.")
                width = APS_WIDTHS_IN[column]
            else:
                width = float(column)
                if width <= 0:
                    raise ValueError("An explicit figure width must be positive.")
            fig_height = float(height) if height is not None else width * float(aspect)
            if fig_height <= 0:
                raise ValueError("height/aspect must produce a positive figure height.")
            fig, ax = plt.subplots(
                figsize=(width, fig_height), constrained_layout=True
            )
        else:
            fig = ax.figure

        for i, ((x, y), kind) in enumerate(zip(datasets, types)):
            color = colors_i[i]
            marker = markers_i[i]
            linestyle = linestyles_i[i]
            linewidth = float(linewidths_i[i])
            markersize = float(markersizes_i[i])

            common: dict[str, Any] = {
                "label": labels_i[i] if labels_i[i] is not None else "_nolegend_",
                "color": color,
                "alpha": alphas_i[i],
                "zorder": 3 if kind != "plot" else 2,
            }
            custom = dict(series_kwargs_i[i] or {})
            has_error = xerr_i[i] is not None or yerr_i[i] is not None

            if has_error:
                error_kw: dict[str, Any] = {
                    **common,
                    "xerr": xerr_i[i],
                    "yerr": yerr_i[i],
                    "capsize": capsize,
                    "elinewidth": max(APS_MIN_LINEWIDTH_PT, 0.8 * linewidth),
                    "capthick": max(APS_MIN_LINEWIDTH_PT, 0.8 * linewidth),
                    "linewidth": linewidth,
                    "markersize": markersize,
                    "markeredgewidth": max(
                        APS_MIN_LINEWIDTH_PT, 0.6 * linewidth
                    ),
                }
                if kind == "scatter":
                    error_kw.update(fmt=marker, linestyle="none")
                elif kind == "plot":
                    error_kw.update(fmt="", linestyle=linestyle)
                else:
                    error_kw.update(fmt=marker, linestyle=linestyle)
                if facecolors_i[i] is not None:
                    error_kw["markerfacecolor"] = facecolors_i[i]
                if edgecolors_i[i] is not None:
                    error_kw["markeredgecolor"] = edgecolors_i[i]
                error_kw.update(custom)
                ax.errorbar(x, y, **error_kw)

            elif kind == "scatter":
                scatter_kw: dict[str, Any] = {
                    **common,
                    "marker": marker,
                    "s": markersize**2,  # scatter uses area in pt^2
                    "linewidths": max(
                        APS_MIN_LINEWIDTH_PT, 0.6 * linewidth
                    ),
                }
                if facecolors_i[i] is not None:
                    scatter_kw["facecolors"] = facecolors_i[i]
                if edgecolors_i[i] is not None:
                    scatter_kw["edgecolors"] = edgecolors_i[i]
                scatter_kw.update(custom)
                ax.scatter(x, y, **scatter_kw)

            else:
                plot_kw: dict[str, Any] = {
                    **common,
                    "linestyle": linestyle,
                    "linewidth": linewidth,
                }
                if kind == "both":
                    plot_kw.update(
                        marker=marker,
                        markersize=markersize,
                        markeredgewidth=max(
                            APS_MIN_LINEWIDTH_PT, 0.6 * linewidth
                        ),
                    )
                    if facecolors_i[i] is not None:
                        plot_kw["markerfacecolor"] = facecolors_i[i]
                    if edgecolors_i[i] is not None:
                        plot_kw["markeredgecolor"] = edgecolors_i[i]
                plot_kw.update(custom)
                ax.plot(x, y, **plot_kw)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(_axis_label(xlabel, xunit))
        ax.set_ylabel(_axis_label(ylabel, yunit))

        if title:
            ax.set_title(title, pad=5)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

        if minor_ticks:
            ax.minorticks_on()
        else:
            ax.minorticks_off()
        ax.tick_params(which="both", top=ticks_top, right=ticks_right)

        if xscale == "linear":
            if x_tick_decimals is not None:
                ax.xaxis.set_major_formatter(
                    FormatStrFormatter(f"%.{x_tick_decimals}f")
                )
            else:
                _apply_scientific_formatter(ax, "x", x_sci_powerlimits)

        if yscale == "linear":
            if y_tick_decimals is not None:
                ax.yaxis.set_major_formatter(
                    FormatStrFormatter(f"%.{y_tick_decimals}f")
                )
            else:
                _apply_scientific_formatter(ax, "y", y_sci_powerlimits)

        if grid:
            ax.grid(
                True,
                which=grid_which,
                color="0.85",
                linewidth=0.5,
                linestyle="-",
                zorder=0,
            )

        ref_kw = {
            "color": "0.35",
            "linewidth": 0.75,
            "linestyle": "--",
            "zorder": 1,
        }
        if reference_line_kwargs:
            ref_kw.update(reference_line_kwargs)
        for yref in () if hlines is None else hlines:
            ax.axhline(yref, **ref_kw)
        for xref in () if vlines is None else vlines:
            ax.axvline(xref, **ref_kw)

        if panel_label:
            ax.text(
                0.025,
                0.975,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.0,
                fontweight="bold",
            )

        if legend and any(label is not None for label in labels_i):
            legend_options = {"loc": legend_loc, "ncol": legend_ncol}
            legend_options.update(legend_kwargs or {})
            ax.legend(**legend_options)

        if save is not None:
            _save_figure(
                fig,
                save=save,
                formats=formats,
                dpi=dpi,
                transparent=transparent,
                savefig_kwargs=savefig_kwargs,
            )

        if show:
            plt.show()

    return fig, ax


if __name__ == "__main__":
    # Minimal runnable example. Replace these arrays and labels with your data.
    rng = np.random.default_rng(7)
    x_data = np.logspace(-3, 1, 24)
    y_model = x_data / (1.0 + x_data)
    y_data = y_model * np.exp(rng.normal(0.0, 0.10, x_data.size))

    aps_plot(
        arrays=[x_data, y_data, x_data, y_model],
        plot_types=["s", "p"],
        labels=["Experiment", "Model"],
        xlabel=r"Mean photon number $\mu$",
        ylabel="Normalized response",
        xscale="log",
        ylim=(0.0, 1.05),
        save="aps_example",
        show=True,
    )
