# aps-plotter

A small Matplotlib utility package for making publication-quality figures with
defaults chosen around the general figure guidance used across APS journals.

The package is intended for journals in the Physical Review family and related
APS publications. It does **not** guarantee journal compliance; always inspect
the final figure at the size at which it will appear in the manuscript.

## Features

- Any number of x/y datasets supplied as `[x1, y1, x2, y2, ...]`
- Scatter, line, or line-plus-marker rendering per dataset
- Linear, log, symlog, and logit axes
- Symmetric or asymmetric error bars
- APS-oriented single-, 1.5-, and double-column figure widths
- Accessible default color palette plus redundant markers/linestyles
- Temporary `rcParams` styling without changing global Matplotlib state
- Panel labels, legends, reference lines, grids, and custom per-series kwargs
- Vector export (PDF/SVG/EPS) and high-resolution raster export
- Fully editable Matplotlib `Figure` and `Axes` returned to the caller

## Installation for collaborators

The recommended collaboration workflow is to clone the repository once and
install it in editable mode inside each Conda environment that needs it. Open a
terminal window and navigate to the directory you ant to install the project
in. Clone the repo by using

```bash
cd path/to/install/location
```
```bash
git clone https://github.com/BryanFRezende/aps-plotter.git
```
Then pip install the module. If you use conda, here's an example of how to
install in a particular environment.

```bash
cd aps-plotter
conda activate YOUR_ENVIRONMENT
python -m pip install -e .
```

Then import it from any notebook using that Conda environment:

```python
from aps_plotter import aps_plot
```

Because the installation is editable, collaborators only need to update their
clone when you publish changes:

```bash
git pull
```

After pulling an update, restart any already-running Jupyter kernel so Python
reloads the package.

## Basic usage

```python
import numpy as np
from aps_plotter import aps_plot

x = np.logspace(-3, 1, 30)
y = x / (1 + x)

fig, ax = aps_plot(
    arrays=[x, y],
    plot_types="p",
    xlabel=r"Mean photon number $\mu$",
    ylabel="Normalized response",
    xscale="log",
    show=True,
)
```

## Multiple datasets

```python
fig, ax = aps_plot(
    arrays=[x_data, y_data, x_fit, y_fit],
    plot_types=["s", "p"],
    labels=["Experiment", "Model"],
    xlabel=r"Mean photon number $\mu$",
    ylabel="Signal-to-noise ratio",
    xscale="log",
    yscale="log",
    show=True,
)
```

## Error bars

```python
fig, ax = aps_plot(
    arrays=[x, y],
    plot_types="scatter",
    yerr=dy,
    xlabel="Time",
    xunit="ms",
    ylabel="Population",
    show=True,
)
```

The axis label will be rendered as `Time (ms)`.

## Figure width

```python
aps_plot(..., column="single")
aps_plot(..., column="onehalf")
aps_plot(..., column="double")
```

You can also give an explicit width in inches:

```python
aps_plot(..., column=4.25)
```

## Saving

If `save` has no extension, one file is written for every format in `formats`:

```python
fig, ax = aps_plot(
    arrays=[x, y],
    save="figures/figure_1",
)
```

By default this writes:

```text
figures/figure_1.pdf
figures/figure_1.png
```

For a single file:

```python
aps_plot(..., save="figure_1.svg")
```

## Editing after plotting

`aps_plot` returns the ordinary Matplotlib figure and axes, so you can continue
editing them:

```python
fig, ax = aps_plot(
    arrays=[x, y],
    show=False,
)

ax.axhline(1.0, linestyle="--")
ax.annotate("crossover", xy=(1, 1))

fig.savefig("figure.pdf")
```

In a Jupyter notebook, either set `show=True` for each call or call
`matplotlib.pyplot.show()` once after creating multiple figures.

## Public style constants

```python
from aps_plotter import (
    APS_COLORS,
    APS_MARKERS,
    APS_LINESTYLES,
    APS_WIDTHS_IN,
    APS_RCPARAMS,
)
```

These can be reused when constructing more specialized Matplotlib figures.

## APS-oriented defaults

The package encodes general APS figure conventions such as:

- 8.5 cm (3 3/8 in) single-column figure width
- data markers at least 1 mm across
- curve linewidths at least 0.5 pt
- axis units in parentheses
- accessible use of color
- 600 dpi default for raster output

These are general-purpose defaults. Journal- or article-specific instructions
and the appearance of the complete figure/caption still take precedence.

## Development

Install the optional test dependency:

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

## Versioning

Use semantic versioning for releases:

- `0.1.1` for bug fixes
- `0.2.0` for backward-compatible feature additions
- `1.0.0` when the public API is considered stable

A typical update workflow is:

```bash
git add .
git commit -m "Describe the change"
git push
```

Collaborators then run:

```bash
git pull
```
