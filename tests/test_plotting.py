import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aps_plotter import APS_WIDTHS_IN, aps_plot


def test_basic_plot_returns_figure_and_axes():
    x = np.linspace(0.0, 1.0, 5)
    y = x**2

    fig, ax = aps_plot([x, y], show=False)

    assert fig is ax.figure
    assert len(ax.collections) == 1
    plt.close(fig)


def test_single_column_width():
    x = np.linspace(0.0, 1.0, 5)

    fig, _ = aps_plot([x, x], column="single", show=False)

    assert fig.get_figwidth() == pytest.approx(APS_WIDTHS_IN["single"])
    plt.close(fig)


def test_log_axis_rejects_nonpositive_values():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="nonpositive"):
        aps_plot([x, y], xscale="log", show=False)
