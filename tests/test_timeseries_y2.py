import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.timeseries import TimeSeriesPlot


def test_timeseries_plot_y2():
    """Test TimeSeriesPlot with secondary y-axis."""
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    obs = np.random.rand(10)
    alt = np.linspace(0, 1000, 10)

    df = pd.DataFrame({"time": dates, "obs": obs, "altitude": alt})

    plot = TimeSeriesPlot(df, x="time", y="obs")
    ax = plot.plot(y2="altitude", y2_label="Alt (m)", y2_kwargs={"color": "red"})

    assert ax is not None
    # Check for two y-axes (ax and ax.twinx())
    assert len(plot.fig.axes) == 2
    assert plot.fig.axes[1].get_ylabel() == "Alt (m)"
    plt.close(plot.fig)


def test_timeseries_y2_uses_matplotlib_for_dataframe(monkeypatch):
    from pandas.plotting import PlotAccessor

    def reject_pandas_plot(*args, **kwargs):
        raise AssertionError("Secondary axis called pandas plotting")

    monkeypatch.setattr(PlotAccessor, "__call__", reject_pandas_plot)
    dates = pd.date_range("2023-01-01", periods=2, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {"time": dates.repeat(2), "obs": [1, 2, 3, 4], "altitude": [10, 20, 30, 40]}
    )
    plot = TimeSeriesPlot(frame)
    try:
        ax = plot.plot(y2="altitude")
        (line,) = plot.fig.axes[1].get_lines()
        np.testing.assert_allclose(line.get_ydata(), [15, 35])
        assert list(line.get_xdata()) == list(dates)
        assert "altitude" in [label.get_text() for label in ax.get_legend().texts]
    finally:
        plot.close()
