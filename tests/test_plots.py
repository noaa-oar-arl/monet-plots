# tests/test_plots.py
import monet_plots
import pandas as pd
import numpy as np
import xarray as xr
import pytest

def test_spatial_plot():
    """Tests the SpatialPlot class."""
    plot = monet_plots.SpatialPlot()
    assert plot.ax is not None
    plot.close()

def test_timeseries_plot():
    """Tests the TimeSeriesPlot class."""
    data = {'time': pd.to_datetime(['2025-01-01', '2025-01-02']), 'obs': [1, 2]}
    df = pd.DataFrame(data)
    plot = monet_plots.TimeSeriesPlot()
    plot.plot(df)
    assert plot.ax is not None
    plot.close()

def test_taylor_diagram_plot():
    """Tests the TaylorDiagramPlot class."""
    obs = np.random.rand(10)
    model = obs + np.random.rand(10) * 0.1
    df = pd.DataFrame({'obs': obs, 'model': model})
    plot = monet_plots.TaylorDiagramPlot(obs.std())
    plot.add_sample(df)
    assert plot.dia is not None
    plot.close()

def test_kde_plot():
    """Tests the KDEPlot class."""
    data = np.random.randn(100)
    plot = monet_plots.KDEPlot()
    plot.plot(data)
    assert plot.ax is not None
    plot.close()

def test_scatter_plot():
    """Tests the ScatterPlot class."""
    data = {'x': np.arange(10), 'y': np.arange(10)}
    df = pd.DataFrame(data)
    plot = monet_plots.ScatterPlot()
    plot.plot(df, 'x', 'y')
    assert plot.ax is not None
    plot.close()

def test_facet_grid_plot():
    """Tests the FacetGridPlot class."""
    data = xr.DataArray(np.random.randn(2, 3, 4),
                        dims=('x', 'y', 'z'),
                        coords={'x': [1, 2], 'y': [1, 2, 3], 'z': [1, 2, 3, 4]})
    plot = monet_plots.FacetGridPlot(data, col='z')
    assert plot.grid is not None
    plot.close()
