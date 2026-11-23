import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from monet_plots.plots.spatial_overlay import SpatialOverlayPlot


def test_spatial_overlay_plot():
    """Test SpatialOverlayPlot with model grid and observation points."""
    # Create model data
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    data = np.random.rand(10, 10)
    model_da = xr.DataArray(
        data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="model"
    )

    # Create observation data
    obs_lon = np.random.uniform(-120, -70, 20)
    obs_lat = np.random.uniform(25, 50, 20)
    obs_val = np.random.rand(20)
    obs_df = pd.DataFrame({"lat": obs_lat, "lon": obs_lon, "obs": obs_val})

    plot = SpatialOverlayPlot(model_da, obs_df, model_var="model", obs_var="obs")
    ax = plot.plot(kind="contourf", coastlines=True)

    assert ax is not None
    assert hasattr(ax, "projection")
    # Check for contourf (collection) and scatter (collection)
    assert len(ax.collections) >= 2
    plt.close(plot.fig)


def test_overlay_marker_options():
    model = xr.DataArray(
        np.ones((2, 2)),
        dims=("lat", "lon"),
        coords={"lat": [30, 40], "lon": [-100, -90]},
    )
    obs = pd.DataFrame({"latitude": [35], "longitude": [-95], "value": [1]})
    plot = SpatialOverlayPlot(model, obs, obs_var="value", coastlines=False)
    try:
        plot.plot(obs_kws={"s": 100, "edgecolors": "white"})
        np.testing.assert_allclose(plot.observation_artist.get_sizes(), [100])
    finally:
        plot.close()
