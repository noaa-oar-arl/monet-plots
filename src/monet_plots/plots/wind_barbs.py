# src/monet_plots/plots/wind_barbs.py
from .base import BasePlot
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class WindBarbsPlot(BasePlot):
    """Creates a wind barbs plot.

    This class creates a wind barbs plot on a map.
    """
    def __init__(self, projection=ccrs.PlateCarree(), **kwargs):
        """Initializes the plot with a cartopy projection.

        Args:
            projection (cartopy.crs): The cartopy projection to use.
            **kwargs: Additional keyword arguments to pass to `subplots`.
        """
        super().__init__(subplot_kw={'projection': projection}, **kwargs)
        self.ax.coastlines()
        self.ax.add_feature(cfeature.BORDERS, linestyle=':')
        self.ax.add_feature(cfeature.STATES, linestyle=':')

    def plot(self, ws, wdir, gridobj, **kwargs):
        """Plots the wind barbs data.

        Args:
            ws (numpy.ndarray): The wind speed data.
            wdir (numpy.ndarray): The wind direction data.
            gridobj (object): The grid object containing the latitude and longitude data.
            **kwargs: Additional keyword arguments to pass to `barbs`.
        """
        from .. import tools
        lat = gridobj.variables['LAT'][0, 0, :, :].squeeze()
        lon = gridobj.variables['LON'][0, 0, :, :].squeeze()

        if 'transform' not in kwargs:
            kwargs['transform'] = ccrs.PlateCarree()

        u, v = tools.wsdir2uv(ws, wdir)
        self.ax.barbs(lon[::15, ::15], lat[::15, ::15], u[::15, ::15], v[::15, ::15], **kwargs)
