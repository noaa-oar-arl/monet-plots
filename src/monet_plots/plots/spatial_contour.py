# src/monet_plots/plots/spatial_contour.py
from .base import BasePlot
from ..colorbars import colorbar_index
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class SpatialContourPlot(BasePlot):
    """Creates a spatial contour plot.

    This class creates a spatial contour plot on a map.
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

    def plot(self, modelvar, gridobj, date, discrete=True, ncolors=None, dtype='int', **kwargs):
        """Plots the spatial contour data.

        Args:
            modelvar (numpy.ndarray): The 2D model variable to plot.
            gridobj (object): The grid object containing the latitude and longitude data.
            date (datetime): The date to plot.
            discrete (bool, optional): Whether to use a discrete colorbar. Defaults to True.
            ncolors (int, optional): The number of colors to use for a discrete colorbar. Defaults to None.
            dtype (str, optional): The data type for the colorbar ticks. Defaults to 'int'.
            **kwargs: Additional keyword arguments to pass to `contourf`.
        """
        lat = gridobj.variables['LAT'][0, 0, :, :].squeeze()
        lon = gridobj.variables['LON'][0, 0, :, :].squeeze()

        if 'transform' not in kwargs:
            kwargs['transform'] = ccrs.PlateCarree()

        self.ax.contourf(lon, lat, modelvar, **kwargs)

        cmap = kwargs.get('cmap', 'viridis')
        levels = kwargs.get('levels')

        if discrete and levels:
            c, cmap = colorbar_index(ncolors, cmap, minval=levels[0], maxval=levels[-1], dtype=dtype)
            self.fig.colorbar(c, ticks=c.get_ticks())
        else:
            self.fig.colorbar()

        self.ax.set_title(date.strftime('%B %d %Y %H'))
