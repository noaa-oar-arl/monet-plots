# src/monet_plots/plots/spatial.py
from .base import BasePlot
from ..colorbars import colorbar_index
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class SpatialPlot(BasePlot):
    """Creates a spatial plot using cartopy.

    This class creates a spatial plot of a 2D model variable on a map.
    It can handle both discrete and continuous colorbars.
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

    def plot(self, modelvar, plotargs={}, ncolors=15, discrete=False, **kwargs):
        """Plots the spatial data.

        Args:
            modelvar (numpy.ndarray): The 2D model variable to plot.
            plotargs (dict, optional): Keyword arguments to pass to `imshow`. Defaults to {}.
            ncolors (int, optional): The number of colors to use for a discrete colorbar. Defaults to 15.
            discrete (bool, optional): Whether to use a discrete colorbar. Defaults to False.
            **kwargs: Additional keyword arguments to pass to `imshow`.
        """
        if 'cmap' not in plotargs:
            plotargs['cmap'] = 'viridis'

        if 'transform' not in kwargs:
            kwargs['transform'] = ccrs.PlateCarree()

        if discrete:
            vmin = plotargs.get('vmin', modelvar.min())
            vmax = plotargs.get('vmax', modelvar.max())
            c, cmap = colorbar_index(ncolors, plotargs['cmap'], minval=vmin, maxval=vmax)
            plotargs['cmap'] = cmap
            im = self.ax.imshow(modelvar, **plotargs, **kwargs)
            self.cbar = self.fig.colorbar(im, ticks=c.get_ticks())
        else:
            im = self.ax.imshow(modelvar, **plotargs, **kwargs)
            self.cbar = self.fig.colorbar(im)

        return self.ax
