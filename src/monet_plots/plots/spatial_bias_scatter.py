# src/monet_plots/plots/spatial_bias_scatter.py
from .base import BasePlot
from ..colorbars import colorbar_index
from numpy import around
from scipy.stats import scoreatpercentile as score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

class SpatialBiasScatterPlot(BasePlot):
    """Creates a spatial bias scatter plot.

    This class creates a spatial bias scatter plot on a map.
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
        self.ax.set_facecolor('white')

    def plot(self, df, date, vmin=None, vmax=None, ncolors=15, fact=1.5, cmap='RdBu_r', **kwargs):
        """Plots the spatial bias scatter data.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            date (datetime): The date to plot.
            vmin (float, optional): The minimum value for the colorbar. Defaults to None.
            vmax (float, optional): The maximum value for the colorbar. Defaults to None.
            ncolors (int, optional): The number of colors to use for the colorbar. Defaults to 15.
            fact (float, optional): The factor to scale the scatter points by. Defaults to 1.5.
            cmap (str, optional): The colormap to use. Defaults to 'RdBu_r'.
            **kwargs: Additional keyword arguments to pass to `scatter`.
        """
        diff = df.CMAQ - df.Obs
        top = around(score(diff.abs(), per=95))
        new = df[df.datetime == date]

        if 'transform' not in kwargs:
            kwargs['transform'] = ccrs.PlateCarree()

        c, cmap_obj = colorbar_index(ncolors, cmap, minval=top * -1, maxval=top)

        colors = new.CMAQ - new.Obs
        ss = (new.CMAQ - new.Obs).abs() / top * 100.0
        ss[ss > 300] = 300.0

        p = self.ax.scatter(new.longitude.values, new.latitude.values, c=colors, s=ss, vmin=-1.0 * top, vmax=top, cmap=cmap_obj, edgecolors='k', linewidths=0.25, alpha=0.7, **kwargs)

        cbar = self.fig.colorbar(p, orientation='horizontal', pad=0.05, aspect=50, extend='both')
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label('Bias')
