# src/monet_plots/plots/xarray_spatial.py
from .base import BasePlot

class XarraySpatialPlot(BasePlot):
    """Creates a spatial plot from an xarray DataArray.

    This class creates a spatial plot from an xarray DataArray.
    """
    def __init__(self, **kwargs):
        """Initializes the plot."""
        super().__init__(**kwargs)

    def plot(self, modelvar, **kwargs):
        """Plots the xarray data.

        Args:
            modelvar (xarray.DataArray): The DataArray to plot.
            **kwargs: Additional keyword arguments to pass to the plot function.
        """
        modelvar.plot(ax=self.ax, **kwargs)
        self.fig.tight_layout()
