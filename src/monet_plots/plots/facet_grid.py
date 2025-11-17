# src/monet_plots/plots/facet_grid.py
import xarray as xr

class FacetGridPlot:
    """Creates a facet grid plot.

    This class creates a facet grid plot using xarray's FacetGrid.
    """
    def __init__(self, data, **kwargs):
        """Initializes the facet grid.

        Args:
            data (xarray.DataArray): The data to plot.
            **kwargs: Additional keyword arguments to pass to `FacetGrid`.
        """
        self.grid = xr.plot.FacetGrid(data, **kwargs)

    def map_dataframe(self, plot_func, *args, **kwargs):
        """Maps a plotting function to the facet grid.

        Args:
            plot_func (function): The plotting function to map.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        self.grid.map_dataframe(plot_func, *args, **kwargs)

    def set_titles(self, *args, **kwargs):
        """Sets the titles of the facet grid.

        Args:
            *args: Positional arguments to pass to `set_titles`.
            **kwargs: Keyword arguments to pass to `set_titles`.
        """
        self.grid.set_titles(*args, **kwargs)

    def save(self, filename, **kwargs):
        """Saves the plot to a file.

        Args:
            filename (str): The name of the file to save the plot to.
            **kwargs: Additional keyword arguments to pass to `savefig`.
        """
        self.grid.savefig(filename, **kwargs)

    def close(self):
        """Closes the plot."""
        import matplotlib.pyplot as plt
        plt.close(self.grid.fig)
