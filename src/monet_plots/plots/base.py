# src/monet_plots/plots/base.py
import matplotlib.pyplot as plt
from ..style import wiley_style

class BasePlot:
    """Base class for all plots.

    Handles figure and axis creation, applies the Wiley style,
    and provides a consistent method for saving figures.
    """
    def __init__(self, fig=None, ax=None, **kwargs):
        """Initializes the plot with a Wiley-compliant style."""
        plt.style.use(wiley_style)
        if fig is not None and ax is not None:
            # Use provided figure and axes
            self.fig = fig
            self.ax = ax
        else:
            # Create new figure and axes
            self.fig, self.ax = plt.subplots(**kwargs)

    def save(self, filename, **kwargs):
        """Saves the plot to a file.

        Args:
            filename (str): The name of the file to save the plot to.
            **kwargs: Additional keyword arguments to pass to `savefig`.
        """
        self.fig.savefig(filename, **kwargs)

    def close(self):
        """Closes the plot."""
        plt.close(self.fig)
