# src/monet_plots/plots/scatter.py
from .base import BasePlot
import seaborn as sns

class ScatterPlot(BasePlot):
    """Creates a scatter plot.

    This class creates a scatter plot with a regression line.
    """
    def __init__(self, **kwargs):
        """Initializes the plot."""
        super().__init__(**kwargs)

    def plot(self, df, x, y, title=None, label=None, **kwargs):
        """Plots the scatter data.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            x (str): The column for the x-axis.
            y (str): The column for the y-axis.
            title (str, optional): The title of the plot. Defaults to None.
            label (str, optional): The label for the legend. Defaults to None.
            **kwargs: Additional keyword arguments to pass to `regplot`.
        """
        sns.regplot(data=df, x=x, y=y, ax=self.ax, label=label, **kwargs)
        self.ax.set_title(title)
