# src/monet_plots/plots/kde.py
from .base import BasePlot
import seaborn as sns

class KDEPlot(BasePlot):
    """Creates a kernel density estimate plot.

    This class creates a kernel density estimate (KDE) plot.
    """
    def __init__(self, **kwargs):
        """Initializes the plot."""
        super().__init__(**kwargs)
        sns.despine()

    def plot(self, df, title=None, label=None, **kwargs):
        """Plots the KDE data.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            title (str, optional): The title of the plot. Defaults to None.
            label (str, optional): The label for the legend. Defaults to None.
            **kwargs: Additional keyword arguments to pass to `kdeplot`.
        """
        sns.kdeplot(df, ax=self.ax, label=label, **kwargs)
        self.ax.set_title(title)
