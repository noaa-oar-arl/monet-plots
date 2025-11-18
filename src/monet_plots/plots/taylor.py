# src/monet_plots/plots/taylor.py
from .base import BasePlot
from .. import taylordiagram as td
from numpy import corrcoef
import seaborn as sns

class TaylorDiagramPlot(BasePlot):
    """Creates a Taylor diagram.

    This class creates a Taylor diagram to compare a model to observations.
    """
    def __init__(self, obsstd, scale=1.5, label='OBS', **kwargs):
        """Initializes the Taylor diagram.

        Args:
            obsstd (float): The standard deviation of the observations.
            scale (float, optional): The scale of the diagram. Defaults to 1.5.
            label (str, optional): The label for the observations. Defaults to 'OBS'.
            **kwargs: Additional keyword arguments to pass to `subplots`.
        """
        super().__init__(**kwargs)
        self.dia = td.TaylorDiagram(obsstd, scale=scale, fig=self.fig, rect=111, label=label)
        self.fig.sca(self.ax) # Set the current axes to the one created by BasePlot

    def add_sample(self, df, col1='obs', col2='model', marker='o', label='MODEL'):
        """Adds a model sample to the diagram.

        Args:
            df (pandas.DataFrame): The DataFrame containing the model and observation data.
            col1 (str, optional): The column for the observations. Defaults to 'obs'.
            col2 (str, optional): The column for the model. Defaults to 'model'.
            marker (str, optional): The marker to use for the model. Defaults to 'o'.
            label (str, optional): The label for the model. Defaults to 'MODEL'.
        """
        df = df.drop_duplicates().dropna(subset=[col1, col2])
        cc = corrcoef(df[col1].values, df[col2].values)[0, 1]
        self.dia.add_sample(df[col2].std(), cc, marker=marker, zorder=9, ls='none', label=label)

    def add_contours(self, **kwargs):
        """Adds contours to the diagram.

        Args:
            **kwargs: Additional keyword arguments to pass to `add_contours`.
        """
        return self.dia.add_contours(**kwargs)

    def finish_plot(self):
        """Finishes the plot by adding a legend and tight layout."""
        self.ax.legend(fontsize='small', loc='best')
        self.fig.tight_layout()
