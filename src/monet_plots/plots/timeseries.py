# src/monet_plots/plots/timeseries.py
from .base import BasePlot
import seaborn as sns

class TimeSeriesPlot(BasePlot):
    """Creates a time series plot.

    This class creates a time series plot of a variable with a shaded
    region for the standard deviation.
    """
    def __init__(self, **kwargs):
        """Initializes the plot."""
        super().__init__(**kwargs)

    def plot(self, df, x='time', y='obs', plotargs={}, fillargs={'alpha': 0.2}, title='', ylabel=None, label=None):
        """Plots the time series data.

        Args:
            df (pandas.DataFrame): The DataFrame containing the time series data.
            x (str, optional): The column to use for the x-axis. Defaults to 'time'.
            y (str, optional): The column to use for the y-axis. Defaults to 'obs'.
            plotargs (dict, optional): Keyword arguments to pass to the plot function. Defaults to {}.
            fillargs (dict, optional): Keyword arguments to pass to the fill_between function. Defaults to {'alpha': 0.2}.
            title (str, optional): The title of the plot. Defaults to ''.
            ylabel (str, optional): The label for the y-axis. Defaults to None.
            label (str, optional): The label for the legend. Defaults to None.
        """
        # Only apply mean and std to numeric columns, keeping the grouping column
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if x not in numeric_cols:
            numeric_cols.append(x)
        
        m = df[numeric_cols].groupby(x).mean()
        e = df[numeric_cols].groupby(x).std()

        upper = m[y] + e[y]
        lower = m[y] - e[y]
        lower.loc[lower < 0] = 0

        if label:
            m = m.rename(columns={y: label})
            y = label

        m[y].plot(ax=self.ax, **plotargs)
        self.ax.fill_between(m.index, lower, upper, **fillargs)

        if ylabel:
            self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel('')
        self.ax.set_title(title)
        self.ax.legend()
        self.fig.tight_layout()
