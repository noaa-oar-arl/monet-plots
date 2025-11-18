# Time Series Plots

Time series plots visualize data that changes over time. MONET Plots provides specialized time series plotting capabilities with statistical analysis features, including mean, standard deviation, and confidence bands.

## Overview

The time series functionality in MONET Plots is designed for meteorological and climate data analysis, with built-in statistical analysis and publication-ready styling.

| Feature | Description |
|---------|-------------|
| Statistical bands | Automatic mean and standard deviation calculation |
| Date formatting | Smart date axis formatting |
| Multiple data series | Support for multiple time series comparison |
| Custom styling | Publication-ready formatting |

## TimeSeriesPlot Class

`TimeSeriesPlot` creates time series plots with statistical bands and professional styling.

### Class Signature

```python
class TimeSeriesPlot(BasePlot):
    """Creates a time series plot with statistical bands.
    
    This class creates a time series plot of a variable with a shaded
    region for the standard deviation.
    """
    
    def __init__(self, **kwargs):
        """Initialize the time series plot.
        
        Args:
            **kwargs: Additional keyword arguments to pass to `subplots`
        """
        pass
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `**kwargs` | `dict` | `{}` | Additional matplotlib figure parameters |

### Methods

#### `plot(df, x='time', y='obs', plotargs={}, fillargs={'alpha': 0.2}, title='', ylabel=None, label=None)`

Plot time series data with statistical bands.

```python
def plot(self, df, x='time', y='obs', plotargs={}, fillargs={'alpha': 0.2}, title='', ylabel=None, label=None):
    """Plot the time series data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing time series data
        x (str, optional): Column for x-axis. Defaults to 'time'
        y (str, optional): Column for y-axis. Defaults to 'obs'
        plotargs (dict, optional): Additional plot parameters. Defaults to {}
        fillargs (dict, optional): Fill between parameters. Defaults to {'alpha': 0.2}
        title (str, optional): Plot title. Defaults to ''
        ylabel (str, optional): Y-axis label. Defaults to None
        label (str, optional): Legend label. Defaults to None
    """
    pass
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pandas.DataFrame` | Required | DataFrame with time series data |
| `x` | `str` | `'time'` | Column name for x-axis (time) |
| `y` | `str` | `'obs'` | Column name for y-axis (values) |
| `plotargs` | `dict` | `{}` | Additional plotting parameters |
| `fillargs` | `dict` | `{'alpha': 0.2}` | Fill between parameters |
| `title` | `str` | `''` | Plot title |
| `ylabel` | `str` | `None` | Y-axis label |
| `label` | `str` | `None` | Legend label |

**Returns:**
- `matplotlib.axes.Axes`: The axes object with the plot

**Example:**
```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Create time series plot
plot = TimeSeriesPlot(figsize=(12, 6))

# Generate sample data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
data = pd.DataFrame({
    'time': dates,
    'temperature': 15 + 10 * np.sin(np.arange(365) * 0.1) + np.random.normal(0, 2, 365),
    'humidity': 60 + 20 * np.cos(np.arange(365) * 0.08) + np.random.normal(0, 5, 365)
})

# Plot temperature
plot.plot(
    data, 
    x='time', 
    y='temperature',
    title="Daily Temperature Variation",
    ylabel="Temperature (°C)",
    label="Temperature"
)

plot.save('timeseries_basic.png')
plot.close()
```

### Common Usage Patterns

#### Basic Time Series Plot

```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Create sample data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.normal(0, 1, 100)) + 50  # Random walk

df = pd.DataFrame({
    'date': dates,
    'value': values,
    'category': ['A'] * 50 + ['B'] * 50
})

# Create plot
plot = TimeSeriesPlot(figsize=(12, 6))

plot.plot(
    df,
    x='date',
    y='value',
    title="Basic Time Series Plot",
    ylabel="Value",
    plotargs={'linewidth': 2, 'color': 'blue'},
    fillargs={'alpha': 0.3, 'color': 'lightblue'}
)

plot.xlabel("Date")
plot.ylabel("Value")
plot.save('basic_timeseries.png')
```

#### Multiple Time Series

```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Create multiple time series
dates = pd.date_range('2023-01-01', periods=200, freq='D')

data = pd.DataFrame({
    'date': dates,
    'series1': np.cumsum(np.random.normal(0, 1, 200)) + 10,
    'series2': np.cumsum(np.random.normal(0, 1.5, 200)) + 20,
    'series3': np.cumsum(np.random.normal(0, 0.8, 200)) + 15
})

# Create plot
plot = TimeSeriesPlot(figsize=(14, 8))

# Plot multiple series with different styling
for series, color, label in [
    ('series1', 'blue', 'Model A'),
    ('series2', 'red', 'Model B'), 
    ('series3', 'green', 'Model C')
]:
    plot.plot(
        data,
        x='date',
        y=series,
        title="Multiple Time Series Comparison",
        ylabel="Value",
        label=label,
        plotargs={'linewidth': 2, 'color': color},
        fillargs={'alpha': 0.2, 'color': color}
    )

plot.legend()
plot.save('multiple_timeseries.png')
```

#### Statistical Analysis Plot

```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Create monthly data with seasonal pattern
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

# Generate realistic seasonal temperature data
day_of_year = dates.dayofyear
years = dates.year

# Temperature with seasonal cycle and trend
base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365.25)
trend = (years - 2020) * 0.1  # Warming trend
noise = np.random.normal(0, 2, len(dates))

temperature = base_temp + trend + noise

df = pd.DataFrame({
    'date': dates,
    'temperature': temperature,
    'month': dates.month,
    'year': dates.year
})

# Create plot
plot = TimeSeriesPlot(figsize=(15, 10))

# Plot full time series
plot.plot(
    df,
    x='date',
    y='temperature',
    title="Temperature Time Series with Statistical Analysis",
    ylabel="Temperature (°C)",
    plotargs={'linewidth': 1, 'color': 'darkblue', 'alpha': 0.7},
    fillargs={'alpha': 0.2, 'color': 'lightblue'}
)

# Add monthly averages
monthly_avg = df.groupby('month')['temperature'].mean()
monthly_std = df.groupby('month')['temperature'].std()

plot.ax.plot(
    [pd.Timestamp(f'2023-{month:02d}-01') for month in monthly_avg.index],
    monthly_avg,
    'ro-',
    markersize=8,
    linewidth=2,
    label='Monthly Average'
)

plot.legend()
plot.save('statistical_timeseries.png')
```

## Advanced Features

### Custom Statistical Bands

```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Create data with multiple statistical measures
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.random.normal(0, 1, 365)

# Create moving statistics
window_size = 30
df = pd.DataFrame({
    'date': dates,
    'value': values,
    'rolling_mean': values.rolling(window=window_size, min_periods=1).mean(),
    'rolling_std': values.rolling(window=window_size, min_periods=1).std(),
    'rolling_min': values.rolling(window=window_size, min_periods=1).min(),
    'rolling_max': values.rolling(window=window_size, min_periods=1).max()
})

# Create plot with custom statistical bands
plot = TimeSeriesPlot(figsize=(14, 8))

# Plot main series
plot.plot(
    df,
    x='date',
    y='value',
    title="Time Series with Custom Statistical Bands",
    ylabel="Value",
    plotargs={'linewidth': 1, 'color': 'gray', 'alpha': 0.6, 'label': 'Daily Values'}
)

# Add confidence intervals
plot.ax.fill_between(
    df['date'],
    df['rolling_mean'] - df['rolling_std'],
    df['rolling_mean'] + df['rolling_std'],
    alpha=0.3,
    color='blue',
    label='±1 Std Dev'
)

# Add range bands
plot.ax.fill_between(
    df['date'],
    df['rolling_min'],
    df['rolling_max'],
    alpha=0.1,
    color='red',
    label='Min-Max Range'
)

# Plot rolling mean
plot.ax.plot(df['date'], df['rolling_mean'], 'b-', linewidth=2, label='30-day Rolling Mean')

plot.legend()
plot.save('custom_bands_timeseries.png')
```

### Subplots for Multiple Variables

```python
import matplotlib.pyplot as plt
from monet_plots import TimeSeriesPlot

# Create figure with multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Create different time series for each subplot
dates = pd.date_range('2023-01-01', periods=365, freq='D')

data1 = pd.DataFrame({
    'date': dates,
    'value': 20 + 5 * np.sin(np.arange(365) * 0.1) + np.random.normal(0, 1, 365)
})

data2 = pd.DataFrame({
    'date': dates,
    'value': 50 + 10 * np.cos(np.arange(365) * 0.08) + np.random.normal(0, 2, 365)
})

data3 = pd.DataFrame({
    'date': dates,
    'value': 100 + 20 * np.sin(np.arange(365) * 0.05) + np.random.normal(0, 3, 365)
})

# Create plots for each subplot
plot1 = TimeSeriesPlot(figure=fig, subplot_kw=dict(ax=axes[0]))
plot2 = TimeSeriesPlot(figure=fig, subplot_kw=dict(ax=axes[1]))
plot3 = TimeSeriesPlot(figure=fig, subplot_kw=dict(ax=axes[2]))

# Plot each variable
plot1.plot(
    data1,
    x='date',
    y='value',
    title="Variable 1",
    ylabel="Value 1",
    plotargs={'color': 'blue'},
    fillargs={'color': 'lightblue'}
)

plot2.plot(
    data2,
    x='date',
    y='value',
    title="Variable 2",
    ylabel="Value 2",
    plotargs={'color': 'red'},
    fillargs={'color': 'lightcoral'}
)

plot3.plot(
    data3,
    x='date',
    y='value',
    title="Variable 3",
    ylabel="Value 3",
    plotargs={'color': 'green'},
    fillargs={'color': 'lightgreen'}
)

# Format x-axis for all subplots
for ax in axes:
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
save_figure(fig, 'multi_variable_timeseries.png')
```

### Interactive Time Series

```python
import matplotlib.pyplot as plt
from monet_plots import TimeSeriesPlot

# Enable interactive mode
plt.ion()

# Create interactive time series plot
plot = TimeSeriesPlot(figsize=(12, 8))

# Generate data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.normal(0, 1, 100)) + 50

df = pd.DataFrame({
    'date': dates,
    'value': values
})

# Initial plot
plot.plot(
    df,
    x='date',
    y='value',
    title="Interactive Time Series",
    ylabel="Value"
)

# Click interaction
def on_click(event):
    if event.inaxes == plot.ax:
        # Add vertical line at clicked position
        plot.ax.axvline(x=event.xdata, color='red', linestyle='--', alpha=0.7)
        plot.fig.canvas.draw()

plot.fig.canvas.mpl_connect('button_press_event', on_click)

# Show plot
plt.show()
```

## Data Requirements

### Input Data Format

Time series plots require pandas DataFrames with specific column names:

```python
import pandas as pd
import numpy as np

# Basic format
df = pd.DataFrame({
    'time': pd.date_range('2023-01-01', periods=100, freq='D'),
    'observed': np.random.normal(0, 1, 100),
    'modeled': np.random.normal(0.1, 1.1, 100)
})

# Multiple series format
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'series_A': np.random.normal(10, 2, 100),
    'series_B': np.random.normal(15, 3, 100),
    'series_C': np.random.normal(8, 1.5, 100)
})
```

### Required Columns

- **Time column**: Must be datetime objects for proper formatting
- **Value column**: Numeric values to plot
- **Optional columns**: Multiple value columns for multiple series

### Data Preprocessing

```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Handle missing values
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.random.normal(20, 5, 365)

# Introduce missing values
values[50:60] = np.nan
values[150:160] = np.nan

df = pd.DataFrame({
    'date': dates,
    'value': values,
    'category': ['A'] * 365
})

# Fill missing values for plotting
df['value_filled'] = df['value'].interpolate()

plot = TimeSeriesPlot(figsize=(12, 6))

plot.plot(
    df,
    x='date',
    y='value_filled',
    title="Time Series with Missing Values",
    ylabel="Value",
    plotargs={'linewidth': 2, 'color': 'blue'},
    fillargs={'alpha': 0.2, 'color': 'lightblue'}
)

# Mark missing values
missing_mask = df['value'].isna()
plot.ax.plot(df.loc[missing_mask, 'date'], 
            df.loc[missing_mask, 'value_filled'], 
            'ro', markersize=4, label='Missing Values')

plot.legend()
plot.save('missing_values_timeseries.png')
```

## Customization Options

### Statistical Band Styling

```python
from monet_plots import TimeSeriesPlot

plot = TimeSeriesPlot(figsize=(12, 8))

# Custom fill styling
custom_fillargs = {
    'alpha': 0.3,
    'color': 'rgba(0, 100, 200, 0.3)',
    'edgecolor': 'none',
    'linewidth': 0
}

plot.plot(
    df,
    x='date',
    y='value',
    title="Custom Statistical Bands",
    ylabel="Value",
    fillargs=custom_fillargs
)

plot.save('custom_bands_style.png')
```

### Line and Marker Customization

```python
from monet_plots import TimeSeriesPlot

plot = TimeSeriesPlot(figsize=(12, 8))

# Custom line styling
custom_plotargs = {
    'linewidth': 3,
    'linestyle': '-',
    'marker': 'o',
    'markersize': 4,
    'markerfacecolor': 'red',
    'markeredgecolor': 'darkred',
    'markeredgewidth': 1
}

plot.plot(
    df,
    x='date',
    y='value',
    title="Custom Line and Markers",
    ylabel="Value",
    plotargs=custom_plotargs
)

plot.save('custom_line_markers.png')
```

### Date Formatting

```python
from monet_plots import TimeSeriesPlot, format_date_axis

plot = TimeSeriesPlot(figsize=(12, 8))

plot.plot(
    df,
    x='date',
    y='value',
    title="Custom Date Formatting",
    ylabel="Value"
)

# Format date axis
format_date_axis(
    plot.ax,
    date_format='%b %Y',  # Month Year format
    rotation=45,
    ha='right'
)

plot.save('custom_date_format.png')
```

## Performance Considerations

### Large Time Series

```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Handle large datasets
dates = pd.date_range('2000-01-01', '2023-12-31', freq='D')  # 24 years of daily data
values = np.cumsum(np.random.normal(0, 1, len(dates))) + 50

df = pd.DataFrame({'date': dates, 'value': values})

# Downsample for plotting
df_downsampled = df.resample('M', on='date').mean()  # Monthly averages

plot = TimeSeriesPlot(figsize=(14, 8))

plot.plot(
    df_downsampled,
    x='date',
    y='value',
    title="Downsampled Time Series",
    ylabel="Monthly Average"
)

plot.save('downsampled_timeseries.png')
```

### Memory Management

```python
from monet_plots import TimeSeriesPlot

# Process multiple time series efficiently
time_series_list = []
for i in range(10):
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.normal(0, 1, 100)) + i * 10
    
    df = pd.DataFrame({'date': dates, 'value': values})
    time_series_list.append(df)

# Create and save plots
for i, df in enumerate(time_series_list):
    plot = TimeSeriesPlot(figsize=(12, 6))
    
    plot.plot(
        df,
        x='date',
        y='value',
        title=f"Time Series {i+1}",
        ylabel="Value"
    )
    
    plot.save(f'timeseries_{i+1}.png')
    plot.close()  # Free memory
```

## Common Issues and Solutions

### Date Axis Formatting

```python
from monet_plots import TimeSeriesPlot, format_date_axis

# Fix date formatting issues
plot = TimeSeriesPlot()

plot.plot(df, x='date', y='value')

# Ensure proper date formatting
format_date_axis(plot.ax, date_format='%Y-%m-%d', rotation=45)

plot.save('proper_date_formatting.png')
```

### Multiple Series Legend

```python
from monet_plots import TimeSeriesPlot

# Handle multiple series with single legend
plot = TimeSeriesPlot(figsize=(12, 8))

# Plot multiple series
for color, label in [('blue', 'Series 1'), ('red', 'Series 2'), ('green', 'Series 3')]:
    df_subset = df[df['category'] == label]
    plot.plot(
        df_subset,
        x='date',
        y='value',
        label=label,
        plotargs={'color': color}
    )

plot.legend(title='Categories')
plot.save('multiple_series_legend.png')
```

### Statistical Band Calculation

```python
# Ensure proper statistical calculation
df['date'] = pd.to_datetime(df['date'])  # Ensure datetime format
df = df.set_index('date')  # Set as index for proper grouping

# Calculate statistics properly
daily_stats = df.groupby(df.index.date)['value'].agg(['mean', 'std', 'count'])

# Filter out insufficient data
daily_stats = daily_stats[daily_stats['count'] >= 10]  # At least 10 points per day
```

---

**Related Resources**:

- [API Reference](../api) - Core functionality and utilities
- [Examples](../examples) - Practical usage examples
- [Style Configuration](../api/style) - Plot styling options
- [Plot Utils](../api/plot_utils) - Date formatting utilities
