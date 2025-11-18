# API Reference

Welcome to the MONET Plots API reference. This section provides detailed documentation for all modules, classes, and functions in the library.

## API Overview

MONET Plots is organized into several key modules:

### Core Modules

- **[`monet_plots`](./module_monet_plots)**: Main package with plot classes and utilities
- **[`monet_plots.base`](./base)**: Base classes and common functionality
- **[`monet_plots.plots`](./plots)**: All plot type implementations
- **[`monet_plots.utils`](./utils)**: Utility functions and helpers
- **[`monet_plots.style`](./style)**: Styling configuration and themes
- **[`monet_plots.colorbars`](./colorbars)**: Colorbar creation utilities
- **[`monet_plots.taylordiagram`](./taylordiagram)**: Taylor diagram functionality
- **[`monet_plots.plot_utils`](./plot_utils)**: Common plotting utilities

### Plot Classes

| Plot Type | Module | Class | Description |
|-----------|--------|-------|-------------|
| Spatial | [`plots.spatial`](../plots/spatial) | [`SpatialPlot`](../plots/spatial) | Geospatial plots with cartopy support |
| Time Series | [`plots.timeseries`](../plots/timeseries) | [`TimeSeriesPlot`](../plots/timeseries) | Time series with statistical bands |
| Scatter | [`plots.scatter`](../plots/scatter) | [`ScatterPlot`](../plots/scatter) | Scatter plots with regression lines |
| Taylor Diagram | [`plots.taylor`](../plots/taylor) | [`TaylorDiagramPlot`](../plots/taylor) | Model evaluation diagrams |
| KDE | [`plots.kde`](../plots/kde) | [`KDEPlot`](../plots/kde) | Kernel density estimation plots |
| Wind Quiver | [`plots.wind`](../plots/wind) | [`WindQuiverPlot`](../plots/wind) | Wind vector plots |
| Wind Barbs | [`plots.wind`](../plots/wind) | [`WindBarbsPlot`](../plots/wind) | Wind barb plots |
| Facet Grid | [`plots.facet_grid`](../plots/facet_grid) | [`FacetGridPlot`](../plots/facet_grid) | Multi-panel figure layouts |

## Quick Navigation

### Base Classes

- **[`BasePlot`](./base)**: Abstract base class for all plots
  - [`__init__()`](./base#BasePlot.__init__)
  - [`plot()`](./base#BasePlot.plot)
  - [`save()`](./base#BasePlot.save)
  - [`close()`](./base#BasePlot.close)
  - [`title()`](./base#BasePlot.title)
  - [`xlabel()`](./base#BasePlot.xlabel)
  - [`ylabel()`](./base#BasePlot.ylabel)

### Main Package

- **[`monet_plots`](./module_monet_plots)**: Main package exports
  - [`__version__`](./module_monet_plots#version)
  - [`wiley_style`](./module_monet_plots#wiley_style)
  - All plot classes

### Plot-Specific APIs

- **[`SpatialPlot`](../plots/spatial)**: Geospatial plotting
- **[`TimeSeriesPlot`](../plots/timeseries)**: Time series analysis
- **[`ScatterPlot`](../plots/scatter)**: Scatter plots and regression
- **[`TaylorDiagramPlot`](../plots/taylor)**: Model evaluation
- **[`KDEPlot`](../plots/kde)**: Density estimation
- **[`WindQuiverPlot`](../plots/wind)**: Vector field plotting
- **[`WindBarbsPlot`](../plots/wind)**: Wind visualization
- **[`FacetGridPlot`](../plots/facet_grid)**: Multi-panel layouts

### Utilities

- **[`style`](./style)**: Styling configuration
  - [`wiley_style`](./style#wiley_style)
  - [`aps_style`](./style#aps_style)
  - [`custom_style()`](./style#custom_style)
- **[`colorbars`](./colorbars)**: Colorbar utilities
  - [`colorbar_index()`](./colorbars#colorbar_index)
  - [`colorbar_from_cmap()`](./colorbars#from_cmap)
- **[`taylordiagram`](./taylordiagram)**: Taylor diagrams
  - [`TaylorDiagram()`](./taylordiagram#TaylorDiagram)
  - [`add_contours()`](./taylordiagram#add_contours)
- **[`plot_utils`](./plot_utils)**: Common utilities
  - [`add_colorbar()`](./plot_utils#add_colorbar)
  - [`format_date_axis()`](./plot_utils#format_date_axis)
  - [`save_figure()`](./plot_utils#save_figure)

## Usage Examples

### Basic Plot Creation

```python
from monet_plots import SpatialPlot, TimeSeriesPlot

# Create a spatial plot
plot = SpatialPlot(figsize=(10, 6))
plot.plot(data, cmap='viridis')
plot.save('spatial.png')

# Create a time series plot
ts_plot = TimeSeriesPlot()
ts_plot.plot(df, x='time', y='value')
ts_plot.save('timeseries.png')
```

### Style Configuration

```python
from monet_plots import style
import matplotlib.pyplot as plt

# Apply Wiley style
plt.style.use(style.wiley_style)

# Create custom style
custom = style.custom_style(
    fontsize=12, 
    figsize=(12, 8)
)
plt.style.use(custom)
```

### Colorbar Creation

```python
from monet_plots import colorbar_index

# Create indexed colorbar
cbar, cmap = colorbar_index(10, 'viridis', minval=0, maxval=100)
```

### Taylor Diagram

```python
from monet_plots import taylordiagram

# Create Taylor diagram
td = taylordiagram.TaylorDiagram(obs_std=1.0)
td.add_model(0.8, 0.5, 'Model 1')
td.add_contours(levels=[0.5, 1.0, 1.5])
```

## Advanced Topics

### Custom Plot Classes

```python
from monet_plots.base import BasePlot

class CustomPlot(BasePlot):
    def plot(self, data, **kwargs):
        # Custom plotting logic
        pass
```

### Integration with Other Libraries

```python
import xarray as xr
import monet_plots as mp

# Use with xarray
data = xr.open_dataset('data.nc')
plot = SpatialPlot()
plot.plot(data.temperature)
```

### Performance Considerations

- Use [`plot.close()`](./base#BasePlot.close) to free memory
- Choose appropriate data types (pandas DataFrames for tabular data)
- Consider downsampling large datasets for interactive use

## Error Handling

### Common Exceptions

- **`ValueError`**: Invalid input data or parameters
- **`ImportError`**: Missing optional dependencies (e.g., cartopy)
- **`AttributeError`**: Invalid attribute access on plot objects
- **`FileNotFoundError`**: File operations on non-existent paths

### Debug Tips

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug output
plot = SpatialPlot(debug=True)
```

## Version Information

```python
import monet_plots
print(f"MONET Plots version: {monet_plots.__version__}")
print(f"Dependencies: {monet_plots.__deps__}")
```

---

**Next Steps**: 

- Browse specific plot types in the [Plots section](../plots)
- Explore [Configuration](../configuration) for customization options
- Check out [Examples](../examples) for practical use cases
- Read about [Performance optimization](../performance) for large datasets