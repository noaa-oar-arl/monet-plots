# Spatial Plots

Spatial plots are geospatial visualizations that display data on geographical maps. MONET Plots provides comprehensive spatial plotting capabilities with cartopy integration for accurate map projections and geographical features.

## Overview

The spatial plotting functionality in MONET Plots includes multiple specialized classes for different types of geospatial data visualization:

| Plot Type | Class | Use Case |
|-----------|-------|----------|
| [`SpatialPlot`](#spatialplot-class) | Basic spatial plotting with cartopy support | General geospatial data visualization |
| [`SpatialContourPlot`](#spatialcontourplot-class) | Contour plots on geographical maps | Continuous data with contour lines |
| [`SpatialBiasScatterPlot`](#spatialbiasscatterplot-class) | Bias analysis scatter plots | Model vs observation comparison |
| [`XarraySpatialPlot`](#xarrayspatialplot-class) | xarray integration for netCDF data | Climate and weather data |

## SpatialPlot Class

`SpatialPlot` is the base class for creating geospatial plots with cartopy support.

### Class Signature

```python
class SpatialPlot(BasePlot):
    """Creates a spatial plot using cartopy.
    
    This class creates a spatial plot of a 2D model variable on a map.
    It can handle both discrete and continuous colorbars.
    """
    
    def __init__(self, projection=ccrs.PlateCarree(), **kwargs):
        """Initialize the plot with a cartopy projection.
        
        Args:
            projection (cartopy.crs): The cartopy projection to use
            **kwargs: Additional keyword arguments to pass to `subplots`
        """
        pass
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `projection` | `cartopy.crs.Projection` | `ccrs.PlateCarree()` | Map projection |
| `**kwargs` | `dict` | `{}` | Additional matplotlib figure parameters |

### Methods

#### `plot(modelvar, plotargs={}, ncolors=15, discrete=False, **kwargs)`

Plot spatial data on the map.

```python
def plot(self, modelvar, plotargs={}, ncolors=15, discrete=False, **kwargs):
    """Plot the spatial data.
    
    Args:
        modelvar (numpy.ndarray): The 2D model variable to plot
        plotargs (dict, optional): Keyword arguments to pass to `imshow`
        ncolors (int, optional): Number of colors for discrete colorbar. Defaults to 15
        discrete (bool, optional): Use discrete colorbar. Defaults to False
        **kwargs: Additional keyword arguments to pass to `imshow`
    """
    pass
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelvar` | `numpy.ndarray` | Required | 2D array of spatial data |
| `plotargs` | `dict` | `{}` | Additional imshow parameters |
| `ncolors` | `int` | `15` | Number of discrete colors |
| `discrete` | `bool` | `False` | Use discrete colorbar |
| `**kwargs` | `dict` | `{}` | Additional plotting parameters |

**Returns:**
- `matplotlib.axes.Axes`: The axes object with the plot

**Example:**
```python
import numpy as np
from monet_plots import SpatialPlot

# Create spatial plot
plot = SpatialPlot(figsize=(12, 8))

# Generate sample data (lat/lon grid)
lat = np.linspace(30, 50, 40)  # 30°N to 50°N
lon = np.linspace(-120, -70, 50)  # 120°W to 70°W
data = np.random.random((40, 50)) * 100  # Random data 0-100

# Plot data
plot.plot(
    data,
    title="Temperature Distribution",
    cmap='viridis',
    discrete=True,
    ncolors=20
)

# Save plot
plot.save('spatial_plot.png')
plot.close()
```

### Common Usage Patterns

#### Basic Spatial Plot

```python
import numpy as np
from monet_plots import SpatialPlot
import cartopy.crs as ccrs

# Create plot with specific projection
plot = SpatialPlot(
    projection=ccrs.Mercator(),
    figsize=(10, 8)
)

# Sample data
lat = np.linspace(25, 55, 30)
lon = np.linspace(-130, -60, 40)
data = np.sin(np.outer(lat, lon)) * 50 + 100

# Plot with custom colormap
plot.plot(
    data,
    cmap='plasma',
    title="Sample Spatial Data",
    interpolation='bilinear'
)

# Add custom formatting
plot.xlabel("Longitude")
plot.ylabel("Latitude")

plot.save('basic_spatial.png')
```

#### Discrete Colorbar

```python
from monet_plots import SpatialPlot, colorbar_index

plot = SpatialPlot(figsize=(12, 9))

# Generate temperature data
lat = np.linspace(20, 60, 35)
lon = np.linspace(-130, -70, 45)
temperature = 20 + 10 * np.sin(np.outer(lat, lon) * 0.1) + np.random.normal(0, 2, (35, 45))

# Plot with discrete colorbar
plot.plot(
    temperature,
    cmap='RdYlBu_r',
    discrete=True,
    ncolors=15,
    title="Temperature Distribution (°C)"
)

# Add colorbar
cbar, cmap = colorbar_index(
    ncolors=15,
    cmap='RdYlBu_r',
    minval=0,
    maxval=40,
    dtype=int
)

plot.save('discrete_spatial.png')
```

#### Custom Projections

```python
import cartopy.crs as ccrs
from monet_plots import SpatialPlot

# Different projections
projections = [
    (ccrs.PlateCarree(), 'Plate Carree'),
    (ccrs.Robinson(), 'Robinson'),
    (ccrs.Mercator(), 'Mercator'),
    (ccrs.LambertConformal(), 'Lambert Conformal'),
    (ccrs.Orthographic(), 'Orthographic')
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                        subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

for i, (proj, name) in enumerate(projections[:5]):
    plot = SpatialPlot(
        projection=proj,
        figsize=(6, 4)
    )
    
    # Generate data for different regions
    if proj == ccrs.Orthographic():
        # Focus on specific region for orthographic
        data = np.random.random((20, 30)) * 100
    else:
        # Global data for other projections
        data = np.random.random((40, 60)) * 100
    
    plot.plot(data, title=name)
    plot.save(f'spatial_{name.replace(" ", "_").lower()}.png')

plt.tight_layout()
```

## SpatialContourPlot Class

`SpatialContourPlot` creates contour plots on geographical maps.

### Class Signature

```python
class SpatialContourPlot(BasePlot):
    """Creates a contour plot on a geographical map."""
    
    def __init__(self, projection=ccrs.PlateCarree(), **kwargs):
        """Initialize with cartopy projection."""
        pass
```

### Methods

#### `plot(modelvar, levels=10, cmap='viridis', **kwargs)`

Plot contour data on the map.

```python
def plot(self, modelvar, levels=10, cmap='viridis', **kwargs):
    """Plot contour data on the map.
    
    Args:
        modelvar (numpy.ndarray): 2D array of contour data
        levels (int or list): Contour levels
        cmap (str): Colormap name
        **kwargs: Additional contour parameters
    """
    pass
```

**Example:**
```python
import numpy as np
from monet_plots import SpatialContourPlot

# Create contour plot
plot = SpatialContourPlot(figsize=(12, 9))

# Generate pressure data for contours
lat = np.linspace(30, 60, 40)
lon = np.linspace(-120, -80, 50)
pressure = 1013 + 20 * np.sin(np.outer(lat, lon) * 0.05)

# Plot with custom contour levels
plot.plot(
    pressure,
    levels=[1000, 1010, 1015, 1020, 1030],
    cmap='RdBu_r',
    title="Pressure Contours (hPa)",
    linewidths=1.5
)

# Add contour labels
plot.ax.clabel(plot.ax.contour_set, inline=True, fontsize=8)

plot.save('contour_plot.png')
```

## SpatialBiasScatterPlot Class

`SpatialBiasScatterPlot` creates scatter plots for bias analysis between model and observation data.

### Class Signature

```python
class SpatialBiasScatterPlot(BasePlot):
    """Creates a spatial bias scatter plot for model-observation comparison."""
    
    def __init__(self, **kwargs):
        """Initialize the bias scatter plot."""
        pass
```

### Methods

#### `plot(model_data, obs_data, bias_bins=20, **kwargs)`

Plot bias scatter analysis.

```python
def plot(self, model_data, obs_data, bias_bins=20, **kwargs):
    """Plot bias scatter analysis.
    
    Args:
        model_data (numpy.ndarray): Model data array
        obs_data (numpy.ndarray): Observation data array
        bias_bins (int): Number of bias bins
        **kwargs: Additional plotting parameters
    """
    pass
```

**Example:**
```python
import numpy as np
from monet_plots import SpatialBiasScatterPlot

# Create bias scatter plot
plot = SpatialBiasScatterPlot(figsize=(12, 10))

# Generate model and observation data
model_data = np.random.normal(25, 5, (30, 40))
obs_data = model_data + np.random.normal(0, 2, (30, 40))  # Add observation error

# Plot bias analysis
plot.plot(
    model_data,
    obs_data,
    bias_bins=15,
    title="Model-Observation Bias Analysis",
    cmap='RdBu_r'
)

# Add reference lines
plot.ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plot.ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

plot.xlabel("Model Values")
plot.ylabel("Observation Values")
plot.save('bias_scatter_plot.png')
```

## XarraySpatialPlot Class

`XarraySpatialPlot` integrates with xarray for handling netCDF and other scientific data formats.

### Class Signature

```python
class XarraySpatialPlot(BasePlot):
    """Creates spatial plots from xarray DataArrays."""
    
    def __init__(self, **kwargs):
        """Initialize the xarray spatial plot."""
        pass
```

### Methods

#### `plot(data_array, x='lon', y='lat', **kwargs)`

Plot xarray DataArray.

```python
def plot(self, data_array, x='lon', y='lat', **kwargs):
    """Plot xarray DataArray.
    
    Args:
        data_array (xarray.DataArray): Data to plot
        x (str): Name of x coordinate
        y (str): Name of y coordinate
        **kwargs: Additional plotting parameters
    """
    pass
```

**Example:**
```python
import xarray as xr
from monet_plots import XarraySpatialPlot

# Create sample xarray data
lat = np.linspace(30, 60, 20)
lon = np.linspace(-120, -80, 30)
time = pd.date_range('2023-01-01', periods=10, freq='D')

data = xr.DataArray(
    np.random.random((10, 20, 30)) * 30 + 10,
    dims=['time', 'lat', 'lon'],
    coords={'time': time, 'lat': lat, 'lon': lon},
    name='temperature'
)

# Create xarray spatial plot
plot = XarraySpatialPlot(figsize=(14, 10))

# Plot time mean
time_mean = data.mean(dim='time')
plot.plot(
    time_mean,
    title="Mean Temperature (°C)",
    cmap='coolwarm',
    discrete=True,
    ncolors=16
)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('xarray_spatial.png')
```

## Advanced Usage

### Custom Map Features

```python
import cartopy.feature as cfeature
from monet_plots import SpatialPlot

plot = SpatialPlot(
    projection=ccrs.Mercator(),
    figsize=(12, 10)
)

# Add custom map features
plot.ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
plot.ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
plot.ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
plot.ax.add_feature(cfeature.RIVERS, linewidth=0.5)
plot.ax.add_feature(cfeature.LAKES, color='lightblue', linewidth=0.5)

# Add custom gridlines
plot.ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Plot data
data = np.random.random((40, 50)) * 100
plot.plot(data, cmap='viridis', title="Custom Map Features")

plot.save('custom_map_features.png')
```

### Multiple Subplots

```python
import matplotlib.pyplot as plt
from monet_plots import SpatialPlot
import cartopy.crs as ccrs

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# Different projections
projections = [
    ccrs.PlateCarree(),
    ccrs.Robinson(),
    ccrrs.Mercator(),
    ccrs.LambertConformal()
]

titles = ['Plate Carree', 'Robinson', 'Mercator', 'Lambert Conformal']

for i, (proj, title) in enumerate(zip(projections, titles)):
    ax = fig.add_subplot(2, 2, i+1, projection=proj)
    
    # Create spatial plot on subplot
    plot = SpatialPlot(figure=fig, subplot_kw={'projection': proj})
    
    # Generate data
    if proj == ccrs.LambertConformal():
        # Lambert conformal focuses on North America
        data = np.random.random((30, 40)) * 100
    else:
        # Other projections show global data
        data = np.random.random((40, 60)) * 100
    
    plot.plot(data, cmap='plasma', title=title)
    plot.ax.set_title(title, fontsize=12)

plt.tight_layout()
save_figure(fig, 'multi_projection_spatial.png')
```

### Animation Support

```python
import matplotlib.animation as animation
from monet_plots import SpatialPlot

# Create animated spatial plot
plot = SpatialPlot(figsize=(12, 9))

# Generate time-varying data
frames = 20
lat = np.linspace(30, 60, 35)
lon = np.linspace(-120, -80, 45)

def animate(frame):
    plot.ax.clear()
    
    # Create time-varying data
    data = np.sin(np.outer(lat, lon) * 0.1 + frame * 0.1) * 50 + 100
    
    plot.plot(data, cmap='viridis', title=f"Frame {frame}")
    plot.ax.set_title(f"Spatial Data - Frame {frame}", fontsize=14)

# Create animation
anim = animation.FuncAnimation(
    plot.fig, animate, frames=frames, interval=200, blit=False
)

# Save animation
anim.save('spatial_animation.gif', writer='pillow', fps=5)
```

## Data Requirements

### Input Data Format

Spatial plots expect 2D numpy arrays with geographical coordinates:

```python
import numpy as np

# Basic 2D array
data = np.random.random((50, 100))  # lat x lon

# With actual coordinates
lat = np.linspace(30, 60, 50)     # Latitude values
lon = np.linspace(-120, -80, 100)   # Longitude values
data = np.random.random((50, 100))  # Corresponding data
```

### Coordinate Systems

- **Latitude**: Typically ranges from -90° to 90°
- **Longitude**: Typically ranges from -180° to 180°
- **Projections**: Use appropriate cartopy projections for your data

### Data Preprocessing

```python
import numpy as np
from monet_plots import SpatialPlot

# Handle missing values
data = np.random.random((40, 50)) * 100
data[data < 20] = np.nan  # Set low values as missing

# Create masked array for better visualization
masked_data = np.ma.masked_invalid(data)

plot = SpatialPlot()
plot.plot(
    masked_data,
    cmap='viridis',
    title="Data with Missing Values",
    interpolation='nearest'
)

plot.save('masked_spatial.png')
```

## Performance Considerations

### Memory Management

```python
from monet_plots import SpatialPlot

# Large data handling
large_data = np.random.random((1000, 1500))  # Large dataset

plot = SpatialPlot(figsize=(16, 12))

# Downsample for display
from scipy.ndimage import zoom
downsampled = zoom(large_data, 0.2)  # Downsample by factor of 5

plot.plot(downsampled, title="Downsampled Data")
plot.save('large_data_plot.png')
```

### Optimization Tips

1. **Use appropriate projections** for your data region
2. **Downsample large datasets** before plotting
3. **Use discrete colorbars** for categorical data
4. **Close plots** when done to free memory: `plot.close()`

## Common Issues and Solutions

### Map Not Showing

```python
# Fix: Ensure cartopy is installed
# pip install cartopy

# Check projection compatibility
import cartopy.crs as ccrs
plot = SpatialPlot(projection=ccrs.PlateCarree())  # Always works
```

### Colorbar Issues

```python
from monet_plots import colorbar_index

# Fix: Create proper colorbar
cbar, cmap = colorbar_index(
    ncolors=10,
    cmap='viridis',
    minval=data.min(),
    maxval=data.max()
)
```

### Coordinate Mismatch

```python
# Fix: Align coordinates with data
lat = np.linspace(data_lat_min, data_lat_max, data.shape[0])
lon = np.linspace(data_lon_min, data_lon_max, data.shape[1])
```

---

**Related Resources**:

- [API Reference](../api) - Core functionality and utilities
- [Examples](../examples) - Practical usage examples
- [Style Configuration](../api/style) - Plot styling options
- [Colorbars](../api/colorbars) - Colorbar customization
