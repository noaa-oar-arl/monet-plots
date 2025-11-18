# Wind Plots

Wind plots visualize wind vector fields using either quiver plots (arrows) or wind barbs. These specialized plots are essential for meteorological analysis and atmospheric visualization.

## Overview

MONET Plots provides comprehensive wind visualization capabilities with support for both quiver and wind barbs, making it suitable for various meteorological applications.

| Plot Type | Class | Use Case | Features |
|-----------|-------|----------|----------|
| [`WindQuiverPlot`](#windquiverplot-class) | Vector field visualization | Wind direction and magnitude | Arrows, color coding, customizable scaling |
| [`WindBarbsPlot`](#windbarbsplot-class) | Traditional meteorological plotting | Standard wind observation display | Barbs, meteorological conventions |

## WindQuiverPlot Class

`WindQuiverPlot` creates wind vector field plots using quiver arrows.

### Class Signature

```python
class WindQuiverPlot(BasePlot):
    """Creates a wind vector field plot using quiver arrows.
    
    This class creates wind vector plots with customizable arrow
    representation and color coding.
    """
    
    def __init__(self, **kwargs):
        """Initialize the wind quiver plot.
        
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

#### `plot(u, v, x=None, y=None, title='', xlabel='X', ylabel='Y', scale=None, scale_units='xy', angles='xy', **kwargs)`

Plot wind vector field.

```python
def plot(self, u, v, x=None, y=None, title='', xlabel='X', ylabel='Y', scale=None, scale_units='xy', angles='xy', **kwargs):
    """Plot the wind vector field.
    
    Args:
        u (numpy.ndarray): U-component (east-west) of wind
        v (numpy.ndarray): V-component (north-south) of wind
        x (numpy.ndarray, optional): X-coordinates. Defaults to None
        y (numpy.ndarray, optional): Y-coordinates. Defaults to None
        title (str, optional): Plot title. Defaults to ''
        xlabel (str, optional): X-axis label. Defaults to 'X'
        ylabel (str, optional): Y-axis label. Defaults to 'Y'
        scale (float, optional): Scale factor for arrows. Defaults to None
        scale_units (str, optional): Scale units. Defaults to 'xy'
        angles (str, optional): Angle units. Defaults to 'xy'
        **kwargs: Additional quiver parameters
    """
    pass
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `u` | `numpy.ndarray` | Required | U-component (east-west) wind |
| `v` | `numpy.ndarray` | Required | V-component (north-south) wind |
| `x` | `numpy.ndarray` | `None` | X-coordinate grid |
| `y` | `numpy.ndarray` | `None` | Y-coordinate grid |
| `title` | `str` | `''` | Plot title |
| `xlabel` | `str` | `'X'` | X-axis label |
| `ylabel` | `str` | `'Y'` | Y-axis label |
| `scale` | `float` | `None` | Arrow scale factor |
| `scale_units` | `str` | `'xy'` | Scale units |
| `angles` | `str` | `'xy'` | Angle units |
| `**kwargs` | `dict` | `{}` | Additional quiver parameters |

**Example:**
```python
import numpy as np
from monet_plots import WindQuiverPlot

# Create wind quiver plot
plot = WindQuiverPlot(figsize=(12, 10))

# Create wind field data
lat = np.linspace(30, 50, 15)  # Latitude
lon = np.linspace(-120, -70, 20)  # Longitude

# Create meshgrid
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Generate wind field (simple circulation pattern)
u = -np.sin(lat_grid * np.pi / 180) * 5  # Easterly component
v = np.cos(lat_grid * np.pi / 180) * 3   # Northerly component
magnitude = np.sqrt(u**2 + v**2)          # Wind speed

# Add some turbulence
u += np.random.normal(0, 0.5, u.shape)
v += np.random.normal(0, 0.5, v.shape)

# Plot wind field
plot.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Wind Vector Field",
    xlabel="Longitude",
    ylabel="Latitude",
    scale=50,  # Scale factor for arrow size
    scale_units='xy',
    angles='xy',
    color=magnitude,  # Color by wind speed
    cmap='viridis',
    width=0.003,  # Arrow width
    headwidth=3,  # Arrow head width
    headlength=4  # Arrow head length
)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('wind_quiver_basic.png')
```

### Common Usage Patterns

#### Basic Wind Field

```python
import numpy as np
from monet_plots import WindQuiverPlot

# Create wind quiver plot
plot = WindQuiverPlot(figsize=(12, 9))

# Create coordinate grids
x = np.linspace(0, 10, 12)
y = np.linspace(0, 8, 10)
X, Y = np.meshgrid(x, y)

# Create wind field (simple flow)
u = np.ones_like(X) * 2  # Constant easterly flow
v = np.sin(X * 0.5) * 1.5  # Sinusoidal northerly component

# Add some variation
u += np.random.normal(0, 0.1, u.shape)
v += np.random.normal(0, 0.1, v.shape)

# Calculate wind speed for coloring
wind_speed = np.sqrt(u**2 + v**2)

plot.plot(
    u, v,
    x=X,
    y=Y,
    title="Basic Wind Vector Field",
    xlabel="X Coordinate",
    ylabel="Y Coordinate",
    color=wind_speed,
    cmap='plasma',
    scale=30,
    scale_units='xy',
    angles='xy',
    alpha=0.8
)

plot.save('basic_wind_field.png')
```

#### Meteorological Wind Analysis

```python
import numpy as np
from monet_plots import WindQuiverPlot

# Create meteorological wind analysis
plot = WindQuiverPlot(figsize=(14, 10))

# Create latitude/longitude grid
lat = np.linspace(25, 55, 18)
lon = np.linspace(-130, -60, 22)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Simulate meteorological wind patterns
# Geostrophic wind approximation
f = 2 * 7.27e-5 * np.sin(lat_grid * np.pi / 180)  # Coriolis parameter
pressure_gradient = np.gradient(np.sin(lat_grid * 0.1) * np.cos(lon_grid * 0.1))

# Calculate geostrophic wind
u_geo = -pressure_gradient[1] / f * 10  # Easterly component
v_geo = pressure_gradient[0] / f * 10   # Northerly component

# Add surface friction effects
friction_factor = np.exp(-np.abs(lat_grid - 40) / 10)  # Stronger near 40°N
u_surface = u_geo * friction_factor
v_surface = v_geo * friction_factor

# Add some random variation
u_surface += np.random.normal(0, 0.5, u_surface.shape)
v_surface += np.random.normal(0, 0.5, v_surface.shape)

# Calculate wind speed and direction
wind_speed = np.sqrt(u_surface**2 + v_surface**2)

plot.plot(
    u_surface, v_surface,
    x=lon_grid,
    y=lat_grid,
    title="Meteorological Wind Analysis",
    xlabel="Longitude",
    ylabel="Latitude",
    color=wind_speed,
    cmap='jet',
    scale=100,
    scale_units='xy',
    angles='xy',
    width=0.004,
    headwidth=3,
    headlength=4,
    alpha=0.8
)

# Add colorbar for wind speed
import matplotlib.pyplot as plt
cbar = plt.colorbar(plot.ax.collections[0], ax=plot.ax, shrink=0.8)
cbar.set_label('Wind Speed (m/s)')

plot.save('meteorological_wind_analysis.png')
```

#### Time-Varying Wind Field

```python
import numpy as np
import matplotlib.animation as animation
from monet_plots import WindQuiverPlot

# Create animated wind field
plot = WindQuiverPlot(figsize=(12, 10))

# Create coordinate grid
x = np.linspace(0, 10, 15)
y = np.linspace(0, 8, 12)
X, Y = np.meshgrid(x, y)

def animate(frame):
    plot.ax.clear()
    
    # Create time-varying wind field
    t = frame * 0.1
    u = np.cos(X * 0.5 + t) * 3 + np.random.normal(0, 0.2, X.shape)
    v = np.sin(Y * 0.3 + t * 1.2) * 2 + np.random.normal(0, 0.2, Y.shape)
    
    wind_speed = np.sqrt(u**2 + v**2)
    
    plot.plot(
        u, v,
        x=X,
        y=Y,
        title=f"Time-Varying Wind Field - Frame {frame}",
        xlabel="X Coordinate",
        ylabel="Y Coordinate",
        color=wind_speed,
        cmap='viridis',
        scale=30,
        scale_units='xy',
        angles='xy',
        alpha=0.7
    )
    
    return plot.ax.collections

# Create animation
anim = animation.FuncAnimation(
    plot.fig, animate, frames=50, interval=200, blit=False
)

# Save animation
anim.save('wind_field_animation.gif', writer='pillow', fps=5)
```

## WindBarbsPlot Class

`WindBarbsPlot` creates wind plots using traditional meteorological wind barbs.

### Class Signature

```python
class WindBarbsPlot(BasePlot):
    """Creates a wind plot using meteorological wind barbs.
    
    This class creates wind plots using standard meteorological
    wind barb notation.
    """
    
    def __init__(self, **kwargs):
        """Initialize the wind barbs plot.
        
        Args:
            **kwargs: Additional keyword arguments to pass to `subplots`
        """
        pass
```

### Methods

#### `plot(u, v, x=None, y=None, title='', xlabel='X', ylabel='Y', **kwargs)`

Plot wind barbs.

```python
def plot(self, u, v, x=None, y=None, title='', xlabel='X', ylabel='Y', **kwargs):
    """Plot the wind barbs.
    
    Args:
        u (numpy.ndarray): U-component of wind
        v (numpy.ndarray): V-component of wind
        x (numpy.ndarray, optional): X-coordinates. Defaults to None
        y (numpy.ndarray, optional): Y-coordinates. Defaults to None
        title (str, optional): Plot title. Defaults to ''
        xlabel (str, optional): X-axis label. Defaults to 'X'
        ylabel (str, optional): Y-axis label. Defaults to 'Y'
        **kwargs: Additional barbs parameters
    """
    pass
```

**Example:**
```python
import numpy as np
from monet_plots import WindBarbsPlot

# Create wind barbs plot
plot = WindBarbsPlot(figsize=(12, 10))

# Create coordinate grid
lat = np.linspace(30, 50, 12)  # Latitude
lon = np.linspace(-120, -70, 16)  # Longitude
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Generate wind data
u = np.random.normal(5, 3, lat_grid.shape)  # Easterly component
v = np.random.normal(-2, 2, lat_grid.shape)  # Southerly component

# Plot wind barbs
plot.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Wind Barbs Plot",
    xlabel="Longitude",
    ylabel="Latitude",
    length=6,  # Barb length
    barbcolor='blue',
    flagcolor='red',
    linewidth=0.8
)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('wind_barbs_basic.png')
```

### Advanced Wind Barb Usage

```python
import numpy as np
from monet_plots import WindBarbsPlot

# Create advanced wind barbs plot
plot = WindBarbsPlot(figsize=(14, 10))

# Create coordinate grid
lat = np.linspace(25, 55, 10)
lon = np.linspace(-130, -60, 14)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Generate realistic wind data
# Create wind patterns based on latitude
wind_speed = 5 + 10 * np.exp(-((lat_grid - 40)**2) / 100)  # Stronger at 40°N
wind_direction = np.random.uniform(0, 360, lat_grid.shape)

# Convert to u, v components
u = wind_speed * np.cos(np.radians(wind_direction))
v = wind_speed * np.sin(np.radians(wind_direction))

# Plot wind barbs with different styling
plot.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Advanced Wind Barbs",
    xlabel="Longitude",
    ylabel="Latitude",
    length=8,  # Longer barbs
    barbcolor='black',
    flagcolor='darkgreen',
    linewidth=1.2,
    sizes=dict(emptybarb=1, halfbarb=2, fullbarb=3, flagbarb=1.5)
)

# Add wind speed contour
wind_speed_mag = np.sqrt(u**2 + v**2)
plot.ax.contour(lon_grid, lat_grid, wind_speed_mag, 
               levels=[5, 10, 15, 20], colors='gray', alpha=0.5)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('advanced_wind_barbs.png')
```

## Advanced Features

### Combined Wind Plots

```python
import numpy as np
import matplotlib.pyplot as plt
from monet_plots import WindQuiverPlot, WindBarbsPlot

# Create figure with combined wind plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Create coordinate grid
lat = np.linspace(30, 50, 12)
lon = np.linspace(-120, -70, 16)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Generate wind data
u = np.random.normal(5, 3, lat_grid.shape)
v = np.random.normal(-2, 2, lat_grid.shape)
wind_speed = np.sqrt(u**2 + v**2)

# Wind quiver plot
plot1 = WindQuiverPlot(figure=fig, subplot_kw=dict(ax=ax1))
plot1.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Wind Quiver Plot",
    xlabel="Longitude",
    ylabel="Latitude",
    color=wind_speed,
    cmap='viridis',
    scale=80,
    scale_units='xy',
    angles='xy',
    alpha=0.8
)

# Wind barbs plot
plot2 = WindBarbsPlot(figure=fig, subplot_kw=dict(ax=ax2))
plot2.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Wind Barbs Plot",
    xlabel="Longitude",
    ylabel="Latitude",
    length=6,
    barbcolor='darkblue',
    flagcolor='red',
    linewidth=0.8
)

plt.tight_layout()
save_figure(fig, 'combined_wind_plots.png')
```

### Interactive Wind Analysis

```python
import matplotlib.pyplot as plt
from monet_plots import WindQuiverPlot

# Enable interactive mode
plt.ion()

# Create interactive wind plot
plot = WindQuiverPlot(figsize=(12, 10))

# Create coordinate grid
x = np.linspace(0, 10, 15)
y = np.linspace(0, 8, 12)
X, Y = np.meshgrid(x, y)

# Initial wind field
u = np.ones_like(X) * 2
v = np.sin(X * 0.5) * 1.5
wind_speed = np.sqrt(u**2 + v**2)

plot.plot(
    u, v,
    x=X,
    y=Y,
    title="Interactive Wind Analysis",
    xlabel="X Coordinate",
    ylabel="Y Coordinate",
    color=wind_speed,
    cmap='plasma',
    scale=30,
    scale_units='xy',
    angles='xy',
    alpha=0.8
)

# Interactive wind modification
def on_click(event):
    if event.inaxes == plot.ax and event.button == 1:  # Left click
        # Find nearest grid point
        distances = np.sqrt((X - event.xdata)**2 + (Y - event.ydata)**2)
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Modify wind at that point
        u[i, j] += 1
        v[i, j] += 1
        
        # Update plot
        wind_speed_new = np.sqrt(u**2 + v**2)
        
        # Clear and redraw
        plot.ax.clear()
        plot.plot(
            u, v,
            x=X,
            y=Y,
            title=f"Interactive Wind Analysis - Click to modify",
            xlabel="X Coordinate",
            ylabel="Y Coordinate",
            color=wind_speed_new,
            cmap='plasma',
            scale=30,
            scale_units='xy',
            angles='xy',
            alpha=0.8
        )
        
        plot.fig.canvas.draw()

plot.fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
```

### Wind Speed and Direction Analysis

```python
import numpy as np
import pandas as pd
from monet_plots import WindQuiverPlot

# Create wind speed and direction analysis
plot = WindQuiverPlot(figsize=(14, 10))

# Create coordinate grid
lat = np.linspace(30, 50, 15)
lon = np.linspace(-120, -70, 20)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Generate realistic wind data
# Create wind patterns based on geography
wind_speed = 5 + 8 * np.exp(-((lat_grid - 40)**2) / 150) + np.random.normal(0, 1, lat_grid.shape)
wind_direction = 270 + 30 * np.sin(lat_grid * np.pi / 180) + np.random.normal(0, 10, lat_grid.shape)

# Convert to u, v components
u = wind_speed * np.cos(np.radians(wind_direction))
v = wind_speed * np.sin(np.radians(wind_direction))

# Calculate additional statistics
wind_direction_deg = np.degrees(np.arctan2(v, u)) % 360
wind_direction_cardinal = np.select(
    [(wind_direction_deg >= 337.5) | (wind_direction_deg < 22.5),
     (wind_direction_deg >= 22.5) & (wind_direction_deg < 67.5),
     (wind_direction_deg >= 67.5) & (wind_direction_deg < 112.5),
     (wind_direction_deg >= 112.5) & (wind_direction_deg < 157.5),
     (wind_direction_deg >= 157.5) & (wind_direction_deg < 202.5),
     (wind_direction_deg >= 202.5) & (wind_direction_deg < 247.5),
     (wind_direction_deg >= 247.5) & (wind_direction_deg < 292.5),
     (wind_direction_deg >= 292.5) & (wind_direction_deg < 337.5)],
    ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
)

# Plot wind vectors with speed-based coloring
plot.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Wind Speed and Direction Analysis",
    xlabel="Longitude",
    ylabel="Latitude",
    color=wind_speed,
    cmap='coolwarm',
    scale=100,
    scale_units='xy',
    angles='xy',
    alpha=0.9,
    width=0.003,
    headwidth=3,
    headlength=4
)

# Add statistics text
stats_text = f"""Wind Statistics:
Max Speed: {wind_speed.max():.1f} m/s
Min Speed: {wind_speed.min():.1f} m/s
Mean Speed: {wind_speed.mean():.1f} m/s
Predominant Direction: {pd.Series(wind_direction_cardinal).mode()[0]}"""

plot.ax.text(0.02, 0.98, stats_text,
            transform=plot.ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('wind_analysis.png')
```

## Data Requirements

### Input Data Format

Wind plots require 2D numpy arrays for wind components:

```python
import numpy as np

# Basic format - regular grid
u = np.random.normal(5, 2, (10, 15))  # U-component
v = np.random.normal(-2, 1, (10, 15))  # V-component

# With coordinates
lat = np.linspace(30, 50, 10)
lon = np.linspace(-120, -70, 15)
lon_grid, lat_grid = np.meshgrid(lon, lat)

u = np.random.normal(5, 2, lat_grid.shape)
v = np.random.normal(-2, 1, lat_grid.shape)

# Irregular grid (for barbs)
x_points = np.random.uniform(0, 10, 50)
y_points = np.random.uniform(0, 8, 50)
u_points = np.random.normal(5, 2, 50)
v_points = np.random.normal(-2, 1, 50)
```

### Data Preprocessing

```python
import numpy as np
from monet_plots import WindQuiverPlot

# Handle missing or invalid wind data
np.random.seed(42)
lat = np.linspace(30, 50, 12)
lon = np.linspace(-120, -70, 16)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Generate wind data with some missing values
u = np.random.normal(5, 3, lat_grid.shape)
v = np.random.normal(-2, 2, lat_grid.shape)

# Introduce missing values
u[3:5, 7:10] = np.nan
v[8:10, 3:6] = np.nan

# Remove invalid wind speeds
wind_speed = np.sqrt(u**2 + v**2)
valid_mask = (wind_speed < 50) & (wind_speed > 0)  # Reasonable wind speed limits
u = np.where(valid_mask, u, np.nan)
v = np.where(valid_mask, v, np.nan)

# Create plot
plot = WindQuiverPlot(figsize=(12, 10))

plot.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Wind Data with Missing Values Handled",
    xlabel="Longitude",
    ylabel="Latitude",
    color=wind_speed,
    cmap='viridis',
    scale=80,
    scale_units='xy',
    angles='xy',
    alpha=0.8
)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('clean_wind_data.png')
```

## Customization Options

### Arrow Styling

```python
from monet_plots import WindQuiverPlot

plot = WindQuiverPlot(figsize=(12, 10))

# Custom arrow styling
custom_kwargs = {
    'scale': 60,
    'scale_units': 'xy',
    'angles': 'xy',
    'width': 0.004,
    'headwidth': 4,
    'headlength': 5,
    'minshaft': 1,
    'minlength': 1,
    'pivot': 'mid',  # Arrow pivot point
    'alpha': 0.9,
    'linewidth': 1.5
}

# Create wind data
lat = np.linspace(30, 50, 15)
lon = np.linspace(-120, -70, 20)
lon_grid, lat_grid = np.meshgrid(lon, lat)

u = np.random.normal(5, 3, lat_grid.shape)
v = np.random.normal(-2, 2, lat_grid.shape)
wind_speed = np.sqrt(u**2 + v**2)

plot.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Custom Arrow Styling",
    xlabel="Longitude",
    ylabel="Latitude",
    color=wind_speed,
    cmap='plasma',
    **custom_kwargs
)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('custom_arrow_styling.png')
```

### Wind Barb Styling

```python
from monet_plots import WindBarbsPlot

plot = WindBarbsPlot(figsize=(12, 10))

# Custom barb styling
custom_kwargs = {
    'length': 8,
    'barbcolor': 'darkblue',
    'flagcolor': 'red',
    'linewidth': 1.5,
    'sizes': {
        'emptybarb': 1,
        'halfbarb': 2,
        'fullbarb': 3,
        'flagbarb': 1.5
    },
    'emptybarb': False,  # Hide empty barbs
    'rounding': False,   # Don't round wind speeds
    'flip_barb': True    # Flip barbs for southern hemisphere
}

# Create wind data
lat = np.linspace(30, 50, 12)
lon = np.linspace(-120, -70, 16)
lon_grid, lat_grid = np.meshgrid(lon, lat)

u = np.random.normal(5, 3, lat_grid.shape)
v = np.random.normal(-2, 2, lat_grid.shape)

plot.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Custom Wind Barb Styling",
    xlabel="Longitude",
    ylabel="Latitude",
    **custom_kwargs
)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('custom_barb_styling.png')
```

## Performance Considerations

### Large Wind Fields

```python
import numpy as np
from monet_plots import WindQuiverPlot

# Handle large wind fields efficiently
n_lat, n_lon = 50, 60  # 50x60 grid = 3000 points

# Create large wind field
lat = np.linspace(30, 50, n_lat)
lon = np.linspace(-120, -70, n_lon)
lon_grid, lat_grid = np.meshgrid(lon, lat)

u = np.random.normal(5, 3, lat_grid.shape)
v = np.random.normal(-2, 2, lat_grid.shape)
wind_speed = np.sqrt(u**2 + v**2)

# Downsample for display
downsample_factor = 3
u_display = u[::downsample_factor, ::downsample_factor]
v_display = v[::downsample_factor, ::downsample_factor]
lon_display = lon_grid[::downsample_factor, ::downsample_factor]
lat_display = lat_grid[::downsample_factor, ::downsample_factor]

plot = WindQuiverPlot(figsize=(14, 10))

plot.plot(
    u_display, v_display,
    x=lon_display,
    y=lat_display,
    title="Large Wind Field (Downsampled)",
    xlabel="Longitude",
    ylabel="Latitude",
    color=wind_speed[::downsample_factor, ::downsample_factor],
    cmap='viridis',
    scale=80,
    scale_units='xy',
    angles='xy',
    alpha=0.8
)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('large_wind_field.png')
```

### Memory Management

```python
from monet_plots import WindQuiverPlot

# Process multiple wind fields efficiently
time_steps = 10

for t in range(time_steps):
    # Generate wind field for this time step
    lat = np.linspace(30, 50, 12)
    lon = np.linspace(-120, -70, 16)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Time-varying wind field
    u = np.random.normal(5 + 2*np.sin(t*0.5), 3, lat_grid.shape)
    v = np.random.normal(-2 + np.cos(t*0.3), 2, lat_grid.shape)
    
    plot = WindQuiverPlot(figsize=(12, 10))
    
    plot.plot(
        u, v,
        x=lon_grid,
        y=lat_grid,
        title=f"Wind Field - Time Step {t}",
        xlabel="Longitude",
        ylabel="Latitude",
        color=np.sqrt(u**2 + v**2),
        cmap='viridis',
        scale=80,
        scale_units='xy',
        angles='xy',
        alpha=0.8
    )
    
    plot.save(f'wind_field_t{t:03d}.png')
    plot.close()  # Free memory
```

## Common Issues and Solutions

### Wind Coordinate Systems

```python
import numpy as np
import pandas as pd
from monet_plots import WindQuiverPlot

# Handle different coordinate systems
def convert_wind_coordinates(u, v, from_crs='math', to_crs='meteo'):
    """Convert wind component coordinate systems.
    
    Args:
        u: U-component
        v: V-component
        from_crs: 'math' (0° = East, +90° = North) or 'meteo' (0° = North, +90° = East)
        to_crs: Target coordinate system
        
    Returns:
        tuple: (u_new, v_new) converted components
    """
    if from_crs == to_crs:
        return u, v
    
    if from_crs == 'math' and to_crs == 'meteo':
        # Rotate 90° clockwise
        u_new = v
        v_new = -u
    elif from_crs == 'meteo' and to_crs == 'math':
        # Rotate 90° counterclockwise
        u_new = -v
        v_new = u
    
    return u_new, v_new

# Example usage
lat = np.linspace(30, 50, 10)
lon = np.linspace(-120, -70, 14)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Generate wind in mathematical coordinates (u=east, v=north)
u_math = np.random.normal(5, 3, lat_grid.shape)
v_math = np.random.normal(-2, 2, lat_grid.shape)

# Convert to meteorological coordinates (u=north, v=east)
u_meteo, v_meteo = convert_wind_coordinates(u_math, v_math, 'math', 'meteo')

plot = WindQuiverPlot(figsize=(12, 10))

plot.plot(
    u_meteo, v_meteo,
    x=lon_grid,
    y=lat_grid,
    title="Wind in Meteorological Coordinate System",
    xlabel="Longitude",
    ylabel="Latitude",
    color=np.sqrt(u_meteo**2 + v_meteo**2),
    cmap='viridis',
    scale=80,
    scale_units='xy',
    angles='xy',
    alpha=0.8
)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('meteorological_coordinates.png')
```

### Wind Speed Limits

```python
import numpy as np
from monet_plots import WindQuiverPlot

# Handle extreme wind speeds
lat = np.linspace(30, 50, 12)
lon = np.linspace(-120, -70, 16)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Generate wind data with some extreme values
u = np.random.normal(5, 10, lat_grid.shape)  # Large variance
v = np.random.normal(-2, 8, lat_grid.shape)

# Cap extreme wind speeds
max_wind_speed = 50  # m/s
wind_speed = np.sqrt(u**2 + v**2)
exceed_mask = wind_speed > max_wind_speed

if np.any(exceed_mask):
    # Scale down extreme values
    scale_factor = max_wind_speed / wind_speed[exceed_mask]
    u[exceed_mask] *= scale_factor
    v[exceed_mask] *= scale_factor

plot = WindQuiverPlot(figsize=(12, 10))

plot.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Wind Speeds Capped at 50 m/s",
    xlabel="Longitude",
    ylabel="Latitude",
    color=np.sqrt(u**2 + v**2),
    cmap='viridis',
    scale=80,
    scale_units='xy',
    angles='xy',
    alpha=0.8
)

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('capped_wind_speeds.png')
```

### Data Quality Checks

```python
import numpy as np
from monet_plots import WindQuiverPlot

# Implement data quality checks
lat = np.linspace(30, 50, 12)
lon = np.linspace(-120, -70, 16)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Generate wind data with quality issues
u = np.random.normal(5, 3, lat_grid.shape)
v = np.random.normal(-2, 2, lat_grid.shape)

# Add data quality issues
u[2:4, 5:8] = np.nan  # Missing data
v[6:8, 2:4] = np.nan
u[10, 10] = 999  # Obvious error
v[10, 10] = 999

# Quality checks
def check_wind_quality(u, v):
    """Check wind data quality and return mask."""
    wind_speed = np.sqrt(u**2 + v**2)
    
    # Create quality mask
    quality_mask = np.ones_like(u, dtype=bool)
    
    # Check for missing values
    quality_mask &= ~np.isnan(u) & ~np.isnan(v)
    
    # Check for extreme values
    quality_mask &= (wind_speed < 100) & (wind_speed >= 0)
    
    # Check for reasonable wind direction (not too close to singularities)
    quality_mask &= (np.abs(u) < 1000) & (np.abs(v) < 1000)
    
    return quality_mask, wind_speed

# Check quality
quality_mask, wind_speed = check_wind_quality(u, v)

# Apply quality mask
u_clean = np.where(quality_mask, u, np.nan)
v_clean = np.where(quality_mask, v, np.nan)

plot = WindQuiverPlot(figsize=(12, 10))

plot.plot(
    u_clean, v_clean,
    x=lon_grid,
    y=lat_grid,
    title="Wind Data Quality Check",
    xlabel="Longitude",
    ylabel="Latitude",
    color=wind_speed,
    cmap='viridis',
    scale=80,
    scale_units='xy',
    angles='xy',
    alpha=0.8
)

# Mark poor quality data points
poor_quality = ~quality_mask
if np.any(poor_quality):
    plot.ax.plot(lon_grid[poor_quality], lat_grid[poor_quality], 
                'rx', markersize=8, markeredgewidth=2, label='Poor Quality')
    plot.ax.legend()

plot.xlabel("Longitude")
plot.ylabel("Latitude")
plot.save('quality_checked_wind.png')
```

---

**Related Resources**:

- [API Reference](../api) - Core functionality and utilities
- [Examples](../examples) - Practical usage examples
- [Style Configuration](../api/style) - Plot styling options
- [Spatial Plots](spatial) - Geospatial plotting capabilities
