# KDE Plots

Kernel Density Estimation (KDE) plots visualize the probability density function of continuous variables. MONET Plots provides enhanced KDE plotting capabilities with multiple visualization modes, statistical analysis, and publication-ready styling.

## Overview

KDE plots are essential for understanding data distribution, identifying patterns, and comparing multiple datasets. MONET Plots builds on seaborn's KDE functionality, offering advanced features for scientific visualization.

| Feature | Description |
|---------|-------------|
| **Multiple KDE modes** | Univariate and bivariate density estimation |
| **Statistical overlays** | Mean, median, and confidence intervals |
| **Multiple distributions** | Easy comparison of multiple datasets |
| **Custom bandwidth** | Adjustable smoothing parameters |
| **Publication styling** | Professional formatting for journals |

## KDEPlot Class

`KDEPlot` creates kernel density estimation plots with advanced statistical visualization capabilities.

### Class Signature

```python
class KDEPlot(BasePlot):
    """Creates a kernel density estimation plot.
    
    This class creates KDE plots for univariate and bivariate
    data analysis with statistical overlays.
    """
    
    def __init__(self, **kwargs):
        """Initialize the KDE plot.
        
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

#### `plot(data, x=None, y=None, title=None, label=None, **kwargs)`

Plot kernel density estimation.

```python
def plot(self, data, x=None, y=None, title=None, label=None, **kwargs):
    """Plot the kernel density estimation.
    
    Args:
        data (pandas.DataFrame): DataFrame containing the data
        x (str, optional): Column name for x-axis. Defaults to None
        y (str, optional): Column name for y-axis. Defaults to None
        title (str, optional): Plot title. Defaults to None
        label (str, optional): Legend label. Defaults to None
        **kwargs: Additional keyword arguments for KDE plotting
    """
    pass
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pandas.DataFrame` | Required | DataFrame with KDE data |
| `x` | `str` | `None` | Column name for x-axis (univariate if None) |
| `y` | `str` | `None` | Column name for y-axis (bivariate if specified) |
| `title` | `str` | `None` | Plot title |
| `label` | `str` | `None` | Legend label |
| `**kwargs` | `dict` | `{}` | Additional seaborn KDE parameters |

**Returns:**
- `matplotlib.axes.Axes`: The axes object with the plot

**Example:**
```python
import pandas as pd
import numpy as np
from monet_plots import KDEPlot

# Create KDE plot
plot = KDEPlot(figsize=(12, 8))

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'values': np.random.normal(0, 1, 1000),
    'category': np.random.choice(['Group A', 'Group B'], 1000)
})

# Plot KDE
plot.plot(
    data,
    x='values',
    title="Kernel Density Estimation",
    label="All Data",
    fillargs={'alpha': 0.7, 'color': 'blue'}
)

plot.xlabel("Value")
plot.ylabel("Density")
plot.save('kde_basic.png')
plot.close()
```

### Common Usage Patterns

#### Univariate KDE Plot

```python
import pandas as pd
import numpy as np
from monet_plots import KDEPlot

# Create sample data
np.random.seed(123)
data = pd.DataFrame({
    'measurement': np.concatenate([
        np.random.normal(0, 1, 500),    # Normal distribution
        np.random.normal(3, 1.5, 300),  # Different normal
        np.random.exponential(1, 200)   # Exponential component
    ])
})

# Create univariate KDE plot
plot = KDEPlot(figsize=(12, 8))

plot.plot(
    data,
    x='measurement',
    title="Univariate Kernel Density Estimation",
    fillargs={'alpha': 0.6, 'color': 'navy'},
    lineargs={'linewidth': 2, 'color': 'darkblue'}
)

# Add statistical markers
mean_val = data['measurement'].mean()
median_val = data['measurement'].median()

plot.ax.axvline(mean_val, color='red', linestyle='--',
               label=f'Mean ({mean_val:.2f})')
plot.ax.axvline(median_val, color='green', linestyle=':',
               label=f'Median ({median_val:.2f})')

plot.xlabel("Measurement Value")
plot.ylabel("Density")
plot.legend()
plot.save('univariate_kde.png')
```

#### Bivariate KDE Plot

```python
import pandas as pd
import numpy as np
from monet_plots import KDEPlot

# Create bivariate data
np.random.seed(456)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]  # Correlated variables
data = np.random.multivariate_normal(mean, cov, 1000)

df = pd.DataFrame({
    'x': data[:, 0],
    'y': data[:, 1],
    'group': np.random.choice(['Type 1', 'Type 2'], 1000)
})

# Create bivariate KDE plot
plot = KDEPlot(figsize=(14, 10))

plot.plot(
    df,
    x='x',
    y='y',
    title="Bivariate Kernel Density Estimation",
    fillargs={'alpha': 0.6, 'cmap': 'viridis'},
    contourargs={'levels': 10, 'colors': 'black', 'alpha': 0.4}
)

plot.xlabel("X Variable")
plot.ylabel("Y Variable")
plot.save('bivariate_kde.png')
```

#### Multiple Distribution Comparison

```python
import pandas as pd
import numpy as np
from monet_plots import KDEPlot

# Create multiple distributions
np.random.seed(789)
data = pd.DataFrame({
    'value': np.concatenate([
        np.random.normal(-2, 1, 300),   # Distribution 1
        np.random.normal(0, 1.5, 400),  # Distribution 2
        np.random.normal(2, 1, 300)    # Distribution 3
    ]),
    'distribution': (['Dist 1'] * 300 +
                    ['Dist 2'] * 400 +
                    ['Dist 3'] * 300)
})

# Create comparison plot
plot = KDEPlot(figsize=(14, 8))

colors = ['blue', 'red', 'green']
for dist, color in zip(['Dist 1', 'Dist 2', 'Dist 3'], colors):
    subset = data[data['distribution'] == dist]
    
    plot.plot(
        subset,
        x='value',
        label=dist,
        fillargs={'alpha': 0.4, 'color': color},
        lineargs={'linewidth': 2, 'color': color}
    )

plot.title("Multiple Distribution Comparison")
plot.xlabel("Value")
plot.ylabel("Density")
plot.legend()
plot.save('multiple_kde_comparison.png')
```

#### KDE with Histogram Overlay

```python
import pandas as pd
import numpy as np
from monet_plots import KDEPlot

# Create sample data
np.random.seed(101)
data = pd.DataFrame({
    'values': np.random.gamma(2, 2, 1000)  # Gamma distribution
})

# Create KDE with histogram
plot = KDEPlot(figsize=(12, 8))

# Plot histogram
plot.ax.hist(data['values'], bins=30, density=True, alpha=0.3,
            color='gray', label='Histogram')

# Plot KDE
plot.plot(
    data,
    x='values',
    title="KDE with Histogram Overlay",
    fillargs={'alpha': 0.7, 'color': 'blue'},
    lineargs={'linewidth': 2, 'color': 'darkblue'},
    label='KDE'
)

plot.xlabel("Value")
plot.ylabel("Density")
plot.legend()
plot.save('kde_histogram_overlay.png')
```

## Advanced Features

### Statistical Overlays

```python
import pandas as pd
import numpy as np
from scipy import stats
from monet_plots import KDEPlot

# Create realistic data
np.random.seed(202)
data = pd.DataFrame({
    'measurements': np.random.normal(10, 2, 1000)
})

# Create KDE with statistical overlays
plot = KDEPlot(figsize=(14, 10))

plot.plot(
    data,
    x='measurements',
    title="KDE with Statistical Analysis",
    fillargs={'alpha': 0.6, 'color': 'lightblue'}
)

# Calculate statistics
mean_val = data['measurements'].mean()
median_val = data['measurements'].median()
std_val = data['measurements'].std()
q25 = data['measurements'].quantile(0.25)
q75 = data['measurements'].quantile(0.75)

# Add statistical markers
plot.ax.axvline(mean_val, color='red', linestyle='--',
               label=f'Mean ({mean_val:.2f})')
plot.ax.axvline(median_val, color='green', linestyle='--',
               label=f'Median ({median_val:.2f})')
plot.ax.axvline(q25, color='orange', linestyle=':',
               label=f'Q1 ({q25:.2f})')
plot.ax.axvline(q75, color='purple', linestyle=':',
               label=f'Q3 ({q75:.2f})')

# Add standard deviation bands
plot.ax.axvspan(mean_val - std_val, mean_val + std_val,
               alpha=0.2, color='red', label='±1σ')
plot.ax.axvspan(mean_val - 2*std_val, mean_val + 2*std_val,
               alpha=0.1, color='red', label='±2σ')

plot.xlabel("Measurement")
plot.ylabel("Density")
plot.legend()
plot.save('kde_statistical_overlays.png')
```

### Interactive KDE Analysis

```python
import matplotlib.pyplot as plt
from monet_plots import KDEPlot

# Enable interactive mode
plt.ion()

# Create interactive KDE plot
plot = KDEPlot(figsize=(12, 8))

# Generate data
import numpy as np
import pandas as pd
np.random.seed(42)
data = pd.DataFrame({
    'values': np.random.normal(0, 1, 1000)
})

# Initial plot
plot.plot(
    data,
    x='values',
    title="Interactive KDE Analysis",
    fillargs={'alpha': 0.6, 'color': 'blue'}
)

# Interactive parameter adjustment
def on_key(event):
    if event.key == 'up':
        # Increase bandwidth
        pass  # Implementation would update KDE bandwidth
    elif event.key == 'down':
        # Decrease bandwidth
        pass

plot.fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
```

## Data Requirements

### Input Data Format

KDE plots work best with continuous data:

```python
import pandas as pd
import numpy as np

# Univariate data
df = pd.DataFrame({
    'values': np.random.normal(0, 1, 1000)
})

# Bivariate data
df = pd.DataFrame({
    'x': np.random.normal(0, 1, 1000),
    'y': np.random.normal(0, 1, 1000)
})

# Labeled data for comparison
df = pd.DataFrame({
    'measurement': np.random.normal(0, 1, 1000),
    'group': np.random.choice(['A', 'B', 'C'], 1000)
})
```

### Data Preprocessing

```python
import pandas as pd
import numpy as np
from monet_plots import KDEPlot

# Handle data quality issues
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
data[100:105] = np.nan  # Introduce missing values
data[200] = 999  # Add outlier

df = pd.DataFrame({
    'values': data,
    'quality': np.random.choice(['Good', 'Poor'], 1000)
})

# Remove outliers and missing values
df_clean = df.dropna()
df_clean = df_clean[np.abs(df_clean['values']) < 5]  # Remove extreme values

plot = KDEPlot(figsize=(12, 8))

plot.plot(
    df_clean,
    x='values',
    title="Cleaned Data for KDE",
    fillargs={'alpha': 0.6, 'color': 'green'}
)

plot.xlabel("Value")
plot.ylabel("Density")
plot.save('cleaned_kde.png')
```

## Customization Options

### Bandwidth Adjustment

```python
from monet_plots import KDEPlot

plot = KDEPlot(figsize=(12, 8))

# Different bandwidth comparisons
bandwidths = [0.1, 0.5, 1.0]
colors = ['red', 'blue', 'green']

for bw, color in zip(bandwidths, colors):
    plot.plot(
        data,
        x='values',
        title=f"KDE with Bandwidth {bw}",
        fillargs={'alpha': 0.4, 'color': color},
        lineargs={'linewidth': 2, 'color': color},
        label=f'bw={bw}'
    )

plot.xlabel("Value")
plot.ylabel("Density")
plot.legend()
plot.save('kde_bandwidth_comparison.png')
```

### Color and Style Customization

```python
from monet_plots import KDEPlot

plot = KDEPlot(figsize=(12, 8))

# Custom styling
custom_fillargs = {
    'alpha': 0.7,
    'color': 'teal',
    'edgecolor': 'darkblue',
    'linewidth': 2
}

custom_lineargs = {
    'linewidth': 3,
    'color': 'navy',
    'linestyle': '--'
}

plot.plot(
    data,
    x='values',
    title="Custom Styled KDE",
    fillargs=custom_fillargs,
    lineargs=custom_lineargs
)

plot.xlabel("Value")
plot.ylabel("Density")
plot.save('custom_styled_kde.png')
```

## Performance Considerations

### Large Datasets

```python
import pandas as pd
import numpy as np
from monet_plots import KDEPlot

# Handle large datasets efficiently
n_samples = 100000  # 100,000 samples
df_large = pd.DataFrame({
    'values': np.random.normal(0, 1, n_samples)
})

# Downsample for KDE plotting (KDE is O(n²))
df_sampled = df_large.sample(n=10000, random_state=42)

plot = KDEPlot(figsize=(12, 8))

plot.plot(
    df_sampled,
    x='values',
    title="Large Dataset (Sampled)",
    fillargs={'alpha': 0.6, 'color': 'purple'}
)

plot.xlabel("Value")
plot.ylabel("Density")
plot.save('large_dataset_kde.png')
```

### Memory Management

```python
from monet_plots import KDEPlot

# Process multiple KDE plots efficiently
categories = df['group'].unique()

for category in categories:
    plot = KDEPlot(figsize=(10, 6))
    
    subset = df[df['group'] == category]
    
    plot.plot(
        subset,
        x='values',
        title=f"KDE - {category}",
        fillargs={'alpha': 0.6, 'color': 'blue'}
    )
    
    plot.xlabel("Value")
    plot.ylabel("Density")
    plot.save(f'kde_{category.lower()}.png')
    plot.close()  # Free memory
```

## Common Issues and Solutions

### Bandwidth Selection

```python
from monet_plots import KDEPlot
import numpy as np
from scipy.stats import gaussian_kde

# Optimal bandwidth selection
plot = KDEPlot(figsize=(12, 8))

# Try different bandwidth selection methods
bandwidth_methods = ['scott', 'silverman', 'isj']
colors = ['blue', 'red', 'green']

data = pd.DataFrame({'values': np.random.normal(0, 1, 1000)})

for method, color in zip(bandwidth_methods, colors):
    plot.plot(
        data,
        x='values',
        title=f"Bandwidth Method: {method}",
        fillargs={'alpha': 0.4, 'color': color},
        lineargs={'linewidth': 2, 'color': color},
        label=method
    )

plot.xlabel("Value")
plot.ylabel("Density")
plot.legend()
plot.save('kde_bandwidth_methods.png')
```

### Boundary Effects

```python
from monet_plots import KDEPlot

# Handle boundary effects in bounded data
plot = KDEPlot(figsize=(12, 8))

# Create bounded data (e.g., 0-100 values)
bounded_data = pd.DataFrame({
    'values': np.random.uniform(0, 100, 1000)
})

# Use transform to handle boundaries
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
transformed_data = pt.fit_transform(bounded_data[['values']])

plot.plot(
    pd.DataFrame({'transformed': transformed_data.flatten()}),
    x='transformed',
    title="KDE with Boundary Handling",
    fillargs={'alpha': 0.6, 'color': 'orange'}
)

plot.xlabel("Transformed Value")
plot.ylabel("Density")
plot.save('kde_boundary_handling.png')
```

### Multiple Comparison Issues

```python
from monet_plots import KDEPlot

# Avoid overplotting with many distributions
plot = KDEPlot(figsize=(14, 8))

# Create many distributions
n_distributions = 6
data = []
for i in range(n_distributions):
    values = np.random.normal(i, 0.5, 200)
    for val in values:
        data.append({'value': val, 'dist': f'Dist {i+1}'})

df = pd.DataFrame(data)

# Use different alpha levels to reduce overlap
colors = plt.cm.tab10(np.linspace(0, 1, n_distributions))
for i, (dist, color) in enumerate(zip([f'Dist {j+1}' for j in range(n_distributions)], colors)):
    subset = df[df['dist'] == dist]
    alpha = 0.3 + (0.5 * (1 - i/n_distributions))  # Decreasing alpha
    
    plot.plot(
        subset,
        x='value',
        label=dist,
        fillargs={'alpha': alpha, 'color': color},
        lineargs={'linewidth': 1.5, 'color': color}
    )

plot.title("Multiple Distributions with Reduced Overlap")
plot.xlabel("Value")
plot.ylabel("Density")
plot.legend()
plot.save('kde_multiple_comparison.png')
```

---

**Related Resources**:

- [API Reference](../api) - Core functionality and utilities
- [Examples](../examples) - Practical usage examples
- [Style Configuration](../api/style) - Plot styling options
- [Statistical Analysis](../api/taylordiagram) - Statistical utilities
