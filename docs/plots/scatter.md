# Scatter Plots

Scatter plots visualize the relationship between two continuous variables. MONET Plots provides enhanced scatter plotting capabilities with regression analysis, confidence intervals, and publication-ready styling.

## Overview

The scatter plot functionality in MONET Plots builds on seaborn's regplot functionality, adding statistical analysis, confidence bands, and professional styling for scientific publications.

| Feature | Description |
|---------|-------------|
| Regression lines | Linear regression with confidence intervals |
| Statistical analysis | Correlation and significance testing |
| Multiple series | Support for multiple scatter series |
| Custom styling | Publication-ready formatting |

## ScatterPlot Class

`ScatterPlot` creates scatter plots with regression lines and statistical analysis.

### Class Signature

```python
class ScatterPlot(BasePlot):
    """Creates a scatter plot with a regression line.
    
    This class creates a scatter plot with a regression line
    and confidence intervals.
    """
    
    def __init__(self, **kwargs):
        """Initialize the scatter plot.
        
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

#### `plot(df, x, y, title=None, label=None, **kwargs)`

Plot scatter data with regression line.

```python
def plot(self, df, x, y, title=None, label=None, **kwargs):
    """Plot the scatter data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        title (str, optional): Plot title. Defaults to None
        label (str, optional): Legend label. Defaults to None
        **kwargs: Additional keyword arguments to pass to `regplot`
    """
    pass
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pandas.DataFrame` | Required | DataFrame with scatter data |
| `x` | `str` | Required | Column name for x-axis |
| `y` | `str` | Required | Column name for y-axis |
| `title` | `str` | `None` | Plot title |
| `label` | `str` | `None` | Legend label |
| `**kwargs` | `dict` | `{}` | Additional seaborn regplot parameters |

**Returns:**
- `matplotlib.axes.Axes`: The axes object with the plot

**Example:**
```python
import pandas as pd
import numpy as np
from monet_plots import ScatterPlot

# Create scatter plot
plot = ScatterPlot(figsize=(10, 8))

# Generate sample data
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 2 * x + np.random.normal(0, 0.5, 100)  # Linear relationship with noise

df = pd.DataFrame({
    'x_variable': x,
    'y_variable': y,
    'category': ['A'] * 50 + ['B'] * 50
})

# Plot scatter with regression
plot.plot(
    df,
    x='x_variable',
    y='y_variable',
    title="Scatter Plot with Regression Line",
    label="Data Points"
)

plot.xlabel("X Variable")
plot.ylabel("Y Variable")
plot.save('scatter_basic.png')
```

### Common Usage Patterns

#### Basic Scatter Plot with Regression

```python
import pandas as pd
import numpy as np
from monet_plots import ScatterPlot

# Create sample data
np.random.seed(123)
x = np.random.uniform(0, 10, 150)
y = 3 + 2 * x + np.random.normal(0, 1.5, 150)  # Linear relationship

df = pd.DataFrame({
    'independent': x,
    'dependent': y,
    'group': np.random.choice(['Group 1', 'Group 2'], 150)
})

# Create plot
plot = ScatterPlot(figsize=(12, 9))

plot.plot(
    df,
    x='independent',
    y='dependent',
    title="Scatter Plot with Linear Regression",
    label="All Data",
    plotargs={
        'scatter_kws': {'alpha': 0.6, 's': 50},
        'line_kws': {'linewidth': 2, 'color': 'red'}
    }
)

plot.xlabel("Independent Variable")
plot.ylabel("Dependent Variable")
plot.save('scatter_regression.png')
```

#### Multiple Scatter Series

```python
import pandas as pd
import numpy as np
from monet_plots import ScatterPlot

# Create multiple series data
np.random.seed(456)
data = []

for group in ['Group A', 'Group B', 'Group C']:
    x = np.random.uniform(0, 10, 50)
    # Different relationships for each group
    if group == 'Group A':
        y = 2 + 1.5 * x + np.random.normal(0, 0.8, 50)
    elif group == 'Group B':
        y = 1 + 2 * x + np.random.normal(0, 1, 50)
    else:
        y = 3 + x + np.random.normal(0, 1.2, 50)
    
    for xi, yi in zip(x, y):
        data.append({'x': xi, 'y': yi, 'group': group})

df = pd.DataFrame(data)

# Create plot with multiple series
plot = ScatterPlot(figsize=(14, 10))

# Plot each series with different colors
colors = ['blue', 'red', 'green']
for i, (group, color) in enumerate(zip(['Group A', 'Group B', 'Group C'], colors)):
    subset = df[df['group'] == group]
    
    plot.plot(
        subset,
        x='x',
        y='y',
        label=group,
        plotargs={
            'scatter_kws': {'alpha': 0.6, 's': 60, 'color': color},
            'line_kws': {'linewidth': 2, 'color': color},
            'ci': 95  # 95% confidence interval
        }
    )

plot.title("Multiple Scatter Series with Regression Lines")
plot.xlabel("X Variable")
plot.ylabel("Y Variable")
plot.legend()
plot.save('multiple_scatter_series.png')
```

#### Custom Regression Analysis

```python
import pandas as pd
import numpy as np
from scipy import stats
from monet_plots import ScatterPlot

# Create data with nonlinear relationship
np.random.seed(789)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)  # Sine wave with noise

df = pd.DataFrame({
    'x': x,
    'y': y,
    'quadrant': np.select(
        [x < 2.5, (x >= 2.5) & (x < 7.5), x >= 7.5],
        ['Early', 'Middle', 'Late']
    )
})

# Create plot
plot = ScatterPlot(figsize=(12, 9))

# Plot overall relationship
plot.plot(
    df,
    x='x',
    y='y',
    title="Nonlinear Relationship with Quadrant Analysis",
    label="All Data",
    plotargs={
        'scatter_kws': {'alpha': 0.4, 's': 40},
        'line_kws': {'linewidth': 2, 'color': 'black'}
    }
)

# Add quadratic fit (custom analysis)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Fit polynomial regression
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
model.fit(df[['x']], df['y'])

# Plot fitted curve
x_fit = np.linspace(0, 10, 200)
y_fit = model.predict(x_fit.reshape(-1, 1))
plot.ax.plot(x_fit, y_fit, 'r--', linewidth=2, label='Quadratic Fit')

# Add statistical information
r_squared = stats.pearsonr(df['x'], df['y'])[0]**2
plot.ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
            transform=plot.ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plot.xlabel("X Variable")
plot.ylabel("Y Variable")
plot.legend()
plot.save('nonlinear_scatter.png')
```

## Advanced Features

### Statistical Analysis Integration

```python
import pandas as pd
import numpy as np
from scipy import stats
from monet_plots import ScatterPlot

# Create realistic data
np.random.seed(101)
n_samples = 200
x = np.random.normal(5, 2, n_samples)
y = 1.5 + 0.8 * x + np.random.normal(0, 1.5, n_samples)

df = pd.DataFrame({
    'x': x,
    'y': y,
    'quality': np.random.choice(['High', 'Medium', 'Low'], n_samples)
})

# Create plot with statistical analysis
plot = ScatterPlot(figsize=(14, 10))

# Plot with quality-based coloring
quality_colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
for quality, color in quality_colors.items():
    subset = df[df['quality'] == quality]
    
    plot.plot(
        subset,
        x='x',
        y='y',
        label=f'{quality} Quality',
        plotargs={
            'scatter_kws': {'alpha': 0.6, 's': 50, 'color': color},
            'line_kws': {'linewidth': 2, 'color': color},
            'ci': None  # No confidence interval for individual groups
        }
    )

# Calculate and display statistics
correlation = stats.pearsonr(df['x'], df['y'])
slope, intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])

# Add statistical summary
stats_text = f"""Statistical Summary:
Correlation: {correlation[0]:.3f} (p = {correlation[1]:.3f})
Slope: {slope:.3f} ± {std_err:.3f}
R²: {r_value**2:.3f}
p-value: {p_value:.3f}"""

plot.ax.text(0.02, 0.98, stats_text,
            transform=plot.ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plot.title("Statistical Analysis of Scatter Plot")
plot.xlabel("Independent Variable")
plot.ylabel("Dependent Variable")
plot.legend()
plot.save('statistical_scatter.png')
```

### Residual Analysis

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from monet_plots import ScatterPlot

# Create data with heteroscedasticity
np.random.seed(202)
x = np.random.uniform(1, 10, 150)
y = 2 + 1.5 * x + np.random.normal(0, 0.5 * x, 150)  # Increasing variance

df = pd.DataFrame({'x': x, 'y': y})

# Fit linear model
model = LinearRegression()
model.fit(df[['x']], df['y'])
df['predicted'] = model.predict(df[['x']])
df['residuals'] = df['y'] - df['predicted']

# Create residual plot
plot = ScatterPlot(figsize=(12, 10))

# Plot residuals
plot.plot(
    df,
    x='predicted',
    y='residuals',
    title="Residual Plot",
    label="Residuals"
)

# Add horizontal line at y=0
plot.ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Add trend line for residuals
residual_model = LinearRegression()
residual_model.fit(df[['predicted']], df['residuals'])
if abs(residual_model.coef_[0]) > 0.01:  # Only show if significant trend
    x_range = np.linspace(df['predicted'].min(), df['predicted'].max(), 100)
    y_trend = residual_model.predict(x_range.reshape(-1, 1))
    plot.ax.plot(x_range, y_trend, 'orange', linewidth=2, label='Residual Trend')

plot.xlabel("Predicted Values")
plot.ylabel("Residuals")
plot.legend()
plot.save('residual_analysis.png')
```

### Interactive Scatter Plot

```python
import matplotlib.pyplot as plt
from monet_plots import ScatterPlot

# Enable interactive mode
plt.ion()

# Create interactive scatter plot
plot = ScatterPlot(figsize=(12, 8))

# Generate data
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = x + np.random.normal(0, 0.5, 100)

df = pd.DataFrame({'x': x, 'y': y})

# Initial plot
plot.plot(
    df,
    x='x',
    y='y',
    title="Interactive Scatter Plot",
    label="Click to add points"
)

# Click interaction
def on_click(event):
    if event.inaxes == plot.ax and event.button == 1:  # Left click
        # Add point at clicked location
        plot.ax.plot(event.xdata, event.ydata, 'go', markersize=8)
        plot.fig.canvas.draw()

plot.fig.canvas.mpl_connect('button_press_event', on_click)

# Right click to remove last point
def on_right_click(event):
    if event.inaxes == plot.ax and event.button == 3:  # Right click
        lines = plot.ax.lines
        if len(lines) > 1:  # Keep regression line
            lines[-2].remove()  # Remove last scatter point
            plot.fig.canvas.draw()

plot.fig.canvas.mpl_connect('button_press_event', on_right_click)

plt.show()
```

## Data Requirements

### Input Data Format

Scatter plots require pandas DataFrames with numeric columns:

```python
import pandas as pd
import numpy as np

# Basic format
df = pd.DataFrame({
    'x': np.random.normal(0, 1, 100),
    'y': np.random.normal(0, 1, 100)
})

# With categories
df = pd.DataFrame({
    'x': np.random.uniform(0, 10, 150),
    'y': np.random.uniform(0, 10, 150),
    'group': np.random.choice(['A', 'B', 'C'], 150)
})

# With weights
df = pd.DataFrame({
    'x': np.random.normal(5, 2, 200),
    'y': np.random.normal(5, 2, 200),
    'weight': np.random.uniform(0.1, 1.0, 200)
})
```

### Data Preprocessing

```python
import pandas as pd
import numpy as np
from monet_plots import ScatterPlot

# Handle missing values
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = x + np.random.normal(0, 0.5, 100)

# Introduce missing values
y[10:15] = np.nan
y[45:50] = np.nan

df = pd.DataFrame({
    'x': x,
    'y': y,
    'category': np.random.choice(['A', 'B'], 100)
})

# Remove rows with missing values
df_clean = df.dropna()

plot = ScatterPlot(figsize=(12, 8))

# Plot with missing values marked
plot.plot(
    df_clean,
    x='x',
    y='y',
    title="Scatter Plot with Missing Values Handled",
    label="Clean Data"
)

# Mark missing values
missing_mask = df['y'].isna()
plot.ax.plot(df.loc[missing_mask, 'x'], 
            np.full_like(df.loc[missing_mask].index, df['y'].min() - 1), 
            'rx', markersize=8, label='Missing Values')

plot.legend()
plot.save('missing_values_scatter.png')
```

## Customization Options

### Regression Line Styling

```python
from monet_plots import ScatterPlot

plot = ScatterPlot(figsize=(12, 9))

# Custom regression styling
custom_plotargs = {
    'scatter_kws': {
        'alpha': 0.7,
        's': 80,
        'facecolors': 'none',
        'edgecolors': 'blue',
        'linewidths': 1.5
    },
    'line_kws': {
        'linewidth': 3,
        'color': 'red',
        'linestyle': '--'
    },
    'ci': 95,  # 95% confidence interval
    'order': 1  # Linear regression
}

plot.plot(
    df,
    x='x',
    y='y',
    title="Custom Regression Styling",
    label="Custom Styled Data",
    **custom_plotargs
)

plot.save('custom_regression_styling.png')
```

### Multiple Series Plotting

```python
from monet_plots import ScatterPlot

plot = ScatterPlot(figsize=(14, 10))

# Plot multiple series with different styles
series_data = [
    (df[df['group'] == 'A'], 'blue', 'Group A', {'marker': 'o'}),
    (df[df['group'] == 'B'], 'red', 'Group B', {'marker': 's'}),
    (df[df['group'] == 'C'], 'green', 'Group C', {'marker': '^'})
]

for subset, color, label, marker_style in series_data:
    plot.plot(
        subset,
        x='x',
        y='y',
        label=label,
        plotargs={
            'scatter_kws': {
                'alpha': 0.6,
                's': 60,
                'color': color,
                **marker_style
            },
            'line_kws': {
                'linewidth': 2,
                'color': color
            }
        }
    )

plot.title("Multiple Series with Different Markers")
plot.xlabel("X Variable")
plot.ylabel("Y Variable")
plot.legend()
plot.save('multiple_series_markers.png')
```

### Confidence Interval Customization

```python
from monet_plots import ScatterPlot

plot = ScatterPlot(figsize=(12, 9))

# Custom confidence interval
plot.plot(
    df,
    x='x',
    y='y',
    title="Custom Confidence Intervals",
    label="95% CI",
    plotargs={
        'scatter_kws': {'alpha': 0.6, 's': 50},
        'line_kws': {'linewidth': 2, 'color': 'blue'},
        'ci': 99,  # 99% confidence interval
        'scatter': False  # Show only confidence band
    )

# Add 95% CI as comparison
plot.ax.plot(df['x'], df['y'], 'ro', alpha=0.3, label='Data Points')
plot.legend()
plot.save('custom_confidence_intervals.png')
```

## Performance Considerations

### Large Datasets

```python
import pandas as pd
import numpy as np
from monet_plots import ScatterPlot

# Handle large datasets
n_points = 100000  # 100,000 points
df = pd.DataFrame({
    'x': np.random.normal(0, 1, n_points),
    'y': np.random.normal(0, 1, n_points),
    'category': np.random.choice(['A', 'B', 'C'], n_points)
})

# Downsample for plotting
df_sampled = df.sample(n=5000, random_state=42)  # Sample 5,000 points

plot = ScatterPlot(figsize=(12, 9))

plot.plot(
    df_sampled,
    x='x',
    y='y',
    title="Large Dataset (Sampled)",
    label="Sampled Data"
)

plot.save('large_dataset_scatter.png')
```

### Memory Management

```python
from monet_plots import ScatterPlot

# Process multiple scatter plots efficiently
categories = df['category'].unique()

for category in categories:
    plot = ScatterPlot(figsize=(10, 8))
    
    subset = df[df['category'] == category]
    
    plot.plot(
        subset,
        x='x',
        y='y',
        title=f"Scatter Plot - {category}",
        label=category
    )
    
    plot.save(f'scatter_{category.lower()}.png')
    plot.close()  # Free memory
```

## Common Issues and Solutions

### Regression Line Issues

```python
from monet_plots import ScatterPlot

# Fix regression line problems
plot = ScatterPlot(figsize=(12, 8))

# Ensure data is numeric
df_clean = df.dropna(subset=['x', 'y'])  # Remove missing values

plot.plot(
    df_clean,
    x='x',
    y='y',
    title="Cleaned Data for Regression",
    label="Clean Data"
)

# Display regression equation
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df_clean[['x']], df_clean['y'])
equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
plot.ax.text(0.05, 0.95, equation,
            transform=plot.ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plot.save('clean_regression.png')
```

### Categorical Data Handling

```python
from monet_plots import ScatterPlot

# Handle categorical variables properly
plot = ScatterPlot(figsize=(12, 9))

# Convert categorical to numeric if needed
if df['category'].dtype == 'object':
    df['category_numeric'] = pd.Categorical(df['category']).codes

# Plot with categorical coloring
scatter = plot.ax.scatter(
    df['x'],
    df['y'],
    c=df['category_numeric'],
    cmap='viridis',
    alpha=0.6,
    s=50
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=plot.ax)
cbar.set_label('Category')

plot.title("Scatter Plot with Categorical Coloring")
plot.xlabel("X Variable")
plot.ylabel("Y Variable")
plot.save('categorical_scatter.png')
```

### Outlier Detection and Handling

```python
import numpy as np
from scipy import stats
from monet_plots import ScatterPlot

# Detect and handle outliers
z_scores = np.abs(stats.zscore(df[['x', 'y']]))
outliers = (z_scores > 3).any(axis=1)

df_clean = df[~outliers]
df_outliers = df[outliers]

plot = ScatterPlot(figsize=(12, 9))

# Plot clean data
plot.plot(
    df_clean,
    x='x',
    y='y',
    title="Scatter Plot with Outliers Removed",
    label="Clean Data"
)

# Mark outliers
plot.ax.plot(df_outliers['x'], df_outliers['y'], 'rx', 
            markersize=10, markeredgewidth=2, label='Outliers')

plot.legend()
plot.save('outlier_removed_scatter.png')
```

---

**Related Resources**:

- [API Reference](../api) - Core functionality and utilities
- [Examples](../examples) - Practical usage examples
- [Style Configuration](../api/style) - Plot styling options
- [Statistical Analysis](../api/taylordiagram) - Statistical utilities
