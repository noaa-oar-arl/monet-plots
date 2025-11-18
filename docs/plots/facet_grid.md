# Facet Grids

Facet grids create matrix layouts of subplots, allowing you to visualize relationships across multiple variables simultaneously. MONET Plots provides enhanced faceting capabilities with customizable layouts, shared axes, and publication-ready styling.

## Overview

Facet grids are powerful for exploring data patterns across categories and conditions. They enable efficient comparison of multiple datasets in a single, coherent visualization.

| Feature | Description |
|---------|-------------|
| **Flexible layouts** | Row, column, or both faceting |
| **Custom mappings** | Map variables to plot aesthetics |
| **Shared axes** | Consistent scaling across subplots |
| **Custom styling** | Publication-ready formatting |
| **Interactive support** | Clickable and hoverable facets |

## FacetGridPlot Class

`FacetGridPlot` creates multi-panel grid layouts for data visualization.

### Class Signature

```python
class FacetGridPlot(BasePlot):
    """Creates a facet grid for multi-panel visualization.
    
    This class creates grids of subplots based on categorical
    variables, enabling comprehensive data exploration.
    """
    
    def __init__(self, row=None, col=None, hue=None, col_wrap=None,
                 height=3, aspect=1, **kwargs):
        """Initialize the facet grid.
        
        Args:
            row (str, optional): Variable to map to row facets. Defaults to None
            col (str, optional): Variable to map to column facets. Defaults to None
            hue (str, optional): Variable to map to color mapping. Defaults to None
            col_wrap (int, optional): Number of columns before wrapping. Defaults to None
            height (float, optional): Height of each facet in inches. Defaults to 3
            aspect (float, optional): Aspect ratio of each facet. Defaults to 1
            **kwargs: Additional keyword arguments for figure creation
        """
        pass
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `row` | `str` | `None` | DataFrame column name for row faceting |
| `col` | `str` | `None` | DataFrame column name for column faceting |
| `hue` | `str` | `None` | DataFrame column name for color mapping |
| `col_wrap` | `int` | `None` | Number of columns before wrapping to new row |
| `height` | `float` | `3` | Height of each subplot in inches |
| `aspect` | `float` | `1` | Aspect ratio (width/height) of each subplot |
| `**kwargs` | `dict` | `{}` | Additional matplotlib figure parameters |

### Methods

#### `map_dataframe(func, **kwargs)`

Apply a function to each subplot using DataFrame columns.

```python
def map_dataframe(self, func, **kwargs):
    """Apply a function to each subplot.
    
    Args:
        func (callable): Function to apply to each subplot
        **kwargs: Additional keyword arguments for the function
    """
    pass
```

#### `map(func, *args, **kwargs)`

Apply a function to each subplot using positional arguments.

```python
def map(self, func, *args, **kwargs):
    """Apply a function to each subplot.
    
    Args:
        func (callable): Function to apply to each subplot
        *args: Positional arguments for the function
        **kwargs: Additional keyword arguments for the function
    """
    pass
```

#### `add_legend(title=None, **kwargs)`

Add a legend to the plot.

```python
def add_legend(self, title=None, **kwargs):
    """Add a legend to the facet grid.
    
    Args:
        title (str, optional): Legend title. Defaults to None
        **kwargs: Additional legend parameters
    """
    pass
```

**Example:**
```python
import pandas as pd
import numpy as np
from monet_plots import FacetGridPlot

# Create facet grid
plot = FacetGridPlot(
    row='category',
    col='region',
    hue='season',
    height=4,
    aspect=1.2
)

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'value': np.random.normal(0, 1, 1000),
    'category': np.random.choice(['Type A', 'Type B', 'Type C'], 1000),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 1000),
    'measurement': np.random.uniform(0, 100, 1000)
})

# Map plotting function to each facet
plot.map_dataframe(
    lambda df: df.plot(kind='hist', y='value', bins=20, alpha=0.7),
    **{'color': 'skyblue', 'edgecolor': 'navy'}
)

# Add shared title and labels
plot.fig.suptitle("Facet Grid Analysis", y=1.02)
plot.set_axis_labels("Value", "Frequency")
plot.add_legend(title="Season")

# Save and close
plot.save('facet_grid_basic.png')
plot.close()
```

### Common Usage Patterns

#### Basic Row Faceting

```python
import pandas as pd
import numpy as np
import seaborn as sns
from monet_plots import FacetGridPlot

# Create sample data
np.random.seed(123)
data = pd.DataFrame({
    'x': np.random.normal(0, 1, 800),
    'y': np.random.normal(0, 1, 800),
    'group': np.random.choice(['Group 1', 'Group 2', 'Group 3', 'Group 4'], 800)
})

# Create row-faceted grid
g = FacetGridPlot(
    row='group',
    height=4,
    aspect=1.5
)

# Map scatter plots to each facet
g.map_dataframe(
    sns.scatterplot,
    x='x',
    y='y',
    alpha=0.6
)

# Add titles and labels
g.fig.suptitle("Row Faceting - Group Analysis", y=1.02)
g.set_axis_labels("X Variable", "Y Variable")

# Add row labels
g.set_titles(row_template='Group {row_name}')

g.save('row_faceting.png')
g.close()
```

#### Column and Row Faceting

```python
import pandas as pd
import numpy as np
from monet_plots import FacetGridPlot

# Create complex dataset
np.random.seed(456)
regions = ['North', 'South', 'East', 'West']
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
categories = ['Type A', 'Type B']

data = []
for region in regions:
    for season in seasons:
        for category in categories:
            n_samples = 25
            values = np.random.normal(
                mean=np.random.uniform(0, 10),
                std=np.random.uniform(1, 3),
                size=n_samples
            )
            
            for val in values:
                data.append({
                    'value': val,
                    'region': region,
                    'season': season,
                    'category': category
                })

df = pd.DataFrame(data)

# Create 4x4 grid (regions x seasons)
g = FacetGridPlot(
    row='season',
    col='region',
    hue='category',
    height=3.5,
    aspect=1.2,
    palette=['blue', 'red']
)

# Map time series plot to each facet
g.map_dataframe(
    lambda df: df.plot(
        kind='line',
        x=range(len(df)),
        y='value',
        alpha=0.7,
        legend=False
    )
)

# Customize the grid
g.fig.suptitle("Regional Seasonal Analysis", y=1.02)
g.set_axis_labels("Time", "Value")
g.set_titles(col_template='{col_name} Region', row_template='{row_name} Season')
g.add_legend(title='Category')

g.save('column_row_faceting.png')
g.close()
```

#### Wrapped Column Faceting

```python
import pandas as pd
import numpy as np
from monet_plots import FacetGridPlot

# Create data with many categories
np.random.seed(789)
n_categories = 12
data = pd.DataFrame({
    'measurement': np.random.normal(0, 1, 2000),
    'category': [f'Cat {i+1}' for i in range(n_categories)] * (2000 // n_categories)
})

# Create wrapped facet grid (wrap after 4 columns)
g = FacetGridPlot(
    col='category',
    col_wrap=4,  # Wrap to new row after 4 columns
    height=3,
    aspect=1.3,
    sharey=True  # Share y-axis across all facets
)

# Map histogram to each facet
g.map_dataframe(
    lambda df: df.plot(kind='hist', y='measurement', bins=15, alpha=0.7),
    **{'color': 'lightgreen', 'edgecolor': 'darkgreen'}
)

# Customize
g.fig.suptitle("Wrapped Facet Grid - 12 Categories", y=1.02)
g.set_axis_labels("Measurement", "Frequency")
g.set_titles(col_template='{col_name}')

g.save('wrapped_faceting.png')
g.close()
```

## Advanced Features

### Custom Function Mapping

```python
import pandas as pd
import numpy as np
from scipy import stats
from monet_plots import FacetGridPlot

# Calculate statistics for each facet
def calculate_statistics(ax, df):
    """Custom function to add statistics to each subplot."""
    mean_val = df['value'].mean()
    std_val = df['value'].std()
    
    # Add mean line
    ax.axhline(mean_val, color='red', linestyle='--',
               label=f'Mean: {mean_val:.2f}')
    
    # Add statistics text
    stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Create data with different distributions
np.random.seed(101)
data = []
for dist_name in ['Normal', 'Exponential', 'Gamma', 'Lognormal']:
    if dist_name == 'Normal':
        values = np.random.normal(5, 2, 200)
    elif dist_name == 'Exponential':
        values = np.random.exponential(2, 200)
    elif dist_name == 'Gamma':
        values = np.random.gamma(2, 2, 200)
    else:  # Lognormal
        values = np.random.lognormal(1, 0.5, 200)
    
    for val in values:
        data.append({
            'value': val,
            'distribution': dist_name,
            'sample_type': np.random.choice(['Sample A', 'Sample B'], 200)
        })

df = pd.DataFrame(data)

# Create facet grid with custom function
g = FacetGridPlot(
    col='distribution',
    row='sample_type',
    height=4,
    aspect=1.2
)

# Map custom statistics function
g.map_dataframe(calculate_statistics, 'value')

# Add boxplots as background
g.map_dataframe(
    lambda df: df.boxplot(column='value', ax=ax, grid=False),
    **{'boxprops': dict(linewidth=2, color='blue')}
)

g.fig.suptitle("Custom Statistics in Facet Grid", y=1.02)
g.set_axis_labels("Value", "")

g.save('custom_function_faceting.png')
g.close()
```

### Mixed Plot Types in Facets

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots import FacetGridPlot

# Create comprehensive dataset
np.random.seed(202)
n_groups = 6
data = []

for i in range(n_groups):
    # Generate data with different characteristics
    x = np.linspace(0, 10, 50)
    y = np.sin(x + i) + np.random.normal(0, 0.2, 50)
    
    for xi, yi in zip(x, y):
        data.append({
            'x': xi,
            'y': yi,
            'group': f'Group {i+1}',
            'type': 'Time Series'
        })
    
    # Add scatter data
    x_scatter = np.random.uniform(0, 10, 30)
    y_scatter = 2 * x_scatter + np.random.normal(0, 1, 30)
    
    for xi, yi in zip(x_scatter, y_scatter):
        data.append({
            'x': xi,
            'y': yi,
            'group': f'Group {i+1}',
            'type': 'Scatter'
        })

df = pd.DataFrame(data)

# Create facet grid
g = FacetGridPlot(
    col='group',
    col_wrap=3,
    height=4,
    aspect=1.2,
    hue='type'
)

# Mixed plot types in facets
def mixed_plot(ax, df):
    """Mixed time series and scatter plot in each facet."""
    time_data = df[df['type'] == 'Time Series']
    scatter_data = df[df['type'] == 'Scatter']
    
    # Plot time series
    ax.plot(time_data['x'], time_data['y'], 'b-', alpha=0.7, label='Time Series')
    
    # Plot scatter points
    ax.scatter(scatter_data['x'], scatter_data['y'],
              c='red', alpha=0.6, s=20, label='Scatter')

g.map_dataframe(mixed_plot)

g.fig.suptitle("Mixed Plot Types in Facets", y=1.02)
g.set_axis_labels("X Variable", "Y Variable")
g.set_titles(col_template='{col_name}')
g.add_legend(title='Data Type')

g.save('mixed_plot_faceting.png')
g.close()
```

## Data Requirements

### Input Data Format

Facet grids work best with long-form DataFrames:

```python
import pandas as pd
import numpy as np

# Basic long-form data
df = pd.DataFrame({
    'x': np.random.normal(0, 1, 1000),
    'y': np.random.normal(0, 1, 1000),
    'group': np.random.choice(['A', 'B', 'C'], 1000),
    'category': np.random.choice(['Type 1', 'Type 2'], 1000)
})

# Multi-level faceting data
df = pd.DataFrame({
    'value': np.random.normal(0, 1, 2000),
    'region': np.random.choice(['North', 'South'], 2000),
    'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 2000),
    'year': np.random.choice([2020, 2021, 2022], 2000),
    'measurement': np.random.choice(['Temp', 'Precip', 'Wind'], 2000)
})
```

### Data Preprocessing

```python
import pandas as pd
import numpy as np
from monet_plots import FacetGridPlot

# Handle missing values and data quality
np.random.seed(42)
data = []

for region in ['North', 'South', 'East', 'West']:
    for season in ['Spring', 'Summer']:
        for i in range(100):
            # Introduce missing values
            if np.random.random() < 0.1:  # 10% missing
                value = np.nan
            else:
                value = np.random.normal(np.random.uniform(0, 10), 2)
            
            data.append({
                'value': value,
                'region': region,
                'season': season,
                'quality': np.random.choice(['High', 'Medium', 'Low'])
            })

df = pd.DataFrame(data)

# Remove rows with missing values in faceting columns
df_clean = df.dropna()

# Create facet grid
g = FacetGridPlot(
    row='quality',
    col='region',
    height=3.5,
    aspect=1.2,
    sharey=True
)

# Plot clean data
g.map_dataframe(
    lambda df: df.plot(kind='hist', y='value', bins=15, alpha=0.7),
    **{'color': 'steelblue', 'edgecolor': 'navy'}
)

# Add mean lines for each facet
def add_mean_line(ax, df):
    mean_val = df['value'].mean()
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8)

g.map_dataframe(add_mean_line, 'value')

g.fig.suptitle("Facet Grid with Clean Data", y=1.02)
g.set_axis_labels("Value", "Frequency")
g.set_titles(col_template='{col_name} Region', row_template='{row_name} Quality')

g.save('clean_facet_grid.png')
g.close()
```

## Customization Options

### Layout and Styling

```python
from monet_plots import FacetGridPlot

# Custom layout and styling
g = FacetGridPlot(
    row='category',
    col='subcategory',
    height=4,
    aspect=1.3,
    linewidth=2,
    edgecolor='gray'
)

# Custom facet styling
def custom_facet_plot(ax, df):
    ax.plot(df['x'], df['y'], 'o-', color='darkblue',
            linewidth=2, markersize=6, alpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f0f0f0')

g.map_dataframe(custom_facet_plot)

# Customize titles and labels
g.fig.suptitle("Custom Styled Facet Grid", y=1.02, fontsize=16, fontweight='bold')
g.set_axis_labels("X Variable", "Y Variable", fontsize=12)
g.set_titles(
    col_template='{col_name} Subcategory',
    row_template='{row_name} Category',
    size=10
)

g.add_legend(title='Legend', fontsize=10)

g.save('custom_styled_faceting.png')
g.close()
```

### Color Mapping and Themes

```python
import matplotlib.pyplot as plt
import seaborn as sns
from monet_plots import FacetGridPlot

# Use seaborn color palette
plt.style.use('seaborn-v0_8')

# Create color-coded facet grid
g = FacetGridPlot(
    col='variable',
    hue='group',
    palette='husl',
    height=4,
    aspect=1.2,
    legend_out=True
)

# Map styled plots
g.map_dataframe(
    sns.regplot,
    x='x',
    y='y',
    scatter_kws={'alpha': 0.6, 's': 30},
    line_kws={'linewidth': 2}
)

# Customize with theme elements
g.fig.suptitle("Color-Coded Facet Grid", y=1.02)
g.set_axis_labels("Independent Variable", "Dependent Variable")
g.set_titles(col_template='{col_name} Analysis')
g.add_legend(title='Group', bbox_to_anchor=(1.05, 1))

g.save('color_coded_faceting.png')
g.close()
```

## Performance Considerations

### Large Facet Grids

```python
import pandas as pd
import numpy as np
from monet_plots import FacetGridPlot

# Handle large facet grids efficiently
n_categories = 16  # 4x4 grid
data = []

for i in range(n_categories):
    values = np.random.normal(0, 1, 100)
    for val in values:
        data.append({
            'value': val,
            'category': f'Category {i+1}'
        })

df = pd.DataFrame(data)

# Create wrapped facet grid for performance
g = FacetGridPlot(
    col='category',
    col_wrap=4,  # Wrap to manage layout
    height=3,
    aspect=1.2,
    sharey=True  # Share axes to save memory
)

# Efficient plotting
g.map_dataframe(
    lambda df: df.plot(kind='hist', y='value', bins=10, alpha=0.7),
    **{'color': 'lightcoral', 'edgecolor': 'darkred'}
)

# Optimize layout
g.fig.tight_layout()
g.fig.suptitle("Large Facet Grid (Optimized)", y=1.02)

g.save('large_facet_grid.png')
g.close()
```

### Memory Management

```python
from monet_plots import FacetGridPlot

# Process multiple facet grids efficiently
categories = ['A', 'B', 'C']
subcategories = ['Type 1', 'Type 2', 'Type 3']

for category in categories:
    # Create subset data for this category
    subset_data = df[df['main_category'] == category]
    
    # Create facet grid for this category
    g = FacetGridPlot(
        row='subcategory',
        col='metric',
        height=3,
        aspect=1.2,
        data=subset_data
    )
    
    # Plot and save
    g.map_dataframe(lambda df: df.plot(kind='box', y='value'))
    g.fig.suptitle(f"Facet Grid - {category}", y=1.02)
    g.save(f'facet_grid_{category.lower()}.png')
    
    # Close to free memory
    g.close()
```

## Common Issues and Solutions

### Aspect Ratio Issues

```python
from monet_plots import FacetGridPlot

# Fix aspect ratio problems
g = FacetGridPlot(
    col='variable',
    height=4,
    aspect=1.5,  # Adjust aspect ratio
    sharey=False  # Allow different y-scales
)

# Custom aspect handling
def fixed_aspect_plot(ax, df):
    ax.plot(df['x'], df['y'], 'b-', linewidth=2)
    ax.set_aspect('auto')  # Let matplotlib handle aspect

g.map_dataframe(fixed_aspect_plot)

g.fig.suptitle("Fixed Aspect Ratio Faceting", y=1.02)
g.save('fixed_aspect_faceting.png')
g.close()
```

### Legend Positioning

```python
from monet_plots import FacetGridPlot

# Handle legend positioning in complex grids
g = FacetGridPlot(
    row='region',
    col='season',
    hue='data_type',
    height=3,
    aspect=1.2,
    legend_out=False  # Legend inside the grid
)

g.map_dataframe(lambda df: df.plot(kind='scatter', x='x', y='y', alpha=0.6))

# Custom legend positioning
g.add_legend(
    title='Data Type',
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    frameon=True
)

g.fig.suptitle("Custom Legend Positioning", y=1.02)
g.fig.tight_layout()  # Adjust layout to make space for legend
g.save('custom_legend_faceting.png')
g.close()
```

### Empty Facets Handling

```python
import pandas as pd
import numpy as np
from monet_plots import FacetGridPlot

# Handle empty facets gracefully
np.random.seed(303)
data = []

# Create unbalanced data (some combinations missing
for region in ['North', 'South', 'East']:
    for season in ['Spring', 'Summer']:
        if not (region == 'East' and season == 'Summer'):  # Missing combination
            values = np.random.normal(0, 1, 50)
            for val in values:
                data.append({
                    'value': val,
                    'region': region,
                    'season': season
                })

df = pd.DataFrame(data)

# Create facet grid
g = FacetGridPlot(
    row='season',
    col='region',
    height=4,
    aspect=1.2
)

# Handle empty facets
def safe_plot(ax, df):
    if not df.empty:
        ax.hist(df['value'], bins=15, alpha=0.7, color='skyblue')
        ax.set_title(ax.get_title())  # Ensure title is set
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(ax.get_title())  # Ensure title is set

g.map_dataframe(safe_plot)

g.fig.suptitle("Empty Facet Handling", y=1.02)
g.set_axis_labels("Value", "Frequency")
g.set_titles(row_template='{row_name}', col_template='{col_name}')

g.save('empty_facet_handling.png')
g.close()
```

---

**Related Resources**:

- [API Reference](../api) - Core functionality and utilities
- [Examples](../examples) - Practical usage examples
- [Style Configuration](../api/style) - Plot styling options
- [Multi-Plot Layouts](../advanced-workflows) - Advanced layout techniques
