# Taylor Diagrams

Taylor diagrams are statistical visualization tools used to evaluate model performance by comparing the standard deviation and correlation coefficient of model predictions against observations. They provide a comprehensive view of model skill in a single plot.

## Overview

Taylor diagrams are particularly valuable in meteorology and climate science for assessing how well different models reproduce observed patterns. The radial distance from the observation point indicates overall skill, while the angle represents the correlation.

| Feature | Description |
|---------|-------------|
| Statistical comparison | Standard deviation and correlation visualization |
| Model evaluation | Quantitative assessment of model performance |
| Multi-model comparison | Easy comparison of multiple models |
| Professional styling | Publication-ready formatting |

## TaylorDiagramPlot Class

`TaylorDiagramPlot` creates Taylor diagrams for model evaluation and comparison.

### Class Signature

```python
class TaylorDiagramPlot(BasePlot):
    """Creates a Taylor diagram to compare a model to observations.
    
    This class creates a Taylor diagram to compare a model to observations
    using standard deviation and correlation coefficient.
    """
    
    def __init__(self, obsstd, scale=1.5, label='OBS', **kwargs):
        """Initialize the Taylor diagram.
        
        Args:
            obsstd (float): Standard deviation of observations
            scale (float, optional): Scale of the diagram. Defaults to 1.5
            label (str, optional): Label for observations. Defaults to 'OBS'
            **kwargs: Additional keyword arguments to pass to `subplots`
        """
        pass
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obsstd` | `float` | Required | Standard deviation of observations |
| `scale` | `float` | `1.5` | Scale factor for diagram size |
| `label` | `str` | `'OBS'` | Label for observations |
| `**kwargs` | `dict` | `{}` | Additional matplotlib figure parameters |

### Methods

#### `add_sample(df, col1='obs', col2='model', marker='o', label='MODEL')`

Add a model sample to the diagram.

```python
def add_sample(self, df, col1='obs', col2='model', marker='o', label='MODEL'):
    """Add a model sample to the diagram.
    
    Args:
        df (pandas.DataFrame): DataFrame containing model and observation data
        col1 (str, optional): Column for observations. Defaults to 'obs'
        col2 (str, optional): Column for model. Defaults to 'model'
        marker (str, optional): Marker style. Defaults to 'o'
        label (str, optional): Model label. Defaults to 'MODEL'
    """
    pass
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pandas.DataFrame` | Required | DataFrame with model and observation data |
| `col1` | `str` | `'obs'` | Column name for observations |
| `col2` | `str` | `'model'` | Column name for model predictions |
| `marker` | `str` | `'o'` | Marker style for the model point |
| `label` | `str` | `'MODEL'` | Label for the model |

**Example:**
```python
import pandas as pd
import numpy as np
from monet_plots import TaylorDiagramPlot

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=1.2,           # Observation standard deviation
    scale=1.8,            # Diagram scale
    label='Observations'  # Observation label
)

# Generate sample model data
np.random.seed(42)
models = []

for i, (model_std, correlation, model_name) in enumerate([
    (1.1, 0.95, 'Model A'),
    (1.3, 0.88, 'Model B'),
    (0.9, 0.92, 'Model C'),
    (1.0, 0.97, 'Model D')
]):
    # Create synthetic data for this model
    n_points = 1000
    obs_data = np.random.normal(0, 1.2, n_points)
    model_data = correlation * obs_data * (model_std / 1.2) + np.random.normal(0, np.sqrt(model_std**2 * (1 - correlation**2)), n_points)
    
    df = pd.DataFrame({
        'obs': obs_data,
        'model': model_data,
        'model_name': model_name
    })
    
    # Add to diagram
    plot.add_sample(
        df,
        col1='obs',
        col2='model',
        marker='o',
        label=model_name
    )

# Add contours and finalize
plot.add_contours(levels=[0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.save('taylor_diagram_basic.png')
```

### Common Usage Patterns

#### Basic Model Comparison

```python
import pandas as pd
import numpy as np
from monet_plots import TaylorDiagramPlot

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=2.5,
    scale=2.0,
    label='Reference'
)

# Sample model data
model_data = [
    (2.4, 0.96, 'High Performance', 'o', 'red'),
    (2.6, 0.92, 'Medium Performance', 's', 'blue'),
    (2.2, 0.85, 'Low Performance', 'D', 'green'),
    (2.8, 0.78, 'Poor Performance', '^', 'orange')
]

# Add models
for model_std, correlation, name, marker, color in model_data:
    # Generate synthetic data
    n_points = 500
    obs = np.random.normal(0, 2.5, n_points)
    model = correlation * obs * (model_std / 2.5) + np.random.normal(0, np.sqrt(model_std**2 * (1 - correlation**2)), n_points)
    
    df = pd.DataFrame({'obs': obs, 'model': model})
    
    plot.add_sample(
        df,
        col1='obs',
        col2='model',
        marker=marker,
        label=name
    )

# Add contours
plot.add_contours(levels=[0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.title("Model Performance Comparison")
plot.save('model_comparison_taylor.png')
```

#### Multi-Category Model Evaluation

```python
import pandas as pd
import numpy as np
from monet_plots import TaylorDiagramPlot

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=1.0,
    scale=1.6,
    label='Observations'
)

# Model categories with different performance levels
categories = {
    'GCM Models': [
        (1.05, 0.94, 'GCM-1', 'o', 'red'),
        (1.15, 0.91, 'GCM-2', 's', 'darkred'),
        (0.95, 0.96, 'GCM-3', 'D', 'lightcoral')
    ],
    'Regional Models': [
        (1.25, 0.87, 'Regional-1', 'o', 'blue'),
        (1.35, 0.84, 'Regional-2', 's', 'darkblue'),
        (1.15, 0.89, 'Regional-3', 'D', 'lightblue')
    ],
    'Statistical Models': [
        (0.85, 0.93, 'Stat-1', 'o', 'green'),
        (0.75, 0.90, 'Stat-2', 's', 'darkgreen'),
        (0.90, 0.94, 'Stat-3', 'D', 'lightgreen')
    ]
}

# Add models by category
colors = {'GCM Models': 'red', 'Regional Models': 'blue', 'Statistical Models': 'green'}
markers = {'GCM Models': 'o', 'Regional Models': 's', 'Statistical Models': 'D'}

for category, models in categories.items():
    for model_std, correlation, name, marker, color in models:
        # Generate synthetic data
        n_points = 300
        obs = np.random.normal(0, 1.0, n_points)
        model = correlation * obs * model_std + np.random.normal(0, np.sqrt(model_std**2 * (1 - correlation**2)), n_points)
        
        df = pd.DataFrame({'obs': obs, 'model': model})
        
        plot.add_sample(
            df,
            col1='obs',
            col2='model',
            marker=markers[category],
            label=name
        )

# Add styled contours
plot.add_contours(
    levels=[0.5, 0.7, 0.8, 0.9, 0.95],
    colors='gray',
    linestyles=['-', '--', ':', '-.', (0, (3, 1, 1, 1))],
    linewidths=1.5,
    alpha=0.7
)

plot.finish_plot()

# Add category legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='GCM Models'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Regional Models'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=8, label='Statistical Models')
]
plot.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

plot.title("Multi-Category Model Evaluation")
plot.save('multi_category_taylor.png')
```

#### Real-World Data Example

```python
import pandas as pd
import numpy as np
from monet_plots import TaylorDiagramPlot

# Load or create realistic climate data
np.random.seed(123)

# Create realistic temperature data
n_years = 30
months = 12
n_points = n_years * months

# Generate realistic temperature patterns
time = np.arange(n_points)
seasonal_cycle = 10 * np.sin(2 * np.pi * time / 12)  # Annual cycle
trend = 0.02 * time  # Warming trend
noise = np.random.normal(0, 2, n_points)  # Natural variability

observed_temps = 15 + seasonal_cycle + trend + noise

# Create model predictions with different characteristics
models = {
    'High-CFM Model': {
        'std_factor': 0.95,
        'correlation': 0.97,
        'color': 'red',
        'marker': 'o'
    },
    'Medium-CFM Model': {
        'std_factor': 1.05,
        'correlation': 0.92,
        'color': 'blue',
        'marker': 's'
    },
    'Low-CFM Model': {
        'std_factor': 1.15,
        'correlation': 0.85,
        'color': 'green',
        'marker': 'D'
    }
}

# Calculate observed standard deviation
obs_std = np.std(observed_temps, ddof=1)

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=obs_std,
    scale=1.8,
    label='Observations'
)

# Add model predictions
for model_name, params in models.items():
    # Generate model predictions
    model_noise = params['std_factor'] * np.random.normal(0, 2, n_points)
    model_temps = params['correlation'] * observed_temps + np.sqrt(1 - params['correlation']**2) * model_noise
    
    df = pd.DataFrame({
        'obs': observed_temps,
        'model': model_temps,
        'model_name': model_name
    })
    
    plot.add_sample(
        df,
        col1='obs',
        col2='model',
        marker=params['marker'],
        label=model_name,
        color=params['color']
    )

# Add professional contours
plot.add_contours(
    levels=[0.5, 0.8, 0.9, 0.95, 0.98],
    colors='gray',
    linestyles=['-', '--', ':', '-.', (0, (5, 5))],
    linewidths=1.2,
    alpha=0.6
)

plot.finish_plot()

plot.title("Climate Model Performance - Temperature")
plot.xlabel("Standard Deviation")
plot.ylabel("Standard Deviation")
plot.save('climate_model_taylor.png')
```

## Advanced Features

### Custom Contour Styling

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from monet_plots import TaylorDiagramPlot

# Create custom colormap for contours
colors = ['#ffffff', '#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c']
n_bins = 100
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=1.0,
    scale=1.7,
    label='Reference'
)

# Add sample models
model_data = [
    (0.9, 0.94, 'Model 1'),
    (1.1, 0.91, 'Model 2'),
    (0.8, 0.89, 'Model 3'),
    (1.2, 0.87, 'Model 4')
]

for std, corr, name in model_data:
    n_points = 400
    obs = np.random.normal(0, 1.0, n_points)
    model = corr * obs * std + np.random.normal(0, np.sqrt(std**2 * (1 - corr**2)), n_points)
    
    df = pd.DataFrame({'obs': obs, 'model': model})
    plot.add_sample(df, col1='obs', col2='model', label=name)

# Add styled contours
plot.add_contours(
    levels=[0.5, 0.7, 0.8, 0.9, 0.95],
    cmap=custom_cmap,
    linewidths=2,
    alpha=0.8
)

plot.finish_plot()

plot.title("Custom Contour Styling")
plot.save('custom_contours_taylor.png')
```

### Interactive Taylor Diagram

```python
import matplotlib.pyplot as plt
from monet_plots import TaylorDiagramPlot

# Enable interactive mode
plt.ion()

# Create interactive Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=1.0,
    scale=1.8,
    label='Observations'
)

# Add initial models
initial_models = [
    (0.9, 0.92, 'Initial Model'),
    (1.1, 0.88, 'Baseline')
]

for std, corr, name in initial_models:
    n_points = 300
    obs = np.random.normal(0, 1.0, n_points)
    model = corr * obs * std + np.random.normal(0, np.sqrt(std**2 * (1 - corr**2)), n_points)
    
    df = pd.DataFrame({'obs': obs, 'model': model})
    plot.add_sample(df, col1='obs', col2='model', label=name)

plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

# Interactive model addition
model_count = len(plot.dia.samples)

def on_click(event):
    if event.inaxes == plot.ax and event.button == 1:  # Left click
        # Convert click to Taylor diagram coordinates
        r = np.sqrt(event.xdata**2 + event.ydata**2)
        if r > 0:
            corr = event.ydata / r
            if abs(corr) <= 1:  # Valid correlation
                model_count += 1
                std = r
                
                # Generate synthetic data for new model
                n_points = 200
                obs = np.random.normal(0, 1.0, n_points)
                model = corr * obs * std + np.random.normal(0, np.sqrt(std**2 * (1 - corr**2)), n_points)
                
                df = pd.DataFrame({'obs': obs, 'model': model})
                plot.add_sample(
                    df,
                    col1='obs',
                    col2='model',
                    marker='o',
                    label=f'Clicked Model {model_count}'
                )
                
                plot.fig.canvas.draw()

plot.fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
```

### Subplot Layout for Multiple Diagrams

```python
import matplotlib.pyplot as plt
from monet_plots import TaylorDiagramPlot

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Different evaluation scenarios
scenarios = [
    ('Temperature', 2.5, 'Temperature Models'),
    ('Precipitation', 15.0, 'Precipitation Models'),
    ('Wind Speed', 5.0, 'Wind Speed Models'),
    ('Pressure', 10.0, 'Pressure Models')
]

colors = ['red', 'blue', 'green', 'orange']
markers = ['o', 's', 'D', '^']

for i, (variable, obs_std, title) in enumerate(scenarios):
    ax = axes[i//2, i%2]
    
    # Create Taylor diagram on subplot
    plot = TaylorDiagramPlot(
        obsstd=obs_std,
        scale=1.5,
        label='Observations',
        figure=fig,
        subplot_kw=dict(ax=ax)
    )
    
    # Add models for this variable
    for j, (model_std, correlation) in enumerate([
        (obs_std * 0.9, 0.92),
        (obs_std * 1.1, 0.88),
        (obs_std * 0.95, 0.95),
        (obs_std * 1.05, 0.90)
    ]):
        n_points = 300
        obs = np.random.normal(0, obs_std, n_points)
        model = correlation * obs * (model_std / obs_std) + np.random.normal(0, np.sqrt(model_std**2 * (1 - correlation**2)), n_points)
        
        df = pd.DataFrame({'obs': obs, 'model': model})
        plot.add_sample(
            df,
            col1='obs',
            col2='model',
            marker=markers[j],
            label=f'Model {j+1}',
            color=colors[j]
        )
    
    # Add contours
    plot.add_contours([0.5, 0.8, 0.9, 0.95])
    plot.finish_plot()
    
    ax.set_title(f'{title}\n({variable})', fontsize=12, pad=20)

plt.tight_layout()
save_figure(fig, 'multi_scenario_taylor.png')
```

## Data Requirements

### Input Data Format

Taylor diagrams require DataFrames with observation and model prediction columns:

```python
import pandas as pd
import numpy as np

# Basic format
df = pd.DataFrame({
    'obs': np.random.normal(0, 1, 1000),      # Observations
    'model': np.random.normal(0, 1, 1000)     # Model predictions
})

# With multiple models
df = pd.DataFrame({
    'obs': np.random.normal(0, 2.5, 1000),
    'model_A': np.random.normal(0, 2.6, 1000),
    'model_B': np.random.normal(0, 2.2, 1000),
    'model_C': np.random.normal(0, 2.8, 1000)
})
```

### Data Preprocessing

```python
import pandas as pd
import numpy as np
from monet_plots import TaylorDiagramPlot

# Handle missing values
np.random.seed(42)
obs = np.random.normal(0, 1, 1000)
model = obs + np.random.normal(0, 0.5, 1000)

# Introduce missing values
model[100:110] = np.nan
model[400:410] = np.nan

df = pd.DataFrame({'obs': obs, 'model': model})

# Remove rows with missing values
df_clean = df.dropna()

# Calculate statistics
obs_std = np.std(df_clean['obs'], ddof=1)
model_std = np.std(df_clean['model'], ddof=1)
correlation = np.corrcoef(df_clean['obs'], df_clean['model'])[0, 1]

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=obs_std,
    scale=1.8,
    label='Observations'
)

plot.add_sample(
    df_clean,
    col1='obs',
    col2='model',
    marker='o',
    label='Cleaned Data'
)

plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.save('cleaned_data_taylor.png')
```

## Statistical Analysis

### Skill Score Calculation

```python
import numpy as np
import pandas as pd
from monet_plots import TaylorDiagramPlot

def calculate_taylor_skill(stddev, corrcoef, obs_std):
    """Calculate Taylor skill score.
    
    Args:
        stddev (float): Model standard deviation
        corrcoef (float): Correlation coefficient
        obs_std (float): Observation standard deviation
        
    Returns:
        float: Skill score (0-1, higher is better)
    """
    # Taylor skill score
    skill_score = (2 * (1 + corrcoef) - (stddev/obs_std)**2 - (obs_std/stddev)**2) / 4
    return max(0, skill_score)  # Ensure non-negative

# Example usage
obs_std = 1.0
model_performance = [
    (0.9, 0.95),  # (stddev, correlation)
    (1.1, 0.88),
    (0.8, 0.92),
    (1.2, 0.85)
]

# Calculate skill scores
skill_scores = [calculate_taylor_skill(std, corr, obs_std) for std, corr in model_performance]

# Create Taylor diagram with skill annotations
plot = TaylorDiagramPlot(obsstd=obs_std, scale=1.7, label='Observations')

for i, ((std, corr), skill) in enumerate(zip(model_performance, skill_scores)):
    n_points = 400
    obs = np.random.normal(0, obs_std, n_points)
    model = corr * obs * (std / obs_std) + np.random.normal(0, np.sqrt(std**2 * (1 - corr**2)), n_points)
    
    df = pd.DataFrame({'obs': obs, 'model': model})
    
    plot.add_sample(
        df,
        col1='obs',
        col2='model',
        marker='o',
        label=f'Model {i+1} (Skill: {skill:.2f})'
    )

plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.title("Taylor Diagram with Skill Scores")
plot.save('skill_score_taylor.png')
```

### Model Ranking

```import numpy as np
import pandas as pd
from monet_plots import TaylorDiagramPlot

def rank_models(model_data, obs_std):
    """Rank models by performance.
    
    Args:
        model_data: List of (stddev, correlation, name) tuples
        obs_std: Observation standard deviation
        
    Returns:
        DataFrame with ranked models
    """
    rankings = []
    
    for std, corr, name in model_data:
        # Calculate multiple metrics
        skill_score = (2 * (1 + corr) - (std/obs_std)**2 - (obs_std/std)**2) / 4
        skill_score = max(0, skill_score)
        
        # Distance from observation point
        distance = np.sqrt((std - obs_std)**2 + (obs_std * np.sqrt(2 * (1 - corr)))**2)
        
        rankings.append({
            'model': name,
            'stddev': std,
            'correlation': corr,
            'skill_score': skill_score,
            'distance': distance,
            'rank': 0  # Will be assigned later
        })
    
    # Rank models
    df_rankings = pd.DataFrame(rankings)
    df_rankings['rank'] = df_rankings['skill_score'].rank(ascending=False).astype(int)
    
    return df_rankings.sort_values('skill_score', ascending=False)

# Example usage
obs_std = 1.5
model_data = [
    (1.4, 0.96, 'Advanced Model'),
    (1.6, 0.92, 'Standard Model'),
    (1.2, 0.94, 'Optimized Model'),
    (1.8, 0.85, 'Baseline Model')
]

# Get rankings
rankings = rank_models(model_data, obs_std)

# Create Taylor diagram with rankings
plot = TaylorDiagramPlot(obsstd=obs_std, scale=1.8, label='Observations')

for _, row in rankings.iterrows():
    n_points = 500
    obs = np.random.normal(0, obs_std, n_points)
    model = row['correlation'] * obs * (row['stddev'] / obs_std) + np.random.normal(0, np.sqrt(row['stddev']**2 * (1 - row['correlation']**2)), n_points)
    
    df = pd.DataFrame({'obs': obs, 'model': model})
    
    plot.add_sample(
        df,
        col1='obs',
        col2='model',
        marker='o',
        label=f"{row['model']} (Rank: {row['rank']})"
    )

plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.title("Ranked Model Performance")
plot.save('ranked_models_taylor.png')

# Display rankings
print("Model Rankings:")
print(rankings[['rank', 'model', 'skill_score', 'correlation', 'stddev']].to_string(index=False))
```

## Performance Considerations

### Large Datasets

```python
import numpy as np
import pandas as pd
from monet_plots import TaylorDiagramPlot

# Handle large datasets efficiently
n_points = 100000  # 100,000 points
obs_std = 2.0

# Generate large dataset
obs = np.random.normal(0, obs_std, n_points)
model = 0.9 * obs * 1.1 + np.random.normal(0, 0.5, n_points)  # Model with some bias

df = pd.DataFrame({'obs': obs, 'model': model})

# Downsample for Taylor diagram calculation
df_sampled = df.sample(n=10000, random_state=42)  # Sample 10,000 points

# Calculate statistics from sample
sample_obs_std = np.std(df_sampled['obs'], ddof=1)
sample_model_std = np.std(df_sampled['model'], ddof=1)
sample_corr = np.corrcoef(df_sampled['obs'], df_sampled['model'])[0, 1]

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=sample_obs_std,
    scale=1.8,
    label='Observations'
)

plot.add_sample(
    df_sampled,
    col1='obs',
    col2='model',
    marker='o',
    label='Model (Sampled)'
)

plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.title("Large Dataset (Sampled)")
plot.save('large_dataset_taylor.png')
```

### Memory Management

```python
from monet_plots import TaylorDiagramPlot

# Process multiple models efficiently
models = [
    (1.1, 0.95, 'Model A'),
    (1.3, 0.88, 'Model B'),
    (0.9, 0.92, 'Model C'),
    (1.0, 0.97, 'Model D'),
    (1.2, 0.90, 'Model E')
]

obs_std = 1.0

# Create Taylor diagram
plot = TaylorDiagramPlot(obsstd=obs_std, scale=1.8, label='Observations')

# Process models one by one
for std, corr, name in models:
    # Generate data for this model only
    n_points = 300
    obs = np.random.normal(0, obs_std, n_points)
    model = corr * obs * (std / obs_std) + np.random.normal(0, np.sqrt(std**2 * (1 - corr**2)), n_points)
    
    df = pd.DataFrame({'obs': obs, 'model': model})
    plot.add_sample(df, col1='obs', col2='model', label=name)

plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.save('efficient_processing_taylor.png')
```

## Common Issues and Solutions

### Invalid Correlation Values

```python
import numpy as np
import pandas as pd
from monet_plots import TaylorDiagramPlot

# Handle edge cases in correlation calculation
np.random.seed(42)
obs = np.random.normal(0, 1, 100)

# Create model with potential numerical issues
model = obs + np.random.normal(0, 0.001, 100)  # Very small noise

df = pd.DataFrame({'obs': obs, 'model': model})

# Calculate correlation with clipping
corr = np.corrcoef(df['obs'], df['model'])[0, 1]
corr = np.clip(corr, -1, 1)  # Ensure valid correlation

print(f"Correlation: {corr:.6f}")

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=np.std(df['obs'], ddof=1),
    scale=1.8,
    label='Observations'
)

plot.add_sample(df, col1='obs', col2='model', label='High Correlation Model')
plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.title("High Correlation Edge Case")
plot.save('high_correlation_taylor.png')
```

### Standard Deviation Scaling

```python
import numpy as np
import pandas as pd
from monet_plots import TaylorDiagramPlot

# Handle different scales of standard deviation
obs_std = 10.0  # Large standard deviation

# Create model with different scale
n_points = 500
obs = np.random.normal(0, obs_std, n_points)
model = 0.95 * obs * 1.05 + np.random.normal(0, 1.0, n_points)  # Model with different scale

df = pd.DataFrame({'obs': obs, 'model': model})

# Calculate statistics
model_std = np.std(df['model'], ddof=1)
correlation = np.corrcoef(df['obs'], df['model'])[0, 1]

print(f"Obs std: {obs_std:.2f}, Model std: {model_std:.2f}, Corr: {correlation:.3f}")

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=obs_std,
    scale=2.0,  # Larger scale for bigger range
    label='Observations'
)

plot.add_sample(df, col1='obs', col2='model', label='Different Scale Model')
plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.title("Different Scale Standard Deviations")
plot.save('different_scale_taylor.png')
```

### Missing Data Handling

```python
import numpy as np
import pandas as pd
from monet_plots import TaylorDiagramPlot

# Handle missing data robustly
np.random.seed(42)
n_points = 1000
obs = np.random.normal(0, 1, n_points)
model = 0.9 * obs + np.random.normal(0, 0.5, n_points)

# Introduce various types of missing data
missing_indices = np.random.choice(n_points, size=100, replace=False)
model[missing_indices] = np.nan

# Additional completely missing rows
completely_missing = np.random.choice(n_points, size=50, replace=False)
obs[completely_missing] = np.nan
model[completely_missing] = np.nan

df = pd.DataFrame({'obs': obs, 'model': model})

# Remove rows with any missing values
df_clean = df.dropna()

# Calculate statistics from clean data
obs_std = np.std(df_clean['obs'], ddof=1)
model_std = np.std(df_clean['model'], ddof=1)
correlation = np.corrcoef(df_clean['obs'], df_clean['model'])[0, 1]

print(f"Original points: {n_points}, Clean points: {len(df_clean)}")
print(f"Obs std: {obs_std:.3f}, Model std: {model_std:.3f}, Corr: {correlation:.3f}")

# Create Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=obs_std,
    scale=1.8,
    label='Observations'
)

plot.add_sample(df_clean, col1='obs', col2='model', label='Cleaned Data')
plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.title("Robust Missing Data Handling")
plot.save('robust_missing_data_taylor.png')
```

---

**Related Resources**:

- [API Reference](../api) - Core functionality and utilities
- [Examples](../examples) - Practical usage examples
- [Statistical Analysis](../api/taylordiagram) - Statistical utilities
- [Style Configuration](../api/style) - Plot styling options
