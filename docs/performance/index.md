# Performance

For measured rendering time, native memory use, and large NumPy/Dask datasets,
see [Large-dataset measurements](large-datasets.md). The repository includes an
isolated-process benchmark harness and a weekly/manual performance workflow.

## Practical starting points

- Select the required times, levels and region before plotting.
- Use meaningful explicit color limits to avoid global scale reductions.
- Use images for regular grids and meshes for irregular or curvilinear grids.
- Aggregate explicitly when the source has many more cells than the display has pixels.
- Close figures and release plot objects after exporting.

MONET Plots is designed with performance in mind, but large datasets and complex visualizations can still present challenges. This guide provides strategies and techniques to optimize your plotting performance.

| Optimization Level | Difficulty | Focus Area |
|-------------------|------------|------------|
| [Basic Optimization](./basic-optimization.md) | Beginner | Quick wins for most users |
| [Dask plot benchmark](./dask-plot-benchmark.md) | Intermediate | Scheduler work and first-render time |
| [Plot benchmark suite](./plot-benchmark-suite.md) | Intermediate | All static plot types and additional Dask workloads |
| [Memory Management](./memory-management) | Intermediate | Large datasets and memory usage |
| [Rendering Optimization](./rendering-optimization) | Advanced | Plot generation speed |
| [Workflow Optimization](./workflow-optimization) | Intermediate | Multi-plot workflows |

## Performance Principles

### 1. **Downsample When Possible**
Reduce data points for interactive viewing while preserving important features.

### 2. **Close Plots Properly**
Always close plots when done to free memory resources.

### 3. **Use Appropriate Data Types**
Choose the right data structure for your use case.

### 4. **Batch Operations**
Group similar operations together for efficiency.

### 5. **Cache Results**
Save intermediate results to avoid recomputation.

## Quick Start: Basic Optimization

```python
import monet_plots as mp

# field is an xarray DataArray on lat/lon dimensions.
# Select aggregation and weighting appropriate to your scientific question.
coarse = field.coarsen(lat=4, lon=4, boundary="trim").mean()
result = mp.spatial(coarse, vmin=0, vmax=80)
result.save("overview.png")
result.close()
```

For common plotting options, see [Spatial comparisons](../spatial-comparisons.md)
and [Observation/model/bias maps](../model-evaluation.md).
