#!/usr/bin/env python3
"""Debug script to understand the SpatialPlot issue."""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from monet_plots.plots.spatial import SpatialPlot

# Create test data
spatial_data = np.random.rand(10, 10) * 100

# Create a figure and axes like in the test
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

print("=== BEFORE SpatialPlot creation ===")
print(f"axes[0].has_data(): {axes[0].has_data()}")
print(f"axes[0]: {axes[0]}")
print(f"axes[0].type: {type(axes[0])}")
print(f"axes[0] in fig.axes: {axes[0] in fig.axes}")

# Create SpatialPlot
spatial_plot = SpatialPlot(fig=fig, ax=axes[0])

print("\n=== AFTER SpatialPlot creation ===")
print(f"axes[0].has_data(): {axes[0].has_data()}")
print(f"spatial_plot.ax.has_data(): {spatial_plot.ax.has_data()}")
print(f"axes[0]: {axes[0]}")
print(f"spatial_plot.ax: {spatial_plot.ax}")
print(f"axes[0] in fig.axes: {axes[0] in fig.axes}")
print(f"spatial_plot.ax in fig.axes: {spatial_plot.ax in fig.axes}")
print(f"len(fig.axes): {len(fig.axes)}")

# Plot data
print("\n=== AFTER plotting ===")
spatial_plot.plot(spatial_data)
print(f"axes[0].has_data(): {axes[0].has_data()}")
print(f"spatial_plot.ax.has_data(): {spatial_plot.ax.has_data()}")
print(f"axes[0]: {axes[0]}")
print(f"spatial_plot.ax: {spatial_plot.ax}")
print(f"axes[0] in fig.axes: {axes[0] in fig.axes}")
print(f"spatial_plot.ax in fig.axes: {spatial_plot.ax in fig.axes}")
print(f"len(fig.axes): {len(fig.axes)}")

# Print all axes in the figure
print("\n=== ALL AXES IN FIGURE ===")
for i, ax in enumerate(fig.axes):
    print(f"  {i}: {ax} - has_data: {ax.has_data()}")

plt.tight_layout()
plt.savefig('/tmp/debug_spatial_plot.png')
print(f"\nSaved debug plot to /tmp/debug_spatial_plot.png")