"""
Advanced Visual Regression Tests for MONET Plots

This module contains advanced visual regression tests to ensure plot appearance
remains consistent across changes and updates.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageChops
import tempfile
import os
from pathlib import Path
import json
import hashlib
from typing import Dict, List, Tuple, Optional


class TestVisualRegressionAdvanced:
    """Advanced visual regression tests for plot appearance consistency."""
    
    def test_plot_style_consistency(self, mock_data_factory):
        """Test that plot styling is consistent across all plot types."""
        plot_styles = {}
        
        # Test different plot types for style consistency
        plot_types_to_test = [
            ("SpatialPlot", lambda: self._create_test_plot("SpatialPlot", mock_data_factory)),
            ("TimeSeriesPlot", lambda: self._create_test_plot("TimeSeriesPlot", mock_data_factory)),
            ("ScatterPlot", lambda: self._create_test_plot("ScatterPlot", mock_data_factory)),
            ("KDEPlot", lambda: self._create_test_plot("KDEPlot", mock_data_factory))
        ]
        
        plots_created = []
        
        try:
            for plot_name, plot_creator in plot_types_to_test:
                try:
                    plot = plot_creator()
                    if plot is not None:
                        plots_created.append(plot)
                        
                        # Check Wiley style consistency
                        ax = plot.ax
                        
                        # Font sizes should be consistent
                        xlabel_size = ax.xaxis.get_label().get_fontsize()
                        ylabel_size = ax.yaxis.get_label().get_fontsize()
                        
                        plot_styles[plot_name] = {
                            'xlabel_fontsize': xlabel_size,
                            'ylabel_fontsize': ylabel_size,
                            'has_grid': ax.get_gridlines() != [],
                            'facecolor': str(ax.get_facecolor())
                        }
                        
                        # Assert style consistency
                        assert xlabel_size == 10, f"{plot_name} xlabel font size {xlabel_size} != 10"
                        assert ylabel_size == 10, f"{plot_name} ylabel font size {ylabel_size} != 10"
                        
                except Exception as e:
                    print(f"Could not test style for {plot_name}: {e}")
                    continue
            
            # Verify that all tested plots have consistent styles
            if plot_styles:
                # Check that font sizes are consistent across plot types
                font_sizes = [style['xlabel_fontsize'] for style in plot_styles.values()]
                assert all(size == 10 for size in font_sizes), "Inconsistent font sizes across plot types"
                
                print("Style consistency check passed:")
                for plot_name, style in plot_styles.items():
                    print(f"  {plot_name}: font_size={style['xlabel_fontsize']}, grid={style['has_grid']}")
                    
        finally:
            # Clean up
            for plot in plots_created:
                try:
                    plot.close()
                except:
                    pass
    
    def test_color_scheme_consistency(self, mock_data_factory):
        """Test that color schemes are consistent across plot instances."""
        # Test that the same data produces the same colors
        data = mock_data_factory.spatial_2d(seed=42)
        
        # Create two identical spatial plots
        plot1 = None
        plot2 = None
        
        try:
            plot1 = self._create_test_plot("SpatialPlot", mock_data_factory)
            plot2 = self._create_test_plot("SpatialPlot", mock_data_factory)
            
            if plot1 and plot2:
                # Both plots should have the same colormap
                img1 = plot1.ax.images[0] if plot1.ax.images else None
                img2 = plot2.ax.images[0] if plot2.ax.images else None
                
                if img1 and img2:
                    cmap1 = img1.get_cmap()
                    cmap2 = img2.get_cmap()
                    
                    # Colormaps should be the same
                    assert cmap1.name == cmap2.name, f"Colormap mismatch: {cmap1.name} != {cmap2.name}"
                    
                    print(f"Color scheme consistency: {cmap1.name}")
                    
        finally:
            if plot1:
                plot1.close()
            if plot2:
                plot2.close()
    
    def test_plot_layout_consistency(self, mock_data_factory):
        """Test that plot layouts are consistent."""
        # Create multiple plots of the same type and verify layout consistency
        plots = []
        
        try:
            for i in range(3):
                plot = self._create_test_plot("TimeSeriesPlot", mock_data_factory)
                if plot:
                    plots.append(plot)
            
            if len(plots) >= 2:
                # Check that layouts are consistent
                axes = [plot.ax for plot in plots if plot.ax]
                
                # All should have similar layout properties
                for ax in axes[1:]:
                    # Check that axis limits are similar (same data, so should be same limits)
                    assert ax.get_xlim() == axes[0].get_xlim(), "X-axis limits inconsistent"
                    assert ax.get_ylim() == axes[0].get_ylim(), "Y-axis limits inconsistent"
                    
        finally:
            for plot in plots:
                try:
                    plot.close()
                except:
                    pass
    
    def test_legend_consistency(self, mock_data_factory):
        """Test that legends are rendered consistently."""
        plot = self._create_test_plot("ScatterPlot", mock_data_factory)
        
        if plot:
            try:
                # Check that legend exists and has consistent properties
                legend = plot.ax.get_legend()
                if legend:
                    legend_items = legend.get_texts()
                    assert len(legend_items) > 0, "Legend should have text items"
                    
                    # Legend text should be readable
                    for item in legend_items:
                        fontsize = item.get_fontsize()
                        assert fontsize >= 8, f"Legend font size {fontsize} too small"
                        
            finally:
                plot.close()
    
    def test_title_and_label_consistency(self, mock_data_factory):
        """Test that titles and labels are rendered consistently."""
        plot = self._create_test_plot("TimeSeriesPlot", mock_data_factory)
        
        if plot:
            try:
                ax = plot.ax
                
                # Check title
                title = ax.get_title()
                assert title, "Plot should have a title"
                
                # Check axis labels
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                
                # Y-label should be set (X might be empty for time series)
                assert ylabel, "Y-axis should have a label"
                
                # Font sizes should be correct
                title_fontsize = ax.title.get_fontsize()
                xlabel_fontsize = ax.xaxis.get_label().get_fontsize()
                ylabel_fontsize = ax.yaxis.get_label().get_fontsize()
                
                assert title_fontsize >= 10, f"Title font size {title_fontsize} too small"
                assert xlabel_fontsize == 10, f"X-label font size {xlabel_fontsize} incorrect"
                assert ylabel_fontsize == 10, f"Y-label font size {ylabel_fontsize} incorrect"
                
            finally:
                plot.close()

    
class TestAccessibilityVisualTests:
    """Visual tests for accessibility compliance."""
    
    def test_colorblind_friendly_colors(self, mock_data_factory):
        """Test that default colors are colorblind friendly."""
        plot = None
        
        try:
            plot = self._create_test_plot("SpatialPlot", mock_data_factory)
            
            if plot and plot.ax.images:
                # Check that colormap is colorblind friendly
                cmap = plot.ax.images[0].get_cmap().name
                colorblind_friendly_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
                
                assert cmap in colorblind_friendly_colormaps, \
                    f"Colormap {cmap} is not colorblind friendly"
                
                print(f"Colorblind friendly colormap: {cmap}")
                
        finally:
            if plot:
                plot.close()
    
    def test_contrast_ratio_visual_check(self, mock_data_factory):
        """Test that plots have sufficient visual contrast."""
        # This is a basic check - in a full implementation, you'd use proper contrast ratio calculations
        plot = self._create_test_plot("ScatterPlot", mock_data_factory)
        
        if plot:
            try:
                # Check that plot elements are visible
                collections = plot.ax.collections
                lines = plot.ax.lines
                
                # Should have visible elements
                assert len(collections) > 0 or len(lines) > 0, "Plot should have visible elements"
                
                # If there are scatter points, they should be colored
                if collections:
                    colors = collections[0].get_facecolors()
                    assert len(colors) > 0, "Scatter points should have colors"
                    
            finally:
                plot.close()

    def _create_test_plot(self, plot_class_name: str, mock_data_factory):
        """Create a test plot for visual regression testing."""
        # Create plots with fixed seed for reproducibility
        np.random.seed(42)
        
        try:
            if plot_class_name == "SpatialPlot":
                from src.monet_plots.plots.spatial import SpatialPlot
                plot = SpatialPlot()
                data = mock_data_factory.spatial_2d(seed=42)
                plot.plot(data, discrete=True, ncolors=12)
                return plot
            
            elif plot_class_name == "TimeSeriesPlot":
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                plot = TimeSeriesPlot()
                data = mock_data_factory.time_series(n_points=50, seed=42)
                plot.plot(data, title='Test Time Series', ylabel='Concentration (ppb)')
                return plot
            
            elif plot_class_name == "ScatterPlot":
                from src.monet_plots.plots.scatter import ScatterPlot
                plot = ScatterPlot()
                data = mock_data_factory.scatter_data(n_points=100, seed=42)
                plot.plot(data, 'x', 'y', title='Test Scatter Plot', label='Test Data')
                return plot
            
            elif plot_class_name == "KDEPlot":
                from src.monet_plots.plots.kde import KDEPlot
                plot = KDEPlot()
                data = mock_data_factory.kde_data(n_points=1000, seed=42)
                plot.plot(data, title='Test KDE Plot', label='Distribution')
                return plot
            
            elif plot_class_name == "TaylorDiagramPlot":
                from src.monet_plots.plots.taylor import TaylorDiagramPlot
                data = mock_data_factory.taylor_data(n_points=100, seed=42)
                obs_std = data['obs'].std()
                plot = TaylorDiagramPlot(obs_std)
                plot.add_sample(data, label='Test Model')
                plot.add_contours(colors='0.5')
                plot.finish_plot()
                return plot
            
            elif plot_class_name == "XarraySpatialPlot":
                from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
                plot = XarraySpatialPlot()
                data = mock_data_factory.xarray_data(seed=42)
                plot.plot(data, cmap='viridis')
                return plot
            
            elif plot_class_name == "FacetGridPlot":
                from src.monet_plots.plots.facet_grid import FacetGridPlot
                data = mock_data_factory.facet_data(seed=42)
                plot = FacetGridPlot(data, col='time')
                plot.plot()
                return plot
            
        except ImportError:
            pytest.skip(f"{plot_class_name} not available")
        
        return None


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_visual_test():
    """Clean up matplotlib figures after each visual test."""
    yield
    plt.close('all')
    plt.clf()