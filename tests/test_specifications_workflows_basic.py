"""
Basic Workflow Test Specifications for MONET Plots Testing Framework

This module contains test specifications for basic integration workflows and performance benchmarks.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import time
import warnings

# Import basic plot classes
from src.monet_plots import (
    SpatialPlot, TimeSeriesPlot, TaylorDiagramPlot, ScatterPlot, 
    KDEPlot
)
from src.monet_plots.plots.base import BasePlot
from tests.test_specifications_base import TestSpecifications


class TestIntegrationWorkflows(TestSpecifications):
    """Test specifications for integration workflows."""
    
    def test_complete_analysis_workflow(self, mock_data_generators, temp_directory):
        """Test complete analysis workflow with multiple plot types."""
        # Generate data
        spatial_data = mock_data_generators.spatial_2d()
        ts_data = mock_data_generators.time_series()
        scatter_data = mock_data_generators.scatter_data()
        taylor_data = mock_data_generators.taylor_data()
        
        output_files = []
        
        # Create spatial plot
        spatial_plot = SpatialPlot()
        spatial_plot.plot(spatial_data)
        spatial_file = os.path.join(temp_directory, 'spatial.png')
        spatial_plot.save(spatial_file)
        output_files.append(spatial_file)
        spatial_plot.close()
        
        # Create time series plot
        ts_plot = TimeSeriesPlot()
        ts_plot.plot(ts_data)
        ts_file = os.path.join(temp_directory, 'timeseries.png')
        ts_plot.save(ts_file)
        output_files.append(ts_file)
        ts_plot.close()
        
        # Create scatter plot
        scatter_plot = ScatterPlot()
        scatter_plot.plot(scatter_data, 'x', 'y')
        scatter_file = os.path.join(temp_directory, 'scatter.png')
        scatter_plot.save(scatter_file)
        output_files.append(scatter_file)
        scatter_plot.close()
        
        # Create Taylor diagram
        obs_std = taylor_data['obs'].std()
        taylor_plot = TaylorDiagramPlot(obs_std)
        taylor_plot.add_sample(taylor_data)
        taylor_plot.finish_plot()
        taylor_file = os.path.join(temp_directory, 'taylor.png')
        taylor_plot.save(taylor_file)
        output_files.append(taylor_file)
        taylor_plot.close()
        
        # Verify all files were created
        for file_path in output_files:
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) > 0
    
    def test_data_pipeline_with_plotting(self, mock_data_generators):
        """Test data processing pipeline that includes plotting."""
        # Simulate a data processing pipeline
        raw_data = mock_data_generators.time_series()
        
        # Process data
        processed_data = raw_data.copy()
        processed_data['model_bias'] = processed_data['model'] - processed_data['obs']
        
        # Create plots at different stages
        plots_created = []
        
        # Initial data plot
        initial_plot = TimeSeriesPlot()
        initial_plot.plot(raw_data)
        plots_created.append(initial_plot)
        
        # Processed data plot
        processed_plot = ScatterPlot()
        processed_plot.plot(processed_data, 'obs', 'model_bias')
        plots_created.append(processed_plot)
        
        # Clean up
        for plot in plots_created:
            plot.close()
    
    def test_multiple_plots_same_figure(self, mock_data_generators):
        """Test creating multiple plots on the same figure."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Create different plot types on subplots
        spatial_plot = SpatialPlot(fig=fig, ax=axes[0, 0])
        spatial_data = mock_data_generators.spatial_2d()
        spatial_plot.plot(spatial_data)
        
        ts_plot = TimeSeriesPlot(fig=fig, ax=axes[0, 1])
        ts_data = mock_data_generators.time_series()
        ts_plot.plot(ts_data)
        
        scatter_plot = ScatterPlot(fig=fig, ax=axes[1, 0])
        scatter_data = mock_data_generators.scatter_data()
        scatter_plot.plot(scatter_data, 'x', 'y')
        
        kde_plot = KDEPlot(fig=fig, ax=axes[1, 1])
        kde_data = mock_data_generators.kde_data()
        kde_plot.plot(kde_data)
        
        plt.tight_layout()
        
        # Clean up
        for plot in [spatial_plot, ts_plot, scatter_plot, kde_plot]:
            plot.close()


class TestPerformanceBenchmarks(TestSpecifications):
    """Test specifications for performance benchmarks."""
    
    @pytest.mark.parametrize("data_size", [100, 1000, 10000])
    def test_spatial_plot_performance(self, mock_data_generators, data_size):
        """Benchmark SpatialPlot performance with different data sizes."""
        # Generate spatial data of different sizes
        shape = (int(data_size**0.5), int(data_size**0.5))
        modelvar = mock_data_generators.spatial_2d(shape=shape)
        
        plot = SpatialPlot()
        
        start_time = time.time()
        plot.plot(modelvar)
        end_time = time.time()
        
        plot.close()
        
        execution_time = end_time - start_time
        
        # Performance assertion (should complete within reasonable time)
        assert execution_time < 10.0, f"SpatialPlot took {execution_time:.2f}s for {data_size} points"
    
    @pytest.mark.parametrize("n_points", [100, 1000, 10000])
    def test_timeseries_plot_performance(self, mock_data_generators, n_points):
        """Benchmark TimeSeriesPlot performance with different data sizes."""
        df = mock_data_generators.time_series(n_points=n_points)
        
        plot = TimeSeriesPlot()
        
        start_time = time.time()
        plot.plot(df)
        end_time = time.time()
        
        plot.close()
        
        execution_time = end_time - start_time
        
        # Performance assertion
        assert execution_time < 5.0, f"TimeSeriesPlot took {execution_time:.2f}s for {n_points} points"
    
    @pytest.mark.parametrize("n_samples", [50, 100, 500])
    def test_taylor_diagram_performance(self, mock_data_generators, n_samples):
        """Benchmark TaylorDiagramPlot performance."""
        df = mock_data_generators.taylor_data(n_points=n_samples)
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        
        start_time = time.time()
        plot.add_sample(df)
        end_time = time.time()
        
        plot.close()
        
        execution_time = end_time - start_time
        
        # Performance assertion
        assert execution_time < 2.0, f"TaylorDiagramPlot took {execution_time:.2f}s for {n_samples} samples"
    
    def test_memory_usage_monitoring(self, mock_data_generators):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple plots
        plots = []
        for i in range(10):
            plot = SpatialPlot()
            data = mock_data_generators.spatial_2d()
            plot.plot(data)
            plots.append(plot)
        
        # Measure memory after creating plots
        during_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Close all plots
        for plot in plots:
            plot.close()
        
        # Measure memory after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should not grow excessively (allow 50MB growth)
        memory_growth = during_memory - initial_memory
        assert memory_growth < 50, f"Memory growth too high: {memory_growth:.1f}MB"
        
        # Memory should be cleaned up reasonably well
        cleanup_ratio = (during_memory - final_memory) / memory_growth
        assert cleanup_ratio > 0.5, f"Memory cleanup insufficient: {cleanup_ratio:.1%}"


class TestErrorHandlingEdgeCases(TestSpecifications):
    """Test specifications for error handling and edge cases."""
    
    def test_nan_inf_handling(self, mock_data_generators):
        """Test handling of NaN and infinite values."""
        # Test SpatialPlot with NaN values
        spatial_data = mock_data_generators.spatial_2d()
        spatial_data[0, 0] = np.nan
        spatial_data[1, 1] = np.inf
        
        plot = SpatialPlot()
        
        # Should handle gracefully or raise specific error
        try:
            plot.plot(spatial_data)
            assert plot.ax is not None
        except (ValueError, TypeError) as e:
            assert "nan" in str(e).lower() or "inf" in str(e).lower()
        
        plot.close()
    
    def test_empty_data_handling(self, mock_data_generators):
        """Test handling of empty datasets."""
        plot_classes = [
            (SpatialPlot, lambda p: p.plot(np.array([]))),
            (ScatterPlot, lambda p: p.plot(pd.DataFrame(), 'x', 'y')),
            (KDEPlot, lambda p: p.plot(np.array([]))),
        ]
        
        for plot_class, plot_method in plot_classes:
            plot = plot_class()
            with pytest.raises((ValueError, TypeError, IndexError)):
                plot_method(plot)
            plot.close()
    
    def test_invalid_projection_handling(self, mock_data_generators):
        """Test SpatialPlot with invalid projections."""
        import cartopy.crs as ccrs
        
        # Test with invalid projection
        try:
            invalid_proj = "not_a_projection"
            SpatialPlot(projection=invalid_proj)
        except (TypeError, AttributeError):
            pass  # Expected to fail
        
        # Test with valid but unusual projection
        try:
            unusual_proj = ccrs.Orthographic(0, 0)
            plot = SpatialPlot(projection=unusual_proj)
            plot.close()
        except Exception:
            pytest.skip("Orthographic projection not available")
    
    def test_colorbar_edge_cases(self, mock_data_generators):
        """Test colorbar edge cases."""
        plot = SpatialPlot()
        
        # Test with constant data (vmin == vmax)
        constant_data = np.ones((10, 10))
        
        try:
            plot.plot(constant_data, discrete=True)
            # Should handle constant data gracefully
            assert plot.ax is not None
        except Exception as e:
            # If it fails, should be a specific, expected error
            assert "colorbar" in str(e).lower() or "bounds" in str(e).lower()
        
        plot.close()
    
    def test_file_save_permissions(self, mock_data_generators):
        """Test error handling when file save fails."""
        plot = BasePlot()
        
        # Try to save to a non-existent directory
        with pytest.raises((PermissionError, FileNotFoundError, OSError)):
            plot.save("/nonexistent/directory/test.png")
        
        plot.close()
    
    def test_matplotlib_state_corruption(self, mock_data_generators):
        """Test matplotlib state handling."""
        # Create plot and verify matplotlib state
        plot1 = SpatialPlot()
        plot1.plot(mock_data_generators.spatial_2d())
        
        initial_figures = len(plt.get_fignums())
        
        # Create another plot
        plot2 = TimeSeriesPlot()
        plot2.plot(mock_data_generators.time_series())
        
        # Should have multiple figures
        assert len(plt.get_fignums()) == initial_figures + 1
        
        # Close first plot
        plot1.close()
        
        # Should have one less figure
        assert len(plt.get_fignums()) == initial_figures
        
        plot2.close()


# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        'tolerance': 1e-10,
        'timeout': 30,
        'retry_attempts': 3
    }


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close('all')
    plt.clf()