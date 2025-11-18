"""
Basic Plot Test Specifications for MONET Plots Testing Framework

This module contains test specifications for basic plot classes.
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


class TestSpatialPlot(TestSpecifications):
    """Test specifications for SpatialPlot class."""
    
    @pytest.mark.parametrize("discrete", [True, False])
    @pytest.mark.parametrize("ncolors", [10, 15, 20])
    def test_spatial_plot_basic_functionality(self, mock_data_generators, discrete, ncolors):
        """Test SpatialPlot basic plotting functionality."""
        plot = SpatialPlot()
        modelvar = mock_data_generators.spatial_2d()
        
        plot.plot(modelvar, discrete=discrete, ncolors=ncolors)
        
        assert plot.ax is not None
        assert hasattr(plot, 'cbar')
        plot.close()
    
    def test_spatial_plot_with_custom_cmap(self, mock_data_generators):
        """Test SpatialPlot with custom colormap."""
        plot = SpatialPlot()
        modelvar = mock_data_generators.spatial_2d()
        
        plot.plot(modelvar, plotargs={'cmap': 'plasma'})
        
        assert plot.ax is not None
        plot.close()
    
    def test_spatial_plot_invalid_data(self, mock_data_generators):
        """Test SpatialPlot error handling with invalid data."""
        plot = SpatialPlot()
        invalid_data = "not_an_array"
        
        with pytest.raises((TypeError, AttributeError)):
            plot.plot(invalid_data)
        plot.close()
    
    def test_spatial_plot_empty_data(self, mock_data_generators):
        """Test SpatialPlot error handling with empty data."""
        plot = SpatialPlot()
        empty_data = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            plot.plot(empty_data)
        plot.close()
    
    def test_spatial_plot_projection(self, mock_data_generators):
        """Test SpatialPlot with different projections."""
        import cartopy.crs as ccrs
        
        projections = [ccrs.PlateCarree(), ccrs.Robinson(), ccrs.Mercator()]
        
        for proj in projections:
            plot = SpatialPlot(projection=proj)
            modelvar = mock_data_generators.spatial_2d()
            plot.plot(modelvar)
            assert plot.ax is not None
            plot.close()


class TestTimeSeriesPlot(TestSpecifications):
    """Test specifications for TimeSeriesPlot class."""
    
    def test_timeseries_plot_basic_functionality(self, mock_data_generators):
        """Test TimeSeriesPlot basic plotting functionality."""
        plot = TimeSeriesPlot()
        df = mock_data_generators.time_series()
        
        plot.plot(df)
        
        assert plot.ax is not None
        assert plot.ax.get_xlabel() == ''
        assert len(plot.ax.lines) > 0
        plot.close()
    
    def test_timeseries_plot_custom_columns(self, mock_data_generators):
        """Test TimeSeriesPlot with custom column names."""
        plot = TimeSeriesPlot()
        df = mock_data_generators.time_series()
        df = df.rename(columns={'obs': 'measurement'})
        
        plot.plot(df, x='time', y='measurement', title='Test Plot', ylabel='ppb')
        
        assert plot.ax is not None
        assert plot.ax.get_title() == 'Test Plot'
        plot.close()
    
    def test_timeseries_plot_no_std_dev_handling(self, mock_data_generators):
        """Test TimeSeriesPlot handling when std dev creates negative values."""
        plot = TimeSeriesPlot()
        # Create data where std dev would make lower bound negative
        df = pd.DataFrame({
            'time': pd.date_range('2025-01-01', periods=10),
            'obs': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Zero std dev
            'model': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })
        
        plot.plot(df)
        assert plot.ax is not None
        plot.close()
    
    def test_timeseries_plot_missing_columns(self, mock_data_generators):
        """Test TimeSeriesPlot error handling with missing columns."""
        plot = TimeSeriesPlot()
        df = pd.DataFrame({'x': [1, 2, 3]})  # Missing 'time' and 'obs'
        
        with pytest.raises(KeyError):
            plot.plot(df)
        plot.close()
    
    def test_timeseries_plot_empty_dataframe(self, mock_data_generators):
        """Test TimeSeriesPlot error handling with empty DataFrame."""
        plot = TimeSeriesPlot()
        df = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            plot.plot(df)
        plot.close()


class TestTaylorDiagramPlot(TestSpecifications):
    """Test specifications for TaylorDiagramPlot class."""
    
    def test_taylor_diagram_basic_functionality(self, mock_data_generators):
        """Test TaylorDiagramPlot basic functionality."""
        df = mock_data_generators.taylor_data()
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        plot.add_sample(df)
        plot.finish_plot()
        
        assert plot.dia is not None
        assert plot.ax is not None
        plot.close()
    
    def test_taylor_diagram_add_contours(self, mock_data_generators):
        """Test TaylorDiagramPlot contour functionality."""
        df = mock_data_generators.taylor_data()
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        plot.add_sample(df)
        plot.add_contours(colors='0.5')
        
        assert plot.dia is not None
        plot.close()
    
    def test_taylor_diagram_multiple_samples(self, mock_data_generators):
        """Test TaylorDiagramPlot with multiple model samples."""
        df1 = mock_data_generators.taylor_data(seed=42)
        df2 = mock_data_generators.taylor_data(seed=123)
        obs_std = df1['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        plot.add_sample(df1, label='Model1')
        plot.add_sample(df2, label='Model2', marker='s')
        plot.finish_plot()
        
        assert plot.dia is not None
        plot.close()
    
    def test_taylor_diagram_invalid_data(self, mock_data_generators):
        """Test TaylorDiagramPlot error handling with invalid data."""
        df = mock_data_generators.taylor_data()
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'x': [1, 2, 3]})
        with pytest.raises((KeyError, TypeError)):
            plot.add_sample(invalid_df)
        
        plot.close()
    
    def test_taylor_diagram_zero_correlation(self, mock_data_generators):
        """Test TaylorDiagramPlot with zero correlation data."""
        # Create data with zero correlation
        df = pd.DataFrame({
            'obs': np.random.randn(100),
            'model': np.random.randn(100) * 10  # Different scale, no correlation
        })
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        plot.add_sample(df)
        
        assert plot.dia is not None
        plot.close()


class TestScatterPlot(TestSpecifications):
    """Test specifications for ScatterPlot class."""
    
    def test_scatter_plot_basic_functionality(self, mock_data_generators):
        """Test ScatterPlot basic functionality."""
        plot = ScatterPlot()
        df = mock_data_generators.scatter_data()
        
        plot.plot(df, 'x', 'y')
        
        assert plot.ax is not None
        assert len(plot.ax.collections) > 0  # Scatter points
        assert len(plot.ax.lines) > 0 # Regression line
        plot.close()
    
    def test_scatter_plot_with_title_and_label(self, mock_data_generators):
        """Test ScatterPlot with custom title and label."""
        plot = ScatterPlot()
        df = mock_data_generators.scatter_data()
        
        plot.plot(df, 'x', 'y', title='Test Scatter Plot', label='Data Points')
        
        assert plot.ax.get_title() == 'Test Scatter Plot'
        assert plot.ax.legend_.get_texts()[0].get_text() == 'Data Points'
        plot.close()
    
    def test_scatter_plot_custom_regression_params(self, mock_data_generators):
        """Test ScatterPlot with custom regression parameters."""
        plot = ScatterPlot()
        df = mock_data_generators.scatter_data()
        
        plot.plot(df, 'x', 'y', ci=95, scatter_kws={'alpha': 0.6})
        
        assert plot.ax is not None
        plot.close()
    
    def test_scatter_plot_invalid_columns(self, mock_data_generators):
        """Test ScatterPlot error handling with invalid columns."""
        plot = ScatterPlot()
        df = mock_data_generators.scatter_data()
        
        with pytest.raises((KeyError, ValueError)):
            plot.plot(df, 'invalid_x', 'y')
        
        with pytest.raises((KeyError, ValueError)):
            plot.plot(df, 'x', 'invalid_y')
        
        plot.close()
    
    def test_scatter_plot_insufficient_data(self, mock_data_generators):
        """Test ScatterPlot error handling with insufficient data."""
        plot = ScatterPlot()
        df = pd.DataFrame({'x': [1], 'y': [2]})  # Only one point
        
        # Should handle gracefully or raise appropriate error
        try:
            plot.plot(df, 'x', 'y')
            assert plot.ax is not None
        except Exception as e:
            assert isinstance(e, (ValueError, TypeError))
        
        plot.close()


class TestKDEPlot(TestSpecifications):
    """Test specifications for KDEPlot class."""
    
    def test_kde_plot_basic_functionality(self, mock_data_generators):
        """Test KDEPlot basic functionality."""
        plot = KDEPlot()
        data = mock_data_generators.kde_data()
        
        plot.plot(data)
        
        assert plot.ax is not None
        assert len(plot.ax.lines) > 0  # KDE line
        plot.close()
    
    def test_kde_plot_with_title_and_label(self, mock_data_generators):
        """Test KDEPlot with custom title and label."""
        plot = KDEPlot()
        data = mock_data_generators.kde_data()
        
        plot.plot(data, title='Test KDE Plot', label='Distribution')
        
        assert plot.ax.get_title() == 'Test KDE Plot'
        assert plot.ax.legend_.get_texts()[0].get_text() == 'Distribution'
        plot.close()
    
    def test_kde_plot_custom_bandwidth(self, mock_data_generators):
        """Test KDEPlot with custom bandwidth."""
        plot = KDEPlot()
        data = mock_data_generators.kde_data()
        
        plot.plot(data, bw=0.5)
        
        assert plot.ax is not None
        plot.close()
    
    def test_kde_plot_dataframe_column(self, mock_data_generators):
        """Test KDEPlot with DataFrame column."""
        plot = KDEPlot()
        df = mock_data_generators.time_series()
        
        plot.plot(df['obs'])
        
        assert plot.ax is not None
        plot.close()
    
    def test_kde_plot_invalid_data(self, mock_data_generators):
        """Test KDEPlot error handling with invalid data."""
        plot = KDEPlot()
        invalid_data = "not_numeric_data"
        
        with pytest.raises((TypeError, ValueError)):
            plot.plot(invalid_data)
        
        plot.close()
    
    def test_kde_plot_empty_data(self, mock_data_generators):
        """Test KDEPlot error handling with empty data."""
        plot = KDEPlot()
        empty_data = np.array([])
        
        with pytest.raises((ValueError, RuntimeError)):
            plot.plot(empty_data)
        
        plot.close()


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close('all')
    plt.clf()