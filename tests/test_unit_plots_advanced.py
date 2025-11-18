"""
Advanced Unit Tests for MONET Plots - Individual Plot Classes

This module contains unit tests for advanced plot classes using TDD approach.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestScatterPlotUnit:
    """Unit tests for ScatterPlot class."""
    
    def test_scatter_plot_initialization(self, mock_data_factory):
        """Test ScatterPlot initialization."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        plot.close()
    
    def test_scatter_plot_basic_functionality(self, mock_data_factory):
        """Test ScatterPlot basic plotting functionality."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data()
        
        result = plot.plot(df, 'x', 'y')
        
        assert result is None  # plot method doesn't return anything
        assert plot.ax is not None
        assert len(plot.ax.collections) > 0  # Should have scatter points
        assert len(plot.ax.lines) > 0  # Should have regression line
        plot.close()
    
    def test_scatter_plot_with_title_and_label(self, mock_data_factory):
        """Test ScatterPlot with custom title and label."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data()
        
        plot.plot(df, 'x', 'y', title='Test Scatter Plot', label='Data Points')
        
        assert plot.ax.get_title() == 'Test Scatter Plot'
        assert plot.ax.legend_ is not None
        plot.close()
    
    def test_scatter_plot_custom_regression_params(self, mock_data_factory):
        """Test ScatterPlot with custom regression parameters."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data()
        
        plot.plot(df, 'x', 'y', ci=95, scatter_kws={'alpha': 0.6, 's': 50},
                  line_kws={'color': 'red'})
        
        assert plot.ax is not None
        plot.close()
    
    def test_scatter_plot_invalid_column_names(self, mock_data_factory):
        """Test ScatterPlot error handling with invalid column names."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data()
        
        with pytest.raises((KeyError, ValueError)):
            plot.plot(df, 'invalid_x', 'y')
        
        with pytest.raises((KeyError, ValueError)):
            plot.plot(df, 'x', 'invalid_y')
        
        plot.close()
    
    def test_scatter_plot_insufficient_data(self, mock_data_factory):
        """Test ScatterPlot with insufficient data."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = pd.DataFrame({'x': [1], 'y': [2]})  # Only one point
        
        # Should handle gracefully or raise appropriate error
        try:
            plot.plot(df, 'x', 'y')
            assert plot.ax is not None
        except Exception as e:
            assert isinstance(e, (ValueError, TypeError))
        
        plot.close()


class TestKDEPlotUnit:
    """Unit tests for KDEPlot class."""
    
    def test_kde_plot_initialization(self, mock_data_factory):
        """Test KDEPlot initialization."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        plot.close()
    
    def test_kde_plot_basic_functionality(self, mock_data_factory):
        """Test KDEPlot basic plotting functionality."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        data = mock_data_factory.kde_data()
        
        result = plot.plot(data)
        
        assert result is None  # plot method doesn't return anything
        assert plot.ax is not None
        assert len(plot.ax.lines) > 0  # Should have KDE line
        plot.close()
    
    def test_kde_plot_with_title_and_label(self, mock_data_factory):
        """Test KDEPlot with custom title and label."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        data = mock_data_factory.kde_data()
        
        plot.plot(data, title='Test KDE Plot', label='Distribution')
        
        assert plot.ax.get_title() == 'Test KDE Plot'
        assert plot.ax.legend_ is not None
        plot.close()
    
    def test_kde_plot_custom_bandwidth(self, mock_data_factory):
        """Test KDEPlot with custom bandwidth."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        data = mock_data_factory.kde_data()
        
        plot.plot(data, bw=0.5)
        
        assert plot.ax is not None
        plot.close()
    
    def test_kde_plot_dataframe_column(self, mock_data_factory):
        """Test KDEPlot with DataFrame column."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        df = mock_data_factory.time_series()
        
        plot.plot(df['obs'])
        
        assert plot.ax is not None
        plot.close()
    
    def test_kde_plot_invalid_data(self, mock_data_factory):
        """Test KDEPlot error handling with invalid data."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        invalid_data = "not_numeric_data"
        
        with pytest.raises((TypeError, ValueError)):
            plot.plot(invalid_data)
        
        plot.close()
    
    def test_kde_plot_empty_data(self, mock_data_factory):
        """Test KDEPlot error handling with empty data."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        empty_data = np.array([])
        
        with pytest.raises((ValueError, RuntimeError)):
            plot.plot(empty_data)
        
        plot.close()


class TestXarraySpatialPlotUnit:
    """Unit tests for XarraySpatialPlot class."""
    
    def test_xarray_spatial_initialization(self, mock_data_factory):
        """Test XarraySpatialPlot initialization."""
        from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
        
        plot = XarraySpatialPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        plot.close()
    
    def test_xarray_spatial_basic_functionality(self, mock_data_factory):
        """Test XarraySpatialPlot basic plotting functionality."""
        from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
        
        plot = XarraySpatialPlot()
        modelvar = mock_data_factory.xarray_data()
        
        result = plot.plot(modelvar)
        
        assert result is None  # plot method doesn't return anything
        assert plot.ax is not None
        plot.close()
    
    def test_xarray_spatial_with_plot_args(self, mock_data_factory):
        """Test XarraySpatialPlot with custom plot arguments."""
        from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
        
        plot = XarraySpatialPlot()
        modelvar = mock_data_factory.xarray_data()
        
        plot.plot(modelvar, cmap='plasma', vmin=-2, vmax=2)
        
        assert plot.ax is not None
        plot.close()
    
    def test_xarray_spatial_invalid_data(self, mock_data_factory):
        """Test XarraySpatialPlot error handling with invalid data."""
        from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
        
        plot = XarraySpatialPlot()
        invalid_data = np.array([1, 2, 3])  # Not an xarray DataArray
        
        with pytest.raises((AttributeError, TypeError)):
            plot.plot(invalid_data)
        
        plot.close()


class TestFacetGridPlotUnit:
    """Unit tests for FacetGridPlot class."""
    
    def test_facet_grid_initialization(self, mock_data_factory):
        """Test FacetGridPlot initialization."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        data = mock_data_factory.facet_data()
        
        plot = FacetGridPlot(data, col='time')
        
        assert plot.grid is not None
        assert hasattr(plot.grid, 'fig')
        plot.close()
    
    def test_facet_grid_with_row_and_col(self, mock_data_factory):
        """Test FacetGridPlot with both row and column faceting."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        data = mock_data_factory.facet_data()
        
        plot = FacetGridPlot(data, row='x', col='y')
        
        assert plot.grid is not None
        plot.close()
    
    def test_facet_grid_plot_method(self, mock_data_factory):
        """Test FacetGridPlot plot method."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        data = mock_data_factory.facet_data()
        
        plot = FacetGridPlot(data, col='time')
        result = plot.plot()
        
        assert result is None  # plot method doesn't return anything
        assert plot.grid is not None
        plot.close()
    
    def test_facet_grid_invalid_dimension(self, mock_data_factory):
        """Test FacetGridPlot error handling with invalid dimension."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        data = mock_data_factory.facet_data()
        
        plot = FacetGridPlot(data, col='invalid_dim')
        
        with pytest.raises((KeyError, ValueError)):
            plot.plot()
        
        plot.close()


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close('all')
    plt.clf()