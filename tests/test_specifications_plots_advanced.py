"""
Advanced Plot Test Specifications for MONET Plots Testing Framework

This module contains test specifications for advanced plot classes.
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

# Import advanced plot classes
from src.monet_plots import (
    WindQuiverPlot, WindBarbsPlot, SpatialBiasScatterPlot,
    SpatialContourPlot, XarraySpatialPlot, FacetGridPlot
)
from src.monet_plots.plots.base import BasePlot
from tests.test_specifications_base import TestSpecifications


class TestWindQuiverPlot(TestSpecifications):
    """Test specifications for WindQuiverPlot class."""
    
    def test_wind_quiver_basic_functionality(self, mock_data_generators):
        """Test WindQuiverPlot basic functionality."""
        # Create mock wind data
        ws = mock_data_generators.spatial_2d()
        wdir = mock_data_generators.spatial_2d() * 360  # Degrees
        
        # Mock grid object
        gridobj = Mock()
        gridobj.variables = {
            'LAT': Mock(),
            'LON': Mock()
        }
        gridobj.variables['LAT'][0, 0, :, :].squeeze.return_value = np.linspace(25, 50, ws.shape[0])
        gridobj.variables['LON'][0, 0, :, :].squeeze.return_value = np.linspace(-120, -70, ws.shape[1])
        
        # Mock basemap
        m = Mock()
        m.return_value = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
        
        plot = WindQuiverPlot()
        
        # Mock the tools.wsdir2uv function
        with patch('src.monet_plots.plots.wind_quiver.tools.wsdir2uv') as mock_wsdir2uv:
            mock_wsdir2uv.return_value = (np.ones_like(ws), np.ones_like(wdir))
            m.quiver.return_value = Mock()
            
            result = plot.plot(ws, wdir, gridobj, m)
            
            assert result is not None
            mock_wsdir2uv.assert_called_once_with(ws, wdir)
            m.quiver.assert_called_once()

    def test_wind_quiver_invalid_data_shapes(self, mock_data_generators):
        """Test WindQuiverPlot error handling with mismatched data shapes."""
        ws = np.ones((5, 5))
        wdir = np.ones((3, 3))  # Different shape
        
        gridobj = Mock()
        m = Mock()
        
        plot = WindQuiverPlot()
        
        with pytest.raises((ValueError, IndexError)):
            plot.plot(ws, wdir, gridobj, m)


class TestWindBarbsPlot(TestSpecifications):
    """Test specifications for WindBarbsPlot class."""
    
    def test_wind_barbs_basic_functionality(self, mock_data_generators):
        """Test WindBarbsPlot basic functionality."""
        # Similar to WindQuiverPlot but testing barbs
        ws = mock_data_generators.spatial_2d()
        wdir = mock_data_generators.spatial_2d() * 360
        
        gridobj = Mock()
        gridobj.variables = {
            'LAT': Mock(),
            'LON': Mock()
        }
        gridobj.variables['LAT'][0, 0, :, :].squeeze.return_value = np.linspace(25, 50, ws.shape[0])
        gridobj.variables['LON'][0, 0, :, :].squeeze.return_value = np.linspace(-120, -70, ws.shape[1])
        
        m = Mock()
        m.return_value = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
        
        plot = WindBarbsPlot()
        
        with patch('src.monet_plots.plots.wind_barbs.tools.wsdir2uv') as mock_wsdir2uv:
            mock_wsdir2uv.return_value = (np.ones_like(ws), np.ones_like(wdir))
            
            result = plot.plot(ws, wdir, gridobj, m)
            assert result is None # WindBarbsPlot.plot doesn't return anything
    
    def test_wind_barbs_custom_parameters(self, mock_data_generators):
        """Test WindBarbsPlot with custom barb parameters."""
        ws = mock_data_generators.spatial_2d()
        wdir = mock_data_generators.spatial_2d() * 360
        
        gridobj = Mock()
        m = Mock()
        
        plot = WindBarbsPlot()
        
        with patch('src.monet_plots.plots.wind_barbs.tools.wsdir2uv'):
            plot.plot(ws, wdir, gridobj, m, length=7, pivot='middle')


class TestSpatialBiasScatterPlot(TestSpecifications):
    """Test specifications for SpatialBiasScatterPlot class."""
    
    def test_spatial_bias_scatter_basic_functionality(self, mock_data_generators):
        """Test SpatialBiasScatterPlot basic functionality."""
        df = mock_data_generators.spatial_dataframe()
        
        # Mock basemap
        m = Mock()
        m.side_effect = lambda x, y: (x, y)
        
        plot = SpatialBiasScatterPlot()
        result = plot.plot(df, m, '2025-01-01')
        
        assert len(result) == 3  # Should return (f, ax, c)
        assert result[1] is not None  # ax
        assert result[2] is not None # colorbar
    
    def test_spatial_bias_scatter_missing_columns(self, mock_data_generators):
        """Test SpatialBiasScatterPlot error handling with missing columns."""
        df = pd.DataFrame({
            'latitude': [1, 2, 3],
            'longitude': [1, 2, 3]
            # Missing CMAQ and Obs columns
        })
        
        m = Mock()
        plot = SpatialBiasScatterPlot()
        
        with pytest.raises((KeyError, ValueError)):
            plot.plot(df, m, '2025-01-01')


class TestSpatialContourPlot(TestSpecifications):
    """Test specifications for SpatialContourPlot class."""
    
    def test_spatial_contour_basic_functionality(self, mock_data_generators):
        """Test SpatialContourPlot basic functionality."""
        modelvar = mock_data_generators.spatial_2d()
        
        # Mock grid object
        gridobj = Mock()
        gridobj.variables = {
            'LAT': Mock(),
            'LON': Mock()
        }
        gridobj.variables['LAT'][0, 0, :, :].squeeze.return_value = np.linspace(25, 50, modelvar.shape[0])
        gridobj.variables['LON'][0, 0, :, :].squeeze.return_value = np.linspace(-120, -70, modelvar.shape[1])
        
        # Mock basemap
        m = Mock()
        m.contourf.return_value = Mock()
        
        date = datetime(2025, 1, 1)
        
        plot = SpatialContourPlot()
        result = plot.plot(modelvar, gridobj, date, m, levels=10, cmap='viridis')
        
        assert result is not None
        m.contourf.assert_called_once()
    
    def test_spatial_contour_invalid_levels(self, mock_data_generators):
        """Test SpatialContourPlot error handling with invalid levels."""
        modelvar = mock_data_generators.spatial_2d()
        gridobj = Mock()
        m = Mock()
        date = datetime(2025, 1, 1)
        
        plot = SpatialContourPlot()
        
        with pytest.raises((ValueError, TypeError)):
            plot.plot(modelvar, gridobj, date, m, levels="invalid")


class TestXarraySpatialPlot(TestSpecifications):
    """Test specifications for XarraySpatialPlot class."""
    
    def test_xarray_spatial_basic_functionality(self, mock_data_generators):
        """Test XarraySpatialPlot basic functionality."""
        modelvar = mock_data_generators.xarray_data()
        
        plot = XarraySpatialPlot()
        plot.plot(modelvar)
        
        assert plot.ax is not None
        plot.close()
    
    def test_xarray_spatial_with_plot_args(self, mock_data_generators):
        """Test XarraySpatialPlot with custom plot arguments."""
        modelvar = mock_data_generators.xarray_data()
        
        plot = XarraySpatialPlot()
        plot.plot(modelvar, cmap='plasma', vmin=-2, vmax=2)
        
        assert plot.ax is not None
        plot.close()
    
    def test_xarray_spatial_invalid_data(self, mock_data_generators):
        """Test XarraySpatialPlot error handling with invalid data."""
        plot = XarraySpatialPlot()
        invalid_data = np.array([1, 2, 3])  # Not an xarray DataArray
        
        with pytest.raises((AttributeError, TypeError)):
            plot.plot(invalid_data)
        
        plot.close()


class TestFacetGridPlot(TestSpecifications):
    """Test specifications for FacetGridPlot class."""
    
    def test_facet_grid_basic_functionality(self, mock_data_generators):
        """Test FacetGridPlot basic functionality."""
        data = mock_data_generators.facet_data()
        
        plot = FacetGridPlot(data, col='time')
        
        assert plot.grid is not None
        assert hasattr(plot.grid, 'fig')
        plot.close()
    
    def test_facet_grid_with_row_and_col(self, mock_data_generators):
        """Test FacetGridPlot with both row and column faceting."""
        data = mock_data_generators.facet_data()
        
        plot = FacetGridPlot(data, row='x', col='y')
        
        assert plot.grid is not None
        plot.close()
    
    def test_facet_grid_plot_functionality(self, mock_data_generators):
        """Test FacetGridPlot plot method."""
        data = mock_data_generators.facet_data()
        
        plot = FacetGridPlot(data, col='time')
        plot.plot()
        
        assert plot.grid is not None
        plot.close()
    
    def test_facet_grid_invalid_dimension(self, mock_data_generators):
        """Test FacetGridPlot error handling with invalid dimension."""
        data = mock_data_generators.facet_data()
        
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