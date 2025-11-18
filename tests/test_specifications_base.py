"""
Base Test Specifications for MONET Plots Testing Framework

This module contains base test specifications and common fixtures.
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

# Import all plot classes
from src.monet_plots import (
    SpatialPlot, TimeSeriesPlot, TaylorDiagramPlot, ScatterPlot, 
    KDEPlot, WindQuiverPlot, WindBarbsPlot, SpatialBiasScatterPlot,
    SpatialContourPlot, XarraySpatialPlot, FacetGridPlot
)
from src.monet_plots.plots.base import BasePlot


class TestSpecifications:
    """Base test specifications class providing common fixtures and utilities."""
    
    @pytest.fixture(scope="class")
    def mock_data_generators(self):
        """Mock data generators for different plot types."""
        class MockData:
            def spatial_2d(self, shape=(10, 10), seed=42):
                """Generate 2D spatial data."""
                np.random.seed(seed)
                return np.random.randn(*shape)
            
            def time_series(self, n_points=100, start_date='2025-01-01', seed=42):
                """Generate time series data."""
                np.random.seed(seed)
                dates = pd.date_range(start=start_date, periods=n_points, freq='D')
                values = np.cumsum(np.random.randn(n_points)) + 20
                return pd.DataFrame({
                    'time': dates,
                    'obs': values,
                    'model': values + np.random.randn(n_points) * 0.5,
                    'units': 'ppb'
                })
            
            def scatter_data(self, n_points=50, seed=42):
                """Generate scatter plot data."""
                np.random.seed(seed)
                x = np.random.randn(n_points)
                y = x * 1.5 + np.random.randn(n_points) * 0.5
                return pd.DataFrame({'x': x, 'y': y})
            
            def kde_data(self, n_points=1000, seed=42):
                """Generate data for KDE plot."""
                np.random.seed(seed)
                return np.random.randn(n_points)
            
            def taylor_data(self, n_points=100, seed=42):
                """Generate data for Taylor diagram."""
                np.random.seed(seed)
                obs = np.random.randn(n_points) * 2 + 20
                model = obs + np.random.randn(n_points) * 0.3
                return pd.DataFrame({'obs': obs, 'model': model})
            
            def spatial_dataframe(self, n_points=50, seed=42):
                """Generate spatial point data."""
                np.random.seed(seed)
                return pd.DataFrame({
                    'latitude': np.random.uniform(25, 50, n_points),
                    'longitude': np.random.uniform(-120, -70, n_points),
                    'CMAQ': np.random.uniform(0, 50, n_points),
                    'Obs': np.random.uniform(0, 50, n_points),
                    'datetime': pd.to_datetime('2025-01-01')
                })
            
            def xarray_data(self, shape=(5, 5), seed=42):
                """Generate xarray DataArray."""
                np.random.seed(seed)
                data = np.random.randn(*shape)
                lat = np.linspace(25, 50, shape[0])
                lon = np.linspace(-120, -70, shape[1])
                return xr.DataArray(
                    data,
                    coords=[('latitude', lat), ('longitude', lon)],
                    dims=['latitude', 'longitude']
                )
            
            def facet_data(self, seed=42):
                """Generate data for facet grid."""
                np.random.seed(seed)
                data = np.random.randn(3, 4, 5)
                return xr.DataArray(
                    data,
                    dims=['x', 'y', 'time'],
                    coords={
                        'x': [1, 2, 3],
                        'y': [1, 2, 3, 4],
                        'time': [1, 2, 3, 4, 5]
                    }
                )
        
        return MockData()
    
    @pytest.fixture(scope="class")
    def temp_directory(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture(autouse=True)
    def cleanup_plots(self):
        """Clean up matplotlib figures after each test."""
        yield
        plt.close('all')
        plt.clf()


class TestBasePlot(TestSpecifications):
    """Test specifications for BasePlot class."""
    
    def test_base_plot_initialization(self, mock_data_generators):
        """Test BasePlot initialization with Wiley style."""
        plot = BasePlot()
        assert plot.fig is not None
        assert plot.ax is not None
        assert hasattr(plot, 'fig')
        assert hasattr(plot, 'ax')
        plot.close()
    
    def test_base_plot_save_functionality(self, temp_directory, mock_data_generators):
        """Test BasePlot save functionality."""
        plot = BasePlot()
        filename = os.path.join(temp_directory, 'test_save.png')
        plot.save(filename, dpi=30)
        assert os.path.exists(filename)
        plot.close()
    
    def test_base_plot_close_functionality(self, mock_data_generators):
        """Test BasePlot close functionality."""
        plot = BasePlot()
        fig_num = plot.fig.number
        plot.close()
        assert plt.fignum_exists(fig_num) is False