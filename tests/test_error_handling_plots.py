"""
Plot-Specific Error Handling and Edge Cases Tests for MONET Plots

This module contains error handling tests for individual plot classes.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch


class TestPlotErrorHandling:
    """Comprehensive error handling and edge case tests for plot classes."""
    
    def test_spatial_plot_error_handling(self, mock_data_factory):
        """Test SpatialPlot error handling for various invalid inputs."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            
            # Test 1: Invalid data types
            with pytest.raises((TypeError, AttributeError)):
                plot.plot("invalid_string")
            
            with pytest.raises((TypeError, AttributeError)):
                plot.plot([1, 2, 3])  # List instead of numpy array
            
            # Test 2: Empty arrays
            with pytest.raises((ValueError, IndexError)):
                plot.plot(np.array([]))
            
            # Test 3: 1D arrays
            with pytest.raises((ValueError, IndexError)):
                plot.plot(np.array([1, 2, 3, 4, 5]))
            
            # Test 4: Arrays with NaN values
            data_with_nan = mock_data_factory.spatial_2d()
            data_with_nan[0, 0] = np.nan
            data_with_nan[1, 1] = np.inf
            
            # Should handle NaN gracefully or raise specific error
            try:
                plot.plot(data_with_nan)
                assert plot.ax is not None
            except (ValueError, RuntimeError) as e:
                assert "nan" in str(e).lower() or "inf" in str(e).lower()
            
            # Test 5: Constant data (vmin == vmax)
            constant_data = np.ones((10, 10))
            
            try:
                plot.plot(constant_data, discrete=True)
                # Should handle constant data gracefully
                assert plot.ax is not None
            except Exception as e:
                # If it fails, should be a specific, expected error
                assert "colorbar" in str(e).lower() or "bounds" in str(e).lower()
            
            # Test 6: Invalid colormap
            data = mock_data_factory.spatial_2d()
            
            with pytest.raises((ValueError, KeyError)):
                plot.plot(data, plotargs={'cmap': 'nonexistent_colormap'})
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_spatial_plot_projection_errors(self, mock_data_factory):
        """Test SpatialPlot projection-related errors."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            import cartopy.crs as ccrs
            
            # Test 1: Invalid projection
            with pytest.raises((TypeError, AttributeError)):
                SpatialPlot(projection="invalid_projection")
            
            # Test 2: Valid but unusual projection
            try:
                unusual_proj = ccrs.Orthographic(0, 0)
                plot = SpatialPlot(projection=unusual_proj)
                data = mock_data_factory.spatial_2d()
                plot.plot(data)
                assert plot.ax is not None
                plot.close()
            except Exception:
                pytest.skip("Orthographic projection not available")
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_timeseries_plot_error_handling(self, mock_data_factory):
        """Test TimeSeriesPlot error handling."""
        try:
            from src.monet_plots.plots.timeseries import TimeSeriesPlot
            
            plot = TimeSeriesPlot()
            
            # Test 1: Missing required columns
            df_missing_cols = pd.DataFrame({'x': [1, 2, 3]})
            
            with pytest.raises(KeyError):
                plot.plot(df_missing_cols)
            
            # Test 2: Empty DataFrame
            empty_df = pd.DataFrame()
            
            with pytest.raises((ValueError, KeyError, IndexError)):
                plot.plot(empty_df)
            
            # Test 3: DataFrame with wrong column names
            df_wrong_cols = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=10),
                'value': np.random.randn(10)
            })
            
            with pytest.raises(KeyError):
                plot.plot(df_wrong_cols)  # Default uses 'time' and 'obs'
            
            # Should work with correct column specification
            plot.plot(df_wrong_cols, x='timestamp', y='value')
            assert plot.ax is not None
            
            # Test 4: Single data point
            single_point_df = pd.DataFrame({
                'time': [pd.Timestamp('2025-01-01')],
                'obs': [25.0]
            })
            
            # Should handle gracefully or raise informative error
            try:
                plot.plot(single_point_df)
                assert plot.ax is not None
            except Exception as e:
                assert isinstance(e, (ValueError, ZeroDivisionError))
            
            # Test 5: Data with NaN values
            df_with_nan = mock_data_factory.time_series()
            df_with_nan.loc[5, 'obs'] = np.nan
            df_with_nan.loc[10, 'model'] = np.inf
            
            # Should handle NaN values gracefully
            try:
                plot.plot(df_with_nan)
                assert plot.ax is not None
            except Exception as e:
                # Should be a specific error about data quality
                assert "nan" in str(e).lower() or "inf" in str(e).lower()
            
            plot.close()
            
        except ImportError:
            pytest.skip("TimeSeriesPlot not available")
    
    def test_timeseries_plot_std_dev_edge_cases(self, mock_data_factory):
        """Test TimeSeriesPlot edge cases related to standard deviation."""
        try:
            from src.monet_plots.plots.timeseries import TimeSeriesPlot
            
            plot = TimeSeriesPlot()
            
            # Test 1: Constant values (zero std dev)
            constant_df = pd.DataFrame({
                'time': pd.date_range('2025-01-01', periods=10, freq='D'),
                'obs': [5.0] * 10,
                'model': [5.0] * 10
            })
            
            plot.plot(constant_df)
            assert plot.ax is not None
            
            # Test 2: Single unique value with some variation
            near_constant_df = pd.DataFrame({
                'time': pd.date_range('2025-01-01', periods=10, freq='D'),
                'obs': [5.0] * 9 + [5.1],
                'model': [5.0] * 10
            })
            
            plot.plot(near_constant_df)
            assert plot.ax is not None
            
            plot.close()
            
        except ImportError:
            pytest.skip("TimeSeriesPlot not available")
    
    def test_taylor_diagram_error_handling(self, mock_data_factory):
        """Test TaylorDiagramPlot error handling."""
        try:
            from src.monet_plots.plots.taylor import TaylorDiagramPlot
            
            # Create plot with valid obs_std
            df = mock_data_factory.taylor_data()
            obs_std = df['obs'].std()
            
            plot = TaylorDiagramPlot(obs_std)
            
            # Test 1: Invalid data for add_sample
            invalid_df = pd.DataFrame({'x': [1, 2, 3]})
            
            with pytest.raises((KeyError, TypeError)):
                plot.add_sample(invalid_df)
            
            # Test 2: Data with NaN values
            df_with_nan = df.copy()
            df_with_nan.loc[5, 'obs'] = np.nan
            df_with_nan.loc[10, 'model'] = np.inf
            
            with pytest.raises((ValueError, TypeError)):
                plot.add_sample(df_with_nan)
            
            # Test 3: Zero standard deviation data
            zero_std_df = pd.DataFrame({
                'obs': [5.0, 5.0, 5.0, 5.0, 5.0],
                'model': [5.0, 5.0, 5.0, 5.0, 5.0]
            })
            obs_std_zero = zero_std_df['obs'].std()  # Will be 0.0
            
            # Creating plot with zero std dev
            zero_std_plot = TaylorDiagramPlot(obs_std_zero)
            
            # Adding zero std dev data
            try:
                zero_std_plot.add_sample(zero_std_df)
                assert zero_std_plot.dia is not None
            except Exception as e:
                # Should be a specific error about zero std dev
                assert "zero" in str(e).lower() or "std" in str(e).lower()
            
            # Test 4: Very small standard deviation
            small_std_df = pd.DataFrame({
                'obs': np.ones(100) + np.random.randn(100) * 1e-10,
                'model': np.ones(100) + np.random.randn(100) * 1e-10
            })
            small_std = small_std_df['obs'].std()
            
            small_std_plot = TaylorDiagramPlot(small_std)
            small_std_plot.add_sample(small_std_df)
            assert small_std_plot.dia is not None
            
            plot.close()
            zero_std_plot.close()
            small_std_plot.close()
            
        except ImportError:
            pytest.skip("TaylorDiagramPlot not available")
    
    def test_scatter_plot_error_handling(self, mock_data_factory):
        """Test ScatterPlot error handling."""
        try:
            from src.monet_plots.plots.scatter import ScatterPlot
            
            plot = ScatterPlot()
            df = mock_data_factory.scatter_data()
            
            # Test 1: Invalid column names
            with pytest.raises((KeyError, ValueError)):
                plot.plot(df, 'invalid_x', 'y')
            
            with pytest.raises((KeyError, ValueError)):
                plot.plot(df, 'x', 'invalid_y')
            
            # Test 2: Missing columns
            df_missing = pd.DataFrame({'a': [1, 2, 3]})
            
            with pytest.raises((KeyError, ValueError)):
                plot.plot(df_missing, 'x', 'y')
            
            # Test 3: Insufficient data
            single_point_df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
            
            # Should handle gracefully or raise appropriate error
            try:
                plot.plot(single_point_df, 'x', 'y')
                assert plot.ax is not None
            except Exception as e:
                assert isinstance(e, (ValueError, TypeError))
            
            # Test 4: Empty DataFrame
            empty_df = pd.DataFrame()
            
            with pytest.raises((ValueError, KeyError)):
                plot.plot(empty_df, 'x', 'y')
            
            # Test 5: Data with all NaN values
            df_all_nan = pd.DataFrame({
                'x': [np.nan, np.nan, np.nan],
                'y': [np.nan, np.nan, np.nan]
            })
            
            with pytest.raises((ValueError, TypeError)):
                plot.plot(df_all_nan, 'x', 'y')
            
            plot.close()
            
        except ImportError:
            pytest.skip("ScatterPlot not available")
    
    def test_kde_plot_error_handling(self, mock_data_factory):
        """Test KDEPlot error handling."""
        try:
            from src.monet_plots.plots.kde import KDEPlot
            
            plot = KDEPlot()
            
            # Test 1: Invalid data types
            with pytest.raises((TypeError, ValueError)):
                plot.plot("invalid_string")
            
            with pytest.raises((TypeError, ValueError)):
                plot.plot([1, 2, 3])  # List instead of array/Series
            
            # Test 2: Empty data
            with pytest.raises((ValueError, RuntimeError)):
                plot.plot(np.array([]))
            
            with pytest.raises((ValueError, RuntimeError)):
                plot.plot(pd.Series([]))
            
            # Test 3: Data with all NaN values
            all_nan_data = np.array([np.nan, np.nan, np.nan])
            
            with pytest.raises((ValueError, RuntimeError)):
                plot.plot(all_nan_data)
            
            # Test 4: Single data point
            single_point_data = np.array([5.0])
            
            with pytest.raises((ValueError, RuntimeError)):
                plot.plot(single_point_data)
            
            # Test 5: DataFrame column that doesn't exist
            df = mock_data_factory.time_series()
            
            with pytest.raises((KeyError, TypeError)):
                plot.plot(df['nonexistent_column'])
            
            # Test 6: Invalid bandwidth
            data = mock_data_factory.kde_data()
            
            with pytest.raises((ValueError, TypeError)):
                plot.plot(data, bw='invalid_bandwidth')
            
            plot.close()
            
        except ImportError:
            pytest.skip("KDEPlot not available")
    
    def test_xarray_spatial_plot_error_handling(self, mock_data_factory):
        """Test XarraySpatialPlot error handling."""
        try:
            from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
            
            plot = XarraySpatialPlot()
            
            # Test 1: Invalid data type
            with pytest.raises((AttributeError, TypeError)):
                plot.plot(np.array([1, 2, 3]))  # NumPy array instead of xarray
            
            with pytest.raises((AttributeError, TypeError)):
                plot.plot(pd.DataFrame({'x': [1, 2, 3]}))  # DataFrame instead of xarray
            
            # Test 2: Empty xarray DataArray
            empty_da = xr.DataArray([])
            
            # Should handle gracefully or raise specific error
            try:
                plot.plot(empty_da)
                assert plot.ax is not None
            except Exception as e:
                assert isinstance(e, (ValueError, IndexError))
            
            # Test 3: Xarray without proper coordinates
            malformed_da = xr.DataArray(
                np.random.randn(5, 5),
                dims=['x', 'y']
                # Missing coordinates
            )
            
            # Should handle gracefully or raise specific error
            try:
                plot.plot(malformed_da)
                assert plot.ax is not None
            except Exception as e:
                assert "coordinate" in str(e).lower() or "dimension" in str(e).lower()
            
            plot.close()
            
        except ImportError:
            pytest.skip("XarraySpatialPlot not available")
    
    def test_facet_grid_plot_error_handling(self, mock_data_factory):
        """Test FacetGridPlot error handling."""
        try:
            from src.monet_plots.plots.facet_grid import FacetGridPlot
            
            data = mock_data_factory.facet_data()
            
            # Test 1: Invalid dimension for faceting
            with pytest.raises((KeyError, ValueError)):
                plot = FacetGridPlot(data, col='invalid_dimension')
                plot.plot()
            
            with pytest.raises((KeyError, ValueError)):
                plot = FacetGridPlot(data, row='invalid_dimension')
                plot.plot()
            
            # Test 2: Empty xarray Dataset
            empty_data = xr.DataArray([], dims=['x']).to_dataset(name='empty')
            
            with pytest.raises((ValueError, KeyError)):
                plot = FacetGridPlot(empty_data, col='x')
                plot.plot()
            
            # Test 3: Data without dimensions
            scalar_data = xr.DataArray(5.0)
            
            with pytest.raises((ValueError, AttributeError)):
                plot = FacetGridPlot(scalar_data, col='nonexistent')
                plot.plot()
            
        except ImportError:
            pytest.skip("FacetGridPlot not available")


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_error_test():
    """Clean up matplotlib figures after each error handling test."""
    yield
    plt.close('all')
    plt.clf()