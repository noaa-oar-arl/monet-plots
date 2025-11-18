"""
General Error Handling and Edge Cases Tests for MONET Plots

This module contains general error handling, edge cases, and boundary condition tests.
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


class TestGeneralErrorHandling:
    """General error handling and edge case tests."""
    
    def test_base_plot_error_handling(self, mock_data_factory):
        """Test BasePlot error handling."""
        try:
            from src.monet_plots.plots.base import BasePlot
            
            plot = BasePlot()
            
            # Test 1: Save to invalid path
            with pytest.raises((PermissionError, FileNotFoundError, OSError)):
                plot.save("/nonexistent/directory/test.png")
            
            with pytest.raises((PermissionError, FileNotFoundError, OSError)):
                plot.save("")  # Empty filename
            
            # Test 2: Save with invalid arguments
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = os.path.join(temp_dir, "test.png")
                
                # Should handle invalid save arguments gracefully
                try:
                    plot.save(test_file, invalid_argument=True)
                except TypeError:
                    pass  # Expected for invalid arguments
            
            plot.close()
            
        except ImportError:
            pytest.skip("BasePlot not available")
    
    def test_file_save_permissions_and_errors(self, mock_data_factory):
        """Test error handling for file save operations."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            data = mock_data_factory.spatial_2d()
            plot.plot(data)
            
            # Test 1: Save to read-only directory (if possible)
            try:
                # Try to save to root directory (should fail on most systems)
                plot.save("/test_readonly.png")
            except (PermissionError, OSError):
                pass  # Expected failure
            
            # Test 2: Invalid file format
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    plot.save(os.path.join(temp_dir, "test.invalid"))
                except (ValueError, KeyError):
                    pass  # Expected for invalid format
            
            # Test 3: Save with invalid DPI
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    plot.save(os.path.join(temp_dir, "test.png"), dpi="invalid")
                except (TypeError, ValueError):
                    pass  # Expected for invalid DPI
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_matplotlib_state_corruption(self, mock_data_factory):
        """Test matplotlib state handling and corruption prevention."""
        # Create multiple plots and verify matplotlib state
        initial_figures = len(plt.get_fignums())
        
        plots_created = []
        
        try:
            # Create several plots
            try:
                from src.monet_plots.plots.spatial import SpatialPlot
                plot1 = SpatialPlot()
                plot1.plot(mock_data_factory.spatial_2d())
                plots_created.append(plot1)
            except ImportError:
                pass
            
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                plot2 = TimeSeriesPlot()
                plot2.plot(mock_data_factory.time_series())
                plots_created.append(plot2)
            except ImportError:
                pass
            
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                plot3 = ScatterPlot()
                df = mock_data_factory.scatter_data()
                plot3.plot(df, 'x', 'y')
                plots_created.append(plot3)
            except ImportError:
                pass
            
            # Verify matplotlib state
            figures_after_creation = len(plt.get_fignums())
            expected_figures = initial_figures + len(plots_created)
            assert figures_after_creation == expected_figures
            
            # Close plots one by one and verify state
            for i, plot in enumerate(plots_created):
                plot.close()
                remaining_figures = len(plt.get_fignums())
                expected_remaining = expected_figures - (i + 1)
                assert remaining_figures == expected_remaining
            
            # Final state should match initial state
            assert len(plt.get_fignums()) == initial_figures
            
        finally:
            # Ensure cleanup
            for plot in plots_created:
                try:
                    plot.close()
                except:
                    pass
    
    def test_memory_error_handling(self, mock_data_factory):
        """Test behavior when memory errors occur."""
        # This test simulates memory pressure scenarios
        plots_created = []
        
        try:
            # Create many plots to simulate memory pressure
            for i in range(10):
                try:
                    from src.monet_plots.plots.spatial import SpatialPlot
                    plot = SpatialPlot()
                    data = mock_data_factory.spatial_2d()
                    plot.plot(data)
                    plots_created.append(plot)
                    
                    # Verify each plot is valid
                    assert plot.ax is not None
                    
                except ImportError:
                    pytest.skip("SpatialPlot not available")
            
            # Verify we can still create plots under "memory pressure"
            assert len(plots_created) > 0
            
        finally:
            # Clean up all plots
            for plot in plots_created:
                try:
                    plot.close()
                except:
                    pass
    
    def test_invalid_plot_arguments(self, mock_data_factory):
        """Test handling of invalid plot arguments."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            data = mock_data_factory.spatial_2d()
            
            # Test invalid plot arguments
            with pytest.raises((TypeError, ValueError, KeyError)):
                plot.plot(data, plotargs={'invalid_param': 'value'})
            
            # Test invalid keyword arguments
            with pytest.raises(TypeError):
                plot.plot(data, invalid_kwarg=True)
            
            # Test invalid discrete parameter
            with pytest.raises((TypeError, ValueError)):
                plot.plot(data, discrete="invalid_boolean")
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_error_test():
    """Clean up matplotlib figures after each error handling test."""
    yield
    plt.close('all')
    plt.clf()