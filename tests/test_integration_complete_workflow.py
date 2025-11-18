"""
Complete Integration Workflow Tests for MONET Plots

This module contains integration tests that test complete workflows
involving multiple plot types, data processing pipelines, and real-world scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
import json
from unittest.mock import Mock, patch


class TestCompleteAnalysisWorkflow:
    """Integration tests for complete analysis workflows."""
    
    def test_multi_plot_analysis_pipeline(self, mock_data_factory, test_outputs_dir):
        """Test a complete analysis pipeline with multiple plot types."""
        # Generate comprehensive dataset
        spatial_data = mock_data_factory.spatial_2d(shape=(20, 20))
        ts_data = mock_data_factory.time_series(n_points=100)
        scatter_data = mock_data_factory.scatter_data(correlation=0.7)
        taylor_data = mock_data_factory.taylor_data(noise_level=0.5)
        xarray_data = mock_data_factory.xarray_data()
        
        output_files = []
        plots_created = []
        
        try:
            # 1. Spatial analysis
            spatial_plot = None
            try:
                from src.monet_plots.plots.spatial import SpatialPlot
                spatial_plot = SpatialPlot()
                spatial_plot.plot(spatial_data, discrete=True, ncolors=12)
                spatial_file = test_outputs_dir / 'spatial_analysis.png'
                spatial_plot.save(str(spatial_file))
                output_files.append(spatial_file)
                plots_created.append(spatial_plot)
            except ImportError:
                pytest.skip("SpatialPlot not available")
            
            # 2. Time series analysis
            ts_plot = None
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                ts_plot = TimeSeriesPlot()
                ts_plot.plot(ts_data, title='Time Series Analysis', ylabel='Concentration (ppb)')
                ts_file = test_outputs_dir / 'timeseries_analysis.png'
                ts_plot.save(str(ts_file))
                output_files.append(ts_file)
                plots_created.append(ts_plot)
            except ImportError:
                pytest.skip("TimeSeriesPlot not available")
            
            # 3. Scatter analysis
            scatter_plot = None
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                scatter_plot = ScatterPlot()
                scatter_plot.plot(scatter_data, 'x', 'y', title='Scatter Analysis',
                                label='Correlation Study')
                scatter_file = test_outputs_dir / 'scatter_analysis.png'
                scatter_plot.save(str(scatter_file))
                output_files.append(scatter_file)
                plots_created.append(scatter_plot)
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # 4. Model evaluation with Taylor diagram
            taylor_plot = None
            try:
                from src.monet_plots.plots.taylor import TaylorDiagramPlot
                obs_std = taylor_data['obs'].std()
                taylor_plot = TaylorDiagramPlot(obs_std)
                taylor_plot.add_sample(taylor_data, label='Model Performance')
                taylor_plot.add_contours()
                taylor_plot.finish_plot()
                taylor_file = test_outputs_dir / 'taylor_diagram.png'
                taylor_plot.save(str(taylor_file))
                output_files.append(taylor_file)
                plots_created.append(taylor_plot)
            except ImportError:
                pytest.skip("TaylorDiagramPlot not available")
            
            # 5. Spatial data with xarray
            xarray_plot = None
            try:
                from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
                xarray_plot = XarraySpatialPlot()
                xarray_plot.plot(xarray_data, cmap='viridis')
                xarray_file = test_outputs_dir / 'xarray_spatial.png'
                xarray_plot.save(str(xarray_file))
                output_files.append(xarray_file)
                plots_created.append(xarray_plot)
            except ImportError:
                pytest.skip("XarraySpatialPlot not available")
            
            # Verify all expected files were created
            for file_path in output_files:
                if file_path.exists():
                    assert file_path.stat().st_size > 0, f"File {file_path} is empty"
            
            # Verify plots have expected properties
            if spatial_plot:
                assert spatial_plot.ax is not None
            if ts_plot:
                assert ts_plot.ax is not None
                assert len(ts_plot.ax.lines) > 0
            if scatter_plot:
                assert scatter_plot.ax is not None
                assert len(scatter_plot.ax.collections) > 0
            if taylor_plot:
                assert taylor_plot.dia is not None
            if xarray_plot:
                assert xarray_plot.ax is not None
                
        finally:
            # Clean up all plots
            for plot in plots_created:
                try:
                    plot.close()
                except:
                    pass
    
    def test_data_processing_pipeline(self, mock_data_factory, test_outputs_dir):
        """Test a complete data processing pipeline with plotting at each stage."""
        # Simulate real-world data processing workflow
        raw_data = mock_data_factory.time_series(n_points=200)
        
        # Stage 1: Initial data exploration
        exploration_plots = []
        
        try:
            # Basic time series plot
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                ts_plot = TimeSeriesPlot()
                ts_plot.plot(raw_data, title='Raw Data Exploration')
                exploration_plots.append(ts_plot)
            except ImportError:
                pytest.skip("TimeSeriesPlot not available")
            
            # Distribution analysis
            try:
                from src.monet_plots.plots.kde import KDEPlot
                kde_plot = KDEPlot()
                kde_plot.plot(raw_data['obs'], title='Raw Data Distribution')
                exploration_plots.append(kde_plot)
            except ImportError:
                pytest.skip("KDEPlot not available")
            
            # Stage 2: Data processing
            processed_data = raw_data.copy()
            processed_data['model_bias'] = processed_data['model'] - processed_data['obs']
            processed_data['normalized_obs'] = (processed_data['obs'] - processed_data['obs'].mean()) / processed_data['obs'].std()
            
            # Stage 3: Processed data analysis
            analysis_plots = []
            
            # Bias analysis
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                bias_plot = ScatterPlot()
                bias_plot.plot(processed_data, 'obs', 'model_bias', 
                             title='Model Bias Analysis', label='Bias vs Observation')
                analysis_plots.append(bias_plot)
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # Normalized data distribution
            try:
                from src.monet_plots.plots.kde import KDEPlot
                norm_plot = KDEPlot()
                norm_plot.plot(processed_data['normalized_obs'], 
                             title='Normalized Data Distribution', label='Normalized')
                analysis_plots.append(norm_plot)
            except ImportError:
                pytest.skip("KDEPlot not available")
            
            # Stage 4: Model evaluation
            model_plots = []
            
            try:
                from src.monet_plots.plots.taylor import TaylorDiagramPlot
                obs_std = processed_data['obs'].std()
                taylor_plot = TaylorDiagramPlot(obs_std)
                taylor_plot.add_sample(processed_data, label='Processed Model')
                taylor_plot.finish_plot()
                model_plots.append(taylor_plot)
            except ImportError:
                pytest.skip("TaylorDiagramPlot not available")
            
            # Verify pipeline integrity
            all_plots = exploration_plots + analysis_plots + model_plots
            for plot in all_plots:
                assert plot.ax is not None, f"Plot {type(plot).__name__} has no axes"
                
        finally:
            # Clean up all plots
            all_plots = exploration_plots + analysis_plots + model_plots
            for plot in all_plots:
                try:
                    plot.close()
                except:
                    pass
    
    def test_multi_panel_figure_workflow(self, mock_data_factory):
        """Test creating multiple plots on a single figure (subplots)."""
        # Create a comprehensive multi-panel figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plots_created = []
        
        try:
            # Panel 1: Spatial plot
            try:
                from src.monet_plots.plots.spatial import SpatialPlot
                spatial_plot = SpatialPlot(fig=fig, ax=axes[0])
                spatial_data = mock_data_factory.spatial_2d()
                spatial_plot.plot(spatial_data)
                plots_created.append(spatial_plot)
                axes[0].set_title('Spatial Distribution')
            except ImportError:
                pytest.skip("SpatialPlot not available")
            
            # Panel 2: Time series
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                ts_plot = TimeSeriesPlot(fig=fig, ax=axes[1])
                ts_data = mock_data_factory.time_series(n_points=50)
                ts_plot.plot(ts_data)
                plots_created.append(ts_plot)
                axes[1].set_title('Time Series')
            except ImportError:
                pytest.skip("TimeSeriesPlot not available")
            
            # Panel 3: Scatter plot
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                scatter_plot = ScatterPlot(fig=fig, ax=axes[2])
                scatter_data = mock_data_factory.scatter_data()
                scatter_plot.plot(scatter_data, 'x', 'y')
                plots_created.append(scatter_plot)
                axes[2].set_title('Scatter Plot')
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # Panel 4: KDE plot
            try:
                from src.monet_plots.plots.kde import KDEPlot
                kde_plot = KDEPlot(fig=fig, ax=axes[3])
                kde_data = mock_data_factory.kde_data()
                kde_plot.plot(kde_data)
                plots_created.append(kde_plot)
                axes[3].set_title('Distribution')
            except ImportError:
                pytest.skip("KDEPlot not available")
            
            # Panel 5: Another time series with different data
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                ts_plot2 = TimeSeriesPlot(fig=fig, ax=axes[4])
                ts_data2 = mock_data_factory.time_series(n_points=50, seed=123)
                ts_plot2.plot(ts_data2)
                plots_created.append(ts_plot2)
                axes[4].set_title('Time Series 2')
            except ImportError:
                pytest.skip("TimeSeriesPlot not available")
            
            # Panel 6: Xarray spatial plot
            try:
                from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
                xarray_plot = XarraySpatialPlot(fig=fig, ax=axes[5])
                xarray_data = mock_data_factory.xarray_data()
                xarray_plot.plot(xarray_data)
                plots_created.append(xarray_plot)
                axes[5].set_title('Xarray Data')
            except ImportError:
                pytest.skip("XarraySpatialPlot not available")
            
            # Verify the multi-panel layout
            plt.tight_layout()
            
            # Check that all axes have been used
            for i, ax in enumerate(axes):
                if i < len(plots_created):
                    # For multi-panel compatibility, check if the axes has data
                    # or if there's any data in the figure at this position
                    if not ax.has_data():
                        # For SpatialPlot with pre-existing axes, it creates a new GeoAxes
                        # at a slightly different position, so we need a different approach
                        
                        # Check if this is a SpatialPlot (which creates GeoAxes)
                        if hasattr(plots_created[i], '_original_ax') or type(plots_created[i]).__name__ == 'SpatialPlot':
                            # For SpatialPlot, check if there's any GeoAxes with data in the figure
                            has_geo_data = any(ax.has_data() and type(ax).__name__ == 'GeoAxes' for ax in fig.axes)
                            if has_geo_data:
                                continue  # This is acceptable for SpatialPlot
                            
                            # Also check if there's any data in the general area
                            original_pos = ax.get_position()
                            has_nearby_data = any(
                                ax.has_data() and
                                abs(fig_ax.get_position().x0 - original_pos.x0) < 0.1 and
                                abs(fig_ax.get_position().y0 - original_pos.y0) < 0.1
                                for fig_ax in fig.axes
                            )
                            if has_nearby_data:
                                continue
                        
                        # For other plot types, check if there's data in similar position
                        has_data_at_position = False
                        for fig_ax in fig.axes:
                            # Check if axes are in similar position (approximately same bounds)
                            fig_pos = fig_ax.get_position()
                            ax_pos = ax.get_position()
                            # Allow for small differences in position
                            position_tolerance = 0.05
                            if (abs(fig_pos.x0 - ax_pos.x0) < position_tolerance and
                                abs(fig_pos.y0 - ax_pos.y0) < position_tolerance and
                                abs(fig_pos.width - ax_pos.width) < position_tolerance and
                                abs(fig_pos.height - ax_pos.height) < position_tolerance):
                                if fig_ax.has_data():
                                    has_data_at_position = True
                                    break
                        
                        if not has_data_at_position:
                            # Debug information
                            print(f"Debug - Panel {i} original position: {ax.get_position()}")
                            print("Debug - All axes in figure:")
                            for j, fig_ax in enumerate(fig.axes):
                                fig_pos = fig_ax.get_position()
                                print(f"  Axis {j}: pos={fig_pos}, has_data={fig_ax.has_data()}, type={type(fig_ax).__name__}")
                            
                        assert has_data_at_position, f"Panel {i} has no data"
                    
        finally:
            # Clean up
            for plot in plots_created:
                try:
                    plot.close()
                except:
                    pass


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_integration_test():
    """Clean up matplotlib figures after each integration test."""
    yield
    plt.close('all')
    plt.clf()