"""
Real-World Scenario Integration Tests for MONET Plots

This module contains integration tests for real-world scientific scenarios.
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


class TestRealWorldScenarios:
    """Integration tests for real-world scientific scenarios."""
    
    def test_air_quality_analysis_workflow(self, mock_data_factory, test_outputs_dir):
        """Test a complete air quality analysis workflow."""
        # Simulate air quality monitoring data
        monitoring_data = mock_data_factory.spatial_dataframe(n_points=100)
        
        # Add realistic air quality data
        monitoring_data['ozone'] = np.random.uniform(0, 100, 100)  # ppb
        monitoring_data['pm25'] = np.random.uniform(0, 50, 100)    # ug/m3
        monitoring_data['no2'] = np.random.uniform(0, 40, 100)     # ppb
        
        plots_created = []
        
        try:
            # 1. Spatial distribution of pollutants
            spatial_plots = []
            pollutants = ['ozone', 'pm25', 'no2']
            
            for i, pollutant in enumerate(pollutants):
                try:
                    from src.monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot
                    spatial_plot = SpatialBiasScatterPlot()
                    
                    # Mock the plotting method for testing
                    with patch.object(spatial_plot, 'plot') as mock_plot:
                        mock_plot.return_value = (plt.gcf(), plt.gca(), plt.colorbar())
                        result = spatial_plot.plot(monitoring_data, None, '2025-01-01')
                        spatial_plots.append(spatial_plot)
                        plots_created.append(spatial_plot)
                except ImportError:
                    pytest.skip("SpatialBiasScatterPlot not available")
            
            # 2. Time series of average concentrations
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                
                # Create time series data
                dates = pd.date_range('2025-01-01', periods=30, freq='D')
                avg_concentrations = pd.DataFrame({
                    'time': dates,
                    'ozone': np.random.uniform(20, 80, 30),
                    'pm25': np.random.uniform(5, 35, 30)
                })
                
                # Plot ozone
                ozone_plot = TimeSeriesPlot()
                ozone_plot.plot(avg_concentrations, x='time', y='ozone', 
                              title='Ozone Concentration Time Series', ylabel='ppb')
                plots_created.append(ozone_plot)
                
                # Plot PM2.5
                pm25_plot = TimeSeriesPlot()
                pm25_plot.plot(avg_concentrations, x='time', y='pm25',
                             title='PM2.5 Concentration Time Series', ylabel='μg/m³')
                plots_created.append(pm25_plot)
                
            except ImportError:
                pytest.skip("TimeSeriesPlot not available")
            
            # 3. Correlation analysis
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                
                correlation_plot = ScatterPlot()
                correlation_plot.plot(monitoring_data, 'ozone', 'pm25',
                                    title='Ozone vs PM2.5 Correlation',
                                    label='Monitoring Sites')
                plots_created.append(correlation_plot)
                
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # 4. Model validation (simulated)
            try:
                from src.monet_plots.plots.taylor import TaylorDiagramPlot
                
                # Simulate model vs observation data
                model_validation = pd.DataFrame({
                    'obs': np.random.uniform(30, 70, 50),
                    'model': np.random.uniform(30, 70, 50)
                })
                
                obs_std = model_validation['obs'].std()
                taylor_plot = TaylorDiagramPlot(obs_std)
                taylor_plot.add_sample(model_validation, label='Air Quality Model')
                taylor_plot.finish_plot()
                plots_created.append(taylor_plot)
                
            except ImportError:
                pytest.skip("TaylorDiagramPlot not available")
            
            # Verify workflow completeness
            assert len(plots_created) >= 3, "Air quality workflow should create multiple plots"
            
            for plot in plots_created:
                assert plot.ax is not None, f"Plot {type(plot).__name__} has no axes"
                
        finally:
            # Clean up
            for plot in plots_created:
                try:
                    plot.close()
                except:
                    pass
    
    def test_climate_data_analysis(self, mock_data_factory, test_outputs_dir):
        """Test climate data analysis workflow."""
        # Generate climate-like data
        time_range = pd.date_range('2024-01-01', '2024-12-31', freq='M')
        
        climate_data = pd.DataFrame({
            'time': time_range,
            'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(12) / 12),
            'precipitation': np.random.gamma(2, 10, 12),
            'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(12) / 12 + 1)
        })
        
        plots_created = []
        
        try:
            # 1. Seasonal temperature analysis
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                
                temp_plot = TimeSeriesPlot()
                temp_plot.plot(climate_data, x='time', y='temperature',
                             title='Monthly Temperature Variation', ylabel='°C')
                plots_created.append(temp_plot)
                
            except ImportError:
                pytest.skip("TimeSeriesPlot not available")
            
            # 2. Climate variable correlations
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                
                temp_humidity_plot = ScatterPlot()
                temp_humidity_plot.plot(climate_data, 'temperature', 'humidity',
                                      title='Temperature vs Humidity',
                                      label='Monthly Data')
                plots_created.append(temp_humidity_plot)
                
                temp_precip_plot = ScatterPlot()
                temp_precip_plot.plot(climate_data, 'temperature', 'precipitation',
                                    title='Temperature vs Precipitation',
                                    label='Monthly Data')
                plots_created.append(temp_precip_plot)
                
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # 3. Distribution analysis
            try:
                from src.monet_plots.plots.kde import KDEPlot
                
                temp_dist_plot = KDEPlot()
                temp_dist_plot.plot(climate_data['temperature'],
                                  title='Temperature Distribution', label='Temperature')
                plots_created.append(temp_dist_plot)
                
                precip_dist_plot = KDEPlot()
                precip_dist_plot.plot(climate_data['precipitation'],
                                    title='Precipitation Distribution', label='Precipitation')
                plots_created.append(precip_dist_plot)
                
            except ImportError:
                pytest.skip("KDEPlot not available")
            
            # Verify climate analysis workflow
            assert len(plots_created) >= 3, "Climate analysis should create multiple plots"
            
            for plot in plots_created:
                assert plot.ax is not None
                
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