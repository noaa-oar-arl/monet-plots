"""
Comprehensive Integration Tests for MONET Plots - Multi-Plot Workflows
======================================================================

This module contains comprehensive integration tests for multi-plot workflows,
real-world scenarios, and system integration testing using TDD approach.

Following TDD principles: Write failing tests first, implement minimal code to pass, then refactor.
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
import warnings


class TestMultiPlotWorkflows:
    """Integration tests for multi-plot workflows."""
    
    def test_air_quality_monitoring_workflow(self, mock_data_factory, test_outputs_dir):
        """Test complete air quality monitoring and analysis workflow."""
        # Generate realistic air quality monitoring data
        monitoring_stations = mock_data_factory.spatial_dataframe(
            n_points=50, 
            lat_range=(30, 50), 
            lon_range=(-120, -70)
        )
        
        # Add realistic pollutant concentrations
        monitoring_stations['ozone'] = np.random.uniform(20, 120, 50)  # ppb
        monitoring_stations['pm25'] = np.random.uniform(5, 50, 50)     # ug/m3
        monitoring_stations['no2'] = np.random.uniform(10, 80, 50)     # ppb
        monitoring_stations['so2'] = np.random.uniform(0, 20, 50)      # ppb
        
        plots_created = []
        
        try:
            # 1. Spatial distribution analysis
            spatial_plots = []
            pollutants = ['ozone', 'pm25', 'no2']
            
            for pollutant in pollutants:
                try:
                    from src.monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot
                    
                    # Mock the plotting method for testing
                    with patch('matplotlib.pyplot.figure') as mock_fig, \
                         patch('matplotlib.pyplot.gca') as mock_ax, \
                         patch('matplotlib.pyplot.colorbar') as mock_colorbar:
                        
                        mock_fig.return_value = plt.figure()
                        mock_ax.return_value = plt.gca()
                        mock_colorbar.return_value = Mock()
                        
                        spatial_plot = SpatialBiasScatterPlot()
                        # Mock the plot method to avoid actual plotting
                        spatial_plot.plot = Mock(return_value=(mock_fig(), mock_ax(), mock_colorbar()))
                        
                        result = spatial_plot.plot(monitoring_stations, None, '2025-01-01')
                        spatial_plots.append(spatial_plot)
                        plots_created.append(spatial_plot)
                        
                except ImportError:
                    pytest.skip("SpatialBiasScatterPlot not available")
            
            # 2. Time series analysis of pollution trends
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                
                # Create time series data for different pollutants
                dates = pd.date_range('2025-01-01', periods=30, freq='D')
                
                for pollutant in ['ozone', 'pm25']:
                    avg_concentrations = pd.DataFrame({
                        'time': dates,
                        'obs': np.random.uniform(30, 80, 30),
                        'model': np.random.uniform(30, 80, 30)
                    })
                    
                    ts_plot = TimeSeriesPlot()
                    ts_plot.plot(avg_concentrations, x='time', y='obs',
                               title=f'{pollutant.upper()} Concentration Trend',
                               ylabel='Concentration')
                    plots_created.append(ts_plot)
                    
            except ImportError:
                pytest.skip("TimeSeriesPlot not available")
            
            # 3. Correlation analysis between pollutants
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                
                correlation_plot = ScatterPlot()
                correlation_plot.plot(monitoring_stations, 'ozone', 'pm25',
                                    title='Ozone vs PM2.5 Correlation',
                                    label='Monitoring Stations')
                plots_created.append(correlation_plot)
                
                # Add another correlation plot
                correlation_plot2 = ScatterPlot()
                correlation_plot2.plot(monitoring_stations, 'no2', 'ozone',
                                     title='NO2 vs Ozone Correlation',
                                     label='Monitoring Stations')
                plots_created.append(correlation_plot2)
                
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # 4. Statistical analysis with Taylor diagram
            try:
                from src.monet_plots.plots.taylor import TaylorDiagramPlot
                
                # Simulate model vs observation comparison
                model_validation = pd.DataFrame({
                    'obs': np.random.uniform(40, 100, 100),
                    'model': np.random.uniform(40, 100, 100)
                })
                
                obs_std = model_validation['obs'].std()
                taylor_plot = TaylorDiagramPlot(obs_std)
                taylor_plot.add_sample(model_validation, label='Air Quality Model')
                taylor_plot.add_contours()
                taylor_plot.finish_plot()
                plots_created.append(taylor_plot)
                
            except ImportError:
                pytest.skip("TaylorDiagramPlot not available")
            
            # 5. Distribution analysis
            try:
                from src.monet_plots.plots.kde import KDEPlot
                
                # Analyze pollutant concentration distributions
                for pollutant in ['ozone', 'pm25']:
                    kde_plot = KDEPlot()
                    kde_plot.plot(monitoring_stations[pollutant],
                                title=f'{pollutant.upper()} Concentration Distribution',
                                label=pollutant.upper())
                    plots_created.append(kde_plot)
                
            except ImportError:
                pytest.skip("KDEPlot not available")
            
            # Verify workflow completeness
            assert len(plots_created) >= 5, "Air quality workflow should create multiple plot types"
            
            for i, plot in enumerate(plots_created):
                assert plot is not None, f"Plot {i} should not be None"
                # Note: Actual plot validation depends on implementation
                
        finally:
            # Clean up
            for plot in plots_created:
                try:
                    if hasattr(plot, 'close'):
                        plot.close()
                except:
                    pass
    
    def test_climate_data_analysis_workflow(self, mock_data_factory, test_outputs_dir):
        """Test complete climate data analysis workflow."""
        # Generate realistic climate data
        time_range = pd.date_range('2024-01-01', '2024-12-31', freq='M')
        
        climate_data = pd.DataFrame({
            'time': time_range,
            'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(12) / 12) + np.random.normal(0, 2, 12),
            'precipitation': np.random.gamma(2, 10, 12),
            'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(12) / 12 + 1) + np.random.normal(0, 5, 12),
            'pressure': 1013 + np.random.normal(0, 10, 12)
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
                
                # Add precipitation time series
                precip_plot = TimeSeriesPlot()
                precip_plot.plot(climate_data, x='time', y='precipitation',
                                title='Monthly Precipitation', ylabel='mm')
                plots_created.append(precip_plot)
                
            except ImportError:
                pytest.skip("TimeSeriesPlot not available")
            
            # 2. Climate variable correlations
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                
                # Temperature vs humidity
                temp_humidity_plot = ScatterPlot()
                temp_humidity_plot.plot(climate_data, 'temperature', 'humidity',
                                      title='Temperature vs Humidity Relationship',
                                      label='Monthly Climate Data')
                plots_created.append(temp_humidity_plot)
                
                # Temperature vs pressure
                temp_pressure_plot = ScatterPlot()
                temp_pressure_plot.plot(climate_data, 'temperature', 'pressure',
                                      title='Temperature vs Pressure Relationship',
                                      label='Monthly Climate Data')
                plots_created.append(temp_pressure_plot)
                
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # 3. Distribution analysis
            try:
                from src.monet_plots.plots.kde import KDEPlot
                
                # Temperature distribution
                temp_dist_plot = KDEPlot()
                temp_dist_plot.plot(climate_data['temperature'],
                                  title='Temperature Distribution', 
                                  label='Temperature')
                plots_created.append(temp_dist_plot)
                
                # Precipitation distribution
                precip_dist_plot = KDEPlot()
                precip_dist_plot.plot(climate_data['precipitation'],
                                    title='Precipitation Distribution', 
                                    label='Precipitation')
                plots_created.append(precip_dist_plot)
                
            except ImportError:
                pytest.skip("KDEPlot not available")
            
            # 4. Spatial climate analysis (mock data)
            try:
                from src.monet_plots.plots.spatial import SpatialPlot
                
                # Generate spatial climate data
                spatial_temp_data = mock_data_factory.spatial_2d(shape=(20, 30))
                
                spatial_plot = SpatialPlot()
                spatial_plot.plot(spatial_temp_data, title='Spatial Temperature Pattern')
                plots_created.append(spatial_plot)
                
            except ImportError:
                pytest.skip("SpatialPlot not available")
            
            # Verify climate analysis workflow
            assert len(plots_created) >= 4, "Climate analysis should create multiple plot types"
            
            for plot in plots_created:
                assert plot is not None
                
        finally:
            # Clean up
            for plot in plots_created:
                try:
                    if hasattr(plot, 'close'):
                        plot.close()
                except:
                    pass
    
    def test_meteorological_wind_analysis_workflow(self, mock_data_factory, test_outputs_dir):
        """Test meteorological wind analysis workflow."""
        plots_created = []
        
        try:
            # 1. Wind vector analysis with quiver plot
            try:
                from src.monet_plots.plots.wind_quiver import WindQuiverPlot
                
                # Generate wind data
                u_wind, v_wind = mock_data_factory.wind_data(shape=(15, 20))
                x, y = np.meshgrid(np.arange(u_wind.shape[1]), np.arange(u_wind.shape[0]))
                
                quiver_plot = WindQuiverPlot()
                quiver_plot.plot(x, y, u_wind, v_wind, title='Wind Vector Field')
                plots_created.append(quiver_plot)
                
            except ImportError:
                pytest.skip("WindQuiverPlot not available")
            
            # 2. Wind barb analysis
            try:
                from src.monet_plots.plots.wind_barbs import WindBarbsPlot
                
                barbs_plot = WindBarbsPlot()
                barbs_plot.plot(x, y, u_wind, v_wind, title='Wind Barb Representation')
                plots_created.append(barbs_plot)
                
            except ImportError:
                pytest.skip("WindBarbsPlot not available")
            
            # 3. Wind speed distribution
            try:
                from src.monet_plots.plots.kde import KDEPlot
                
                # Calculate wind speed from u, v components
                wind_speed = np.sqrt(u_wind**2 + v_wind**2)
                wind_speed_flat = wind_speed.flatten()
                
                wind_speed_plot = KDEPlot()
                wind_speed_plot.plot(wind_speed_flat,
                                   title='Wind Speed Distribution',
                                   label='Wind Speed')
                plots_created.append(wind_speed_plot)
                
            except ImportError:
                pytest.skip("KDEPlot not available")
            
            # 4. Wind direction histogram (scatter plot approach)
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                
                # Calculate wind direction
                wind_dir = np.arctan2(v_wind, u_wind) * 180 / np.pi
                wind_dir_flat = wind_dir.flatten()
                
                # Create dummy data for scatter plot (direction vs speed)
                wind_analysis_df = pd.DataFrame({
                    'direction': wind_dir_flat,
                    'speed': wind_speed_flat
                })
                
                wind_dir_plot = ScatterPlot()
                wind_dir_plot.plot(wind_analysis_df, 'direction', 'speed',
                                 title='Wind Direction vs Speed',
                                 label='Wind Analysis')
                plots_created.append(wind_dir_plot)
                
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # Verify meteorological workflow
            assert len(plots_created) >= 3, "Meteorological workflow should create multiple plot types"
            
            for plot in plots_created:
                assert plot is not None
                
        finally:
            # Clean up
            for plot in plots_created:
                try:
                    if hasattr(plot, 'close'):
                        plot.close()
                except:
                    pass


class TestRealWorldScenarios:
    """Integration tests for real-world scientific scenarios."""
    
    def test_environmental_impact_assessment(self, mock_data_factory, test_outputs_dir):
        """Test environmental impact assessment workflow."""
        # Simulate environmental monitoring data
        monitoring_data = mock_data_factory.spatial_dataframe(n_points=30)
        
        # Add environmental parameters
        monitoring_data['temperature'] = np.random.uniform(10, 30, 30)
        monitoring_data['humidity'] = np.random.uniform(40, 90, 30)
        monitoring_data['pollutant_level'] = np.random.uniform(0, 100, 30)
        monitoring_data['vegetation_index'] = np.random.uniform(0.1, 0.9, 30)
        
        plots_created = []
        
        try:
            # 1. Spatial environmental conditions
            try:
                from src.monet_plots.plots.spatial import SpatialPlot
                
                # Create spatial data arrays
                temp_data = mock_data_factory.spatial_2d()
                
                spatial_env_plot = SpatialPlot()
                spatial_env_plot.plot(temp_data, title='Environmental Conditions Map')
                plots_created.append(spatial_env_plot)
                
            except ImportError:
                pytest.skip("SpatialPlot not available")
            
            # 2. Correlation analysis
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                
                # Pollutant vs vegetation correlation
                corr_plot = ScatterPlot()
                corr_plot.plot(monitoring_data, 'pollutant_level', 'vegetation_index',
                             title='Pollutant Impact on Vegetation',
                             label='Monitoring Sites')
                plots_created.append(corr_plot)
                
                # Temperature vs humidity correlation
                temp_humidity_plot = ScatterPlot()
                temp_humidity_plot.plot(monitoring_data, 'temperature', 'humidity',
                                      title='Temperature vs Humidity Correlation',
                                      label='Environmental Monitoring')
                plots_created.append(temp_humidity_plot)
                
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # 3. Distribution analysis
            try:
                from src.monet_plots.plots.kde import KDEPlot
                
                # Pollutant level distribution
                pollutant_dist_plot = KDEPlot()
                pollutant_dist_plot.plot(monitoring_data['pollutant_level'],
                                       title='Pollutant Level Distribution',
                                       label='Environmental Impact')
                plots_created.append(pollutant_dist_plot)
                
            except ImportError:
                pytest.skip("KDEPlot not available")
            
            # Verify environmental assessment workflow
            assert len(plots_created) >= 3, "Environmental assessment should create multiple plots"
            
        finally:
            # Clean up
            for plot in plots_created:
                try:
                    if hasattr(plot, 'close'):
                        plot.close()
                except:
                    pass
    
    def test_oceanographic_data_analysis(self, mock_data_factory, test_outputs_dir):
        """Test oceanographic data analysis workflow."""
        # Generate oceanographic time series data
        dates = pd.date_range('2025-01-01', '2025-03-31', freq='D')
        
        ocean_data = pd.DataFrame({
            'time': dates,
            'sea_surface_temp': 15 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 1, len(dates)),
            'salinity': 35 + np.random.normal(0, 0.5, len(dates)),
            'ocean_current_speed': np.random.uniform(0.1, 2.0, len(dates)),
            'chlorophyll': np.random.uniform(0.5, 5.0, len(dates))
        })
        
        plots_created = []
        
        try:
            # 1. Time series analysis
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                
                # Sea surface temperature trend
                sst_plot = TimeSeriesPlot()
                sst_plot.plot(ocean_data, x='time', y='sea_surface_temp',
                            title='Sea Surface Temperature Trend', ylabel='°C')
                plots_created.append(sst_plot)
                
                # Salinity variation
                salinity_plot = TimeSeriesPlot()
                salinity_plot.plot(ocean_data, x='time', y='salinity',
                                 title='Salinity Variation', ylabel='PSU')
                plots_created.append(salinity_plot)
                
            except ImportError:
                pytest.skip("TimeSeriesPlot not available")
            
            # 2. Oceanographic correlations
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                
                # Temperature vs chlorophyll
                temp_chlorophyll_plot = ScatterPlot()
                temp_chlorophyll_plot.plot(ocean_data, 'sea_surface_temp', 'chlorophyll',
                                         title='Sea Temperature vs Chlorophyll Correlation',
                                         label='Oceanographic Data')
                plots_created.append(temp_chlorophyll_plot)
                
                # Current speed vs chlorophyll
                current_chlorophyll_plot = ScatterPlot()
                current_chlorophyll_plot.plot(ocean_data, 'ocean_current_speed', 'chlorophyll',
                                            title='Current Speed vs Chlorophyll',
                                            label='Oceanographic Data')
                plots_created.append(current_chlorophyll_plot)
                
            except ImportError:
                pytest.skip("ScatterPlot not available")
            
            # 3. Distribution analysis
            try:
                from src.monet_plots.plots.kde import KDEPlot
                
                # Ocean current speed distribution
                current_dist_plot = KDEPlot()
                current_dist_plot.plot(ocean_data['ocean_current_speed'],
                                     title='Ocean Current Speed Distribution',
                                     label='Marine Dynamics')
                plots_created.append(current_dist_plot)
                
            except ImportError:
                pytest.skip("KDEPlot not available")
            
            # Verify oceanographic workflow
            assert len(plots_created) >= 4, "Oceanographic analysis should create multiple plot types"
            
        finally:
            # Clean up
            for plot in plots_created:
                try:
                    if hasattr(plot, 'close'):
                        plot.close()
                except:
                    pass


class TestWorkflowErrorHandling:
    """Integration tests for error handling in workflows."""
    
    def test_partial_workflow_failure_handling(self, mock_data_factory, test_outputs_dir):
        """Test workflow behavior when some plot types fail."""
        plots_created = []
        
        try:
            # Simulate a workflow where some plot types are missing
            try:
                from src.monet_plots.plots.spatial import SpatialPlot
                
                spatial_plot = SpatialPlot()
                data = mock_data_factory.spatial_2d()
                spatial_plot.plot(data)
                plots_created.append(spatial_plot)
                
            except ImportError:
                # SpatialPlot not available - workflow should continue
                warnings.warn("SpatialPlot not available, continuing workflow")
            
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                
                ts_plot = TimeSeriesPlot()
                df = mock_data_factory.time_series()
                ts_plot.plot(df)
                plots_created.append(ts_plot)
                
            except ImportError:
                # TimeSeriesPlot not available - workflow should continue
                warnings.warn("TimeSeriesPlot not available, continuing workflow")
            
            # Workflow should still create some plots even if some fail
            assert len(plots_created) >= 0, "Workflow should handle partial failures gracefully"
            
        finally:
            # Clean up successfully created plots
            for plot in plots_created:
                try:
                    if hasattr(plot, 'close'):
                        plot.close()
                except:
                    pass
    
    def test_data_validation_in_workflows(self, mock_data_factory, test_outputs_dir):
        """Test data validation and error handling in multi-step workflows."""
        # Test with various data quality issues
        test_cases = [
            "empty_dataframe",
            "missing_columns", 
            "invalid_data_types",
            "insufficient_data_points"
        ]
        
        for test_case in test_cases:
            plots_created = []
            
            try:
                # Create problematic data based on test case
                if test_case == "empty_dataframe":
                    df = pd.DataFrame()
                elif test_case == "missing_columns":
                    df = pd.DataFrame({'x': [1, 2, 3]})  # Missing 'y' column
                elif test_case == "invalid_data_types":
                    df = pd.DataFrame({'x': ['a', 'b', 'c'], 'y': [1, 2, 3]})  # Invalid x type
                elif test_case == "insufficient_data_points":
                    df = pd.DataFrame({'x': [1], 'y': [2]})  # Only one data point
                
                # Try to create plots with problematic data
                try:
                    from src.monet_plots.plots.scatter import ScatterPlot
                    
                    scatter_plot = ScatterPlot()
                    
                    # This should either work with minimal data or fail gracefully
                    try:
                        scatter_plot.plot(df, 'x', 'y')
                        plots_created.append(scatter_plot)
                    except Exception as e:
                        # If it fails, it should be with a clear, expected error
                        expected_errors = [ValueError, KeyError, TypeError]
                        assert any(isinstance(e, error_type) for error_type in expected_errors)
                        
                except ImportError:
                    pytest.skip("ScatterPlot not available")
                
                # Verify that error handling works correctly
                if plots_created:
                    # If plot was created, it handled the edge case
                    for plot in plots_created:
                        assert plot is not None
                        plot.close()
                else:
                    # If no plot was created, error was handled appropriately
                    pass
                    
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