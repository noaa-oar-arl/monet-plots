"""
Comprehensive Performance Benchmarks for MONET Plots
====================================================

This module contains comprehensive performance benchmarks, stress tests, and scalability validation
for all plot classes using TDD approach.

Following TDD principles: Write failing tests first, implement minimal code to pass, then refactor.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import time
import tempfile
import os
from pathlib import Path
import psutil
import gc
import tracemalloc
from typing import Dict, List, Tuple, Optional
import warnings


class TestPerformanceBenchmarks:
    """Performance benchmarks for all plot classes."""
    
    @pytest.mark.parametrize("data_size", [100, 1000, 5000, 10000])
    def test_spatial_plot_performance_scaling(self, mock_data_factory, data_size):
        """Benchmark SpatialPlot performance scaling with data size."""
        # Calculate appropriate shape for given data size
        side_length = int(np.sqrt(data_size))
        shape = (side_length, side_length)
        
        # Generate spatial data
        modelvar = mock_data_factory.spatial_2d(shape=shape)
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            plot = SpatialPlot()
            plot.plot(modelvar, discrete=True, ncolors=15)
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            # Time should scale reasonably with data size
            max_expected_time = data_size * 0.001 + 0.1  # Linear scaling with overhead
            assert execution_time < max_expected_time, \
                f"SpatialPlot took {execution_time:.3f}s for {data_size} points (expected < {max_expected_time:.3f}s)"
            
            # Memory usage should be reasonable
            max_expected_memory = data_size * 0.01  # MB per data point
            assert memory_delta < max_expected_memory, \
                f"SpatialPlot used {memory_delta:.1f}MB for {data_size} points (expected < {max_expected_memory:.1f}MB)"
            
            # Plot should be valid
            assert plot.ax is not None
            assert hasattr(plot, 'cbar')
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    @pytest.mark.parametrize("n_points", [100, 1000, 5000, 10000, 50000])
    def test_timeseries_plot_performance(self, mock_data_factory, n_points):
        """Benchmark TimeSeriesPlot performance with different data sizes."""
        # Generate time series data
        df = mock_data_factory.time_series(n_points=n_points)
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.timeseries import TimeSeriesPlot
            plot = TimeSeriesPlot()
            plot.plot(df)
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            # Time complexity should be reasonable for time series operations
            max_expected_time = n_points * 0.001 + 0.05  # Should be very fast
            assert execution_time < max_expected_time, \
                f"TimeSeriesPlot took {execution_time:.3f}s for {n_points} points"
            
            # Memory should scale linearly
            max_expected_memory = n_points * 0.001
            assert memory_delta < max_expected_memory, \
                f"TimeSeriesPlot used {memory_delta:.1f}MB for {n_points} points"
            
            # Plot should be valid
            assert plot.ax is not None
            assert len(plot.ax.lines) > 0
            
            plot.close()
            
        except ImportError:
            pytest.skip("TimeSeriesPlot not available")
    
    @pytest.mark.parametrize("n_samples", [50, 100, 500, 1000, 5000])
    def test_taylor_diagram_performance(self, mock_data_factory, n_samples):
        """Benchmark TaylorDiagramPlot performance."""
        # Generate data
        df = mock_data_factory.taylor_data(n_points=n_samples)
        obs_std = df['obs'].std()
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.taylor import TaylorDiagramPlot
            plot = TaylorDiagramPlot(obs_std)
            plot.add_sample(df)
            plot.add_contours()
            plot.finish_plot()
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            max_expected_time = n_samples * 0.001 + 0.1
            assert execution_time < max_expected_time, \
                f"TaylorDiagramPlot took {execution_time:.3f}s for {n_samples} samples"
            
            assert memory_delta < 50, f"TaylorDiagramPlot used {memory_delta:.1f}MB"
            
            assert plot.dia is not None
            assert plot.ax is not None
            
            plot.close()
            
        except ImportError:
            pytest.skip("TaylorDiagramPlot not available")
    
    @pytest.mark.parametrize("n_points", [100, 1000, 5000, 10000])
    def test_scatter_plot_performance(self, mock_data_factory, n_points):
        """Benchmark ScatterPlot performance."""
        # Generate scatter data
        df = mock_data_factory.scatter_data(n_points=n_points)
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.scatter import ScatterPlot
            plot = ScatterPlot()
            plot.plot(df, 'x', 'y', ci=None)  # Disable confidence interval for speed
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            max_expected_time = n_points * 0.005 + 0.1
            assert execution_time < max_expected_time, \
                f"ScatterPlot took {execution_time:.3f}s for {n_points} points"
            
            max_expected_memory = n_points * 0.001
            assert memory_delta < max_expected_memory, \
                f"ScatterPlot used {memory_delta:.1f}MB for {n_points} points"
            
            assert plot.ax is not None
            assert len(plot.ax.collections) > 0
            
            plot.close()
            
        except ImportError:
            pytest.skip("ScatterPlot not available")
    
    @pytest.mark.parametrize("n_points", [1000, 5000, 10000, 50000])
    def test_kde_plot_performance(self, mock_data_factory, n_points):
        """Benchmark KDEPlot performance."""
        # Generate data
        data = mock_data_factory.kde_data(n_points=n_points)
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.kde import KDEPlot
            plot = KDEPlot()
            plot.plot(data, bw='scott')  # Use fast bandwidth estimation
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            max_expected_time = n_points * 0.002 + 0.2 # KDE can be slower
            assert execution_time < max_expected_time, \
                f"KDEPlot took {execution_time:.3f}s for {n_points} points"
            
            max_expected_memory = n_points * 0.005
            assert memory_delta < max_expected_memory, \
                f"KDEPlot used {memory_delta:.1f}MB for {n_points} points"
            
            assert plot.ax is not None
            assert len(plot.ax.lines) > 0
            
            plot.close()
            
        except ImportError:
            pytest.skip("KDEPlot not available")
    
    @pytest.mark.parametrize("data_shape", [(10, 10), (20, 20), (50, 50), (100, 100)])
    def test_xarray_spatial_plot_performance(self, mock_data_factory, data_shape):
        """Benchmark XarraySpatialPlot performance with different data shapes."""
        # Generate xarray data
        data = mock_data_factory.xarray_data(shape=data_shape)
        total_points = data_shape[0] * data_shape[1]
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
            plot = XarraySpatialPlot()
            plot.plot(data, cmap='viridis')
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            max_expected_time = total_points * 0.0005 + 0.05
            assert execution_time < max_expected_time, \
                f"XarraySpatialPlot took {execution_time:.3f}s for {total_points} points"
            
            max_expected_memory = total_points * 0.005
            assert memory_delta < max_expected_memory, \
                f"XarraySpatialPlot used {memory_delta:.1f}MB for {total_points} points"
            
            assert plot.ax is not None
            
            plot.close()
            
        except ImportError:
            pytest.skip("XarraySpatialPlot not available")
    
    @pytest.mark.parametrize("n_levels", [2, 3, 4, 5])
    def test_facet_grid_plot_performance(self, mock_data_factory, n_levels):
        """Benchmark FacetGridPlot performance with different numbers of facets."""
        # Generate facet data
        data = mock_data_factory.facet_data()
        
        # Adjust data size based on n_levels
        if n_levels == 2:
            data = data.isel(time=slice(0, 2))
        elif n_levels == 3:
            data = data.isel(time=slice(0, 3))
        elif n_levels == 5:
            data = data  # Use all 5 levels
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.facet_grid import FacetGridPlot
            plot = FacetGridPlot(data, col='time')
            plot.plot()
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            max_expected_time = n_levels * 0.5 + 0.2  # Should scale linearly with facets
            assert execution_time < max_expected_time, \
                f"FacetGridPlot took {execution_time:.3f}s for {n_levels} facets"
            
            max_expected_memory = n_levels * 10  # 10MB per facet
            assert memory_delta < max_expected_memory, \
                f"FacetGridPlot used {memory_delta:.1f}MB for {n_levels} facets"
            
            assert plot.g is not None
            
            plot.close()
            
        except ImportError:
            pytest.skip("FacetGridPlot not available")


class TestMemoryPerformance:
    """Memory performance and leak detection tests."""
    
    def test_memory_leak_detection_spatial_plots(self, mock_data_factory):
        """Test for memory leaks in SpatialPlot creation and cleanup."""
        # Start memory tracking
        tracemalloc.start()
        initial_snapshot = tracemalloc.take_snapshot()
        
        # Create and destroy multiple plots
        for i in range(20):
            try:
                from src.monet_plots.plots.spatial import SpatialPlot
                
                plot = SpatialPlot()
                data = mock_data_factory.spatial_2d()
                plot.plot(data)
                plot.close()
                
                # Force garbage collection
                gc.collect()
                
            except ImportError:
                pytest.skip("SpatialPlot not available")
        
        # Take final snapshot and compare
        final_snapshot = tracemalloc.take_snapshot()
        top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        
        # Check for significant memory growth
        total_memory_growth = sum(stat.size_diff for stat in top_stats)
        assert total_memory_growth < 50 * 1024 * 1024, \
            f"Potential memory leak detected: {total_memory_growth / 1024 / 1024:.1f}MB growth"
        
        tracemalloc.stop()
    
    def test_memory_usage_under_stress(self, mock_data_factory):
        """Test memory usage when creating many plots simultaneously."""
        plots_created = []
        max_memory_usage = 0
        
        try:
            # Create multiple plots to stress test memory
            for i in range(10):
                try:
                    from src.monet_plots.plots.spatial import SpatialPlot
                    
                    plot = SpatialPlot()
                    data = mock_data_factory.spatial_2d()
                    plot.plot(data)
                    plots_created.append(plot)
                    
                    # Monitor memory usage
                    current_memory = self._get_memory_usage()
                    max_memory_usage = max(max_memory_usage, current_memory)
                    
                except ImportError:
                    pytest.skip("SpatialPlot not available")
            
            # Memory should be reasonable even under stress
            assert max_memory_usage < 1000, f"Memory usage too high: {max_memory_usage:.1f}MB"
            
        finally:
            # Clean up all plots
            for plot in plots_created:
                try:
                    plot.close()
                except:
                    pass
    
    def test_large_dataset_handling(self, mock_data_factory):
        """Test handling of large datasets without memory issues."""
        # Create a large dataset
        large_shape = (200, 200)  # 40,000 data points
        large_data = mock_data_factory.spatial_2d(shape=large_shape)
        
        initial_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            plot.plot(large_data)
            
            # Memory usage should be reasonable for large data
            peak_memory = self._get_memory_usage()
            memory_increase = peak_memory - initial_memory
            
            # Should handle large data without excessive memory usage
            assert memory_increase < 200, \
                f"Memory increase too high for large dataset: {memory_increase:.1f}MB"
            
            assert plot.ax is not None
            assert hasattr(plot, 'cbar')
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")


class TestScalabilityValidation:
    """Scalability validation and performance regression tests."""
    
    def test_performance_regression_baseline(self, mock_data_factory):
        """Test that performance doesn't regress beyond acceptable thresholds."""
        performance_thresholds = {
            'spatial_plot_10k': 2.0,      # 2 seconds for 10k points
            'timeseries_plot_10k': 1.0,   # 1 second for 10k points
            'scatter_plot_5k': 1.5,        # 1.5 seconds for 5k points
            'kde_plot_10k': 3.0,           # 3 seconds for 10k points
        }
        
        # Test each plot type against baseline
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            # SpatialPlot performance test
            data = mock_data_factory.spatial_2d(shape=(100, 100))  # 10k points
            start_time = time.time()
            
            plot = SpatialPlot()
            plot.plot(data)
            plot.close()
            
            execution_time = time.time() - start_time
            assert execution_time < performance_thresholds['spatial_plot_10k'], \
                f"SpatialPlot performance regression: {execution_time:.3f}s > {performance_thresholds['spatial_plot_10k']}s"
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
        
        try:
            from src.monet_plots.plots.timeseries import TimeSeriesPlot
            
            # TimeSeriesPlot performance test
            df = mock_data_factory.time_series(n_points=10000)
            start_time = time.time()
            
            plot = TimeSeriesPlot()
            plot.plot(df)
            plot.close()
            
            execution_time = time.time() - start_time
            assert execution_time < performance_thresholds['timeseries_plot_10k'], \
                f"TimeSeriesPlot performance regression: {execution_time:.3f}s > {performance_thresholds['timeseries_plot_10k']}s"
            
        except ImportError:
            pytest.skip("TimeSeriesPlot not available")
    
    def test_concurrent_plot_creation(self, mock_data_factory):
        """Test performance under concurrent plot creation (simulated)."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def create_plot_worker(data, plot_class_name, results_queue):
            """Worker function to create plots."""
            try:
                start_time = time.time()
                
                if plot_class_name == "SpatialPlot":
                    from src.monet_plots.plots.spatial import SpatialPlot
                    plot = SpatialPlot()
                    plot.plot(data)
                    plot.close()
                
                execution_time = time.time() - start_time
                results_queue.put(('success', execution_time, plot_class_name))
                
            except Exception as e:
                results_queue.put(('error', str(e), plot_class_name))
        
        # Test concurrent SpatialPlot creation
        threads = []
        test_data = [mock_data_factory.spatial_2d() for _ in range(5)]
        
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            # Create multiple threads
            for i, data in enumerate(test_data):
                thread = threading.Thread(
                    target=create_plot_worker,
                    args=(data, "SpatialPlot", results)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)  # 10 second timeout
            
            # Check results
            successful_plots = 0
            total_time = 0
            
            while not results.empty():
                status, result, plot_type = results.get()
                if status == 'success':
                    successful_plots += 1
                    total_time += result
            
            # Should complete all plots successfully
            assert successful_plots == len(threads), \
                f"Only {successful_plots}/{len(threads)} plots completed successfully"
            
            # Average time should be reasonable
            avg_time = total_time / successful_plots if successful_plots > 0 else 0
            assert avg_time < 5.0, f"Average plot creation time too slow: {avg_time:.3f}s"
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_performance_stability_over_time(self, mock_data_factory):
        """Test that performance remains stable over multiple iterations."""
        execution_times = []
        
        # Run the same operation multiple times
        for i in range(10):
            start_time = time.time()
            
            try:
                from src.monet_plots.plots.spatial import SpatialPlot
                
                plot = SpatialPlot()
                data = mock_data_factory.spatial_2d()
                plot.plot(data)
                plot.close()
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
            except ImportError:
                pytest.skip("SpatialPlot not available")
        
        # Analyze performance stability
        times_array = np.array(execution_times)
        coefficient_of_variation = np.std(times_array) / np.mean(times_array)
        
        # Performance should be stable (low coefficient of variation)
        assert coefficient_of_variation < 0.2, \
            f"Performance unstable: CV = {coefficient_of_variation:.3f} > 0.2"
        
        # No single run should be extremely slow (outlier detection)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        slowest_time = np.max(times_array)
        
        assert slowest_time < mean_time + 3 * std_time, \
            f"Performance outlier detected: {slowest_time:.3f}s vs mean {mean_time:.3f}s"


class TestResourceManagement:
    """Resource management and cleanup validation."""
    
    def test_matplotlib_resource_cleanup(self, mock_data_factory):
        """Test that matplotlib resources are properly cleaned up."""
        initial_figures = len(plt.get_fignums())
        initial_axes = len(plt.get_fignums())  # Each figure has one axes initially
        
        plots_created = []
        
        try:
            # Create multiple plots
            for i in range(5):
                try:
                    from src.monet_plots.plots.spatial import SpatialPlot
                    
                    plot = SpatialPlot()
                    data = mock_data_factory.spatial_2d()
                    plot.plot(data)
                    plots_created.append(plot)
                    
                except ImportError:
                    pytest.skip("SpatialPlot not available")
            
            # Should have created new figures
            assert len(plt.get_fignums()) > initial_figures
            
            # Close plots one by one
            for i, plot in enumerate(plots_created):
                plot.close()
                remaining_figures = len(plt.get_fignums())
                expected_figures = initial_figures + len(plots_created) - (i + 1)
                
                # Each close should reduce figure count
                assert remaining_figures == expected_figures, \
                    f"Figure cleanup failed: expected {expected_figures}, got {remaining_figures}"
            
            # Final state should match initial state
            assert len(plt.get_fignums()) == initial_figures
            
        finally:
            # Ensure all figures are closed
            plt.close('all')
    
    def test_file_handle_management(self, mock_data_factory, temp_directory):
        """Test that file handles are properly managed during save operations."""
        # Create a plot and save it multiple times
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            data = mock_data_factory.spatial_2d()
            plot.plot(data)
            
            # Save the plot multiple times
            for i in range(5):
                filepath = os.path.join(temp_directory, f'test_plot_{i}.png')
                plot.save(filepath)
                
                # File should exist and be readable
                assert os.path.exists(filepath)
                assert os.path.getsize(filepath) > 0
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")


    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0  # Fallback if psutil not available


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_performance_test():
    """Clean up matplotlib figures after each performance test."""
    yield
    plt.close('all')
    plt.clf()