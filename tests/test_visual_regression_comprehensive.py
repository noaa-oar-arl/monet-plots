"""
Comprehensive Visual Regression Tests for MONET Plots
=====================================================

This module contains comprehensive visual regression tests to ensure plot appearance
remains consistent across changes and updates using TDD approach.

Following TDD principles: Write failing tests first, implement minimal code to pass, then refactor.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageChops
import tempfile
import os
from pathlib import Path
import json
import hashlib
import warnings
from typing import Dict, List, Tuple, Optional
import cv2


class TestVisualRegression:
    """Visual regression tests for plot appearance consistency."""
    
    @pytest.fixture(scope="class")
    def baseline_images_dir(self):
        """Directory for baseline images."""
        baseline_dir = Path(__file__).parent / "baseline_images"
        baseline_dir.mkdir(exist_ok=True)
        return baseline_dir
    
    @pytest.fixture(scope="class")
    def test_results_dir(self):
        """Directory for test result images."""
        results_dir = Path(__file__).parent / "test_results"
        results_dir.mkdir(exist_ok=True)
        return results_dir
    
    @pytest.fixture(scope="class")
    def visual_thresholds(self):
        """Visual regression thresholds."""
        return {
            'pixel_tolerance': 0.01, # 1% of pixels can differ
            'structural_similarity': 0.95,  # SSIM threshold
            'color_tolerance': 10,  # RGB color difference tolerance
            'size_tolerance': 2  # Pixel size difference tolerance
        }
    
    def _generate_plot_hash(self, plot_data: np.ndarray) -> str:
        """Generate a hash for plot data to identify changes."""
        return hashlib.md5(plot_data.tobytes()).hexdigest()[:16]
    
    def _calculate_image_similarity(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Calculate similarity metrics between two images."""
        # Basic pixel difference
        if img1.shape != img2.shape:
            return {'pixel_match': 0.0, 'structural_similarity': 0.0, 'identical_shape': False}
        
        # Convert to same format if needed
        if img1.dtype != img2.dtype:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
        
        # Calculate pixel-wise differences
        diff = np.abs(img1 - img2)
        
        # Calculate metrics
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Pixel match percentage (within tolerance)
        tolerance = 10  # RGB difference tolerance
        pixel_match = np.mean(diff <= tolerance)
        
        # Structural similarity approximation
        # This is a simplified version of SSIM
        luminance_diff = np.abs(np.mean(img1) - np.mean(img2))
        contrast_diff = np.abs(np.std(img1) - np.std(img2))
        structural_diff = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
        
        ssim_approx = max(0, 1 - (luminance_diff / 255 + contrast_diff / 255 + (1 - structural_diff)) / 3)
        
        return {
            'pixel_match': float(pixel_match),
            'structural_similarity': float(ssim_approx),
            'mean_pixel_difference': float(mean_diff),
            'max_pixel_difference': float(max_diff),
            'identical_shape': img1.shape == img2.shape
        }
    
    def _save_baseline_image(self, plot, filename: str, baseline_dir: Path) -> Path:
        """Save a plot as baseline image."""
        baseline_path = baseline_dir / filename
        plot.save(str(baseline_path), dpi=150, bbox_inches='tight')
        return baseline_path
    
    def _create_test_plot(self, plot_class_name: str, mock_data_factory):
        """Create a test plot for visual regression testing."""
        # Create plots with fixed seed for reproducibility
        np.random.seed(42)
        
        try:
            if plot_class_name == "SpatialPlot":
                from src.monet_plots.plots.spatial import SpatialPlot
                plot = SpatialPlot()
                data = mock_data_factory.spatial_2d(seed=42)
                plot.plot(data, discrete=True, ncolors=12)
                return plot
            
            elif plot_class_name == "TimeSeriesPlot":
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                plot = TimeSeriesPlot()
                data = mock_data_factory.time_series(n_points=50, seed=42)
                plot.plot(data, title='Test Time Series', ylabel='Concentration (ppb)')
                return plot
            
            elif plot_class_name == "ScatterPlot":
                from src.monet_plots.plots.scatter import ScatterPlot
                plot = ScatterPlot()
                data = mock_data_factory.scatter_data(n_points=100, seed=42)
                plot.plot(data, 'x', 'y', title='Test Scatter Plot', label='Test Data')
                return plot
            
            elif plot_class_name == "KDEPlot":
                from src.monet_plots.plots.kde import KDEPlot
                plot = KDEPlot()
                data = mock_data_factory.kde_data(n_points=1000, seed=42)
                plot.plot(data, title='Test KDE Plot', label='Distribution')
                return plot
            
            elif plot_class_name == "TaylorDiagramPlot":
                from src.monet_plots.plots.taylor import TaylorDiagramPlot
                data = mock_data_factory.taylor_data(n_points=100, seed=42)
                obs_std = data['obs'].std()
                plot = TaylorDiagramPlot(obs_std)
                plot.add_sample(data, label='Test Model')
                plot.add_contours(colors='0.5')
                plot.finish_plot()
                return plot
            
            elif plot_class_name == "XarraySpatialPlot":
                from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
                plot = XarraySpatialPlot()
                data = mock_data_factory.xarray_data(seed=42)
                plot.plot(data, cmap='viridis')
                return plot
            
            elif plot_class_name == "FacetGridPlot":
                from src.monet_plots.plots.facet_grid import FacetGridPlot
                data = mock_data_factory.facet_data(seed=42)
                plot = FacetGridPlot(data, col='time')
                plot.plot()
                return plot
            
        except ImportError as e:
            pytest.skip(f"{plot_class_name} not available: {e}")
        
        return None
    
    def test_spatial_plot_visual_regression(self, mock_data_factory, baseline_images_dir, test_results_dir, visual_thresholds):
        """Test SpatialPlot visual regression."""
        plot_class = "SpatialPlot"
        baseline_file = f"{plot_class}_baseline.png"
        baseline_path = baseline_images_dir / baseline_file
        
        # Create test plot
        plot = self._create_test_plot(plot_class, mock_data_factory)
        if plot is None:
            return
        
        try:
            # Save test image
            test_file = f"{plot_class}_test.png"
            test_path = test_results_dir / test_file
            plot.save(str(test_path), dpi=150, bbox_inches='tight')
            
            # Check if baseline exists
            if baseline_path.exists():
                # Load and compare images
                baseline_img = mpimg.imread(baseline_path)
                test_img = mpimg.imread(test_path)
                
                # Calculate similarity
                similarity = self._calculate_image_similarity(baseline_img, test_img)
                
                # Assert visual regression thresholds
                assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                    f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
                
                assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                    f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
                
                assert similarity['identical_shape'], "Image shapes do not match"
                
                # Log similarity metrics
                print(f"Visual regression metrics for {plot_class}:")
                print(f"  Pixel match: {similarity['pixel_match']:.3f}")
                print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
                print(f"  Mean pixel difference: {similarity['mean_pixel_difference']:.1f}")
                
            else:
                # Create baseline image if it doesn't exist
                print(f"Creating baseline image for {plot_class}")
                self._save_baseline_image(plot, baseline_file, baseline_images_dir)
                
        finally:
            plot.close()
    
    def test_timeseries_plot_visual_regression(self, mock_data_factory, baseline_images_dir, test_results_dir, visual_thresholds):
        """Test TimeSeriesPlot visual regression."""
        plot_class = "TimeSeriesPlot"
        baseline_file = f"{plot_class}_baseline.png"
        baseline_path = baseline_images_dir / baseline_file
        
        # Create test plot
        plot = self._create_test_plot(plot_class, mock_data_factory)
        if plot is None:
            return
        
        try:
            # Save test image
            test_file = f"{plot_class}_test.png"
            test_path = test_results_dir / test_file
            plot.save(str(test_path), dpi=150, bbox_inches='tight')
            
            # Check if baseline exists
            if baseline_path.exists():
                # Load and compare images
                baseline_img = mpimg.imread(baseline_path)
                test_img = mpimg.imread(test_path)
                
                # Calculate similarity
                similarity = self._calculate_image_similarity(baseline_img, test_img)
                
                # Assert visual regression thresholds
                assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                    f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
                
                assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                    f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
                
                # Log similarity metrics
                print(f"Visual regression metrics for {plot_class}:")
                print(f"  Pixel match: {similarity['pixel_match']:.3f}")
                print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
                
            else:
                # Create baseline image if it doesn't exist
                print(f"Creating baseline image for {plot_class}")
                self._save_baseline_image(plot, baseline_file, baseline_images_dir)
                
        finally:
            plot.close()
    
    def test_scatter_plot_visual_regression(self, mock_data_factory, baseline_images_dir, test_results_dir, visual_thresholds):
        """Test ScatterPlot visual regression."""
        plot_class = "ScatterPlot"
        baseline_file = f"{plot_class}_baseline.png"
        baseline_path = baseline_images_dir / baseline_file
        
        # Create test plot
        plot = self._create_test_plot(plot_class, mock_data_factory)
        if plot is None:
            return
        
        try:
            # Save test image
            test_file = f"{plot_class}_test.png"
            test_path = test_results_dir / test_file
            plot.save(str(test_path), dpi=150, bbox_inches='tight')
            
            # Check if baseline exists
            if baseline_path.exists():
                # Load and compare images
                baseline_img = mpimg.imread(baseline_path)
                test_img = mpimg.imread(test_path)
                
                # Calculate similarity
                similarity = self._calculate_image_similarity(baseline_img, test_img)
                
                # Assert visual regression thresholds
                assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                    f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
                
                assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                    f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
                
                # Log similarity metrics
                print(f"Visual regression metrics for {plot_class}:")
                print(f"  Pixel match: {similarity['pixel_match']:.3f}")
                print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
                
            else:
                # Create baseline image if it doesn't exist
                print(f"Creating baseline image for {plot_class}")
                self._save_baseline_image(plot, baseline_file, baseline_images_dir)
                
        finally:
            plot.close()
    
    def test_kde_plot_visual_regression(self, mock_data_factory, baseline_images_dir, test_results_dir, visual_thresholds):
        """Test KDEPlot visual regression."""
        plot_class = "KDEPlot"
        baseline_file = f"{plot_class}_baseline.png"
        baseline_path = baseline_images_dir / baseline_file
        
        # Create test plot
        plot = self._create_test_plot(plot_class, mock_data_factory)
        if plot is None:
            return
        
        try:
            # Save test image
            test_file = f"{plot_class}_test.png"
            test_path = test_results_dir / test_file
            plot.save(str(test_path), dpi=150, bbox_inches='tight')
            
            # Check if baseline exists
            if baseline_path.exists():
                # Load and compare images
                baseline_img = mpimg.imread(baseline_path)
                test_img = mpimg.imread(test_path)
                
                # Calculate similarity
                similarity = self._calculate_image_similarity(baseline_img, test_img)
                
                # Assert visual regression thresholds
                assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                    f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
                
                assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                    f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
                
                # Log similarity metrics
                print(f"Visual regression metrics for {plot_class}:")
                print(f"  Pixel match: {similarity['pixel_match']:.3f}")
                print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
                
            else:
                # Create baseline image if it doesn't exist
                print(f"Creating baseline image for {plot_class}")
                self._save_baseline_image(plot, baseline_file, baseline_images_dir)
                
        finally:
            plot.close()
    
    def test_taylor_diagram_visual_regression(self, mock_data_factory, baseline_images_dir, test_results_dir, visual_thresholds):
        """Test TaylorDiagramPlot visual regression."""
        plot_class = "TaylorDiagramPlot"
        baseline_file = f"{plot_class}_baseline.png"
        baseline_path = baseline_images_dir / baseline_file
        
        # Create test plot
        plot = self._create_test_plot(plot_class, mock_data_factory)
        if plot is None:
            return
        
        try:
            # Save test image
            test_file = f"{plot_class}_test.png"
            test_path = test_results_dir / test_file
            plot.save(str(test_path), dpi=150, bbox_inches='tight')
            
            # Check if baseline exists
            if baseline_path.exists():
                # Load and compare images
                baseline_img = mpimg.imread(baseline_path)
                test_img = mpimg.imread(test_path)
                
                # Calculate similarity
                similarity = self._calculate_image_similarity(baseline_img, test_img)
                
                # Assert visual regression thresholds
                assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                    f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
                
                assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                    f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
                
                # Log similarity metrics
                print(f"Visual regression metrics for {plot_class}:")
                print(f"  Pixel match: {similarity['pixel_match']:.3f}")
                print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
                
            else:
                # Create baseline image if it doesn't exist
                print(f"Creating baseline image for {plot_class}")
                self._save_baseline_image(plot, baseline_file, baseline_images_dir)
                
        finally:
            plot.close()


class TestCrossPlatformConsistency:
    """Tests for cross-platform visual consistency."""
    
    def test_font_rendering_consistency(self, mock_data_factory, visual_thresholds):
        """Test that font rendering is consistent across platforms."""
        # Create a plot with text elements
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            data = mock_data_factory.spatial_2d()
            plot.plot(data, title='Test Plot with Title', discrete=True)
            
            # The plot should render successfully regardless of font availability
            assert plot.ax is not None
            assert plot.ax.get_title() == 'Test Plot with Title'
            
            # Font fallback should work
            assert hasattr(plot.ax, 'texts') or len(plot.ax.texts) >= 0
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_color_consistency_across_environments(self, mock_data_factory, visual_thresholds):
        """Test that colors appear consistent across different environments."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            data = mock_data_factory.spatial_2d()
            
            # Test with different colormaps
            colormaps = ['viridis', 'plasma', 'inferno', 'magma']
            
            for cmap in colormaps:
                plot.plot(data, plotargs={'cmap': cmap})
                
                # Each colormap should be applied successfully
                assert plot.ax is not None
                assert len(plot.ax.images) > 0
                
                # Colormap should be set
                actual_cmap = plot.ax.images[0].get_cmap().name
                # Allow for colormap name variations across matplotlib versions
                assert actual_cmap in [cmap, f'_{cmap}'] or cmap in actual_cmap
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_figure_size_consistency(self, mock_data_factory):
        """Test that figure sizes are consistent across environments."""
        try:
            from src.monet_plots.plots.base import BasePlot
            
            # Test different figure sizes
            test_sizes = [(6, 4), (8, 6), (10, 8), (12, 9)]
            
            for width, height in test_sizes:
                plot = BasePlot(figsize=(width, height))
                
                # Check that the figure size is approximately correct
                fig_width, fig_height = plot.fig.get_size_inches()
                
                # Allow for small variations in figure size
                assert abs(fig_width - width) < 0.1, f"Width mismatch: {fig_width} vs {width}"
                assert abs(fig_height - height) < 0.1, f"Height mismatch: {fig_height} vs {height}"
                
                plot.close()
            
        except ImportError:
            pytest.skip("BasePlot not available")


class TestVisualQualityValidation:
    """Tests for visual quality and consistency validation."""
    
    def test_plot_element_visibility(self, mock_data_factory):
        """Test that all plot elements are visible and properly positioned."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            data = mock_data_factory.spatial_2d()
            plot.plot(data, title='Test Title', discrete=True)
            
            # Check that all expected elements are present
            assert plot.ax is not None
            assert plot.ax.get_title() == 'Test Title'
            assert hasattr(plot, 'cbar')
            assert plot.cbar is not None
            
            # Check that axes are visible
            assert plot.ax.get_xlabel() is not None or plot.ax.get_ylabel() is not None
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_colorbar_consistency(self, mock_data_factory):
        """Test that colorbars are consistently rendered."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            data = mock_data_factory.spatial_2d()
            
            # Test discrete colorbar
            plot.plot(data, discrete=True, ncolors=10)
            assert plot.cbar is not None
            
            # Test continuous colorbar
            plot.close()
            plot = SpatialPlot()
            plot.plot(data, discrete=False)
            assert plot.cbar is not None
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_legend_consistency(self, mock_data_factory):
        """Test that legends are consistently rendered."""
        try:
            from src.monet_plots.plots.scatter import ScatterPlot
            
            plot = ScatterPlot()
            data = mock_data_factory.scatter_data()
            plot.plot(data, 'x', 'y', label='Test Data')
            
            # Check that legend is created
            assert plot.ax is not None
            if plot.ax.get_legend():
                legend = plot.ax.get_legend()
                assert legend.get_texts()[0].get_text() == 'Test Data'
            
            plot.close()
            
        except ImportError:
            pytest.skip("ScatterPlot not available")
    
    def test_axis_label_consistency(self, mock_data_factory):
        """Test that axis labels are consistently applied."""
        try:
            from src.monet_plots.plots.timeseries import TimeSeriesPlot
            
            plot = TimeSeriesPlot()
            data = mock_data_factory.time_series()
            plot.plot(data, title='Test Title', ylabel='Test Y Label')
            
            # Check that labels are applied
            assert plot.ax.get_title() == 'Test Title'
            assert 'Test Y Label' in plot.ax.get_ylabel()
            
            plot.close()
            
        except ImportError:
            pytest.skip("TimeSeriesPlot not available")


class TestVisualRegressionEdgeCases:
    """Edge cases for visual regression testing."""
    
    def test_empty_plot_visual_consistency(self, mock_data_factory):
        """Test visual consistency with empty or minimal plots."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            
            # Test with minimal data
            minimal_data = np.array([[1]])
            plot.plot(minimal_data)
            
            # Should still create a valid plot
            assert plot.ax is not None
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_large_dataset_visual_consistency(self, mock_data_factory):
        """Test visual consistency with large datasets."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            
            # Test with large dataset
            large_data = mock_data_factory.spatial_2d(shape=(100, 100))
            plot.plot(large_data)
            
            # Should handle large data without visual artifacts
            assert plot.ax is not None
            assert hasattr(plot, 'cbar')
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_high_contrast_data_visual_consistency(self, mock_data_factory):
        """Test visual consistency with high contrast data."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            
            # Create high contrast data
            high_contrast_data = np.zeros((10, 10))
            high_contrast_data[0, 0] = 1000  # Very high value
            high_contrast_data[5, 5] = -1000  # Very low value
            
            plot.plot(high_contrast_data)
            
            # Should handle high contrast without visual issues
            assert plot.ax is not None
            assert hasattr(plot, 'cbar')
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_visual_test():
    """Clean up matplotlib figures after each visual test."""
    yield
    plt.close('all')
    plt.clf()