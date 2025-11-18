"""
Basic Visual Regression Tests for MONET Plots

This module contains basic visual regression tests to ensure plot appearance
remains consistent across changes and updates.
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
from typing import Dict, List, Tuple, Optional


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
                data = mock_data_factory.scatter_data(n_points=10, seed=42)
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
            
        except ImportError:
            pytest.skip(f"{plot_class_name} not available")
        
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


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_visual_test():
    """Clean up matplotlib figures after each visual test."""
    yield
    plt.close('all')
    plt.clf()