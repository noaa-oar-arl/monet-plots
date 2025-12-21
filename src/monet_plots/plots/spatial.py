# src/monet_plots/plots/spatial.py

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .base import BasePlot


class SpatialPlot(BasePlot):
    """Base class for spatial plots using cartopy.

    Handles the creation of cartopy axes and the drawing of common
    map features like coastlines, states, etc., via keyword arguments.
    """

    def __init__(self, projection=ccrs.PlateCarree(), resolution="50m", *args, **kwargs):
        """
        Initialize the spatial plot.

        Args:
            projection (ccrs.Projection): The cartopy projection for the map.
            resolution (str): Resolution for cartopy features ('10m', '50m', '110m').
            **kwargs: Additional keyword arguments for plotting, including cartopy features
                      like 'coastlines', 'countries', 'states', 'borders', 'counties', 'ocean',
                      'land', 'rivers', 'lakes', 'gridlines', 'natural_earth'. These can be True for default
                      styling or a dict for custom styling.
        """
        # The 'projection' kwarg is passed to subplot creation via 'subplot_kw'.
        subplot_kw = kwargs.pop("subplot_kw", {})
        subplot_kw["projection"] = projection

        # Separate BasePlot kwargs from feature kwargs
        fig = kwargs.pop("fig", None)
        ax = kwargs.pop("ax", None)
        figsize = kwargs.pop("figsize", None)

        base_plot_kwargs = {}
        if fig:
            base_plot_kwargs["fig"] = fig
        if ax:
            base_plot_kwargs["ax"] = ax
        if figsize:
            base_plot_kwargs["figsize"] = figsize
        if subplot_kw:
            base_plot_kwargs["subplot_kw"] = subplot_kw

        super().__init__(*args, **base_plot_kwargs)

        # Store resolution and feature kwargs passed at initialization
        self.resolution = resolution
        self.feature_kwargs = kwargs

    def _add_natural_earth_features(self):
        """Add natural earth features (ocean, land, lakes, rivers)."""
        self.ax.add_feature(cfeature.OCEAN)
        self.ax.add_feature(cfeature.LAND)
        self.ax.add_feature(cfeature.LAKES)
        self.ax.add_feature(cfeature.RIVERS)

    def _add_coastlines(self, coastlines_style, resolution):
        """Add coastlines with proper resolution."""
        if isinstance(coastlines_style, dict):
            linewidth = coastlines_style.pop("linewidth", 0.5)
            self.ax.coastlines(resolution=resolution, linewidth=linewidth, **coastlines_style)
        elif coastlines_style:
            self.ax.coastlines(resolution=resolution, linewidth=0.5)

    def _add_counties(self, counties_style, resolution):
        """Add US counties feature."""
        if counties_style:
            counties_feature = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_2_counties",
                scale=resolution,
                facecolor="none",
                edgecolor="k",
            )
            if isinstance(counties_style, dict):
                self.ax.add_feature(counties_feature, **counties_style)
            else:
                self.ax.add_feature(counties_feature, linewidth=0.5)

    def _add_standard_features(self, feature_map, combined_kwargs):
        """Add standard cartopy features."""
        for key, feature in feature_map.items():
            if key in combined_kwargs:
                style = combined_kwargs.pop(key)
                if isinstance(style, dict):
                    self.ax.add_feature(feature, **style)
                elif style:
                    self.ax.add_feature(feature, edgecolor="black", linewidth=0.5)

    def _add_gridlines(self, gl_style):
        """Add gridlines to the map."""
        if isinstance(gl_style, dict):
            self.ax.gridlines(**gl_style)
        else:
            self.ax.gridlines(draw_labels=True, linestyle="--", color="gray")

    def _draw_features(self, **kwargs):
        """Draw cartopy features on the map axes based on kwargs."""
        # Combine kwargs from __init__ and the plot call, with plot() taking precedence
        combined_kwargs = {**self.feature_kwargs, **kwargs}

        # Get resolution from kwargs or use instance default
        resolution = combined_kwargs.pop("resolution", self.resolution)

        # Handle natural earth features first
        if combined_kwargs.pop("natural_earth", False):
            self._add_natural_earth_features()

        # Handle coastlines
        if "coastlines" in combined_kwargs:
            self._add_coastlines(combined_kwargs.pop("coastlines"), resolution)

        # Handle counties
        if "counties" in combined_kwargs:
            self._add_counties(combined_kwargs.pop("counties"), resolution)

        # Define feature mapping
        feature_map = {
            "countries": cfeature.BORDERS.with_scale(resolution),
            "states": cfeature.STATES.with_scale(resolution),
            "borders": cfeature.BORDERS,
            "ocean": cfeature.OCEAN,
            "land": cfeature.LAND,
            "rivers": cfeature.RIVERS,
            "lakes": cfeature.LAKES,
        }

        # Add standard features
        self._add_standard_features(feature_map, combined_kwargs)

        # Handle gridlines
        if "gridlines" in combined_kwargs:
            self._add_gridlines(combined_kwargs.pop("gridlines"))

        # Handle extent
        if "extent" in combined_kwargs:
            extent = combined_kwargs.pop("extent")
            if extent is not None:
                self.ax.set_extent(extent)

        # Return remaining kwargs
        return combined_kwargs

    @classmethod
    def draw_map(
        cls,
        *,
        crs=None,
        natural_earth=False,
        coastlines=True,
        states=False,
        counties=False,
        countries=True,
        resolution="10m",
        extent=None,
        figsize=(10, 5),
        linewidth=0.25,
        return_fig=False,
        **kwargs,
    ):
        """Draw a map with Cartopy - compatibility method for draw_map function.

        Creates a map using Cartopy with configurable features like coastlines,
        borders, and natural earth elements. This method provides compatibility
        with the legacy draw_map function.

        Parameters
        ----------
        crs : cartopy.crs.Projection
            The map projection. Defaults to PlateCarree.
        natural_earth : bool
            Add the Cartopy Natural Earth ocean, land, lakes, and rivers features.
        coastlines : bool
            Add coastlines (linewidth applied).
        states : bool
            Add states/provinces (linewidth applied).
        counties : bool
            Add US counties (linewidth applied).
        countries : bool
            Add country borders (linewidth applied).
        resolution : {'10m', '50m', '110m'}
            The resolution of the Natural Earth features.
        extent : array-like
            Set the map extent with [lon_min, lon_max, lat_min, lat_max].
        figsize : tuple
            Figure size (width, height).
        linewidth : float
            Line width for coastlines, states, counties, and countries.
        return_fig : bool
            Return the figure and axes objects.
        **kwargs
            Additional arguments passed to plt.subplots().

        Returns
        -------
        matplotlib.axes.Axes or tuple
            By default, returns just the axes. If return_fig is True,
            returns a tuple of (fig, ax).
        """
        projection = crs or ccrs.PlateCarree()

        # Prepare feature kwargs
        feature_kwargs = {
            "natural_earth": natural_earth,
            "coastlines": {"linewidth": linewidth} if coastlines else False,
            "states": {"linewidth": linewidth} if states else False,
            "counties": {"linewidth": linewidth} if counties else False,
            "countries": {"linewidth": linewidth} if countries else False,
            "extent": extent,
        }

        # Create SpatialPlot instance
        spatial_plot = cls(projection=projection, resolution=resolution, figsize=figsize, **feature_kwargs, **kwargs)

        # Draw the features
        spatial_plot._draw_features()

        if return_fig:
            return spatial_plot.fig, spatial_plot.ax
        else:
            return spatial_plot.ax


class SpatialTrack(SpatialPlot):
    """Plot a trajectory on a map, with points colored by a variable."""

    def __init__(self, longitude, latitude, data, *args, **kwargs):
        """
        Initialize the spatial track plot.
        Args:
            longitude (np.ndarray, pd.Series, xr.DataArray): Longitude values.
            latitude (np.ndarray, pd.Series, xr.DataArray): Latitude values.
            data (np.ndarray, pd.Series, xr.DataArray): Data to use for coloring the track.
            **kwargs: Additional keyword arguments passed to SpatialPlot.
        """
        super().__init__(*args, **kwargs)
        self.longitude = longitude
        self.latitude = latitude
        self.data = data

    def plot(self, **kwargs):
        """
        Plot the trajectory.
        Args:
            **kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`.
        """
        if self.ax is None:
            self.ax = self.fig.add_subplot(projection=self.projection)

        plot_kwargs = self._draw_features(**kwargs)
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        sc = self.ax.scatter(self.longitude, self.latitude, c=self.data, **plot_kwargs)
        return sc
