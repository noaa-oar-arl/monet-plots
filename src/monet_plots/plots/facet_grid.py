"""Faceting for tabular plots and geographically comparable spatial panels."""

from __future__ import annotations

from inspect import Parameter, signature
from itertools import cycle, islice, product
from math import ceil
from typing import Any, Callable

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize
from matplotlib.patches import Patch
from xarray.plot import FacetGrid as XarrayFacetGrid

from ..geo import identify_coords, validate_data_crs, variable_label
from ..plot_utils import normalize_data, to_dataframe
from ..style import style_context
from .base import BasePlot


def _facet_levels(data, column, order):
    if column is None:
        return []
    if order is not None:
        return list(order)
    values = data[column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        return list(values.cat.categories)
    levels = list(values.dropna().unique())
    return sorted(levels) if pd.api.types.is_numeric_dtype(values) else levels


def _facet_colors(palette, levels):
    if isinstance(palette, dict):
        return [palette[level] for level in levels]
    if isinstance(palette, str):
        # Matplotlib bundles the six-color palettes used by older seaborn styles.
        legacy = plt.style.library.get(f"seaborn-v0_8-{palette.removesuffix('6')}", {})
        if "axes.prop_cycle" in legacy:
            palette = legacy["axes.prop_cycle"].by_key()["color"]
        else:
            cmap = colormaps["Dark2" if palette in {"dark", "dark6"} else palette]
            if cmap.N > 20:
                return list(cmap(np.linspace(0, 1, len(levels) + 2)[1:-1]))
            palette = cmap.colors
    if palette is None:
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return list(islice(cycle(palette), len(levels)))


class _TableFacetGrid:
    """Create Matplotlib panels and dispatch callbacks for tabular facets."""

    def __init__(
        self,
        data,
        *,
        row,
        col,
        hue,
        col_wrap,
        height,
        aspect,
        subplot_kws,
        row_order=None,
        col_order=None,
        hue_order=None,
        sharex=True,
        sharey=True,
        palette=None,
        hue_kws=None,
        legend_out=True,
        despine=True,
        margin_titles=False,
        dropna=False,
        xlim=None,
        ylim=None,
        gridspec_kws=None,
    ):
        if col_wrap is not None and (row is not None or col is None or col_wrap < 1):
            raise ValueError("col_wrap requires column facets and a positive width.")
        self.data = data
        self._row, self._col, self._hue = row, col, hue
        self._wrap = col_wrap
        self._dropna = dropna
        self._legend_out = legend_out
        self._margin_titles = margin_titles
        self.hue_kws = hue_kws or {}
        self.row_names = _facet_levels(data, row, row_order)
        self.col_names = _facet_levels(data, col, col_order)
        self.hue_names = (
            _facet_levels(data, hue, hue_order) if hue is not None else None
        )

        self._colors = _facet_colors(palette, self.hue_names or [None])

        nrow, ncol = len(self.row_names) or 1, len(self.col_names) or 1
        self._facets = list(product(range(nrow), range(ncol)))
        if col_wrap:
            nrow, ncol = ceil(ncol / col_wrap), col_wrap
        self.fig, self.axs = plt.subplots(
            nrow,
            ncol,
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            figsize=(ncol * height * aspect, nrow * height),
            subplot_kw=subplot_kws,
            gridspec_kw=gridspec_kws,
        )
        self.figure = self.fig
        self.axes = self.axs.ravel()[: len(self._facets)] if col_wrap else self.axs
        self.axes_dict = {}
        for (i, j), ax in zip(self._facets, self.axs.flat):
            key = tuple(
                names[k]
                for names, k in ((self.row_names, i), (self.col_names, j))
                if names
            )
            if key:
                self.axes_dict[key[0] if len(key) == 1 else key] = ax
            if despine:
                ax.spines[["right", "top"]].set_visible(False)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
        for ax in self.axs.ravel()[len(self._facets) :]:
            ax.set_visible(False)
        if col_wrap and sharex:
            for ax in self._outer_axes("x"):
                ax.tick_params(labelbottom=True)
        self._margin_annotations = []
        self._legend = None
        self.set_titles()

    @property
    def ax(self):
        if self.axes.size != 1:
            raise AttributeError("Use .axes for a grid with multiple facets.")
        return self.axes.flat[0]

    def facet_axis(self, row_i, col_j, modify_state=True):
        ax = self.axes[col_j] if self._wrap else self.axes[row_i, col_j]
        if modify_state:
            plt.sca(ax)
        return ax

    def facet_data(self):
        """Yield each row, column and hue subset with its facet indices."""

        def masks(column, levels):
            if column is None:
                return [np.ones(len(self.data), dtype=bool)]
            return [self.data[column].eq(level).fillna(False) for level in levels]

        groups = zip(
            (self._row, self._col, self._hue),
            (self.row_names, self.col_names, self.hue_names),
        )
        for (i, row), (j, col), (k, hue) in product(
            *(enumerate(masks(column, levels)) for column, levels in groups)
        ):
            yield (i, j, k), self.data.loc[row & col & hue]

    def _map(self, func, args, kwargs, *, frame):
        color = kwargs.pop("color", None)
        for (i, j, k), subset in self.facet_data():
            if subset.empty:
                continue
            self.facet_axis(i, j)
            options = dict(kwargs, color=self._colors[k] if color is None else color)
            options.update({name: values[k] for name, values in self.hue_kws.items()})
            if self.hue_names is not None:
                options["label"] = str(self.hue_names[k])
            if frame:
                options["data"] = subset.dropna() if self._dropna else subset
                func(*args, **options)
            else:
                columns = subset[list(args)]
                if self._dropna:
                    columns = columns.dropna()
                values = (column for _, column in columns.items())
                if getattr(func, "__module__", "").startswith("matplotlib"):
                    values = (column.to_numpy() for _, column in columns.items())
                func(*values, **options)
        labels = [kwargs.get("x"), kwargs.get("y")]
        labels[: len(args[:2])] = args[:2]
        self.set_axis_labels(*labels)
        return self.set_titles().tight_layout()

    def map(self, func, *args, **kwargs):
        """Call a plotting function with the selected columns as arrays."""
        return self._map(func, args, kwargs, frame=False)

    def map_dataframe(self, func, *args, **kwargs):
        """Call a plotting function with a subset passed as the data keyword."""
        return self._map(func, args, kwargs, frame=True)

    def _outer_axes(self, axis):
        if axis == "y":
            return [ax for ax in self.axs[:, 0] if ax.get_visible()]
        bottom = []
        for column in self.axs.T:
            visible = [ax for ax in column if ax.get_visible()]
            if visible:
                bottom.append(visible[-1])
        return bottom

    def set_xlabels(self, label=None, clear_inner=True, **kwargs):
        return self._set_labels("x", label, clear_inner, **kwargs)

    def set_ylabels(self, label=None, clear_inner=True, **kwargs):
        return self._set_labels("y", label, clear_inner, **kwargs)

    def _set_labels(self, axis, label, clear_inner, **kwargs):
        if label is not None:
            outer = self._outer_axes(axis)
            for ax in self.axes.flat:
                if ax in outer or clear_inner:
                    getattr(ax, f"set_{axis}label")(
                        label if ax in outer else "", **kwargs
                    )
        return self

    def set_axis_labels(self, x_var=None, y_var=None, clear_inner=True, **kwargs):
        return self.set_xlabels(x_var, clear_inner, **kwargs).set_ylabels(
            y_var, clear_inner, **kwargs
        )

    def set_titles(self, template=None, row_template=None, col_template=None, **kwargs):
        row_template = row_template or "{row_var} = {row_name}"
        col_template = col_template or "{col_var} = {col_name}"
        for artist in self._margin_annotations:
            artist.remove()
        self._margin_annotations.clear()
        for i, j in self._facets:
            ax = self.facet_axis(i, j, modify_state=False)
            values = {
                "row_var": self._row,
                "col_var": self._col,
                "row_name": self.row_names[i] if self.row_names else None,
                "col_name": self.col_names[j] if self.col_names else None,
            }
            if self._margin_titles:
                if i == 0 and self.col_names:
                    ax.set_title(col_template.format(**values), **kwargs)
                if j == self.axs.shape[1] - 1 and self.row_names:
                    self._margin_annotations.append(
                        ax.annotate(
                            row_template.format(**values),
                            xy=(1.02, 0.5),
                            xycoords="axes fraction",
                            rotation=270,
                            va="center",
                            **kwargs,
                        )
                    )
            else:
                parts = [
                    part
                    for names, part in (
                        (self.row_names, row_template),
                        (self.col_names, col_template),
                    )
                    if names
                ]
                if parts:
                    ax.set_title(
                        (template or " | ".join(parts)).format(**values), **kwargs
                    )
        return self

    def add_legend(self, legend_data=None, title=None, label_order=None, **kwargs):
        """Add a legend using handles from all the facet axes."""
        if legend_data is None:
            legend_data = {}
            for ax in self.axes.flat:
                handles, labels = ax.get_legend_handles_labels()
                legend_data.update(zip(labels, handles))
        if label_order is None:
            label_order = self.hue_names if self.hue_names is not None else legend_data
        labels = list(map(str, label_order))
        handles = [legend_data.get(label, Patch(visible=False)) for label in labels]
        title = self._hue if title is None else title
        if self._legend is not None:
            self._legend.remove()
        if self._legend_out:
            kwargs.setdefault("loc", "center right")
            self._legend = self.fig.legend(handles, labels, title=title, **kwargs)
            width = self._legend.get_window_extent().width / self.fig.bbox.width
            self._layout_rect = (0, 0, max(0.1, 1 - width), 1)
            self.tight_layout()
        else:
            self._legend = self.axes.flat[0].legend(
                handles, labels, title=title, **kwargs
            )
        return self

    def tight_layout(self, *args, **kwargs):
        kwargs.setdefault("rect", getattr(self, "_layout_rect", None))
        self.fig.tight_layout(*args, **kwargs)
        return self

    def set(self, **kwargs):
        for ax in self.axes.flat:
            ax.set(**kwargs)
        return self

    def savefig(self, *args, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        return self.fig.savefig(*args, **kwargs)


class FacetGridPlot(BasePlot):
    """A consistent figure wrapper around Matplotlib and xarray facet grids."""

    def __init__(
        self,
        data: Any,
        row: str | None = None,
        col: str | None = None,
        hue: str | None = None,
        col_wrap: int | None = None,
        size: float | None = None,
        aspect: float = 1,
        subplot_kws: dict | None = None,
        style: str | None = "wiley",
        **kwargs,
    ):
        self.row, self.col, self.hue, self.col_wrap = row, col, hue, col_wrap
        self.size = size if size is not None else kwargs.pop("height", 3)
        self.aspect = aspect
        self.raw_data = data
        self.is_xarray = isinstance(data, (xr.DataArray, xr.Dataset))
        if row is not None and col_wrap is not None:
            raise ValueError("col_wrap cannot be combined with row faceting.")
        if self.is_xarray:
            if row is None and col is None:
                raise ValueError("Specify row or col to create an xarray facet grid.")
            if hue is not None:
                raise ValueError("hue is supported for tabular facets only.")
            self.data = normalize_data(data)
            # Fail with the actual xarray error; never silently create an empty figure.
            self.grid = XarrayFacetGrid(
                self.data,
                row=row,
                col=col,
                col_wrap=col_wrap,
                size=self.size,
                aspect=aspect,
                subplot_kws=subplot_kws,
                **kwargs,
            )
            self.grid.set_titles()
        else:
            self.data = to_dataframe(data).reset_index()
            self.grid = _TableFacetGrid(
                self.data,
                row=row,
                col=col,
                hue=hue,
                col_wrap=col_wrap,
                height=self.size,
                aspect=aspect,
                subplot_kws=subplot_kws,
                **kwargs,
            )
        axes = self.grid.axs if self.is_xarray else self.grid.axes
        super().__init__(fig=self.grid.fig, ax=axes.flat[0], style=style)
        self.axes = axes
        self.g = self.grid

    def map_dataframe(self, plot_func: Callable, *args, **kwargs):
        """Map a function over tabular subsets."""
        if self.is_xarray:
            raise TypeError(
                "map_dataframe requires tabular data; use map_monet for spatial xarray facets."
            )
        self.grid.map_dataframe(plot_func, *args, **kwargs)
        return self

    def set_titles(self, *args, **kwargs):
        """Set panel titles with the underlying backend's formatting options."""
        self.grid.set_titles(*args, **kwargs)
        return self

    def plot(self, plot_func=None, *args, **kwargs):
        """Map a function over the facet grid."""
        if plot_func is not None:
            self.grid.map(plot_func, *args, **kwargs)
        return self


class SpatialFacetGridPlot(FacetGridPlot):
    """Spatial panels with explicit CRS and scientifically comparable scales.

    ``colorbar='shared'`` (default) uses one norm across all panels. ``'each'``
    permits independent scales, including variables with incompatible units.
    ``False`` hides colorbars while retaining a common scale. Initialization
    preserves lazy arrays; rendering computes the global scale and panel values.
    """

    def __init__(
        self,
        data,
        *,
        row=None,
        col=None,
        col_wrap=None,
        projection=None,
        data_crs=None,
        extent_crs=None,
        size=None,
        aspect=1.2,
        style="wiley",
        colorbar="shared",
        _plotter_class=None,
        _plotter_kwargs=None,
        **kwargs,
    ):
        self.original_data = data
        self.is_dataset = isinstance(data, xr.Dataset)
        self.projection = projection or ccrs.PlateCarree()
        self._data_crs_explicit = data_crs is not None
        self.data_crs = data_crs or ccrs.PlateCarree()
        self.extent_crs = extent_crs or ccrs.PlateCarree()
        self.default_plotter = _plotter_class
        self.plotter_kwargs = _plotter_kwargs or {}
        self.colorbar_mode = colorbar
        self.plotters, self.mappables = [], []
        self._variable_facet = self.is_dataset and "variable" in (row, col)
        if self._variable_facet:
            if len({frozenset(v.dims) for v in data.data_vars.values()}) > 1:
                raise ValueError(
                    "Variables must have matching dimensions to facet; select compatible fields or regrid explicitly."
                )
            # Keep metadata on original variables for labels and unit validation.
            data = data.to_array(dim="variable", name="value")
        elif self.is_dataset and len(data.data_vars) == 1:
            data = data[next(iter(data.data_vars))]
        height = kwargs.pop("height", 4)
        size = size if size is not None else height
        grid_keys = (
            {"figsize", "sharex", "sharey"}
            if isinstance(data, (xr.Dataset, xr.DataArray))
            else {
                "row_order",
                "col_order",
                "hue",
                "hue_order",
                "palette",
                "sharex",
                "sharey",
                "margin_titles",
                "despine",
                "dropna",
                "legend_out",
                "gridspec_kws",
            }
        )
        grid_options = {
            key: kwargs.pop(key) for key in list(kwargs) if key in grid_keys
        }
        self.render_options = {**(self.plotter_kwargs.get("plotargs") or {}), **kwargs}
        super().__init__(
            data,
            row=row,
            col=col,
            col_wrap=col_wrap,
            size=size,
            aspect=aspect,
            subplot_kws={"projection": self.projection},
            style=style,
            **grid_options,
        )
        self.fig.set_layout_engine("constrained")
        self._set_default_titles()

    def _panels(self, data=None):
        data = self.data if data is None else data
        if self.is_xarray:
            for ax, selection in zip(self.axes.flat, self.grid.name_dicts.flat):
                if selection is None:
                    ax.set_visible(False)
                    continue
                subset = data.sel(selection)
                if self._variable_facet and "variable" in selection:
                    subset = subset.assign_attrs(
                        self.original_data[selection["variable"]].attrs
                    )
                    subset.name = str(selection["variable"])
                yield ax, subset, selection
        else:
            for (i, j, _), subset in self.grid.facet_data():
                if subset.empty:
                    continue
                ax = self.axes.flat[j] if self.col_wrap else self.axes[i, j]
                selection = {}
                if self.row:
                    selection[self.row] = self.grid.row_names[i]
                if self.col:
                    selection[self.col] = self.grid.col_names[j]
                yield ax, subset, selection

    def _set_default_titles(self):
        if not self.is_xarray:
            return
        for annotation in self.grid.row_labels:
            if annotation is not None and annotation.axes is not None:
                annotation.remove()
        self.grid.row_labels = []
        for ax, _, selection in self._panels():
            parts = []
            for dim, value in selection.items():
                if dim == "variable" and self._variable_facet:
                    parts.append(variable_label(self.original_data[value]))
                else:
                    coord = self.original_data.coords.get(dim)
                    label = (
                        coord.attrs.get("long_name", dim) if coord is not None else dim
                    )
                    if isinstance(value, (np.datetime64, pd.Timestamp)):
                        value = pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
                    parts.append(f"{label} = {value}")
            ax.set_title(" | ".join(parts))

    def add_map_features(self, **kwargs):
        """Add features to occupied panels, leaving wrapped empty cells hidden."""
        from .spatial import SpatialPlot

        for ax, _, _ in self._panels():
            SpatialPlot(
                ax=ax,
                data_crs=self.data_crs,
                extent_crs=self.extent_crs,
                style=self._style,
                **kwargs,
            )
        return self

    def plot(self, plotter_class=None, **kwargs):
        """Render using the originating spatial class, or SpatialMeshPlot."""
        from .spatial_mesh import SpatialMeshPlot

        return self.map_monet(
            plotter_class or self.default_plotter or SpatialMeshPlot, **kwargs
        )

    def map_monet(
        self, plotter_class: type, *, x=None, y=None, var_name=None, **kwargs
    ):
        """Render each subset using a spatial plotter, returning this grid.

        Constructor-only arguments (e.g. col1/col2, gridobj) are routed to the
        constructor. Rendering options (e.g. norm, cmap) are sent to ``plot``.
        ``colorbar`` accepts 'shared', 'each', or False; ``add_colorbar`` is a
        boolean compatibility alias. ``cbar_kwargs`` configure the real mappable.
        """
        options = {**self.render_options, **kwargs}
        mode = options.pop("colorbar", self.colorbar_mode)
        if "add_colorbar" in options:
            mode = "shared" if options.pop("add_colorbar") else False
        if mode not in ("shared", "each", False):
            raise ValueError("colorbar must be 'shared', 'each', or False.")
        cbar_kwargs = options.pop("cbar_kwargs", {})
        label = options.pop("label", None)
        if self._variable_facet and mode != "each":
            units = {
                str(v.attrs.get("units", ""))
                for v in self.original_data.data_vars.values()
            }
            if len(units) > 1:
                raise ValueError(
                    "Variables have incompatible units; use colorbar='each' or convert units first."
                )
        x = x or options.pop("lon_coord", None)
        y = y or options.pop("lat_coord", None)
        x, y = identify_coords(self.data, x, y)
        validate_data_crs(
            self.data, x, y, self.data_crs if self._data_crs_explicit else None
        )
        # Render the scale and panels from one snapshot while keeping the source lazy.
        render_data = self.data
        if isinstance(render_data, (xr.DataArray, xr.Dataset)) and render_data.chunks:
            if isinstance(render_data, xr.Dataset):
                selected = [options[key] for key in ("col1", "col2") if key in options]
                if not selected and var_name is not None:
                    selected = [var_name]
                if selected:
                    render_data = render_data[list(dict.fromkeys(selected))]
            render_data = render_data.compute()
        # Determine the field used for normalization, including model-minus-obs bias.
        if "col1" in options and "col2" in options:
            scale_data = render_data[options["col2"]] - render_data[options["col1"]]
        elif isinstance(render_data, xr.Dataset):
            if var_name is None:
                raise ValueError(
                    "Specify var_name when plotting a Dataset with multiple variables."
                )
            scale_data = render_data[var_name]
        elif isinstance(render_data, xr.DataArray):
            scale_data = render_data
        else:
            if var_name is None:
                raise ValueError("Specify var_name for tabular spatial facets.")
            scale_data = render_data[var_name]
        if mode != "each":
            options = _shared_scale(scale_data, options, plotter_class)
        constructor_parameters = signature(plotter_class.__init__).parameters
        constructor_keys = {
            name
            for name, param in constructor_parameters.items()
            if name not in {"self", "data", "modelvar"}
            and param.kind not in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}
        }
        constructor_options = dict(self.plotter_kwargs)
        for key in list(options):
            # Render vmin/vmax/norm/cmap together so a shared norm stays intact.
            if key in constructor_keys and key not in {"vmin", "vmax", "cmap"}:
                constructor_options[key] = options.pop(key)
        feature_keys = {
            "coastlines",
            "states",
            "countries",
            "ocean",
            "land",
            "lakes",
            "rivers",
            "borders",
            "gridlines",
            "extent",
            "resolution",
            "natural_earth",
        }
        features = {
            key: options.pop(key) for key in list(options) if key in feature_keys
        }
        features.setdefault("coastlines", True)
        for ax, subset, _ in self._panels(render_data):
            if isinstance(subset, xr.Dataset) and var_name is not None:
                subset = subset[var_name]
            elif (
                isinstance(subset, pd.DataFrame)
                and not {"col1", "col2"} <= constructor_options.keys()
            ):
                from .spatial import SpatialTrack

                if issubclass(plotter_class, SpatialTrack):
                    subset = xr.DataArray(
                        subset[var_name].to_numpy(),
                        dims="sample",
                        coords={
                            x: ("sample", subset[x].to_numpy()),
                            y: ("sample", subset[y].to_numpy()),
                        },
                    )
                else:
                    subset = subset.set_index([y, x]).to_xarray()[var_name]
            with style_context(self._style):
                plotter = plotter_class(
                    subset,
                    ax=ax,
                    style=self._style,
                    data_crs=self.data_crs,
                    extent_crs=self.extent_crs,
                    lon_coord=x,
                    lat_coord=y,
                    **features,
                    **constructor_options,
                )
                plotter.plot(**options, add_colorbar=False)
            if hasattr(plotter, "gridliner") and any(
                (
                    plotter.gridliner.top_labels,
                    plotter.gridliner.right_labels,
                    plotter.gridliner.left_labels,
                    plotter.gridliner.bottom_labels,
                )
            ):
                index = next(
                    i for i, candidate in enumerate(self.axes.flat) if candidate is ax
                )
                rows, cols = (
                    self.axes.shape if self.axes.ndim == 2 else (1, self.axes.size)
                )
                row_index, col_index = divmod(index, cols)
                plotter.gridliner.top_labels = False
                plotter.gridliner.right_labels = False
                plotter.gridliner.left_labels = col_index == 0
                plotter.gridliner.bottom_labels = (
                    row_index == rows - 1
                    or index + cols >= sum(1 for _ in self._panels())
                )
            self.plotters.append(plotter)
            if plotter.mappable is not None:
                self.mappables.append(plotter.mappable)
                if mode == "each":
                    panel_label = label
                    if panel_label is None and isinstance(subset, xr.DataArray):
                        panel_label = variable_label(subset)
                    cb = self.fig.colorbar(
                        plotter.mappable,
                        ax=ax,
                        **{"label": panel_label or "", **cbar_kwargs},
                    )
                    self.colorbars.append(cb)
        if self.mappables:
            self.mappable = self.mappables[-1]
        if mode == "shared" and self.mappables:
            if label is None and isinstance(scale_data, xr.DataArray):
                label = variable_label(scale_data)
            cb = self.fig.colorbar(
                self.mappables[-1],
                ax=[ax for ax, _, _ in self._panels()],
                **{"label": label or "", **cbar_kwargs},
            )
            self.colorbars.append(cb)
        self._set_default_titles()
        return self


def _shared_scale(data, options, plotter_class):
    """Resolve one norm and contour levels without materializing all panel values."""
    from .spatial_bias_scatter import SpatialBiasScatterPlot
    from .spatial_contour import SpatialContourPlot
    from .spatial_mesh import SpatialMeshPlot

    options = options.copy()
    norm = options.get("norm")
    cmap = options.get("cmap")
    if isinstance(cmap, tuple):
        options["cmap"], tuple_norm = cmap
        norm = norm or tuple_norm
    vmin, vmax = options.pop("vmin", None), options.pop("vmax", None)
    explicit_limits = vmin is not None or vmax is not None
    levels = options.get("levels")
    if levels is not None and not np.isscalar(levels):
        vmin = levels[0] if vmin is None else vmin
        vmax = levels[-1] if vmax is None else vmax
    if norm is not None and explicit_limits:
        raise ValueError("Supply norm or vmin/vmax, not both.")
    norm = norm or Normalize(vmin, vmax)
    if norm.vmin is None or norm.vmax is None:
        if isinstance(norm, LogNorm):
            data = (
                data.where(data > 0)
                if isinstance(data, xr.DataArray)
                else data[data > 0]
            )
        if isinstance(data, xr.DataArray):
            import dask

            low, high = dask.compute(data.min(skipna=True), data.max(skipna=True))
            low, high = float(low), float(high)
        else:
            low, high = float(data.min()), float(data.max())
        if not np.isfinite([low, high]).all():
            raise ValueError(
                "Cannot determine a color scale from entirely missing/non-finite data; supply explicit limits."
            )
        if issubclass(plotter_class, SpatialBiasScatterPlot):
            low, high = -max(abs(low), abs(high)), max(abs(low), abs(high))
        if norm.vmin is None:
            norm.vmin = low
        if norm.vmax is None:
            norm.vmax = high
    if norm.vmin == norm.vmax:
        delta = abs(norm.vmin) * 0.01 or 0.5
        norm.vmin -= delta
        norm.vmax += delta
    options["norm"] = norm
    if issubclass(plotter_class, SpatialContourPlot) and not issubclass(
        plotter_class, SpatialMeshPlot
    ):
        if levels is None and isinstance(norm, BoundaryNorm):
            options["levels"] = norm.boundaries
        elif levels is None or np.isscalar(levels):
            options["levels"] = np.linspace(
                norm.vmin,
                norm.vmax,
                int(levels) if levels is not None else options.get("ncolors", 10) + 1,
            )
    return options
