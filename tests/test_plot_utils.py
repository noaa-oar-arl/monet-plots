import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_plots import plot_utils


def test_to_dataframe():
    """Test the to_dataframe function."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert plot_utils.to_dataframe(df) is df

    da = xr.DataArray(np.random.rand(3, 3), name="data")
    assert isinstance(plot_utils.to_dataframe(da), pd.DataFrame)

    arr = np.random.rand(3, 3)
    assert isinstance(plot_utils.to_dataframe(arr), pd.DataFrame)

    with pytest.raises(TypeError):
        plot_utils.to_dataframe(1)

    # Test 1D and 2D numpy arrays
    df1 = plot_utils.to_dataframe(np.array([1, 2, 3]))
    assert "col_0" in df1.columns
    assert len(df1) == 3

    df2 = plot_utils.to_dataframe(np.array([[1, 2], [3, 4]]))
    assert "col_0" in df2.columns
    assert "col_1" in df2.columns
    assert len(df2) == 2

    # Test numpy array with > 2 dims
    with pytest.raises(ValueError, match="dimensions not supported"):
        plot_utils.to_dataframe(np.zeros((2, 2, 2)))


def test_validate_dataframe():
    """Test the validate_dataframe function."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    plot_utils.validate_dataframe(df, required_columns=["a"])

    with pytest.raises(ValueError):
        plot_utils.validate_dataframe(df, required_columns=["b"])

    with pytest.raises(ValueError):
        plot_utils.validate_dataframe(pd.DataFrame(), required_columns=["a"])


def test_validate_plot_parameters():
    """Test the validate_plot_parameters function."""
    plot_utils.validate_plot_parameters(
        "SpatialPlot", "plot", discrete=True, ncolors=10, plotargs={"cmap": "viridis"}
    )

    with pytest.raises(TypeError, match="discrete parameter must be boolean"):
        plot_utils.validate_plot_parameters("SpatialPlot", "plot", discrete="true")

    with pytest.raises(TypeError, match="ncolors parameter must be integer"):
        plot_utils.validate_plot_parameters("SpatialPlot", "plot", ncolors=10.5)

    with pytest.raises(ValueError, match="ncolors parameter must be between"):
        plot_utils.validate_plot_parameters("SpatialPlot", "plot", ncolors=0)

    with pytest.raises(TypeError, match="plotargs parameter must be dict"):
        plot_utils.validate_plot_parameters(
            "SpatialPlot", "plot", plotargs="not a dict"
        )

    with pytest.raises(TypeError, match="colormap must be string"):
        plot_utils.validate_plot_parameters(
            "SpatialPlot", "plot", plotargs={"cmap": 123}
        )

    # TimeSeriesPlot validation
    plot_utils.validate_plot_parameters(
        "TimeSeriesPlot", "plot", x="time", y="obs", fillargs={"alpha": 0.5}
    )

    with pytest.raises(TypeError, match="x parameter must be string"):
        plot_utils.validate_plot_parameters("TimeSeriesPlot", "plot", x=123)

    with pytest.raises(TypeError, match="y parameter must be string"):
        plot_utils.validate_plot_parameters("TimeSeriesPlot", "plot", y=123)

    with pytest.raises(TypeError, match="fillargs parameter must be dict"):
        plot_utils.validate_plot_parameters(
            "TimeSeriesPlot", "plot", fillargs="not a dict"
        )

    with pytest.raises(TypeError, match="alpha must be numeric"):
        plot_utils.validate_plot_parameters(
            "TimeSeriesPlot", "plot", fillargs={"alpha": "high"}
        )

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        plot_utils.validate_plot_parameters(
            "TimeSeriesPlot", "plot", fillargs={"alpha": 1.5}
        )


def test_validate_data_array():
    """Test the validate_data_array function."""
    with pytest.raises(ValueError, match="data cannot be None"):
        plot_utils.validate_data_array(None)

    with pytest.raises(TypeError, match="data must have a shape attribute"):
        plot_utils.validate_data_array(123)

    da = xr.DataArray(np.random.rand(3, 3), dims=["x", "y"])
    plot_utils.validate_data_array(da, required_dims=["x", "y"])

    with pytest.raises(ValueError):
        plot_utils.validate_data_array(da, required_dims=["z"])

    with pytest.raises(TypeError, match="data must have dims attribute"):
        plot_utils.validate_data_array(np.zeros((3, 3)), required_dims=["x"])


def test_dynamic_fig_size():
    """Test the _dynamic_fig_size function."""
    da = xr.DataArray(np.random.rand(10, 20), dims=["y", "x"])
    width, height = plot_utils._dynamic_fig_size(da)
    assert width > 0
    assert height > 0

    da2 = xr.DataArray(
        np.random.rand(10, 20),
        dims=["latitude", "longitude"],
        coords={"latitude": np.arange(10), "longitude": np.arange(20)},
    )
    assert plot_utils._dynamic_fig_size(da2) == (10, 10 * 0.5)

    da3 = xr.DataArray(
        np.random.rand(10, 20),
        dims=["lat", "lon"],
        coords={"lat": np.arange(10), "lon": np.arange(20)},
    )
    assert plot_utils._dynamic_fig_size(da3) == (10, 10 * 0.5)


def test_set_outline_patch_alpha():
    """Test the _set_outline_patch_alpha function."""
    from unittest.mock import MagicMock

    ax = MagicMock()
    plot_utils._set_outline_patch_alpha(ax, 0)
    ax.axes.outline_patch.set_alpha.assert_called_with(0)


@pytest.mark.parametrize(
    "convert", [plot_utils.to_dataframe, plot_utils.normalize_data]
)
def test_dask_dataframe_preserves_pandas_index_and_nullable_values(convert):
    dd = pytest.importorskip("dask.dataframe")

    data = pd.DataFrame(
        {"score": pd.array([1, pd.NA, -2, 4], dtype="Int64")},
        index=pd.DatetimeIndex(
            ["2025-01-07", "2025-01-06", "2025-01-03", "2025-01-05"],
            tz="UTC",
            name="time",
        ),
    )
    chunked = dd.from_pandas(data, npartitions=2, sort=False)
    pd.testing.assert_frame_equal(convert(chunked), data)
    pd.testing.assert_frame_equal(convert(chunked["score"]), data)


@pytest.mark.parametrize(
    "convert", [plot_utils.to_dataframe, plot_utils.normalize_data]
)
@pytest.mark.parametrize("dimensions", [1, 2])
def test_dask_array_uses_numpy_tabular_column_semantics(convert, dimensions):
    import dask.array as da

    matrix = np.arange(20, dtype=np.int64).reshape(10, 2)
    values = matrix if dimensions == 2 else matrix[:, 0]
    chunked = da.from_array(values, chunks=3)
    # Filtering on the Dask collection makes its row chunk sizes unknown.
    subset = (
        chunked[chunked[:, 0] % 4 == 0]
        if dimensions == 2
        else chunked[chunked % 4 == 0]
    )
    expected = values[matrix[:, 0] % 4 == 0]
    pd.testing.assert_frame_equal(convert(subset), plot_utils.to_dataframe(expected))


def test_invalid_dask_array_shape_is_rejected_without_computing():
    import dask.array as da
    from dask import delayed

    @delayed
    def fail_if_evaluated():
        raise AssertionError("Unsupported shape should be rejected before computation.")

    chunked = da.from_delayed(fail_if_evaluated(), shape=(2, 2, 2), dtype=float)
    with pytest.raises(ValueError, match="3 dimensions not supported"):
        plot_utils.to_dataframe(chunked)


def test_normalize_data_keeps_chunked_xarray_native():
    import dask.array as da

    array = xr.DataArray(da.arange(12, chunks=4), dims="time", name="concentration")
    dataset = array.to_dataset()
    for container in [array, dataset]:
        for prefer_xarray in (True, False):
            result = plot_utils.normalize_data(container, prefer_xarray=prefer_xarray)
            assert type(result) is type(container)
            original = (
                container
                if isinstance(container, xr.DataArray)
                else container["concentration"]
            )
            normalized = (
                result if isinstance(result, xr.DataArray) else result["concentration"]
            )
            assert normalized.data is original.data
            result.attrs["plot_history"] = "updated"
            assert "plot_history" not in container.attrs
    assert plot_utils.normalize_data(array).chunks == ((4, 4, 4),)


@pytest.mark.parametrize("backend", ["array", "dataframe"])
def test_rank_histogram_accepts_chunked_inputs(backend):
    import dask.array as da

    from monet_plots.plots.rank_histogram import RankHistogramPlot

    ranks = np.array([0, 1, 1, 2])
    if backend == "dataframe":
        dd = pytest.importorskip("dask.dataframe")
        chunked = dd.from_pandas(pd.DataFrame({"rank": ranks}), npartitions=2)
        rank_col = "rank"
    else:
        chunked = da.from_array(ranks[:, None], chunks=(2, 1))
        rank_col = "col_0"
    plot = RankHistogramPlot()
    try:
        plot.plot(chunked, rank_col=rank_col, n_members=2)
        np.testing.assert_allclose(
            [bar.get_height() for bar in plot.ax.patches], [0.25, 0.5, 0.25]
        )
    finally:
        plot.close()
