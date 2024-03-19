#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `xarrayfits` package."""

from contextlib import ExitStack

from astropy.io import fits
from dask.distributed import Client, LocalCluster
import numpy as np
import pytest
import xarray

from xarrayfits import xds_from_fits


@pytest.fixture(scope="session")
def multiple_files(tmp_path_factory):
    path = tmp_path_factory.mktemp("globbing")
    shape = (10, 10)
    data = np.arange(np.prod(shape), dtype=np.float64)
    data = data.reshape(shape)

    for i in range(3):
        filename = str(path / f"data-{i}.fits")
        primary_hdu = fits.PrimaryHDU(data)
        primary_hdu.writeto(filename, overwrite=True)

    return str(path / f"data*.fits")


def test_globbing(multiple_files):
    datasets = xds_from_fits(multiple_files)
    assert len(datasets) == 3

    for xds in datasets:
        expected = np.arange(np.prod(xds.hdu0.shape), dtype=np.float64)
        expected = expected.reshape(xds.hdu0.shape)
        np.testing.assert_array_equal(xds.hdu0.data, expected)

    combined = xarray.concat(datasets, dim="hdu0-0")
    np.testing.assert_array_equal(
        combined.hdu0.data, np.concatenate([expected] * 3, axis=0)
    )
    combined = xarray.concat(datasets, dim="hdu0-1")
    np.testing.assert_array_equal(
        combined.hdu0.data, np.concatenate([expected] * 3, axis=1)
    )


@pytest.fixture(scope="session")
def beam_cube(tmp_path_factory):
    frequency = np.linspace(0.856e9, 0.856e9 * 2, 32, endpoint=True)
    bandwidth_delta = (frequency[-1] - frequency[0]) / frequency.size
    dtype = np.float64

    # List of key values of the form:
    #
    #    (key, None)
    #    (key, (value,))
    #    (key, (value, comment))
    #

    # We put them in a list so that they are added to the
    # FITS header in the correct order
    axis1 = [
        ("CTYPE", ("X", "points right on the sky")),
        ("CUNIT", ("DEG", "degrees")),
        ("NAXIS", (257, "number of X")),
        ("CRPIX", (129, "reference pixel (one relative)")),
        ("CRVAL", (0.0110828777007, "degrees")),
        ("CDELT", (0.011082, "degrees")),
    ]

    axis2 = [
        ("CTYPE", ("Y", "points up on the sky")),
        ("CUNIT", ("DEG", "degrees")),
        ("NAXIS", (257, "number of Y")),
        ("CRPIX", (129, "reference pixel (one relative)")),
        ("CRVAL", (-2.14349358381e-07, "degrees")),
        ("CDELT", (0.011082, "degrees")),
    ]

    axis3 = [
        ("CTYPE", ("FREQ",)),
        ("CUNIT", None),
        ("NAXIS", (frequency.shape[0], "number of FREQ")),
        ("CRPIX", (1, "reference frequency position")),
        ("CRVAL", (frequency[0], "reference frequency")),
        ("CDELT", (bandwidth_delta, "frequency step in Hz")),
    ]

    axes = [axis1, axis2, axis3]

    metadata = [
        ("SIMPLE", True),
        ("BITPIX", -64),
        ("NAXIS", len(axes)),
        ("OBSERVER", "Observer"),
        ("ORIGIN", "Artificial"),
        ("TELESCOP", "Telescope"),
        ("OBJECT", "beam"),
        ("EQUINOX", 2000.0),
    ]

    # Create header and set metadata
    header = fits.Header(metadata)

    shape = tuple(v[0] for _, _, (_, v), _, _, _ in axes if v is not None)

    # Now set the key value entries for each axis
    # Note that the axes are reversed, compared
    # to the numpy shape
    ax_info = [
        (f"{k}{a}",) + vt
        for a, axis_data in enumerate(reversed(axes), 1)
        for k, vt in axis_data
        if vt is not None
    ]
    header.update(ax_info)

    filename = tmp_path_factory.mktemp("beam") / "beam.fits"
    filename = str(filename)
    # Write some data to it
    data = np.arange(np.prod(shape), dtype=dtype)
    primary_hdu = fits.PrimaryHDU(data.reshape(shape), header=header)
    primary_hdu.writeto(filename, overwrite=True)

    yield filename


def test_beam_creation(beam_cube):
    (xds,) = xds_from_fits(beam_cube)
    cmp_data = np.arange(np.prod(xds.hdu0.shape), dtype=np.float64)
    cmp_data = cmp_data.reshape(xds.hdu0.shape)
    np.testing.assert_array_equal(xds.hdu0.data, cmp_data)
    assert xds.hdu0.data.shape == (257, 257, 32)
    assert xds.hdu0.dims == ("hdu0-0", "hdu0-1", "hdu0-2")
    assert xds.hdu0.attrs == {
        "BITPIX": -64,
        "EQUINOX": 2000.0,
        "OBJECT": "beam",
        "OBSERVER": "Observer",
        "ORIGIN": "Artificial",
        "SIMPLE": True,
        "TELESCOP": "Telescope",
        "CDELT1": 26750000.0,
        "CDELT2": 0.011082,
        "CDELT3": 0.011082,
        "CRPIX1": 1,
        "CRPIX2": 129,
        "CRPIX3": 129,
        "CRVAL1": 856000000.0,
        "CRVAL2": -2.14349358381e-07,
        "CRVAL3": 0.0110828777007,
        "CTYPE1": "FREQ",
        "CTYPE2": "Y",
        "CTYPE3": "X",
        "CUNIT2": "DEG",
        "CUNIT3": "DEG",
        "NAXIS": 3,
        "NAXIS1": 32,
        "NAXIS2": 257,
        "NAXIS3": 257,
    }


def test_distributed(beam_cube):
    """Sanity check for the distributed case"""
    with ExitStack() as stack:
        cluster = stack.enter_context(LocalCluster(n_workers=8, processes=True))
        stack.enter_context(Client(cluster))

        (xds,) = xds_from_fits(beam_cube, chunks={0: 100, 1: 100, 2: 15})
        expected = np.arange(np.prod(xds.hdu0.shape)).reshape(xds.hdu0.shape)
        np.testing.assert_array_equal(expected, xds.hdu0.data)
        assert xds.hdu0.data.chunks == ((100, 100, 57), (100, 100, 57), (15, 15, 2))
