#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `xarrayfits` package."""

from contextlib import ExitStack
import mmap
import os.path

from astropy.io import fits
from dask.distributed import Client, LocalCluster
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import xarray

from xarrayfits import xds_from_fits
from xarrayfits.fits_proxy import FitsProxy


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


@pytest.fixture(scope="session")
def multiple_hdu_file(tmp_path_factory):
    ctypes = ["X", "Y", "FREQ", "STOKES"]

    def make_hdu(hdu_cls, shape):
        data = np.arange(np.prod(shape), dtype=np.float64)
        data = data.reshape(shape)
        header = {
            # "SIMPLE": True,
            # "BITPIX": -64,
            # "NAXIS": len(data),
            # **{f"NAXIS{data.ndim - i}": d for i, d in enumerate(data.shape)},
            **{f"CTYPE{data.ndim - i}": ctypes[i] for i in range(data.ndim)},
        }

        return hdu_cls(data, header=fits.Header(header))

    hdu1 = make_hdu(fits.PrimaryHDU, (10, 10))
    hdu2 = make_hdu(fits.ImageHDU, (10, 20, 30))
    hdu3 = make_hdu(fits.ImageHDU, (30, 40, 50))

    filename = str(tmp_path_factory.mktemp("multihdu") / "data.fits")
    hdu_list = fits.HDUList([hdu1, hdu2, hdu3])
    hdu_list.writeto(filename, overwrite=True)

    return filename


@pytest.fixture(scope="session")
def multiple_files(tmp_path_factory):
    path = tmp_path_factory.mktemp("globbing")
    shape = (10, 10)
    data = np.arange(np.prod(shape), dtype=np.float64)
    data = data.reshape(shape)

    filenames = []

    for i in range(3):
        filename = str(path / f"data-{i}.fits")
        filenames.append(filename)
        primary_hdu = fits.PrimaryHDU(data)
        primary_hdu.writeto(filename, overwrite=True)

    return filenames


def multiple_dataset_tester(datasets):
    assert len(datasets) == 3

    for xds in datasets:
        expected = np.arange(np.prod(xds.hdu.shape), dtype=np.float64)
        expected = expected.reshape(xds.hdu.shape)
        assert_array_equal(xds.hdu.data, expected)

    combined = xarray.concat(datasets, dim="hdu-0")
    assert_array_equal(combined.hdu.data, np.concatenate([expected] * 3, axis=0))
    assert combined.hdu.dims == ("hdu-0", "hdu-1")

    combined = xarray.concat(datasets, dim="hdu-1")
    assert_array_equal(combined.hdu.data, np.concatenate([expected] * 3, axis=1))
    assert combined.hdu.dims == ("hdu-0", "hdu-1")

    tds = [ds.expand_dims(dim="time", axis=0) for ds in datasets]
    combined = xarray.concat(tds, dim="time")
    assert_array_equal(combined.hdu.data, np.stack([expected] * 3, axis=0))
    assert combined.hdu.dims == ("time", "hdu-0", "hdu-1")


def test_list_files(multiple_files):
    datasets = xds_from_fits(multiple_files)
    return multiple_dataset_tester(datasets)


def test_globbing(multiple_files):
    path, _ = os.path.split(multiple_files[0])
    datasets = xds_from_fits(f"{path}{os.sep}data*.fits")
    return multiple_dataset_tester(datasets)


def test_multiple_unnamed_hdus(multiple_hdu_file):
    """Test hdu requests with hdu indexes"""
    (ds,) = xds_from_fits(multiple_hdu_file, hdus=0)
    assert len(ds.data_vars) == 1
    assert ds.hdu.shape == (10, 10)
    assert ds.hdu.dims == ("X", "Y")

    (ds,) = xds_from_fits(multiple_hdu_file, hdus=[0, 2])
    assert len(ds.data_vars) == 2

    assert ds.hdu0.shape == (10, 10)
    assert ds.hdu0.dims == ("hdu0-X", "hdu0-Y")

    assert ds.hdu2.shape == (30, 40, 50)
    assert ds.hdu2.dims == ("hdu2-X", "hdu2-Y", "hdu2-FREQ")


def test_multiple_named_hdus(multiple_hdu_file):
    """Test hdu requests with named hdus"""
    (ds,) = xds_from_fits(multiple_hdu_file, hdus={0: "beam"})
    assert ds.beam.dims == ("X", "Y")
    assert ds.beam.shape == (10, 10)

    (ds,) = xds_from_fits(multiple_hdu_file, hdus=["beam"])
    assert ds.beam.dims == ("X", "Y")
    assert ds.beam.shape == (10, 10)

    (ds,) = xds_from_fits(multiple_hdu_file, hdus={0: "beam", 2: "3C147"})
    assert ds.beam.dims == ("beam-X", "beam-Y")
    assert ds.beam.shape == (10, 10)
    assert ds["3C147"].dims == ("3C147-X", "3C147-Y", "3C147-FREQ")
    assert ds["3C147"].shape == (30, 40, 50)

    (ds,) = xds_from_fits(multiple_hdu_file, hdus=["beam", "3C147"])
    assert ds.beam.dims == ("beam-X", "beam-Y")
    assert ds.beam.shape == (10, 10)
    assert ds["3C147"].dims == ("3C147-X", "3C147-Y", "3C147-FREQ")
    assert ds["3C147"].shape == (10, 20, 30)


def test_beam_creation(beam_cube):
    (xds,) = xds_from_fits(beam_cube)
    cmp_data = np.arange(np.prod(xds.hdu.shape), dtype=np.float64)
    cmp_data = cmp_data.reshape(xds.hdu.shape)
    assert_array_equal(xds.hdu.data, cmp_data)
    assert xds.hdu.data.shape == (257, 257, 32)
    assert xds.hdu.dims == ("X", "Y", "FREQ")
    assert xds.hdu.attrs == {
        "header": {
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
    }


def test_distributed(beam_cube):
    """Sanity check for the distributed case"""
    with ExitStack() as stack:
        cluster = stack.enter_context(LocalCluster(n_workers=8, processes=True))
        stack.enter_context(Client(cluster))

        (xds,) = xds_from_fits(beam_cube, chunks={0: 100, 1: 100, 2: 15})
        expected = np.arange(np.prod(xds.hdu.shape)).reshape(xds.hdu.shape)
        assert_array_equal(expected, xds.hdu.data)
        assert xds.hdu.data.chunks == ((100, 100, 57), (100, 100, 57), (15, 15, 2))


def test_memory_mapped(beam_cube):
    with fits.open(beam_cube, memmap=True) as hdu_list:
        hdu_list[0].data[:]
        astropy_file = hdu_list.fileinfo(0)["file"]
        assert isinstance(astropy_file._mmap, mmap.mmap)

    proxy = FitsProxy(beam_cube)
    assert proxy.is_memory_mapped
    proxy.hdu_list[0].data[:]
    astropy_file = proxy.hdu_list.fileinfo(0)["file"]
    assert isinstance(astropy_file._mmap, mmap.mmap)

    proxy = FitsProxy(beam_cube, memmap=True)
    assert proxy.is_memory_mapped
    proxy.hdu_list[0].data[:]
    astropy_file = proxy.hdu_list.fileinfo(0)["file"]
    assert isinstance(astropy_file._mmap, mmap.mmap)

    proxy = FitsProxy(beam_cube, memmap=None)
    assert proxy.is_memory_mapped
    proxy.hdu_list[0].data[:]
    astropy_file = proxy.hdu_list.fileinfo(0)["file"]
    assert isinstance(astropy_file._mmap, mmap.mmap)

    proxy = FitsProxy(beam_cube, memmap=False)
    assert not proxy.is_memory_mapped
    proxy.hdu_list[0].data[:]
    astropy_file = proxy.hdu_list.fileinfo(0)["file"]
    assert astropy_file._mmap is None
