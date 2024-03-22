# -*- coding: utf-8 -*-

"""Main module."""

from functools import reduce
from itertools import product
import logging
import os
import os.path
from collections.abc import Sequence

import dask
import dask.array as da
import fsspec
from fsspec.implementations.local import LocalFileSystem
import numpy as np

import xarray as xr

from xarrayfits.fits_proxy import FitsProxy

log = logging.getLogger("xarray-fits")


def short_fits_file(table_name):
    """
    Returns the last part

    Parameters
    ----------
    table_name : str
        CASA table path

    Returns
    -------
    str
        Shortenend path

    """
    return os.path.split(table_name.rstrip(os.sep))[1]


BITPIX_MAP = {
    np.int8: 8,
    np.int16: 16,
    np.int32: 32,
    np.float32: -32,
    np.float64: -64,
}

INV_BITPIX_MAP = {v: k for k, v in list(BITPIX_MAP.items())}


def ranges(chunks):
    return reduce(lambda i, x: i + [i[-1] + x], chunks, [0])


def slices(r):
    return (slice(s, e) for s, e in zip(r[:-1], r[1:]))


# https://docs.astropy.org/en/stable/io/fits/index.html#working-with-large-files
# https://docs.astropy.org/en/stable/io/fits/index.html#working-with-remote-and-cloud-hosted-files


def _get_data_function(fits_proxy, h, i, dt):
    if fits_proxy.is_memory_mapped:
        data = fits_proxy.hdu_list[h].data[i]
    else:
        data = fits_proxy.hdu_list[h].section[i]

    return data.astype(dt.newbyteorder("="))


def generate_slice_gets(fits_proxy, hdu, shape, dtype, chunks):
    """
    Parameters
    ----------
    fits_proxy : FitsProxy
        FITS Proxy
    hdu : integer
        FITS HDU for which to generate a dask array
    shape : tuple
        Shape of the resulting array
    dtype : np.dtype
        Numpy dtype
    chunks : tuple
        Chunks associated with each shape dimension

    Returns
    -------
    :class:`dask.array.Array`
        Dask array representing the data associated
        with the ``hdu``.
    """

    token = dask.base.tokenize(fits_proxy, hdu, dtype)
    name = "-".join((short_fits_file(fits_proxy._filename), "slice", token))
    dsk_chunks = da.core.normalize_chunks(chunks, shape)

    # Produce keys and slices
    keys = product([name], *[list(range(len(bd))) for bd in dsk_chunks])
    slices_ = product(*[slices(tuple(ranges(c))) for c in dsk_chunks])
    dt = np.dtype(dtype)

    # Create dask graph
    dsk = {
        key: (_get_data_function, fits_proxy, hdu, slice_, dt)
        for key, slice_ in zip(keys, slices_)
    }

    return da.Array(dsk, name, dsk_chunks, dtype)


def array_from_fits_hdu(
    fits_proxy,
    prefix,
    hdu_list,
    hdu_index,
    chunks,
):
    """
    Parameters
    ----------
    fits_proxy : FitsProxy
        The FITS proxy
    hdu_list : :class:`astropy.io.fits.hdu.hdulist.HDUList`
        FITS HDU list
    hdu_index : integer
        HDU index for which to generate an :class:`xarray.DataArray`
    chunks : list of dictionaries

    Returns
    -------
    :class:`xarray.DataArray`
        Array associated with ``hdu_index``
    """

    try:
        hdu = hdu_list[hdu_index]
    except IndexError as e:
        raise IndexError(f"Invalid hdu {hdu_index}") from e

    naxis = hdu.header["NAXIS"]
    bitpix = hdu.header["BITPIX"]
    simple = hdu.header["SIMPLE"]

    if simple is False:
        raise ValueError(
            f"HDU {hdu} doesn't conform " f"to the FITS standard: " f"SIMPLE={simple}"
        )

    try:
        dtype = INV_BITPIX_MAP[bitpix]
    except KeyError:
        raise ValueError(
            f"Couldn't find a numpy type associated "
            f"with BITPIX {bitpix}. Ignoring hdu {hdu_index}"
        )

    shape = []
    flat_chunks = []

    # At this point we are dealing with FORTRAN ordered axes
    for i in range(naxis):
        ax_key = f"NAXIS{naxis - i}"
        ax_shape = hdu.header[ax_key]
        shape.append(ax_shape)

        try:
            # Try add existing chunking strategies to the list
            flat_chunks.append(chunks[i])
        except KeyError:
            flat_chunks.append(ax_shape)

    array = generate_slice_gets(
        fits_proxy,
        hdu_index,
        tuple(shape),
        dtype,
        tuple(flat_chunks),
    )

    dims = tuple(f"{prefix}{hdu_index}-{i}" for i in range(0, naxis))
    attrs = {"header": {k: v for k, v in sorted(hdu.header.items())}}
    return xr.DataArray(array, dims=dims, attrs=attrs)


def xds_from_fits(fits_filename, hdus=None, prefix="hdu", chunks=None):
    """
    Parameters
    ----------
    fits_filename : str or list of str
        FITS filename or a list of FITS filenames.
        The first case supports a globbed pattern.
    hdus : integer or list of integers, optional
        hdus to represent on the returned Dataset.
        If ``None``, all HDUs are selected
    prefix : str, optional
        Array name prefix
    chunks : dictionary or list of dictionaries, optional
        Chunking strategy for each dimension of each hdu.
        Dimensions should be specified via the
        C order dimensions
        :code:`{0: 513, 1: 513, 2: 33}`

    Returns
    -------
    list of :class:`xarray.Dataset`
        A list of xarray Datasets corresponding to glob matches
        in the ``fits_filename`` parameter.
        Each Dataset contains the DataArray's corresponding
        to each HDU on the FITS file.
    """

    if isinstance(fits_filename, str):
        openfiles = fsspec.open_files(fits_filename)
    elif isinstance(fits_filename, Sequence):
        openfiles = fsspec.open_files(fits_filename)
    else:
        raise TypeError(f"{type(fits_filename)} is not a " f"string or Sequence")

    datasets = []

    for of in openfiles:
        fits_proxy = FitsProxy(
            of.full_name, use_fsspec=True, memmap=isinstance(of.fs, LocalFileSystem)
        )

        # Take all hdus if None specified
        if hdus is None:
            hdus = list(range(len(fits_proxy.hdu_list)))
        # promote to list in case of single integer
        elif isinstance(hdus, int):
            hdus = [hdus]

        if chunks is None:
            chunks = [{} for _ in hdus]
        # Promote to list in case of single dict
        elif isinstance(chunks, dict):
            chunks = [chunks]

        if not len(hdus) == len(chunks):
            raise ValueError(
                f"Number of requested hdus ({len(hdus)}) "
                f"does not match the number of "
                f"chunks ({len(chunks)})"
            )

        # Generate xarray datavars for each hdu
        xarrays = {
            f"{prefix}{hdu_index}": array_from_fits_hdu(
                fits_proxy,
                prefix,
                fits_proxy.hdu_list,
                hdu_index,
                hdu_chunks,
            )
            for hdu_index, hdu_chunks in zip(hdus, chunks)
        }

        datasets.append(xr.Dataset(xarrays))

    return datasets
