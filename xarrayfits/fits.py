# -*- coding: utf-8 -*-

"""Main module."""

from collections import Counter
from functools import reduce
from itertools import product
from numbers import Integral
import logging
import os
import os.path
from collections.abc import Sequence, Mapping

import dask
import dask.array as da
import fsspec
from fsspec.implementations.local import LocalFileSystem
import numpy as np

import xarray as xr

from xarrayfits.grid import AffineGrid
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

    token = dask.base.tokenize(fits_proxy, shape, chunks, hdu, dtype)
    name = "-".join((short_fits_file(fits_proxy._filename), "slice", token))
    dsk_chunks = da.core.normalize_chunks(chunks, shape, dtype=dtype)

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
    hdu_list,
    hdu_index,
    hdu_name,
    chunks,
    singleton,
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
    hdu_name : str
        HDU name
    chunks : list of dictionaries
    singleton : bool
        True if only a single hdu is selected.
        If False, dimensions will be suffixed with hdu indices

    Returns
    -------
    :class:`xarray.DataArray`
        Array associated with ``hdu_index``
    """

    try:
        hdu = hdu_list[hdu_index]
    except IndexError as e:
        raise IndexError(f"Invalid hdu {hdu_index}") from e

    try:
        is_simple = hdu.header["SIMPLE"] is True
    except KeyError:
        try:
            ext = hdu.header["XTENSION"]
        except KeyError:
            raise ValueError(
                f"Neither SIMPLE of XTENSION header card is present"
            ) from e
        else:
            if ext != "IMAGE":
                raise ValueError(f"{ext} XTENSION is not supported")
    else:
        if not is_simple:
            raise ValueError(f"SIMPLE is not True")

    bitpix = hdu.header["BITPIX"]

    try:
        dtype = INV_BITPIX_MAP[bitpix]
    except KeyError:
        raise ValueError(
            f"Couldn't find a numpy type associated "
            f"with BITPIX {bitpix}. Ignoring hdu {hdu_index}"
        )

    shape = []
    flat_chunks = []
    grid = AffineGrid(hdu.header)

    # Determine shapes and apply chunking
    for d in range(grid.ndims):
        shape.append(grid.naxis[d])

        try:
            # Try add existing chunking strategies to the list
            flat_chunks.append(chunks[d])
        except KeyError:
            flat_chunks.append(grid.naxis[d])

    array = generate_slice_gets(
        fits_proxy,
        hdu_index,
        tuple(shape),
        dtype,
        tuple(flat_chunks),
    )

    dims = []

    for d in range(grid.ndims):
        if name := grid.name(d):
            dim_name = name if singleton else f"{hdu_name}-{name}"
        else:
            dim_name = f"{hdu_name}-{d}"

        dims.append(dim_name)

    coords = {d: (d, grid.coords(i)) for i, d in enumerate(dims)}
    attrs = {"header": {k: v for k, v in sorted(hdu.header.items())}}
    return xr.DataArray(array, dims=dims, coords=coords, attrs=attrs)


def xds_from_fits(fits_filename, hdus=None, chunks=None):
    """
    Parameters
    ----------
    fits_filename : str or list of str
        FITS filename or a list of FITS filenames.
        The first case supports a globbed pattern.
    hdus : Int or List[Int] or str or List[str] or Dict[Int, str], optional
        hdus to store on the returned datasets
        If ``None``, all HDUs are selected.

        if integers, the DataArray's will be named ``hdu{h}``, where h
        is the hdu index.

        In strings are provided, the DataArray's will be named by name.

        A Dict[int, str] will name DataArry hdus at specific indices.
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

        nhdus = len(fits_proxy.hdu_list)

        type_err_msg = (
            f"hdus must a int, str, "
            f"Sequence[int], Sequence[str], "
            f"or a Mapping[int, str]"
        )

        # Take all hdus if None specified
        if hdus is None:
            hdus = list(range(nhdus))
        # promote to list in case of single integer or string
        elif isinstance(hdus, (Integral, str)):
            hdus = [hdus]
        elif isinstance(hdus, (Sequence, Mapping)):
            pass
        else:
            raise TypeError(type_err_msg)

        if isinstance(hdus, Mapping):
            if not all(isinstance(i, int) and i < nhdus for i in hdus.keys()):
                raise ValueError(f"{hdus} keys must be integers")
            if any(v > 1 for v in Counter(hdus.values()).values()):
                raise ValueError(f"{hdus} values must be unique strings")
        elif isinstance(hdus, Sequence):
            if all(isinstance(h, str) for h in hdus):
                hdus = {i: h for i, h in enumerate(hdus)}
            elif all(isinstance(h, int) for h in hdus):
                if len(hdus) == 1:
                    hdus = {0: "hdu"}
                else:
                    hdus = {i: f"hdu{i}" for i in hdus}
        else:
            raise TypeError(type_err_msg)

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

        singleton = len(hdus) == 1

        # Generate xarray datavars for each hdu
        xarrays = {
            f"{name}": array_from_fits_hdu(
                fits_proxy, fits_proxy.hdu_list, index, name, hdu_chunks, singleton
            )
            for (index, name), hdu_chunks in zip(sorted(hdus.items()), chunks)
        }

        datasets.append(xr.Dataset(xarrays))

    return datasets
