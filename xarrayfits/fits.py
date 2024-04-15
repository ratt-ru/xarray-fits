# -*- coding: utf-8 -*-

"""Main module."""

from collections.abc import Iterable, Sequence
from functools import reduce
from itertools import product
import logging
import os
import os.path
from typing import Union

import dask
import dask.array as da
import fsspec  # type: ignore
from fsspec.implementations.local import LocalFileSystem  # type: ignore
import numpy as np

import xarray as xr

from xarrayfits.grid import AffineGrid
from xarrayfits.fits_proxy import FitsProxy
from xarrayfits.typing import HduType, ChunksType
from xarrayfits.utils import promote_chunks, promote_hdus

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


FilenameType = Union[str, Iterable[str]]


def xds_from_fits(
    fits_filename: FilenameType, hdus: HduType = None, chunks: ChunksType = None
) -> Iterable[xr.Dataset]:
    """
    Parameters
    ----------
    fits_filename: :code:`str` or :code:`Iterable` of :code:`str`.
        FITS filename or a list of FITS filenames.
        The first case supports a globbed pattern.
    hdus : ``int`` or ``Iterable`` of ``int`` or ``str`` or ``Iterable`` of ``str`` or ``Mapping[int, str]``, optional
        Specifies which HDUs are stored on the returned datasets.

        - If ``None``, all HDUs are selected.
        - If integers, the DataArray's will be named ``hdu{h}``, where ``h`` is the hdu index.
        - If strings are provided, the DataArray's will be named by them.
        - A ``Mapping[int, str]`` will name DataArry hdus at specific indices.
    chunks : ``Mapping[str, int]`` or ``Iterable`` of ``Mapping[str, int]``, optional
        Chunking strategy for each dimension of each hdu.
        Dimensions should be specified via the
        C order dimensions
        :code:`{0: 513, 1: 513, 2: 33}`

    Returns
    -------
    datasets: list of :class:`xarray.Dataset`
        A list of xarray Datasets corresponding to glob matches
        in the ``fits_filename`` parameter.
        Each Dataset contains :class:`xarray.DataArray` 's corresponding
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
        phdus = promote_hdus(hdus, nhdus)
        pchunks = promote_chunks(chunks, len(phdus))

        if len(phdus) > len(pchunks):
            raise ValueError(
                f"Number of requested hdus ({len(phdus)}) "
                f"does not match the number of "
                f"chunks ({len(pchunks)})"
            )

        singleton = len(phdus) == 1

        # Generate xarray datavars for each hdu
        xarrays = {
            f"{name}": array_from_fits_hdu(
                fits_proxy, fits_proxy.hdu_list, index, name, hdu_chunks, singleton
            )
            for (index, name), hdu_chunks in zip(sorted(phdus.items()), pchunks)
        }

        datasets.append(xr.Dataset(xarrays))

    return datasets
