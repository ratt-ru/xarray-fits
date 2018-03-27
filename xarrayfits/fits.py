# -*- coding: utf-8 -*-

"""Main module."""

from future_builtins import zip, map

from collections import OrderedDict
from functools import partial
from itertools import product
import logging
from operator import add
import os
import os.path

from six.moves import range

from astropy.io import fits
import dask
import dask.array as da
import numpy as np

try:
    from cytoolz import accumulate, cons, merge
except ImportError:
    from toolz import accumulate, cons, merge

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

def fits_open_graph(fits_file, **kwargs):
    """
    Generate a dask graph containing fits open commands

    Parameters
    ----------
    fits_file : str
        FITS filename
    **kwargs (optional) :
        Keywords arguments passed to the :meth:`astropy.io.fits.open`
        command.`

    Returns
    -------
    tuple
        Graph key associated with the opened file
    dict
        Dask graph containing the graph open command

    """
    token = dask.base.tokenize(fits_file, kwargs)
    fits_key = ('open', short_fits_file(fits_file), token)
    fits_graph = { fits_key: (partial(FitsProxy, **kwargs), fits_file) }
    return fits_key, fits_graph

BITPIX_MAP = {
    np.int8: 8,
    np.int16: 16,
    np.int32: 32,
    np.float32: -32,
    np.float64: -64,
}

INV_BITPIX_MAP = { v: k for k, v in BITPIX_MAP.items() }

ranges = lambda c: accumulate(add, cons(0, c))
slices = lambda r: (slice(s,e) for s, e in zip(r[:-1], r[1:]))
_get_fn = lambda fp, h, i: fp("__getitem__", h).data.__getitem__(i)

def generate_slice_gets(fits_filename, fits_key, fits_graph,
                        hdu, shape, dtype, chunks):
    """
    Parameters
    ----------
    fits_filename : str
        FITS filename
    fits_key : tuple
        dask key referencing an opened FITS file object
    fits_graph : dict
        dask graph containing ``fits_key`` referencing an
        opened FITS file object
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

    token = dask.base.tokenize(fits_filename)
    name = "-".join((short_fits_file(fits_filename), "slice", token))

    dsk_chunks = da.core.normalize_chunks(chunks, shape)

    # Produce keys and slices
    keys = product([name], *[range(len(bd)) for bd in dsk_chunks])
    slices_ = product(*[slices(tuple(ranges(c))) for c in dsk_chunks])

    # Create dask graph
    dsk = {key: (_get_fn, fits_key, hdu, slice_)
                        for key, slice_ in zip(keys, slices_)}

    return da.Array(merge(dsk, fits_graph), name, dsk_chunks, dtype)

def _xarray_from_fits_hdu(fits_filename, fits_key, fits_graph,
                        name_prefix, hdu_list, hdu_index, chunks):

    """
    Parameters
    ----------
    fits_filename : str
        FITS filename
    fits_key : tuple
        dask key referencing an opened FITS file object
    fits_graph : dict
        dask graph containing ``fits_key`` referencing an
        opened FITS file object
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
    except IndexError:
        raise IndexError("Invalid hdu %d" % hdu_index)

    naxis = hdu.header['NAXIS']
    bitpix = hdu.header['BITPIX']
    simple = hdu.header['SIMPLE']

    if not simple == True:
        raise ValueError("Non-fits conforming hdu %d "
                         "header['SIMPLE']=%s." % (hdu, simple))

    try:
        dtype = INV_BITPIX_MAP[bitpix]
    except KeyError:
        raise ValueError("Couldn't find a numpy type associated "
                         "with BITPIX %d. Ignoring hdu %d" %
                            (bitpix, hdu_index))

    shape = []
    flat_chunks = []

    # At this point we are dealing with FORTRAN ordered axes
    for i in range(1, naxis+1):
        ax_key = 'NAXIS%d' % i
        s = hdu.header[ax_key]
        shape.append(s)

        try:
            # Try add existing chunking strategies to the list
            flat_chunks.append(chunks[ax_key])
        except KeyError:
            # Otherwise do single slices of some row major axes
            flat_chunks.append(1 if i >  2 else s)

    # Reverse to get C major ordering
    shape = tuple(reversed(shape))
    flat_chunks = tuple(reversed(flat_chunks))

    array = generate_slice_gets(fits_filename, fits_key, fits_graph,
                                    hdu_index, shape, dtype, flat_chunks)

    name = '%s%d' % (name_prefix, hdu_index)
    dims = tuple("%s-%d" % (name, i) for i in range(3, 0, -1))
    attrs = {"fits_header": OrderedDict((k, v) for k,v in hdu.header.items())}
    return xr.DataArray(array, dims=dims, attrs=attrs)


def xds_from_fits(fits_filename, hdus=None, name_prefix="hdu", chunks=None):
    """
    Parameters
    ----------
    fits_filename : str
        FITS filename
    hdus : integer or list of integers, optional
        hdus to represent on the returned Dataset.
        If ``None``, all HDUs are selected
    name_prefix : str, optional
        Array name prefix
    chunks : dictionary or list of dictionaries, optional
        Chunking strategy for each dimension of each hdu.
        Dimensions should be specified via the
        standard FITS dimensions
        :code:`{'NAXIS1' : 513, 'NAXIS2' : 513, 'NAXIS3' : 33}`

    Returns
    -------
    :class:`xarray.Dataset`
        xarray Dataset containing DataArray's representing the
        specified HDUs on the FITS file.
    """

    with fits.open(fits_filename) as hdu_list:
        # Take all hdus if None specified
        if hdus is None:
            hdus = list(range(len(hdu_list)))
        # promote to list in case of single integer
        elif isinstance(hdus, int):
            hdus = [hdus]

        if chunks is None:
            chunks = [{} for h in hdus]
        # Promote to list in case of single dict
        elif isinstance(chunks, dict):
            chunks = [chunks]

        if not len(hdus) == len(chunks):
            raise ValueError("Number of requested hdus (%d) "
                             "does not match the number of "
                             "chunks (%d)" %
                                (len(hdus), len(chunks)))

        fits_key, fits_graph = fits_open_graph(fits_filename)

        fn = _xarray_from_fits_hdu

        xarrays = { "%s%d" % (name_prefix, hdu_index) : fn(fits_filename,
                                        fits_key, fits_graph, name_prefix,
                                        hdu_list, hdu_index, hdu_chunks)
                    for hdu_index, hdu_chunks in zip(hdus, chunks) }

        return xr.Dataset(xarrays)
