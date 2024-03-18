#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `xarrayfits` package."""

import os

from astropy.io import fits
import numpy as np
import pytest

from xarrayfits import xds_from_fits


@pytest.fixture
def data_cube(tmp_path):
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
    header = fits.Header()
    header.update(metadata)

    shape = tuple(
        reversed([nax[1][0] for _, _, nax, _, _, _ in axes if nax[1] is not None])
    )

    # Now set the key value entries for each axis
    ax_info = [
        ("%s%d" % (k, a),) + vt
        for a, axis_data in enumerate(axes, 1)
        for k, vt in axis_data
        if vt is not None
    ]
    header.update(ax_info)

    filename = os.path.join(str(tmp_path), "beam.fits")
    # Write some data to it
    data = np.arange(np.prod(shape), dtype=dtype)
    primary_hdu = fits.PrimaryHDU(data.reshape(shape), header=header)
    primary_hdu.writeto(filename, overwrite=True)

    yield filename


def test_beam_creation(data_cube):
    xds = xds_from_fits(data_cube)
    data = xds.hdu0.data.compute()

    cmp_data = np.arange(np.prod(xds.hdu0.shape), dtype=np.float64)
    assert np.all(data.ravel() == cmp_data)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("fits")
    args = p.parse_args()

    chunks = {"NAXIS1": 513, "NAXIS2": 513, "NAXIS3": 1}
    data = xds_from_fits(args.fits, chunks=chunks)

    print(data)

    from pprint import pprint

    print(data.hdu0.data.compute().shape)
    pprint(dict(data.hdu0.attrs["fits_header"]))
