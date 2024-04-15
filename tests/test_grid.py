from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from xarrayfits.grid import AffineGrid


@pytest.fixture(params=[(10, 20, 30)])
def header(request):
    # Reverse into FORTRAN order
    rev_dims = list(reversed(request.param))
    naxes = {f"NAXIS{d + 1}": s for d, s in enumerate(rev_dims)}
    crpix = {f"CRPIX{d + 1}": 5 + d for d, _ in enumerate(rev_dims)}
    crval = {f"CRVAL{d + 1}": 1.0 + d for d, _ in enumerate(rev_dims)}
    cdelt = {f"CDELT{d + 1}": 2.0 + d for d, _ in enumerate(rev_dims)}
    cunit = {f"CUNIT{d + 1}": f"UNIT-{len(rev_dims) - d}" for d in range(len(rev_dims))}
    ctype = {f"CTYPE{d + 1}": f"TYPE-{len(rev_dims) - d}" for d in range(len(rev_dims))}
    cname = {f"CNAME{d + 1}": f"NAME-{len(rev_dims) - d}" for d in range(len(rev_dims))}

    return fits.Header(
        {
            "NAXIS": len(request.param),
            **naxes,
            **crpix,
            **crval,
            **cdelt,
            **cname,
            **ctype,
            **cunit,
        }
    )


def test_affine_grid(header):
    grid = AffineGrid(header)
    ndims = grid.ndims
    assert ndims == header["NAXIS"]
    assert grid.naxis == [10, 20, 30]
    assert grid.crpix == [7, 6, 5]
    assert grid.crval == [3.0, 2.0, 1.0]
    assert grid.cdelt == [4.0, 3.0, 2.0]
    assert grid.cname == [header[f"CNAME{ndims - i}"] for i in range(ndims)]
    assert grid.cunit == [header[f"CUNIT{ndims - i}"] for i in range(ndims)]
    assert grid.ctype == [header[f"CTYPE{ndims - i}"] for i in range(ndims)]

    # Worked coordinate example
    assert_array_equal(grid.coords(0), 3.0 + (np.arange(1, 10 + 1) - 7) * 4.0)
    assert_array_equal(grid.coords(1), 2.0 + (np.arange(1, 20 + 1) - 6) * 3.0)
    assert_array_equal(grid.coords(2), 1.0 + (np.arange(1, 30 + 1) - 5) * 2.0)

    # More automatic version
    for d in range(ndims):
        assert_array_equal(
            grid.coords(d),
            grid.crval[d]
            + (np.arange(1, grid.naxis[d] + 1) - grid.crpix[d]) * grid.cdelt[d],
        )
