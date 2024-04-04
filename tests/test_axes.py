from astropy.io import fits
import pytest

from xarrayfits.axes import Axes


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


def test_axes(header):
    axes = Axes(header)
    ndims = axes.ndims
    assert ndims == header["NAXIS"]
    assert axes.naxis == [10, 20, 30]
    assert axes.crpix == [7, 6, 5]
    assert axes.crval == [3.0, 2.0, 1.0]
    assert axes.cdelt == [4.0, 3.0, 2.0]
    assert axes.cname == [header[f"CNAME{ndims - i}"] for i in range(ndims)]
    assert axes.cunit == [header[f"CUNIT{ndims - i}"] for i in range(ndims)]
    assert axes.ctype == [header[f"CTYPE{ndims - i}"] for i in range(ndims)]
