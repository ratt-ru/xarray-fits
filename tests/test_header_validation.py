import warnings

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits
from numpy.testing import assert_array_equal

from xarrayfits.errors import (
  IgnoredHduWarning,
  InvalidFitsImage,
  UnsupportedFitsImage,
)
from xarrayfits.testing.simulator import (
  DEC,
  FREQ,
  RA,
  FitsAxis,
  simulate_fits_image,
  stokes_axis,
)

ENGINE = "xarray-fits:fits"

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingMetadataWarning"
)


def open_image(path):
  return xr.open_dataset(path, engine=ENGINE)


@pytest.mark.parametrize(
  "dtype", [np.uint8, np.int16, np.int32, np.int64, np.float32, np.float64]
)
def test_every_bitpix_reads_in_native_byte_order(tmp_path, dtype):
  path = simulate_fits_image(tmp_path / "image.fits", dtype=dtype)
  sky = open_image(path).SKY

  assert sky.dtype == dtype
  assert sky.values.dtype.isnative
  assert_array_equal(sky.values.ravel()[:4], [0, 6, 12, 18])


def test_compressed_hdus_are_rejected(tmp_path):
  compressed = fits.CompImageHDU(np.zeros((4, 4), dtype=np.float32))
  path = simulate_fits_image(tmp_path / "image.fits", extra_hdus=[compressed])

  with pytest.raises(UnsupportedFitsImage, match="compressed"):
    open_image(path)


def test_scaled_pixels_are_rejected(tmp_path):
  data = np.arange(90, dtype=np.uint16).reshape(1, 3, 5, 6)
  path = simulate_fits_image(tmp_path / "image.fits", data=data)

  with pytest.raises(UnsupportedFitsImage, match="BSCALE/BZERO"):
    open_image(path)


def test_primaries_without_an_image_are_rejected(tmp_path):
  one_axis = tmp_path / "one_axis.fits"
  fits.PrimaryHDU(np.zeros(4, dtype=np.float32)).writeto(one_axis)

  groups = tmp_path / "groups.fits"
  data = fits.GroupData(
    np.zeros((2, 1, 1, 1, 3)), parnames=["UU"], pardata=[np.zeros(2)], bitpix=-32
  )
  fits.GroupsHDU(data).writeto(groups)

  for path in (one_axis, groups):
    with pytest.raises(InvalidFitsImage, match="holds no image"):
      open_image(str(path))


@pytest.mark.parametrize(
  "axes, match",
  [
    (
      (
        FitsAxis("GLON-SIN", 6, 0.0, 1.0, 1.0),
        FitsAxis("GLAT-SIN", 5, 0.0, 1.0, 1.0),
      ),
      "GLON-SIN is an unsupported axis",
    ),
    ((FitsAxis("UU", 6, 0.0, 1.0, 1.0), FitsAxis("VV", 5, 0.0, 1.0, 1.0)), "UU"),
    ((RA, DEC, FitsAxis("VOPT-F2W", 3, 0.0, 1.0, 1.0)), "unsupported spectral"),
    ((RA, DEC, FitsAxis("VELO", 3, 0.0, 1.0, 1.0)), "unsupported spectral"),
    ((RA, FREQ), "both direction axes"),
  ],
)
def test_unsupported_axes_are_rejected(tmp_path, axes, match):
  path = simulate_fits_image(tmp_path / "image.fits", axes=axes)

  with pytest.raises(UnsupportedFitsImage, match=match):
    open_image(path)


def test_cd_matrices_are_rejected(tmp_path):
  path = simulate_fits_image(
    tmp_path / "image.fits",
    cards={"CD1_1": -1.0 / 60.0, "CD2_2": 1.0 / 60.0},
    remove=("CDELT1", "CDELT2"),
  )

  with pytest.raises(UnsupportedFitsImage, match="CDi_j matrix"):
    open_image(path)


def test_other_extensions_are_ignored_with_a_warning(tmp_path):
  other = fits.ImageHDU(np.zeros((2, 2), dtype=np.float32), name="OTHER")
  beams = fits.BinTableHDU.from_columns(
    [fits.Column(name="BMAJ", format="E", array=np.ones(1))], name="BEAMS"
  )
  path = simulate_fits_image(
    tmp_path / "image.fits", axes=(RA, DEC, FREQ, stokes_axis()), extra_hdus=[beams]
  )
  with warnings.catch_warnings():
    warnings.simplefilter("error", IgnoredHduWarning)
    open_image(path)

  path = simulate_fits_image(tmp_path / "other.fits", extra_hdus=[other])
  with pytest.warns(IgnoredHduWarning, match="OTHER"):
    open_image(path)
