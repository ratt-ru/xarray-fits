import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from xarrayfits.errors import UnsupportedFitsImage
from xarrayfits.testing.simulator import (
  DEC,
  FREQ,
  RA,
  simulate_fits_image,
  stokes_axis,
)

ENGINE = "xarray-fits:fits"

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingMetadataWarning"
)


@pytest.mark.parametrize(
  "stokes, labels",
  [
    (stokes_axis(1.0, 1.0, 4), ["I", "Q", "U", "V"]),
    (stokes_axis(4.0), ["V"]),
    (stokes_axis(-1.0, -1.0, 4), ["RR", "RL", "LR", "LL"]),
    (stokes_axis(-5.0, -1.0, 4), ["XX", "XY", "YX", "YY"]),
    (stokes_axis(-5.0, -1.0, 2), ["XX", "YY"]),
    (stokes_axis(2.0, 1.0, 2, crpix=2.0), ["I", "Q"]),
  ],
)
def test_polarizations_are_in_canonical_order(
  tmp_path, assert_conforms, stokes, labels
):
  path = simulate_fits_image(tmp_path / "image.fits", axes=(RA, DEC, FREQ, stokes))
  ds = xr.open_dataset(path, engine=ENGINE)

  assert ds.polarization.values.tolist() == labels
  assert_conforms(ds, path)


def test_images_without_a_stokes_axis_are_stokes_i(tmp_path, assert_conforms):
  path = simulate_fits_image(tmp_path / "image.fits", axes=(RA, DEC, FREQ))
  ds = xr.open_dataset(path, engine=ENGINE)

  assert ds.polarization.values.tolist() == ["I"]
  assert ds.SKY.shape == (1, 3, 1, 6, 5)
  assert_conforms(ds, path)


def test_unsupported_stokes_codes_are_rejected(tmp_path):
  path = simulate_fits_image(
    tmp_path / "image.fits", axes=(RA, DEC, FREQ, stokes_axis(5.0))
  )

  with pytest.raises(UnsupportedFitsImage, match="STOKES axis value 5"):
    xr.open_dataset(path, engine=ENGINE)


def test_pixels_follow_their_polarization(tmp_path):
  # FITS planes are RR, LL, RL, LR
  data = np.arange(4, dtype=np.float32)[:, None, None, None] * np.ones((4, 3, 5, 6))
  path = simulate_fits_image(
    tmp_path / "image.fits",
    axes=(RA, DEC, FREQ, stokes_axis(-1.0, -1.0, 4)),
    data=data.astype(np.float32),
  )
  ds = xr.open_dataset(path, engine=ENGINE)

  plane = {"RR": 0, "LL": 1, "RL": 2, "LR": 3}
  for label, value in plane.items():
    assert np.all(ds.SKY.sel(polarization=label).values == value)

  assert_array_equal(ds.SKY.isel(polarization=[3, 0])[0, 0, :, 0, 0], [1, 0])
  chunked = xr.open_dataset(path, engine=ENGINE, chunks={})
  xr.testing.assert_identical(chunked.SKY.compute(), ds.SKY.load())
