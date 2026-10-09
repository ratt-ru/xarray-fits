import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from xarrayfits.errors import MissingObservationDateWarning, MissingSpectralFrameWarning
from xarrayfits.testing.simulator import simulate_fits_image

ENGINE = "xarray-fits:fits"


def open_simple_image(path):
  with pytest.warns(MissingObservationDateWarning):
    with pytest.warns(MissingSpectralFrameWarning):
      return xr.open_dataset(path, engine=ENGINE)


def test_simple_fits_image_opens_as_an_image_dataset(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits")
  ds = open_simple_image(path)

  assert ds.attrs["type"] == "image_dataset"
  assert ds.attrs["schema_version"] == "0.0.2"
  assert ds.attrs["data_groups"] == {"base": {"sky": "SKY"}}
  assert ds.SKY.dims == ("time", "frequency", "polarization", "l", "m")
  assert ds.SKY.dtype == np.float32
  assert ds.polarization.values.tolist() == ["I"]
  assert ds.time.values.tolist() == [0.0]
  assert_array_equal(ds.frequency, [1.414999e9, 1.415e9, 1.415001e9])


def test_sky_holds_the_fits_pixels(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits")
  ds = open_simple_image(path)

  # FITS pixels are (stokes, freq, dec, ra): SKY is (time, freq, pol, l, m)
  pixels = np.arange(3 * 5 * 6, dtype=np.float32).reshape(1, 3, 5, 6)
  expected = pixels.transpose(1, 0, 3, 2)[None]
  assert_array_equal(ds.SKY.values, expected)
  assert_array_equal(ds.SKY.isel(frequency=1, l=slice(2, 4)), expected[:, 1, :, 2:4])


def test_l_keeps_the_sign_of_cdelt(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits")
  ds = open_simple_image(path)

  assert np.all(np.diff(ds.l) < 0)
  assert np.all(np.diff(ds.m) > 0)
  assert ds.l.values[3] == 0.0
  assert ds.m.values[2] == 0.0


def test_simple_fits_image_conforms(tmp_path, assert_conforms):
  path = simulate_fits_image(tmp_path / "image.fits")
  assert_conforms(open_simple_image(path), path)
