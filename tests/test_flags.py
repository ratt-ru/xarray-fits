import numpy as np
import pytest
import xarray as xr
from astropy.io import fits
from numpy.testing import assert_array_equal

from xarrayfits.testing.simulator import simulate_fits_image

try:
  from xradio.image import check_image
except ImportError:
  check_image = None

ENGINE = "xarray-fits:fits"

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingMetadataWarning"
)


def nan_corner_data():
  data = np.arange(90, dtype=np.float32).reshape(1, 3, 5, 6)
  data[..., :2, :2] = np.nan
  return data


def test_float_images_get_a_lazy_flag(tmp_path, assert_conforms):
  path = simulate_fits_image(tmp_path / "image.fits", data=nan_corner_data())
  ds = xr.open_dataset(path, engine=ENGINE)
  flag = ds.FLAG_SKY

  assert flag.dtype == bool
  assert flag.dims == ds.SKY.dims
  assert flag.attrs == {"type": "flag"}
  assert ds.SKY.attrs["flag"] == "FLAG_SKY"
  assert ds.attrs["data_groups"]["base"]["flag"] == "FLAG_SKY"
  assert_array_equal(flag, np.isnan(ds.SKY))
  assert flag.values.sum() == 3 * 4
  assert_conforms(ds, path)


def test_images_without_nans_get_an_unset_flag(tmp_path, assert_conforms):
  path = simulate_fits_image(tmp_path / "image.fits")
  ds = xr.open_dataset(path, engine=ENGINE, chunks={})

  assert ds.FLAG_SKY.chunks == ds.SKY.chunks
  assert not ds.FLAG_SKY.values.any()
  assert_conforms(ds, path)


def test_flags_are_read_with_the_pixels(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits")
  ds = xr.open_dataset(path, engine=ENGINE)

  with fits.open(path, mode="update") as hdu_list:
    hdu_list[0].data[0, 1, 2, 3] = np.nan

  assert ds.FLAG_SKY.values.sum() == 1
  assert ds.FLAG_SKY.isel(frequency=1, l=3, m=2).item()


def test_integer_images_have_no_flag(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits", dtype=np.int16)
  ds = xr.open_dataset(path, engine=ENGINE)

  assert "FLAG_SKY" not in ds
  assert "flag" not in ds.SKY.attrs
  assert "flag" not in ds.attrs["data_groups"]["base"]


def test_dropped_flags_are_unwired(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits")
  ds = xr.open_dataset(path, engine=ENGINE, drop_variables=["FLAG_SKY"])

  assert "FLAG_SKY" not in ds
  assert "flag" not in ds.SKY.attrs
  assert "flag" not in ds.attrs["data_groups"]["base"]
  if check_image is not None:
    assert list(check_image(ds)) == []
