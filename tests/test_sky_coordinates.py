import tracemalloc

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits
from numpy.testing import assert_array_equal

from xarrayfits.testing.simulator import simulate_fits_image

ENGINE = "xarray-fits:fits"

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingMetadataWarning"
)


def large_sparse_image(path, n=16384):
  """Writes the header of an n x n float32 FITS Image and sizes the file
  sparsely, without writing its pixels"""
  header = fits.Header()
  header["SIMPLE"] = True
  header["BITPIX"] = -32
  header["NAXIS"] = 2
  header["NAXIS1"] = n
  header["NAXIS2"] = n
  for i, (ctype, crval, cdelt) in enumerate(
    [("RA---SIN", 105.0, -1e-4), ("DEC--SIN", -40.0, 1e-4)], 1
  ):
    header[f"CTYPE{i}"] = ctype
    header[f"CRVAL{i}"] = crval
    header[f"CDELT{i}"] = cdelt
    header[f"CRPIX{i}"] = n / 2
    header[f"CUNIT{i}"] = "deg"
  header.tofile(path)
  with open(path, "r+b") as f:
    f.truncate(len(header.tostring()) + ((4 * n * n + 2879) // 2880) * 2880)
  return str(path)


def test_opening_allocates_no_sky_coordinates(tmp_path):
  path = large_sparse_image(tmp_path / "large.fits")

  tracemalloc.start()
  ds = xr.open_dataset(path, engine=ENGINE)
  _, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  assert ds.right_ascension.shape == (16384, 16384)
  assert peak < 64 * 2**20
  assert ds.declination[8191, 8191].values == pytest.approx(np.deg2rad(-40.0))


def test_sky_coordinate_blocks_equal_the_full_grid(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits", cards={"CROTA2": 10.0})
  ds = xr.open_dataset(path, engine=ENGINE)
  ra = ds.right_ascension.values
  dec = ds.declination.values

  assert ra.shape == dec.shape == (6, 5)
  assert_array_equal(ds.right_ascension[2:4, 1:3], ra[2:4, 1:3])
  assert_array_equal(ds.declination[[5, 0], 3], dec[[5, 0], 3])
  assert ds.right_ascension[3, 2].values == pytest.approx(np.deg2rad(105.0))

  chunked = xr.open_dataset(path, engine=ENGINE, chunks={"l": 4, "m": 2})
  assert_array_equal(chunked.right_ascension.compute(), ra)


def test_sky_coordinates_can_be_dropped(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits")
  ds = xr.open_dataset(
    path, engine=ENGINE, drop_variables=["right_ascension", "declination"]
  )

  assert "right_ascension" not in ds.coords
  assert "declination" not in ds.coords
