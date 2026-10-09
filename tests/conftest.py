import copy
import gc

import pytest
import xarray as xr
from rarg_python_patterns.multiton import Multiton

try:
  # The conformance oracle: a reference reader of FITS Images and the
  # Image Schema checker, only available to the tests
  from xradio.image import check_image
  from xradio.image import open_image as reference_open_image
except ImportError:
  reference_open_image = None


@pytest.fixture(autouse=True)
def clear_multiton_cache():
  yield
  Multiton._INSTANCE_CACHE.clear()
  Multiton._EXPIRY_HEAP.clear()
  gc.collect()


def without_flags(ds: xr.Dataset) -> xr.Dataset:
  """Removes flags and their wiring, which ADR 0001 lets differ
  from the reference"""
  ds = ds.drop_vars([n for n in ds.data_vars if str(n).startswith("FLAG_")])
  ds.attrs = copy.deepcopy(ds.attrs)

  for group in ds.attrs.get("data_groups", {}).values():
    group.pop("flag", None)

  for var in ds.data_vars.values():
    var.attrs.pop("flag", None)

  return ds


@pytest.fixture
def assert_conforms():
  """Asserts that an Image Dataset passes the Image Schema checker and
  equals the reference reader's Image Dataset for the same FITS Images,
  apart from flags"""
  if reference_open_image is None:
    pytest.skip("The conformance oracle is not installed")

  def check(ds: xr.Dataset, store):
    assert list(check_image(ds)) == []
    expected = reference_open_image(store)
    xr.testing.assert_identical(
      without_flags(ds).load(), without_flags(expected).load()
    )

  return check
