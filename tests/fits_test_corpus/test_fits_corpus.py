import pytest
import xarray as xr

ENGINE = "xarray-fits:fits"

TCLEAN_PRODUCTS = "3c286_Band6_5chans_lsrk_robust_0.5_niter_99.fits"
CORPUS = ["test_image.fits", TCLEAN_PRODUCTS]

pytestmark = [
  pytest.mark.fits_test_corpus,
  pytest.mark.filterwarnings("ignore::xarrayfits.errors.UnknownRoleWarning"),
]


@pytest.mark.parametrize("fits_corpus_images", CORPUS, indirect=True)
def test_corpus_images_conform(fits_corpus_images, assert_conforms):
  assert len(fits_corpus_images) > 0

  for path in fits_corpus_images:
    ds = xr.open_dataset(path, engine=ENGINE, chunks={})
    assert_conforms(ds, path)


@pytest.mark.parametrize("fits_corpus_images", [TCLEAN_PRODUCTS], indirect=True)
def test_tclean_products_conform_as_one_image_dataset(
  fits_corpus_images, assert_conforms
):
  # The un-suffixed image duplicates the .image.fits sky image
  paths = [p for p in fits_corpus_images if not p.endswith("_99.fits")]
  ds = xr.open_dataset(paths, engine=ENGINE, chunks={})

  assert {"SKY", "SKY_RESIDUAL", "SKY_MODEL", "VISIBILITY_NORMALIZATION"} <= set(
    ds.data_vars
  )
  assert_conforms(ds, paths)
