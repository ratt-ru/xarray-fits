import pickle

import pytest
import xarray as xr

from xarrayfits.testing.simulator import simulate_fits_image

ENGINE = "xarray-fits:fits"

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingMetadataWarning"
)


@pytest.fixture
def image(tmp_path):
  return simulate_fits_image(tmp_path / "image.fits")


def test_engine_recognises_fits_images(image, tmp_path):
  entrypoint = xr.backends.list_engines()[ENGINE]
  not_fits = tmp_path / "not.fits"
  not_fits.write_bytes(b"\x00" * 2880)

  assert entrypoint.guess_can_open(image)
  assert not entrypoint.guess_can_open(str(not_fits))
  assert not entrypoint.guess_can_open(str(tmp_path / "missing.fits"))


def test_preferred_chunks_default_to_image_planes(image):
  ds = xr.open_dataset(image, engine=ENGINE, chunks={})
  assert ds.SKY.chunks == ((1,), (1, 1, 1), (1,), (6,), (5,))


def test_preferred_chunks_can_be_overridden(image):
  ds = xr.open_dataset(
    image, engine=ENGINE, chunks={}, preferred_chunks={"frequency": 2, "l": 4}
  )
  assert ds.SKY.chunks == ((1,), (2, 1), (1,), (4, 2), (5,))


@pytest.mark.parametrize("chunks", [None, {}])
def test_image_datasets_pickle(image, chunks):
  ds = xr.open_dataset(image, engine=ENGINE, chunks=chunks)
  xr.testing.assert_identical(pickle.loads(pickle.dumps(ds)).load(), ds.load())


def test_dask_and_distributed_reads_equal_eager_reads(image):
  distributed = pytest.importorskip("dask.distributed")
  eager = xr.open_dataset(image, engine=ENGINE).load()
  chunked = xr.open_dataset(image, engine=ENGINE, chunks={})

  xr.testing.assert_identical(chunked.compute(), eager)

  with distributed.LocalCluster(
    n_workers=2, processes=True, threads_per_worker=1, dashboard_address=":0"
  ) as cluster:
    with distributed.Client(cluster):
      xr.testing.assert_identical(chunked.compute(), eager)
