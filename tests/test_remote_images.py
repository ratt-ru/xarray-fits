import os
import pickle
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import fsspec
import numpy as np
import pytest
import xarray as xr

from xarrayfits.testing.simulator import (
  DEC,
  FREQ,
  RA,
  simulate_fits_image,
  stokes_axis,
)

ENGINE = "xarray-fits:fits"
CIRCULAR = (RA, DEC, FREQ, stokes_axis(-1.0, -1.0, 4))

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingMetadataWarning"
)


class RangeRequestHandler(SimpleHTTPRequestHandler):
  """Serves files, honouring single byte range requests"""

  def send_head(self):
    match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
    path = self.translate_path(self.path)

    if match is None or not os.path.isfile(path):
      return super().send_head()

    size = os.path.getsize(path)
    start = int(match.group(1))
    end = min(int(match.group(2) or size - 1), size - 1)
    f = open(path, "rb")
    f.seek(start)
    self.send_response(206)
    self.send_header("Content-Type", "application/octet-stream")
    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    self.send_header("Content-Length", str(end - start + 1))
    self.end_headers()
    self._remaining = end - start + 1
    return f

  def copyfile(self, source, outputfile):
    if (remaining := getattr(self, "_remaining", None)) is None:
      return super().copyfile(source, outputfile)
    outputfile.write(source.read(remaining))

  def log_message(self, *args):
    pass


@pytest.fixture
def http_directory(tmp_path):
  def handler(*args, **kwargs):
    return RangeRequestHandler(*args, directory=str(tmp_path), **kwargs)

  server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  yield f"http://127.0.0.1:{server.server_address[1]}"
  server.shutdown()
  server.server_close()


def image_data():
  data = np.arange(4 * 3 * 5 * 6, dtype=np.float32).reshape(4, 3, 5, 6)
  data[1, :, 0, 0] = np.nan
  return data


@pytest.fixture
def local_image(tmp_path):
  return simulate_fits_image(
    tmp_path / "image.fits",
    axes=CIRCULAR,
    data=image_data(),
    beams=np.ones((3, 4, 3)),
  )


def test_memory_filesystem_images_equal_local_images(local_image):
  fs = fsspec.filesystem("memory")
  with open(local_image, "rb") as f:
    fs.pipe("/remote/image.fits", f.read())

  local = xr.open_dataset(local_image, engine=ENGINE).load()
  remote = xr.open_dataset("memory://remote/image.fits", engine=ENGINE)

  xr.testing.assert_identical(remote.load(), local)
  xr.testing.assert_identical(
    remote.SKY.isel(frequency=[2, 0], l=slice(1, 4)).load(),
    local.SKY.isel(frequency=[2, 0], l=slice(1, 4)),
  )


def test_http_images_read_on_distributed_workers(local_image, http_directory):
  distributed = pytest.importorskip("dask.distributed")
  url = f"{http_directory}/image.fits"
  local = xr.open_dataset(local_image, engine=ENGINE).load()
  remote = xr.open_dataset(url, engine=ENGINE, chunks={})

  xr.testing.assert_identical(pickle.loads(pickle.dumps(remote)).compute(), local)

  with distributed.LocalCluster(
    n_workers=2, processes=True, threads_per_worker=2, dashboard_address=":0"
  ) as cluster:
    with distributed.Client(cluster):
      xr.testing.assert_identical(remote.compute(), local)
