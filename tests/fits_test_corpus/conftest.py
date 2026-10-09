from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from contextlib import ExitStack
from hashlib import sha256

import platformdirs
import pytest

BASE_URL = "https://downloadnrao.org"
METADATA_URL = f"{BASE_URL}/file.download.json"
HEADERS = {"user-agent": "Wget/1.16 (linux-gnu)"}
ONE_MB = 1024**2


def download_item(url: str, checksum: str, output_file: str, checksum_file: str):
  """Downloads ``url`` to ``output_file``, verifying its sha256 checksum
  and storing it in ``checksum_file``"""
  with ExitStack() as stack:
    request = urllib.request.Request(url, headers=HEADERS)
    response = stack.enter_context(urllib.request.urlopen(request))
    archive = stack.enter_context(open(output_file, "wb"))
    digest = sha256()

    while data := response.read(ONE_MB):
      archive.write(data)
      digest.update(data)

  if (actual := digest.hexdigest()) != checksum:
    os.remove(output_file)
    raise ValueError(f"Checksum {actual} of {url} does not match {checksum}")

  with open(checksum_file, "w") as f:
    f.write(actual)


@pytest.fixture(scope="session")
def fits_corpus_metadata():
  request = urllib.request.Request(METADATA_URL, headers=HEADERS)
  with urllib.request.urlopen(request) as response:
    return json.loads(response.read())["metadata"]


@pytest.fixture(scope="session")
def fits_corpus_images(request, fits_corpus_metadata, tmp_path_factory):
  """Downloads a corpus item into the user cache, and returns the
  FITS Images it holds"""
  metadata = fits_corpus_metadata[request.param]
  cache_dir = os.path.join(
    platformdirs.user_cache_dir("xarray-fits", ensure_exists=True),
    "fits-test-data",
    metadata["path"],
  )
  os.makedirs(cache_dir, exist_ok=True)
  archive_file = os.path.join(cache_dir, metadata["file"])
  checksum_file = f"{archive_file}.sha256sum"
  cached_checksum = None

  if os.path.isfile(checksum_file) and os.path.isfile(archive_file):
    with open(checksum_file) as f:
      cached_checksum = f.read().strip()

  if cached_checksum != metadata["hash"]:
    url = f"{BASE_URL}/{metadata['path']}/{metadata['file']}"
    download_item(url, metadata["hash"], archive_file, checksum_file)

  directory = tmp_path_factory.mktemp(request.param)

  with zipfile.ZipFile(archive_file) as archive:
    archive.extractall(directory)

  # Skip the resource forks of archives made on macOS
  return sorted(str(p) for p in directory.rglob("*.fits") if "__MACOSX" not in p.parts)
