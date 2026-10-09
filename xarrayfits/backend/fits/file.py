from __future__ import annotations

import weakref
from threading import Lock
from typing import TYPE_CHECKING, Tuple

import fsspec
import numpy as np
from astropy.io import fits
from fsspec.implementations.local import LocalFileSystem

if TYPE_CHECKING:
  import numpy.typing as npt


def file_version(url: str) -> str:
  """Returns a token that changes when the file at ``url`` changes"""
  fs, path = fsspec.core.url_to_fs(url)
  return fs.ukey(path)


class FitsFile:
  """An open FITS file whose primary HDU pixels can be read
  from several threads. Local files are memory mapped, while
  remote files are read in sections through fsspec"""

  __slots__ = ("_url", "_hdu_list", "_memmap", "_lock", "__weakref__")

  _url: str
  _hdu_list: fits.HDUList
  _memmap: bool
  _lock: Lock

  def __init__(self, url: str, version: str | None = None):
    """Opens the FITS file.

    Args:
      url: Path or fsspec URL of the FITS file.
      version: Version token of the FITS file, which distinguishes
        cached handles of a file that has since been rewritten.
    """
    fs, path = fsspec.core.url_to_fs(url)
    self._url = url
    self._memmap = isinstance(fs, LocalFileSystem)

    if self._memmap:
      self._hdu_list = fits.open(path, memmap=True)
    else:
      self._hdu_list = fits.open(url, use_fsspec=True)

    self._lock = Lock()
    weakref.finalize(self, self._hdu_list.close)

  @property
  def url(self) -> str:
    return self._url

  @property
  def hdu_list(self) -> fits.HDUList:
    return self._hdu_list

  def read(self, region: Tuple[slice, ...]) -> npt.NDArray:
    """Reads a region of the primary HDU pixels, given as slices
    in numpy (reversed FITS) axis order, in native byte order"""
    with self._lock:
      primary = self._hdu_list[0]
      data = primary.data[region] if self._memmap else primary.section[region]
      return np.ascontiguousarray(data, dtype=data.dtype.newbyteorder("="))
