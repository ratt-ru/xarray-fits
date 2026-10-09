from __future__ import annotations

import weakref
from threading import Lock
from typing import TYPE_CHECKING, Tuple

import numpy as np
from astropy.io import fits

if TYPE_CHECKING:
  import numpy.typing as npt


class FitsFile:
  """An open FITS file whose primary HDU pixels can be read
  from several threads"""

  __slots__ = ("_path", "_hdu_list", "_lock", "__weakref__")

  _path: str
  _hdu_list: fits.HDUList
  _lock: Lock

  def __init__(self, path: str):
    self._path = path
    self._hdu_list = fits.open(path, memmap=True)
    self._lock = Lock()
    weakref.finalize(self, self._hdu_list.close)

  @property
  def path(self) -> str:
    return self._path

  @property
  def hdu_list(self) -> fits.HDUList:
    return self._hdu_list

  def read(self, region: Tuple[slice, ...]) -> npt.NDArray:
    """Reads a region of the primary HDU pixels, given as slices
    in numpy (reversed FITS) axis order, in native byte order"""
    with self._lock:
      data = self._hdu_list[0].data[region]
      return np.ascontiguousarray(data, dtype=data.dtype.newbyteorder("="))
