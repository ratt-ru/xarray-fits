from __future__ import annotations

from functools import reduce
from operator import mul
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
from xarray.backends import BackendArray
from xarray.core.indexing import IndexingSupport, explicit_indexing_adapter

from xarrayfits.backend.fits.coordinate_system import DEG_TO_RAD

if TYPE_CHECKING:
  import numpy.typing as npt

  from xarrayfits.backend.fits.structure import FitsFileFactory


def outer_index(key, shape: Tuple[int, ...]) -> Tuple[list, Tuple[int, ...]]:
  """Converts an outer indexing key into a 1D index array per dimension,
  and the dimensions selected by integers, which are squeezed out"""
  squeeze = tuple(i for i, k in enumerate(key) if isinstance(k, (int, np.integer)))
  index = [np.atleast_1d(np.arange(n)[k]) for k, n in zip(key, shape)]
  return index, squeeze


class FitsImageArray(BackendArray):
  """Lazily reads the primary HDU pixels of a FITS Image as an array
  of sky-plane dimensions"""

  __slots__ = ("shape", "dtype", "_file_factory", "_numpy_axes", "_orders")

  shape: Tuple[int, ...]
  dtype: np.dtype
  _file_factory: FitsFileFactory
  _numpy_axes: Tuple[int | None, ...]
  _orders: Tuple[npt.NDArray[np.intp] | None, ...]

  def __init__(
    self,
    file_factory: FitsFileFactory,
    numpy_axes: Tuple[int | None, ...],
    orders: Tuple[npt.NDArray[np.intp] | None, ...],
    shape: Tuple[int, ...],
    dtype: npt.DTypeLike,
  ):
    """Creates the array.

    Args:
      file_factory: The FITS file.
      numpy_axes: For each dimension, the numpy axis of the pixels
        holding it, or ``None`` for a dimension of length one
        that the FITS Image does not have.
      orders: For each dimension, the pixel index of each element,
        or ``None`` if they are in the same order.
      shape: Shape of the array.
      dtype: Data type of the array.
    """
    self._file_factory = file_factory
    self._numpy_axes = numpy_axes
    self._orders = orders
    self.shape = shape
    self.dtype = np.dtype(dtype)

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    index, squeeze = outer_index(key, self.shape)
    index = [i if order is None else order[i] for i, order in zip(index, self._orders)]
    expected_shape = tuple(len(i) for i in index)

    if reduce(mul, expected_shape, 1) == 0:
      return np.empty(expected_shape, dtype=self.dtype).squeeze(axis=squeeze)

    present = [a for a in self._numpy_axes if a is not None]
    region = [slice(None)] * len(present)
    within = [np.arange(0)] * len(present)

    for i, axis in zip(index, self._numpy_axes):
      if axis is not None:
        start = int(i.min())
        region[axis] = slice(start, int(i.max()) + 1)
        within[axis] = i - start

    data = self._file_factory.instance.read(tuple(region))
    data = data[np.ix_(*within)].transpose(present)

    for dim, axis in enumerate(self._numpy_axes):
      if axis is None:
        data = np.expand_dims(data, dim)

    return data.squeeze(axis=squeeze)


class SkyCoordinateArray(BackendArray):
  """Lazily computes a sky coordinate of every (l, m) pixel in radians"""

  __slots__ = ("shape", "dtype", "_wcs_cards", "_component")

  shape: Tuple[int, ...]
  dtype: np.dtype
  _wcs_cards: Dict[str, Any]
  _component: int

  def __init__(self, wcs_cards: Dict[str, Any], component: int):
    """Creates the array.

    Args:
      wcs_cards: Cards of a two axis celestial WCS, longitude first.
      component: 0 for the longitude, 1 for the latitude.
    """
    self._wcs_cards = wcs_cards
    self._component = component
    self.shape = (wcs_cards["NAXIS1"], wcs_cards["NAXIS2"])
    self.dtype = np.dtype(np.float64)

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    from astropy.wcs import WCS

    (x, y), squeeze = outer_index(key, self.shape)

    if x.size == 0 or y.size == 0:
      return np.empty((x.size, y.size), self.dtype).squeeze(axis=squeeze)

    wcs = WCS(self._wcs_cards)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    world = wcs.pixel_to_world_values(xx, yy)[self._component]
    return (world * DEG_TO_RAD).squeeze(axis=squeeze)
