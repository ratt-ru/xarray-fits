from __future__ import annotations

from functools import reduce
from operator import mul
from typing import TYPE_CHECKING, Tuple

import numpy as np
from xarray.backends import BackendArray
from xarray.core.indexing import IndexingSupport, explicit_indexing_adapter

if TYPE_CHECKING:
  import numpy.typing as npt

  from xarrayfits.backend.fits.structure import FitsFileFactory


class FitsImageArray(BackendArray):
  """Lazily reads the primary HDU pixels of a FITS Image as an array
  of sky-plane dimensions"""

  __slots__ = ("shape", "dtype", "_file_factory", "_numpy_axes")

  shape: Tuple[int, ...]
  dtype: np.dtype
  _file_factory: FitsFileFactory
  _numpy_axes: Tuple[int | None, ...]

  def __init__(
    self,
    file_factory: FitsFileFactory,
    numpy_axes: Tuple[int | None, ...],
    shape: Tuple[int, ...],
    dtype: npt.DTypeLike,
  ):
    """Creates the array.

    Args:
      file_factory: The FITS file.
      numpy_axes: For each dimension, the numpy axis of the pixels
        holding it, or ``None`` for a dimension of length one
        that the FITS Image does not have.
      shape: Shape of the array.
      dtype: Data type of the array.
    """
    self._file_factory = file_factory
    self._numpy_axes = numpy_axes
    self.shape = shape
    self.dtype = np.dtype(dtype)

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    squeeze = tuple(i for i, k in enumerate(key) if isinstance(k, (int, np.integer)))
    index = [np.atleast_1d(np.arange(n)[k]) for k, n in zip(key, self.shape)]
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
