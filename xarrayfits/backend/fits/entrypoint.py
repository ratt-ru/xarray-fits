from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Iterable

from rarg_python_patterns.multiton import Multiton
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractDataStore
from xarray.backends.common import _normalize_path as _xr_normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.utils import try_read_magic_number_from_file_or_path

from xarrayfits.backend.fits.factories import ImageFactory
from xarrayfits.backend.fits.file import FitsFile
from xarrayfits.backend.fits.structure import FitsImageStructure
from xarrayfits.msv4_image_types import IMAGE_DATASET_TYPE, IMAGE_SCHEMA_VERSION

#: Magic number at the start of every FITS file
FITS_MAGIC = b"SIMPLE  ="

#: Default preferred chunks: one chunk per image plane
DEFAULT_PREFERRED_CHUNKS = {"frequency": 1, "polarization": 1}

if TYPE_CHECKING:
  from io import BufferedIOBase

  from xarray import Dataset

  from xarrayfits.backend.fits.structure import (
    FitsFileFactory,
    FitsImageStructureFactory,
  )


class FitsStore(AbstractDataStore):
  """Store reading Image Datasets from FITS Images"""

  __slots__ = (
    "_file_factory",
    "_structure_factory",
    "_preferred_chunks",
    "_drop_variables",
  )

  _file_factory: FitsFileFactory
  _structure_factory: FitsImageStructureFactory
  _preferred_chunks: Dict[str, int]
  _drop_variables: FrozenSet[str]

  def __init__(
    self,
    file_factory: FitsFileFactory,
    structure_factory: FitsImageStructureFactory,
    preferred_chunks: Dict[str, int],
    drop_variables: FrozenSet[str],
  ):
    self._file_factory = file_factory
    self._structure_factory = structure_factory
    self._preferred_chunks = preferred_chunks
    self._drop_variables = drop_variables

  @classmethod
  def open(
    cls,
    path: str,
    drop_variables: str | Iterable[str] | None = None,
    preferred_chunks: Dict[str, int] | None = None,
  ) -> FitsStore:
    file_factory = Multiton(FitsFile, path, os.stat(path).st_mtime_ns)
    structure_factory = Multiton(FitsImageStructure, file_factory)
    preferred_chunks = {**DEFAULT_PREFERRED_CHUNKS, **(preferred_chunks or {})}

    if drop_variables is None:
      drop_variables = ()
    elif isinstance(drop_variables, str):
      drop_variables = (drop_variables,)

    return cls(
      file_factory, structure_factory, preferred_chunks, frozenset(drop_variables)
    )

  def image_factory(self) -> ImageFactory:
    return ImageFactory(
      "SKY",
      self._file_factory,
      self._structure_factory,
      self._preferred_chunks,
      self._drop_variables,
    )

  def close(self, **kwargs) -> None:
    self._file_factory.release()
    self._structure_factory.release()

  def get_variables(self):
    """Overrides AbstractDataStore.get_variables"""
    return self.image_factory().get_variables()

  def get_attrs(self) -> Dict[str, Any]:
    """Overrides AbstractDataStore.get_attrs"""
    structure = self._structure_factory.instance
    group = {"sky": "SKY"}

    factory = self.image_factory()

    if (flag := factory.flag) is not None:
      group["flag"] = flag

    if (beam_fit_params := factory.beam_fit_params) is not None:
      group["beam_fit_params_sky"] = beam_fit_params

    return {
      "coordinate_system_info": structure.coordinate_system.to_attrs(),
      "data_groups": {"base": group},
      "schema_version": IMAGE_SCHEMA_VERSION,
      "type": IMAGE_DATASET_TYPE,
    }

  def get_dimensions(self):
    """Overrides AbstractDataStore.get_dimensions"""
    return None

  def get_encoding(self):
    """Overrides AbstractDataStore.get_encoding"""
    return {}


class FitsEntryPoint(BackendEntrypoint):
  open_dataset_parameters = ["filename_or_obj", "drop_variables", "preferred_chunks"]
  description = "Opens FITS Images as MSv4 Image Datasets in Xarray"
  url = "https://xarray-fits.readthedocs.io/"

  def guess_can_open(
    self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore
  ) -> bool:
    if not isinstance(filename_or_obj, (str, os.PathLike)):
      return False

    magic = try_read_magic_number_from_file_or_path(filename_or_obj, count=9)
    return magic == FITS_MAGIC

  def open_dataset(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables: str | Iterable[str] | None = None,
    preferred_chunks: Dict[str, int] | None = None,
  ) -> Dataset:
    """Opens a FITS Image as an Image Dataset.

    Args:
      filename_or_obj: Path of the FITS Image.
      drop_variables: Variables to omit from the Image Dataset.
      preferred_chunks: Chunk sizes by dimension, which xarray uses
        when ``chunks={}`` is passed. Defaults to one chunk per
        frequency and polarization plane.

    Returns:
      An Image Dataset.
    """
    path = _xr_normalize_path(filename_or_obj)
    store = FitsStore.open(
      path, drop_variables=drop_variables, preferred_chunks=preferred_chunks
    )
    store_entrypoint = StoreBackendEntrypoint()
    return store_entrypoint.open_dataset(store, drop_variables=drop_variables)
