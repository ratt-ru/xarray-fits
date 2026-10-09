from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Iterable

from rarg_python_patterns.multiton import Multiton
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractDataStore
from xarray.backends.common import _normalize_path as _xr_normalize_path
from xarray.backends.store import StoreBackendEntrypoint

from xarrayfits.backend.fits.factories import ImageFactory
from xarrayfits.backend.fits.file import FitsFile
from xarrayfits.backend.fits.structure import FitsImageStructure
from xarrayfits.msv4_image_types import IMAGE_DATASET_TYPE, IMAGE_SCHEMA_VERSION

if TYPE_CHECKING:
  from io import BufferedIOBase

  from xarray import Dataset

  from xarrayfits.backend.fits.structure import (
    FitsFileFactory,
    FitsImageStructureFactory,
  )


class FitsStore(AbstractDataStore):
  """Store reading Image Datasets from FITS Images"""

  __slots__ = ("_file_factory", "_structure_factory")

  _file_factory: FitsFileFactory
  _structure_factory: FitsImageStructureFactory

  def __init__(
    self,
    file_factory: FitsFileFactory,
    structure_factory: FitsImageStructureFactory,
  ):
    self._file_factory = file_factory
    self._structure_factory = structure_factory

  @classmethod
  def open(cls, path: str) -> FitsStore:
    file_factory = Multiton(FitsFile, path)
    structure_factory = Multiton(FitsImageStructure, file_factory)
    return cls(file_factory, structure_factory)

  def close(self, **kwargs) -> None:
    self._file_factory.release()
    self._structure_factory.release()

  def get_variables(self):
    """Overrides AbstractDataStore.get_variables"""
    factory = ImageFactory("SKY", self._file_factory, self._structure_factory)
    return factory.get_variables()

  def get_attrs(self) -> Dict[str, Any]:
    """Overrides AbstractDataStore.get_attrs"""
    structure = self._structure_factory.instance
    return {
      "coordinate_system_info": structure.coordinate_system.to_attrs(),
      "data_groups": {"base": {"sky": "SKY"}},
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
  open_dataset_parameters = ["filename_or_obj", "drop_variables"]
  description = "Opens FITS Images as MSv4 Image Datasets in Xarray"
  url = "https://xarray-fits.readthedocs.io/"

  def open_dataset(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables: str | Iterable[str] | None = None,
  ) -> Dataset:
    """Opens a FITS Image as an Image Dataset.

    Args:
      filename_or_obj: Path of the FITS Image.
      drop_variables: Variables to omit from the Image Dataset.

    Returns:
      An Image Dataset.
    """
    path = _xr_normalize_path(filename_or_obj)
    store = FitsStore.open(path)
    store_entrypoint = StoreBackendEntrypoint()
    return store_entrypoint.open_dataset(store, drop_variables=drop_variables)
