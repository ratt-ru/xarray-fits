from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Iterable, List, Mapping, Tuple

from rarg_python_patterns.multiton import Multiton
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractDataStore
from xarray.backends.common import _normalize_path as _xr_normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.utils import try_read_magic_number_from_file_or_path

from xarrayfits.backend.fits.factories import ImageDatasetFactory, ImageFactory
from xarrayfits.backend.fits.file import FitsFile, file_version
from xarrayfits.backend.fits.roles import resolve_roles
from xarrayfits.backend.fits.structure import FitsImageStructure

#: Magic number at the start of every FITS file
FITS_MAGIC = b"SIMPLE  ="

#: Default preferred chunks: one chunk per image plane
DEFAULT_PREFERRED_CHUNKS = {"frequency": 1, "polarization": 1}

if TYPE_CHECKING:
  from io import BufferedIOBase

  from xarray import Dataset, Variable

  from xarrayfits.backend.fits.structure import (
    FitsFileFactory,
    FitsImageStructureFactory,
  )


class FitsStore(AbstractDataStore):
  """Store reading an Image Dataset from FITS Images"""

  __slots__ = (
    "_urls",
    "_file_factories",
    "_structure_factories",
    "_preferred_chunks",
    "_drop_variables",
    "_assembled",
  )

  _urls: Dict[str, str]
  _file_factories: Dict[str, FitsFileFactory]
  _structure_factories: Dict[str, FitsImageStructureFactory]
  _preferred_chunks: Dict[str, int]
  _drop_variables: FrozenSet[str]
  _assembled: Tuple[Dict[str, Variable], Dict[str, Any]] | None

  def __init__(
    self,
    urls: Dict[str, str],
    file_factories: Dict[str, FitsFileFactory],
    structure_factories: Dict[str, FitsImageStructureFactory],
    preferred_chunks: Dict[str, int],
    drop_variables: FrozenSet[str],
  ):
    self._urls = urls
    self._file_factories = file_factories
    self._structure_factories = structure_factories
    self._preferred_chunks = preferred_chunks
    self._drop_variables = drop_variables
    self._assembled = None

  @classmethod
  def open(
    cls,
    urls: Dict[str, str],
    drop_variables: str | Iterable[str] | None = None,
    preferred_chunks: Dict[str, int] | None = None,
  ) -> FitsStore:
    """Opens the FITS Image of each Role"""
    file_factories: Dict[str, FitsFileFactory] = {
      role: Multiton(FitsFile, url, file_version(url)) for role, url in urls.items()
    }
    structure_factories: Dict[str, FitsImageStructureFactory] = {
      role: Multiton(FitsImageStructure, f) for role, f in file_factories.items()
    }
    preferred_chunks = {**DEFAULT_PREFERRED_CHUNKS, **(preferred_chunks or {})}

    if drop_variables is None:
      drop_variables = ()
    elif isinstance(drop_variables, str):
      drop_variables = (drop_variables,)

    return cls(
      urls,
      file_factories,
      structure_factories,
      preferred_chunks,
      frozenset(drop_variables),
    )

  def dataset_factory(self) -> ImageDatasetFactory:
    images = {
      role: ImageFactory(
        role,
        self._file_factories[role],
        self._structure_factories[role],
        self._preferred_chunks,
        self._drop_variables,
      )
      for role in self._urls
    }
    return ImageDatasetFactory(images, self._urls)

  def close(self, **kwargs) -> None:
    for factory in self._file_factories.values():
      factory.release()
    for factory in self._structure_factories.values():
      factory.release()

  def assemble(self) -> Tuple[Dict[str, Variable], Dict[str, Any]]:
    """Returns the variables and attributes of the Image Dataset,
    assembling them once"""
    if self._assembled is None:
      self._assembled = self.dataset_factory().assemble()
    return self._assembled

  def get_variables(self):
    """Overrides AbstractDataStore.get_variables"""
    variables, _ = self.assemble()
    return variables

  def get_attrs(self) -> Dict[str, Any]:
    """Overrides AbstractDataStore.get_attrs"""
    _, attrs = self.assemble()
    return attrs

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
  supports_groups = False

  def guess_can_open(
    self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore
  ) -> bool:
    if not isinstance(filename_or_obj, (str, os.PathLike)):
      return False

    magic = try_read_magic_number_from_file_or_path(filename_or_obj, count=9)
    return magic == FITS_MAGIC

  def open_dataset(  # type: ignore[override]
    self,
    filename_or_obj: str
    | os.PathLike[Any]
    | List[str | os.PathLike[Any]]
    | Mapping[str, str | os.PathLike[Any]],
    *,
    drop_variables: str | Iterable[str] | None = None,
    preferred_chunks: Dict[str, int] | None = None,
  ) -> Dataset:
    """Opens FITS Images as an Image Dataset.

    Args:
      filename_or_obj: Path or fsspec URL of a FITS Image, a list of them,
        or a mapping of Roles to them. The Role of each FITS Image in a
        list is taken from its file name, for example ``cube.psf.fits``
        holds the ``POINT_SPREAD_FUNCTION``.
      drop_variables: Variables to omit from the Image Dataset.
      preferred_chunks: Chunk sizes by dimension, which xarray uses
        when ``chunks={}`` is passed. Defaults to one chunk per
        frequency and polarization plane.

    Returns:
      An Image Dataset.
    """
    urls = {
      role: _xr_normalize_path(url)
      for role, url in resolve_roles(filename_or_obj).items()
    }
    store = FitsStore.open(
      urls, drop_variables=drop_variables, preferred_chunks=preferred_chunks
    )
    store_entrypoint = StoreBackendEntrypoint()
    return store_entrypoint.open_dataset(store, drop_variables=drop_variables)
