from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

import numpy as np
from xarray import Variable
from xarray.core.indexing import LazilyIndexedArray

from xarrayfits.backend.fits.array import FitsImageArray
from xarrayfits.backend.fits.coordinate_system import sky_coordinates
from xarrayfits.msv4_image_types import BEAM_PARAMS_LABELS, L_M_NOTES, SKY_DIMS

if TYPE_CHECKING:
  from xarrayfits.backend.fits.structure import (
    FitsFileFactory,
    FitsImageStructureFactory,
  )


def coordinate(
  name: str, dims: Tuple[str, ...], values, attrs: Dict[str, Any] | None = None
) -> Variable:
  """Returns a coordinate variable. Coordinates that are not dimensions
  name themselves in a ``coordinates`` attribute so that xarray
  decodes them as coordinates"""
  attrs = dict(attrs or {})

  if dims != (name,):
    attrs["coordinates"] = name

  return Variable(dims, values, attrs)


class ImageFactory:
  """Creates the variables of an Image held by a FITS Image"""

  __slots__ = ("_role", "_file_factory", "_structure_factory", "_preferred_chunks")

  _role: str
  _file_factory: FitsFileFactory
  _structure_factory: FitsImageStructureFactory
  _preferred_chunks: Dict[str, int]

  def __init__(
    self,
    role: str,
    file_factory: FitsFileFactory,
    structure_factory: FitsImageStructureFactory,
    preferred_chunks: Dict[str, int],
  ):
    self._role = role
    self._file_factory = file_factory
    self._structure_factory = structure_factory
    self._preferred_chunks = preferred_chunks

  def get_variables(self) -> Mapping[str, Variable]:
    """Returns the Image's data variables and coordinates"""
    structure = self._structure_factory.instance
    layout = structure.layout
    observation = structure.observation
    spectral = structure.spectral
    lon = structure.direction_values(layout.lon)
    lat = structure.direction_values(layout.lat)
    polarizations = np.asarray(structure.polarizations)
    ra, dec = sky_coordinates(layout, structure.coordinate_system)

    numpy_axes = (
      None,
      layout.numpy_axis(layout.frequency),
      layout.numpy_axis(layout.polarization),
      layout.numpy_axis(layout.lon),
      layout.numpy_axis(layout.lat),
    )
    orders = (None, None, np.asarray(structure.polarization_order), None, None)
    shape = (1, spectral.frequency.size, polarizations.size, lon.size, lat.size)
    array = FitsImageArray(
      self._file_factory, numpy_axes, orders, shape, structure.dtype
    )
    attrs = {"type": self._role.lower(), **observation.image_attrs}
    encoding = {
      "preferred_chunks": {
        d: c for d, c in self._preferred_chunks.items() if d in SKY_DIMS
      }
    }

    return {
      self._role: Variable(SKY_DIMS, LazilyIndexedArray(array), attrs, encoding),
      "time": coordinate(
        "time", ("time",), [observation.mjd], observation.time_attrs()
      ),
      "frequency": coordinate(
        "frequency", ("frequency",), spectral.frequency, spectral.frequency_attrs()
      ),
      "polarization": coordinate("polarization", ("polarization",), polarizations),
      "l": coordinate("l", ("l",), lon, {"note": L_M_NOTES["l"]}),
      "m": coordinate("m", ("m",), lat, {"note": L_M_NOTES["m"]}),
      "right_ascension": coordinate("right_ascension", ("l", "m"), ra),
      "declination": coordinate("declination", ("l", "m"), dec),
      "beam_params_label": coordinate(
        "beam_params_label", ("beam_params_label",), np.asarray(BEAM_PARAMS_LABELS)
      ),
    }
