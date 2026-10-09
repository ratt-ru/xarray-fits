from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Tuple

import numpy as np
from xarray import Variable
from xarray.core.indexing import LazilyIndexedArray

from xarrayfits.backend.fits.array import (
  FitsImageArray,
  FlagArray,
  SkyCoordinateArray,
)
from xarrayfits.backend.fits.coordinate_system import sky_wcs_cards
from xarrayfits.backend.fits.roles import VISIBILITY_NORMALIZATION, is_sky
from xarrayfits.errors import InvalidFitsImage
from xarrayfits.msv4_image_types import L_M_NOTES, SKY_DIMS

if TYPE_CHECKING:
  from xarrayfits.backend.fits.structure import (
    FitsFileFactory,
    FitsImageStructure,
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


def image_type(role: str) -> str:
  """Returns the ``type`` attribute of an Image with the given Role"""
  return "sky" if is_sky(role) else role.lower()


class ImageFactory:
  """Creates the variables of an Image held by a FITS Image"""

  __slots__ = (
    "_role",
    "_file_factory",
    "_structure_factory",
    "_preferred_chunks",
    "_drop_variables",
  )

  _role: str
  _file_factory: FitsFileFactory
  _structure_factory: FitsImageStructureFactory
  _preferred_chunks: Dict[str, int]
  _drop_variables: FrozenSet[str]

  def __init__(
    self,
    role: str,
    file_factory: FitsFileFactory,
    structure_factory: FitsImageStructureFactory,
    preferred_chunks: Dict[str, int],
    drop_variables: FrozenSet[str] = frozenset(),
  ):
    self._role = role
    self._file_factory = file_factory
    self._structure_factory = structure_factory
    self._preferred_chunks = preferred_chunks
    self._drop_variables = drop_variables

  @property
  def structure(self) -> FitsImageStructure:
    """The parsed structure of the FITS Image"""
    return self._structure_factory.instance

  @property
  def flag(self) -> str | None:
    """Name of the Image's flag variable. Floating point Images are
    flagged where they are NaN (see ADR 0001)"""
    name = f"FLAG_{self._role}"
    structure = self._structure_factory.instance
    if structure.dtype.kind != "f" or name in self._drop_variables:
      return None
    return name

  @property
  def beam_fit_params(self) -> str | None:
    """Name of the Image's beam variable, if it has beams"""
    name = f"BEAM_FIT_PARAMS_{self._role}"
    structure = self._structure_factory.instance
    if structure.beams is None or name in self._drop_variables:
      return None
    return name

  @property
  def dims(self) -> Tuple[str, ...]:
    """Dimensions of the Image"""
    return SKY_DIMS[:3] if self._role == VISIBILITY_NORMALIZATION else SKY_DIMS

  def get_variables(self) -> Dict[str, Variable]:
    """Returns the Image's data variables and coordinates"""
    structure = self._structure_factory.instance
    layout = structure.layout
    observation = structure.observation
    spectral = structure.spectral
    dims = self.dims
    lon = structure.direction_values(layout.lon)
    lat = structure.direction_values(layout.lat)
    polarizations = np.asarray(structure.polarizations)

    if "l" not in dims and (lon.size, lat.size) != (1, 1):
      raise InvalidFitsImage(
        f"The {self._role} image {self._file_factory.instance.url} must have "
        f"direction axes of one pixel, found {(lon.size, lat.size)} pixels"
      )

    # Sky-plane dimensions, of which an Image without directions
    # has the first three
    ndim = len(dims)
    numpy_axes: Tuple[int | None, ...] = (
      None,
      layout.numpy_axis(layout.frequency),
      layout.numpy_axis(layout.polarization),
      layout.numpy_axis(layout.lon),
      layout.numpy_axis(layout.lat),
    )
    shape = (1, spectral.frequency.size, polarizations.size, lon.size, lat.size)
    orders = (None, None, np.asarray(structure.polarization_order), None, None)
    array = FitsImageArray(
      self._file_factory,
      len(layout.axes),
      numpy_axes[:ndim],
      orders[:ndim],
      shape[:ndim],
      structure.dtype,
    )
    attrs = {**observation.image_attrs, "type": image_type(self._role)}

    if observation.sub_type is not None:
      attrs["sub_type"] = observation.sub_type

    if (flag := self.flag) is not None:
      attrs["flag"] = flag

    if (beam_fit_params := self.beam_fit_params) is not None:
      attrs["beam_fit_params"] = beam_fit_params

    encoding = {
      "preferred_chunks": {d: c for d, c in self._preferred_chunks.items() if d in dims}
    }

    variables = {
      self._role: Variable(dims, LazilyIndexedArray(array), attrs, encoding),
      "time": coordinate(
        "time", ("time",), [observation.mjd], observation.time_attrs()
      ),
      "frequency": coordinate(
        "frequency", ("frequency",), spectral.frequency, spectral.frequency_attrs()
      ),
      "polarization": coordinate("polarization", ("polarization",), polarizations),
    }

    if spectral.velocity is not None:
      variables["velocity"] = coordinate(
        "velocity", ("frequency",), spectral.velocity, spectral.velocity_attrs()
      )

    if "l" in dims:
      wcs_cards = sky_wcs_cards(layout, structure.coordinate_system)
      ra = LazilyIndexedArray(SkyCoordinateArray(wcs_cards, 0))
      dec = LazilyIndexedArray(SkyCoordinateArray(wcs_cards, 1))
      variables["l"] = coordinate("l", ("l",), lon, {"note": L_M_NOTES["l"]})
      variables["m"] = coordinate("m", ("m",), lat, {"note": L_M_NOTES["m"]})
      variables["right_ascension"] = coordinate("right_ascension", ("l", "m"), ra)
      variables["declination"] = coordinate("declination", ("l", "m"), dec)

    if flag is not None:
      flags = LazilyIndexedArray(FlagArray(array))
      variables[flag] = Variable(dims, flags, {"type": "flag"}, encoding)

    if beam_fit_params is not None:
      variables[beam_fit_params] = Variable(
        ("time", "frequency", "polarization", "beam_params_label"),
        structure.beams,
        {"units": "rad", "type": f"beam_fit_params_{self._role.lower()}"},
      )

    return variables
