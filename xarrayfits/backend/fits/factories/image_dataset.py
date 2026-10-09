from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
from astropy import units as u
from astropy.time import Time
from xarray import Variable

from xarrayfits.backend.fits.factories.image import coordinate
from xarrayfits.backend.fits.roles import (
  VISIBILITY_NORMALIZATION,
  data_groups,
  is_sky,
)
from xarrayfits.errors import IncompatibleImages
from xarrayfits.msv4_image_types import (
  BEAM_PARAMS_LABELS,
  IMAGE_DATASET_TYPE,
  IMAGE_SCHEMA_VERSION,
)

if TYPE_CHECKING:
  from xarrayfits.backend.fits.factories.image import ImageFactory

#: Coordinates of an Image, shared by the Images of an Image Dataset
IMAGE_COORDINATES = frozenset(
  {
    "time",
    "frequency",
    "polarization",
    "velocity",
    "l",
    "m",
    "right_ascension",
    "declination",
  }
)

#: Shared coordinate values that agree within this fraction of the axis
#: increment are taken as equal
COORD_MATCH_TOLERANCE = 1e-6

#: Shared times that agree within this many seconds are taken as equal
TIME_MATCH_TOLERANCE_S = 1e-3

#: Shared coordinates of a single value that agree within this fraction
#: of the value (or absolutely, for values below one) are taken as equal
SINGLE_VALUE_TOLERANCE = 1e-9


def time_offsets(reference: Variable, other: Variable) -> np.ndarray:
  """Returns the offsets of two time coordinates in seconds"""
  keys = ("units", "format", "scale")

  if all(reference.attrs.get(k) == other.attrs.get(k) for k in keys):
    seconds = float(u.Unit(reference.attrs["units"]).to(u.s))
    return (other.values - reference.values) * seconds

  def to_time(v: Variable) -> Time:
    values = np.asarray(v.values, dtype=float) * u.Unit(v.attrs["units"])
    return Time(values, format=v.attrs["format"], scale=v.attrs["scale"])

  return np.atleast_1d((to_time(other) - to_time(reference)).sec)


def snap(dim: str, reference: Variable, other: Variable, prefix: str) -> Variable:
  """Returns the ``other`` coordinate of an Image with the values of the
  ``reference`` coordinate of the Images opened before it.

  Raises:
    IncompatibleImages: If the coordinates differ by more than round-off.
  """
  if reference.size != other.size:
    raise IncompatibleImages(
      f"{prefix} has {other.size} values, theirs has {reference.size}"
    )

  if reference.dtype.kind not in "iuf" or other.dtype.kind not in "iuf":
    if sorted(reference.values.tolist()) != sorted(other.values.tolist()):
      raise IncompatibleImages(
        f"{prefix} has the labels {other.values.tolist()}, "
        f"theirs has {reference.values.tolist()}"
      )
    return other

  increment = None

  if dim == "time":
    offsets = time_offsets(reference, other)
    tolerance = TIME_MATCH_TOLERANCE_S
    units = " s"
  else:
    ref_values = np.asarray(reference.values, dtype=np.float64)
    offsets = np.asarray(other.values, dtype=np.float64) - ref_values
    units = f" {reference.attrs['units']}" if "units" in reference.attrs else ""

    if dim == "frequency" and ref_values.size < 2:
      width = reference.attrs["channel_width"]
      increment = abs(float(width["data"]))
    elif ref_values.size >= 2:
      increment = abs(ref_values[-1] - ref_values[0]) / (ref_values.size - 1)

    if increment:
      tolerance = COORD_MATCH_TOLERANCE * increment
    else:
      increment = None
      tolerance = SINGLE_VALUE_TOLERANCE * max(1.0, float(np.abs(ref_values).max()))

  largest = float(np.abs(offsets).max()) if offsets.size else 0.0

  if largest > tolerance:
    fraction = (
      f" ({largest / increment:.3g} of the {dim} increment)" if increment else ""
    )
    raise IncompatibleImages(
      f"{prefix} differs from theirs by up to {largest:g}{units}{fraction}, "
      f"more than the tolerance of {tolerance:g}{units}. "
      f"Images opened together must share their coordinates."
    )

  if largest > 0:
    return Variable(other.dims, reference.values, other.attrs)

  return other


class ImageDatasetFactory:
  """Creates an Image Dataset from the Images of several FITS Images"""

  __slots__ = ("_images", "_urls")

  _images: Dict[str, ImageFactory]
  _urls: Dict[str, str]

  def __init__(self, images: Dict[str, ImageFactory], urls: Dict[str, str]):
    """Creates the factory.

    Args:
      images: The Image of each Role, in the order they are opened.
      urls: The FITS Image holding each Role.
    """
    self._images = images
    self._urls = urls

  def assemble(self) -> Tuple[Dict[str, Variable], Dict[str, Any]]:
    """Returns the variables and attributes of the Image Dataset"""
    groups = data_groups(list(self._images))
    coords: Dict[str, Variable] = {}
    data_vars: Dict[str, Variable] = {}
    coordinate_system_info = None
    dated_before = False

    for role, factory in self._images.items():
      structure = factory.structure
      variables = factory.get_variables()
      image_coords = {n: v for n, v in variables.items() if n in IMAGE_COORDINATES}
      dated = structure.observation.known

      # Images without an observation date take the date of the others
      if "time" in coords:
        if dated_before and not dated:
          image_coords["time"] = coords["time"]
        elif dated and not dated_before:
          coords["time"] = image_coords["time"]

      prefix = (
        f"Cannot open the {role} image {self._urls[role]} with the images "
        f"opened before it: its {{dim}} coordinate"
      )

      for dim in factory.dims:
        if dim in coords:
          image_coords[dim] = snap(
            dim, coords[dim], image_coords[dim], prefix.format(dim=dim)
          )

      coords.update(image_coords)
      dated_before = dated_before or dated
      coordinate_system_info = structure.coordinate_system.to_attrs()
      data_vars[role] = variables[role]
      beam_fit_params = factory.beam_fit_params
      group = next((g for g in groups.values() if g.get("sky") == role), None)

      if is_sky(role) and beam_fit_params is not None:
        data_vars[beam_fit_params] = variables[beam_fit_params]
        if group is not None:
          group["beam_fit_params_sky"] = beam_fit_params

      if (flag := factory.flag) is not None:
        data_vars[flag] = variables[flag]
        if group is not None:
          group["flag"] = flag

      psf = role == "POINT_SPREAD_FUNCTION"

      if psf and beam_fit_params is not None:
        data_vars[beam_fit_params] = variables[beam_fit_params]

      if not is_sky(role):
        for g in groups.values():
          g[role.lower()] = role
          if psf and beam_fit_params is not None:
            g["beam_fit_params_point_spread_function"] = beam_fit_params

    images = [n for n in data_vars if not n.startswith("FLAG_")]

    # Images have beam parameters, except for a sum of weights alone
    if role != VISIBILITY_NORMALIZATION or len(images) > 1:
      coords["beam_params_label"] = coordinate(
        "beam_params_label", ("beam_params_label",), np.asarray(BEAM_PARAMS_LABELS)
      )

    attrs = {
      "coordinate_system_info": coordinate_system_info,
      "data_groups": groups,
      "schema_version": IMAGE_SCHEMA_VERSION,
      "type": IMAGE_DATASET_TYPE,
    }

    return {**data_vars, **coords}, attrs
