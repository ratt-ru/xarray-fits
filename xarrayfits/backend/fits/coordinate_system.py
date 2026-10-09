from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from astropy import units as u
from astropy.io import fits

from xarrayfits.measures import direction_location, sky_coord

if TYPE_CHECKING:
  from astropy.io.fits import Header

  from xarrayfits.backend.fits.axes import AxisLayout

#: Conversion factor from degrees to radians
DEG_TO_RAD = np.pi / 180.0


@dataclasses.dataclass(frozen=True, slots=True)
class CoordinateSystem:
  """The celestial coordinate system of a FITS Image"""

  projection: str
  """Projection code, for example ``"SIN"``"""
  frame: str
  """Astropy sky frame, for example ``"fk5"``"""
  equinox: str | None
  """Equinox, ``"j2000.0"`` or ``"b1950.0"``, or ``None`` for ICRS"""
  reference_direction: Tuple[float, float]
  """Longitude and latitude of the reference pixel in radians"""
  native_pole: Tuple[float, float]
  """``LONPOLE`` and ``LATPOLE`` in degrees"""
  pc: Tuple[Tuple[float, float], Tuple[float, float]]
  """PC matrix of the celestial axes, in (longitude, latitude) order"""
  projection_parameters: Tuple[float, ...]
  """Projection parameters of the latitude axis"""

  def to_attrs(self) -> Dict[str, Any]:
    """Returns the ``coordinate_system_info`` attribute"""
    reference_direction = sky_coord(self.reference_direction, "rad", self.frame)

    if self.equinox is not None:
      reference_direction["attrs"]["equinox"] = self.equinox

    return {
      "projection": self.projection,
      "reference_direction": reference_direction,
      "native_pole_direction": direction_location(
        [x * DEG_TO_RAD for x in self.native_pole], "rad", "native_projection"
      ),
      "pixel_coordinate_transformation_matrix": [list(row) for row in self.pc],
      "projection_parameters": list(self.projection_parameters),
    }


def to_radians(value: float, unit: str) -> float:
  """Converts an angle into radians"""
  return float((value * u.Unit(unit)).to("rad").value)


def native_pole(header: Header, layout: AxisLayout) -> Tuple[float, float]:
  """Returns ``LONPOLE`` and ``LATPOLE``, computed by wcslib
  if the header lacks them"""
  if "LONPOLE" in header and "LATPOLE" in header:
    return float(header["LONPOLE"]), float(header["LATPOLE"])

  from astropy.wcs import WCS

  celestial = fits.Header()

  for new, old in ((1, layout.lon + 1), (2, layout.lat + 1)):
    for key in ("CTYPE", "CRVAL", "CDELT", "CRPIX", "CUNIT"):
      if f"{key}{old}" in header:
        celestial[f"{key}{new}"] = header[f"{key}{old}"]

  for key in ("LONPOLE", "LATPOLE"):
    if key in header:
      celestial[key] = header[key]

  with warnings.catch_warnings():
    # wcslib "fixes" are irrelevant to the native pole
    warnings.simplefilter("ignore")
    wcs = WCS(celestial)
    wcs.wcs.set()

  return float(wcs.wcs.lonpole), float(wcs.wcs.latpole)


def read_coordinate_system(header: Header, layout: AxisLayout) -> CoordinateSystem:
  """Reads the celestial coordinate system of the primary HDU header"""
  lon = layout.axes[layout.lon]
  lat = layout.axes[layout.lat]
  identity: List[Tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0)]

  return CoordinateSystem(
    projection=lon.ctype[-3:],
    frame="icrs",
    equinox=None,
    reference_direction=(
      to_radians(lon.crval, lon.cunit),
      to_radians(lat.crval, lat.cunit),
    ),
    native_pole=native_pole(header, layout),
    pc=(identity[0], identity[1]),
    projection_parameters=(0.0, 0.0),
  )


def sky_coordinates(
  layout: AxisLayout, coordinate_system: CoordinateSystem
) -> Tuple[np.ndarray, np.ndarray]:
  """Returns the right ascension and declination of every pixel,
  in radians, on (l, m) dimensions"""
  from astropy.wcs import WCS

  wcs_cards: Dict[str, Any] = {}

  for n, (i, name) in enumerate(((layout.lon, "RA--"), (layout.lat, "DEC-")), 1):
    axis = layout.axes[i]
    wcs_cards[f"CTYPE{n}"] = f"{name}-{coordinate_system.projection}"
    wcs_cards[f"NAXIS{n}"] = axis.naxis
    wcs_cards[f"CUNIT{n}"] = axis.cunit
    wcs_cards[f"CDELT{n}"] = axis.cdelt
    wcs_cards[f"CRPIX{n}"] = axis.crpix + 1
    wcs_cards[f"CRVAL{n}"] = axis.crval

  wcs_cards["LONPOLE"] = coordinate_system.native_pole[0]
  wcs_cards["LATPOLE"] = coordinate_system.native_pole[1]

  wcs = WCS(wcs_cards)
  x, y = np.indices(wcs.pixel_shape)
  ra, dec = wcs.pixel_to_world_values(x, y)
  return ra * DEG_TO_RAD, dec * DEG_TO_RAD
