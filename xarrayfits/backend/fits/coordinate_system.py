from __future__ import annotations

import dataclasses
import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
from astropy import units as u
from astropy.io import fits

from xarrayfits.errors import UnsupportedFitsImage
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


#: FITS ``RADESYS`` -> astropy sky frame
RADESYS_FRAMES = {
  "ICRS": "icrs",
  "FK5": "fk5",
  "FK4": "fk4",
  "FK4-NO-E": "fk4noterms",
}

#: Projection parameter cards, ``PVi_m`` and ``PV0i_0m``
PV_CARD = re.compile(r"PV0?(\d+)_0?(\d+)")


def equinox_year(value: Any) -> float:
  """Returns the year of an ``EQUINOX`` (or ``EPOCH``) value,
  a number such as 2000.0 or a string such as ``"J2000"``"""
  if isinstance(value, str):
    match = re.fullmatch(r"\s*[JjBb]?\s*(\d+(?:\.\d*)?)\s*", value)
    if match is None:
      raise UnsupportedFitsImage(f"Cannot interpret the FITS EQUINOX {value!r}")
    return float(match.group(1))
  return float(value)


def reference_frame(header: Header) -> Tuple[str, str | None]:
  """Returns the sky frame and equinox of the celestial axes, taking
  the FITS WCS Paper II defaults for missing values"""
  equinox = header.get("EQUINOX", header.get("EPOCH"))
  year = None if equinox is None else equinox_year(equinox)
  radesys = str(header.get("RADESYS", header.get("RADECSYS", ""))).strip().upper()

  if not radesys:
    if year is None:
      radesys = "ICRS"
    else:
      radesys = "FK4" if year < 1984.0 else "FK5"

  try:
    frame = RADESYS_FRAMES[radesys]
  except KeyError:
    raise UnsupportedFitsImage(
      f"Unsupported FITS RADESYS {radesys!r}; supported reference systems "
      f"are {', '.join(RADESYS_FRAMES)}"
    ) from None

  if frame == "icrs":
    return frame, None
  if frame == "fk5":
    return frame, f"j{2000.0 if year is None else year:.1f}"
  return frame, f"b{1950.0 if year is None else year:.1f}"


def projection_parameters(
  header: Header, lat_axis: int, projection: str
) -> Tuple[float, ...]:
  """Returns the projection parameters ``PVi_m`` of the 1-based latitude
  axis: ``PVi_1, PVi_2, ...`` (from ``PVi_0`` for ZPN), with 0.0 for
  missing parameters in between"""
  values = {}

  for key in header.keys():
    match = PV_CARD.fullmatch(key)
    if match is not None and int(match.group(1)) == lat_axis:
      values[int(match.group(2))] = float(header[key])

  if not values:
    return (0.0, 0.0)

  first = 0 if projection == "ZPN" else 1
  return tuple(values.get(m, 0.0) for m in range(first, max(values) + 1))


def pc_matrix(
  header: Header, layout: AxisLayout
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
  """Returns the PC matrix of the celestial axes, in (longitude, latitude)
  order. Missing elements default to the identity matrix, and a header
  without PC cards may give an AIPS ``CROTAi`` rotation instead"""
  axes = (layout.lon + 1, layout.lat + 1)
  pc = np.eye(2)
  has_pc = False

  for i in (0, 1):
    for j in (0, 1):
      a, b = axes[i], axes[j]
      for key in (f"PC{a}_{b}", f"PC0{a}_0{b}", f"PC{a:03d}{b:03d}"):
        if key in header:
          pc[i, j] = float(header[key])
          has_pc = True
          break

  crota = float(header.get(f"CROTA{axes[1]}", 0.0))

  if not has_pc and crota != 0.0:
    rho = crota * DEG_TO_RAD
    ratio = layout.axes[layout.lat].cdelt / layout.axes[layout.lon].cdelt
    pc = np.array(
      [[np.cos(rho), -np.sin(rho) * ratio], [np.sin(rho) / ratio, np.cos(rho)]]
    )

  (a, b), (c, d) = pc.tolist()
  return (a, b), (c, d)


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

  for key in header.keys():
    match = PV_CARD.fullmatch(key)
    if match is not None and int(match.group(1)) == layout.lat + 1:
      celestial[f"PV2_{int(match.group(2))}"] = header[key]

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

  if lon.ctype[-3:] != lat.ctype[-3:]:
    raise UnsupportedFitsImage(
      f"Projections for direction axes ({lon.ctype[-3:]}, {lat.ctype[-3:]}) "
      f"differ, but they must be the same"
    )

  projection = lon.ctype[-3:]
  frame, equinox = reference_frame(header)

  return CoordinateSystem(
    projection=projection,
    frame=frame,
    equinox=equinox,
    reference_direction=(
      to_radians(lon.crval, lon.cunit),
      to_radians(lat.crval, lat.cunit),
    ),
    native_pole=native_pole(header, layout),
    pc=pc_matrix(header, layout),
    projection_parameters=projection_parameters(header, layout.lat + 1, projection),
  )


def sky_wcs_cards(
  layout: AxisLayout, coordinate_system: CoordinateSystem
) -> Dict[str, Any]:
  """Returns the cards of a two axis celestial WCS, longitude first,
  of the FITS Image"""
  cards: Dict[str, Any] = {}

  for n, (i, name) in enumerate(((layout.lon, "RA--"), (layout.lat, "DEC-")), 1):
    axis = layout.axes[i]
    cards[f"CTYPE{n}"] = f"{name}-{coordinate_system.projection}"
    cards[f"NAXIS{n}"] = axis.naxis
    cards[f"CUNIT{n}"] = axis.cunit
    cards[f"CDELT{n}"] = axis.cdelt
    cards[f"CRPIX{n}"] = axis.crpix + 1
    cards[f"CRVAL{n}"] = axis.crval

  first = 0 if coordinate_system.projection.upper() == "ZPN" else 1

  for m, value in enumerate(coordinate_system.projection_parameters, start=first):
    if value != 0:
      cards[f"PV2_{m}"] = float(value)

  if not np.array_equal(np.asarray(coordinate_system.pc), np.eye(2)):
    for i in range(2):
      for j in range(2):
        cards[f"PC{i + 1}_{j + 1}"] = float(coordinate_system.pc[i][j])

  cards["LONPOLE"] = coordinate_system.native_pole[0]
  cards["LATPOLE"] = coordinate_system.native_pole[1]
  return cards
