from __future__ import annotations

import dataclasses
import re
import warnings
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from astropy.time import Time
from erfa import ErfaWarning

from xarrayfits.backend.fits.coordinate_system import to_radians
from xarrayfits.errors import (
  InvalidObservationDateWarning,
  MissingObservationDateWarning,
  UnknownTimeScaleWarning,
)
from xarrayfits.measures import DEG_TO_RAD, sky_coord, time_attrs, time_measure

if TYPE_CHECKING:
  from astropy.io.fits import Header

  from xarrayfits.backend.fits.axes import AxisLayout
  from xarrayfits.backend.fits.coordinate_system import CoordinateSystem

#: MJD of the time coordinate of a FITS Image without an observation date,
#: casacore's value for an unset observation date
UNKNOWN_OBSDATE_MJD = 0.0

#: Header cards that are interpreted, or describe the file layout,
#: and are excluded from the ``user`` attribute
USER_EXCLUDED_CARDS = frozenset(
  {
    "ALTRPIX",
    "ALTRVAL",
    "BITPIX",
    "BLANK",
    "BMAJ",
    "BMIN",
    "BPA",
    "BSCALE",
    "BTYPE",
    "BUNIT",
    "BZERO",
    "CASAMBM",
    "CHECKSUM",
    "DATASUM",
    "DATE",
    "DATE-OBS",
    "EPOCH",
    "EQUINOX",
    "EXTEND",
    "HISTORY",
    "LATPOLE",
    "LONPOLE",
    "MJD-OBS",
    "OBSDEC",
    "OBSERVER",
    "OBSRA",
    "ORIGIN",
    "TELESCOP",
    "OBJECT",
    "RADECSYS",
    "RADESYS",
    "RESTFREQ",
    "RESTFRQ",
    "RESTWAV",
    "SIMPLE",
    "SPECSYS",
    "TIMESYS",
    "VELREF",
  }
)

#: Header card patterns excluded from the ``user`` attribute
USER_EXCLUDED_PATTERN = re.compile(
  r"|".join(
    [
      r"^NAXIS\d?$",
      r"^CRVAL\d$",
      r"^CRPIX\d$",
      r"^CTYPE\d$",
      r"^CDELT\d$",
      r"^CUNIT\d$",
      r"^OBSGEO-(X|Y|Z)$",
      r"^P(C|V)0?\d_0?\d",
      r"^PC\d{6}$",
      r"^CD\d_\d$",
      r"^CROTA\d$",
    ]
  )
)


#: FITS ``TIMESYS`` -> astropy time scale
TIMESYS_SCALES = {
  "UTC": "utc",
  "TAI": "tai",
  "IAT": "tai",
  "TT": "tt",
  "TDT": "tt",
  "ET": "tt",
  "TDB": "tdb",
  "TCB": "tcb",
  "TCG": "tcg",
  "UT1": "ut1",
  "UT": "ut1",
  "GMT": "utc",
}

#: casacore image types, which ``BTYPE`` may give
CASACORE_IMAGE_TYPES = (
  "Undefined",
  "Intensity",
  "Beam",
  "Column Density",
  "Depolarization Ratio",
  "Kinetic Temperature",
  "Magnetic Field",
  "Optical Depth",
  "Rotation Measure",
  "Rotational Temperature",
  "Spectral Index",
  "Velocity",
  "Velocity Dispersion",
)


def image_type_key(name: Any) -> str:
  return re.sub(r"[\s_\-]", "", str(name)).lower()


IMAGE_TYPES_BY_KEY = {image_type_key(name): name for name in CASACORE_IMAGE_TYPES}


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
  """Observation metadata of a FITS Image"""

  mjd: float
  """Observation date in MJD days"""
  scale: str
  """Time scale of the observation date"""
  known: bool
  """Whether the header gives the observation date"""
  sub_type: str | None
  """casacore image type given by ``BTYPE``, without spaces"""
  image_attrs: Dict[str, Any]
  """Descriptive attributes of the image variable"""

  def time_attrs(self) -> Dict[str, str]:
    """Returns the attributes of the ``time`` coordinate"""
    return time_attrs(self.scale)


def sub_type(btype: Any) -> str | None:
  """Returns the casacore image type of a ``BTYPE`` without spaces,
  matching case insensitively and ignoring spaces, underscores and hyphens.
  Returns ``None`` for an empty, ``Undefined`` or unknown ``BTYPE``"""
  if not btype:
    return None
  name = IMAGE_TYPES_BY_KEY.get(image_type_key(btype))
  if name is None or name == "Undefined":
    return None
  return name.replace(" ", "")


def time_scale(header: Header) -> str:
  """Returns the time scale of the header's dates, UTC by default"""
  timesys = str(header.get("TIMESYS", "")).strip().upper() or "UTC"

  try:
    return TIMESYS_SCALES[timesys]
  except KeyError:
    warnings.warn(
      f"The FITS TIMESYS {timesys!r} is not supported; assuming UTC",
      UnknownTimeScaleWarning,
      stacklevel=2,
    )
    return "utc"


def mjd_from_date_obs(date_obs: Any, scale: str) -> float | None:
  """Returns the MJD of a ``DATE-OBS``, or ``None`` if it cannot be
  interpreted"""
  text = str(date_obs).strip()
  # DD/MM/YY, the FITS date format before 1999
  if (old_style := re.fullmatch(r"(\d{2})/(\d{2})/(\d{2})", text)) is not None:
    text = f"19{old_style.group(3)}-{old_style.group(2)}-{old_style.group(1)}"

  try:
    with warnings.catch_warnings():
      # ERFA warns that UTC before 1960 is "dubious";
      # the date is only parsed here
      warnings.simplefilter("ignore", ErfaWarning)
      return float(Time(text, format="fits", scale=scale).mjd)
  except ValueError as e:
    warnings.warn(
      f"Cannot interpret the FITS DATE-OBS {date_obs!r}: {e}",
      InvalidObservationDateWarning,
      stacklevel=2,
    )
    return None


def telescope(header: Header) -> Dict[str, Any]:
  """Returns the telescope name and, given ``OBSGEO-X/Y/Z``,
  its geocentric location"""
  result: Dict[str, Any] = {
    "name": str(header.get("TELESCOP", "")).strip() or "UNKNOWN"
  }

  if all(k in header for k in ("OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z")):
    xyz = np.array([header["OBSGEO-X"], header["OBSGEO-Y"], header["OBSGEO-Z"]])
    r = np.sqrt(np.sum(xyz * xyz))
    attrs = {
      "coordinate_system": "geocentric",
      "frame": "ITRF",
      "origin_object_name": "earth",
      "type": "location",
    }
    result["direction"] = {
      "attrs": {**attrs, "units": "rad"},
      "data": [np.arctan2(xyz[1], xyz[0]), np.arcsin(xyz[2] / r)],
      "dims": ["ellipsoid_dir_label"],
      "coords": {
        "ellipsoid_dir_label": {"dims": ["ellipsoid_dir_label"], "data": ["lon", "lat"]}
      },
    }
    result["distance"] = {
      "attrs": {**attrs, "units": "m"},
      "data": [r],
      "dims": ["ellipsoid_dis_label"],
      "coords": {
        "ellipsoid_dis_label": {"dims": ["ellipsoid_dis_label"], "data": ["dist"]}
      },
    }

  return result


def user_attrs(header: Header) -> Dict[str, Any]:
  """Returns the header cards that are not otherwise interpreted"""
  return {
    key.lower(): value
    for key, value in header.items()
    if not (USER_EXCLUDED_PATTERN.search(key) or key in USER_EXCLUDED_CARDS)
  }


def pointing_center(
  header: Header, layout: AxisLayout, coordinate_system: CoordinateSystem
) -> Dict[str, Any]:
  """Returns the pointing center from ``OBSRA`` and ``OBSDEC``,
  else the reference direction"""
  if "OBSRA" in header and "OBSDEC" in header:
    direction = [
      float(header["OBSRA"]) * DEG_TO_RAD,
      float(header["OBSDEC"]) * DEG_TO_RAD,
    ]
  else:
    lon = layout.axes[layout.lon]
    lat = layout.axes[layout.lat]
    direction = [to_radians(lon.crval, lon.cunit), to_radians(lat.crval, lat.cunit)]

  return sky_coord(direction, "rad", coordinate_system.frame)


def read_observation(
  header: Header, layout: AxisLayout, coordinate_system: CoordinateSystem
) -> Observation:
  """Reads the observation metadata of the primary HDU header"""
  scale = time_scale(header)
  mjd = None

  if "DATE-OBS" in header:
    mjd = mjd_from_date_obs(header["DATE-OBS"], scale)

  if mjd is None and "MJD-OBS" in header:
    mjd = float(header["MJD-OBS"])

  known = mjd is not None and mjd != UNKNOWN_OBSDATE_MJD

  if not known:
    warnings.warn(
      "The FITS header has no usable observation date (DATE-OBS or "
      f"MJD-OBS other than MJD {UNKNOWN_OBSDATE_MJD}); its time "
      f"coordinate is MJD {UNKNOWN_OBSDATE_MJD} (1858-11-17), the value "
      "casacore uses for an unset observation date, and the image has no "
      "obsdate attribute",
      MissingObservationDateWarning,
      stacklevel=2,
    )
    mjd = UNKNOWN_OBSDATE_MJD

  assert mjd is not None
  image_attrs: Dict[str, Any] = {}

  for card, attr in (
    ("BUNIT", "units"),
    ("OBJECT", "object_name"),
    ("OBSERVER", "observer"),
  ):
    if card in header:
      image_attrs[attr] = header[card]

  if known:
    image_attrs["obsdate"] = time_measure(mjd, scale)

  image_attrs["pointing_center"] = pointing_center(header, layout, coordinate_system)
  image_attrs["telescope"] = telescope(header)
  image_attrs["description"] = None
  image_attrs["user"] = user_attrs(header)

  return Observation(
    mjd=mjd,
    scale=scale,
    known=known,
    sub_type=sub_type(header.get("BTYPE")),
    image_attrs=image_attrs,
  )
