from __future__ import annotations

import dataclasses
import re
import warnings
from typing import TYPE_CHECKING, Any, Dict

from xarrayfits.backend.fits.coordinate_system import to_radians
from xarrayfits.errors import MissingObservationDateWarning
from xarrayfits.measures import sky_coord, time_attrs

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


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
  """Observation metadata of a FITS Image"""

  mjd: float
  """Observation date in MJD days"""
  scale: str
  """Time scale of the observation date"""
  known: bool
  """Whether the header gives the observation date"""
  image_attrs: Dict[str, Any]
  """Descriptive attributes of the image variable"""

  def time_attrs(self) -> Dict[str, str]:
    """Returns the attributes of the ``time`` coordinate"""
    return time_attrs(self.scale)


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
  """Returns the pointing center, the reference direction by default"""
  lon = layout.axes[layout.lon]
  lat = layout.axes[layout.lat]
  direction = [to_radians(lon.crval, lon.cunit), to_radians(lat.crval, lat.cunit)]
  return sky_coord(direction, "rad", coordinate_system.frame)


def read_observation(
  header: Header, layout: AxisLayout, coordinate_system: CoordinateSystem
) -> Observation:
  """Reads the observation metadata of the primary HDU header"""
  scale = "utc"
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
  known = False

  image_attrs: Dict[str, Any] = {}
  image_attrs["pointing_center"] = pointing_center(header, layout, coordinate_system)
  image_attrs["telescope"] = {"name": "UNKNOWN"}
  image_attrs["description"] = None
  image_attrs["user"] = user_attrs(header)

  return Observation(mjd=mjd, scale=scale, known=known, image_attrs=image_attrs)
