from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from astropy import units as u

from xarrayfits.backend.fits.axes import CTYPE_FRAME_TAGS, spectral_ctype_tag
from xarrayfits.errors import (
  MissingSpectralFrameWarning,
  UnknownSpectralFrameWarning,
  UnsupportedFitsImage,
)
from xarrayfits.measures import quantity, spectral_coord

if TYPE_CHECKING:
  import numpy.typing as npt
  from astropy.io.fits import Header

  from xarrayfits.backend.fits.axes import AxisLayout

#: Speed of light
C = 2.99792458e08 * u.m / u.s

#: Frequency of the single channel of an image without a spectral axis,
#: the reference value of casacore's default spectral coordinate
DEFAULT_FREQUENCY_HZ = 1415000000.0

#: Channel width of casacore's default spectral coordinate
DEFAULT_CHANNEL_WIDTH_HZ = 1000.0

#: Rest frequency of casacore's default spectral coordinate (HI)
DEFAULT_REST_FREQUENCY_HZ = 1420405751.7860003

#: Rest frequency recorded when it is unknown
UNKNOWN_REST_FREQUENCY_HZ = 0.0

#: casacore spectral frame -> FITS ``SPECSYS``
CASACORE_TO_FITS_SPECSYS = {
  "REST": "SOURCE",
  "LSRK": "LSRK",
  "LSRD": "LSRD",
  "BARY": "BARYCENT",
  "GEO": "GEOCENTR",
  "TOPO": "TOPOCENT",
  "GALACTO": "GALACTOC",
  "LGROUP": "LOCALGRP",
  "CMB": "CMBDIPOL",
}

#: casacore spectral frame -> ``reference_frequency`` observer
CASACORE_TO_OBSERVER = {
  "REST": "REST",
  "LSRK": "lsrk",
  "LSRD": "lsrd",
  "BARY": "BARY",
  "GEO": "gcrs",
  "TOPO": "TOPO",
  "GALACTO": "GALACTO",
  "LGROUP": "LGROUP",
  "CMB": "CMB",
}

#: Lower case spectral frame names -> casacore spectral frame
SPECTRAL_FRAME_ALIASES = {
  **{frame.lower(): frame for frame in CASACORE_TO_FITS_SPECSYS},
  **{specsys.lower(): frame for frame, specsys in CASACORE_TO_FITS_SPECSYS.items()},
  **{observer.lower(): frame for frame, observer in CASACORE_TO_OBSERVER.items()},
  "heliocen": "BARY",
}

#: AIPS ``VELREF`` frame code (modulo 256) -> casacore spectral frame
VELREF_FRAMES = {
  1: "LSRK",
  2: "BARY",
  3: "TOPO",
  4: "LSRD",
  5: "GEO",
  6: "REST",
  7: "GALACTO",
}

#: Spellings of spectral axis units that astropy does not parse
FITS_UNIT_ALIASES = {
  "HZ": "Hz",
  "KHZ": "kHz",
  "MHZ": "MHz",
  "GHZ": "GHz",
  "THZ": "THz",
  "M/S": "m/s",
  "KM/S": "km/s",
}


@dataclasses.dataclass(frozen=True, slots=True)
class Spectral:
  """The spectral axis of a FITS Image"""

  frequency: npt.NDArray[np.float64]
  """Channel frequencies in Hz"""
  velocity: npt.NDArray[np.float64] | None
  """Channel velocities in m/s, or ``None`` without a rest frequency"""
  doppler_type: str
  """Doppler convention of the velocities"""
  frame: str
  """casacore spectral frame"""
  rest_frequency: float
  """Rest frequency in Hz, 0.0 when unknown"""
  reference_frequency: float
  """Frequency of the reference channel in Hz"""
  channel_width: float
  """Channel width in Hz"""

  def frequency_attrs(self) -> Dict[str, Any]:
    """Returns the attributes of the ``frequency`` coordinate"""
    return {
      "rest_frequency": quantity(self.rest_frequency, "Hz"),
      "type": "spectral_coord",
      "units": "Hz",
      "frame": self.frame,
      "wave_units": "mm",
      "reference_frequency": spectral_coord(
        self.reference_frequency, "Hz", CASACORE_TO_OBSERVER[self.frame]
      ),
      "channel_width": quantity(self.channel_width, "Hz"),
    }

  def velocity_attrs(self) -> Dict[str, Any]:
    """Returns the attributes of the ``velocity`` coordinate"""
    return {"units": "m/s", "doppler_type": self.doppler_type, "type": "doppler"}


def parse_unit(unit: str, ctype: str) -> u.UnitBase:
  """Parses an axis unit, accepting spellings astropy does not parse"""
  try:
    return u.Unit(unit)
  except ValueError:
    if unit.upper() in FITS_UNIT_ALIASES:
      return u.Unit(FITS_UNIT_ALIASES[unit.upper()])
    raise UnsupportedFitsImage(
      f"Cannot interpret the unit {unit!r} of the {ctype} axis"
    ) from None


def unit_scale(unit: str, target: u.UnitBase, ctype: str) -> float:
  """Returns the factor converting the values of an axis to ``target``"""
  try:
    return float((1.0 * parse_unit(unit, ctype)).to(target).value)
  except u.UnitConversionError as e:
    raise UnsupportedFitsImage(
      f"The {ctype} axis has unit {unit!r}, which cannot be converted to {target}"
    ) from e


def rest_frequency(header: Header) -> float | None:
  """Returns the rest frequency in Hz from ``RESTFRQ``, ``RESTFREQ`` or
  ``RESTWAV``, or ``None`` if the header has no positive value"""
  for key in ("RESTFRQ", "RESTFREQ"):
    if key in header and float(header[key]) > 0:
      return float(header[key])

  if "RESTWAV" in header and float(header["RESTWAV"]) > 0:
    return float((C / (float(header["RESTWAV"]) * u.m)).to(u.Hz).value)

  return None


def velref(header: Header) -> int | None:
  """Returns the AIPS ``VELREF``, or ``None`` if absent or not an integer"""
  try:
    return int(header["VELREF"])
  except (KeyError, TypeError, ValueError):
    return None


def spectral_frame(header: Header, ctype: str) -> str:
  """Returns the casacore spectral frame of the image from ``SPECSYS``,
  else the AIPS frame suffix of the spectral ``CTYPE``, else ``VELREF``"""
  specsys = str(header.get("SPECSYS", "")).strip()

  if specsys:
    try:
      return SPECTRAL_FRAME_ALIASES[specsys.lower()]
    except KeyError:
      warnings.warn(
        f"The FITS SPECSYS {specsys!r} is not a known spectral "
        f"reference frame; ignoring it",
        UnknownSpectralFrameWarning,
        stacklevel=2,
      )

  if (tag := spectral_ctype_tag(ctype)) in CTYPE_FRAME_TAGS:
    return CTYPE_FRAME_TAGS[tag]

  if (code := velref(header)) is not None and code >= 0:
    if code % 256 in VELREF_FRAMES:
      return VELREF_FRAMES[code % 256]

  warnings.warn(
    "The FITS header gives no spectral reference frame (SPECSYS, a "
    "frame suffix of the spectral CTYPE or VELREF); assuming LSRK",
    MissingSpectralFrameWarning,
    stacklevel=2,
  )
  return "LSRK"


def doppler_type(header: Header, axis_type: str) -> str:
  """Returns the doppler convention of the velocities: optical for VOPT
  and FELO axes, radio for VRAD axes, and for frequency axes the AIPS
  ``VELREF`` convention, radio without ``VELREF``"""
  if axis_type in ("VOPT", "FELO"):
    return "z"
  if axis_type == "VRAD":
    return "radio"
  code = velref(header)
  return "radio" if code is None or code > 256 else "z"


def velocities(
  rest: float, frequency: npt.NDArray[np.float64], doppler: str
) -> npt.NDArray[np.float64]:
  """Converts frequencies into velocities in m/s"""
  c = C.value
  if doppler == "radio":
    return np.asarray([(1 - f / rest) * c for f in frequency])
  return np.asarray([(rest / f - 1) * c for f in frequency])


def default_spectral() -> Spectral:
  """Returns casacore's default spectral coordinate, given to images
  without a spectral axis"""
  frequency = np.array([DEFAULT_FREQUENCY_HZ])
  return Spectral(
    frequency=frequency,
    velocity=velocities(DEFAULT_REST_FREQUENCY_HZ, frequency, "radio"),
    doppler_type="radio",
    frame="LSRK",
    rest_frequency=DEFAULT_REST_FREQUENCY_HZ,
    reference_frequency=DEFAULT_FREQUENCY_HZ,
    channel_width=DEFAULT_CHANNEL_WIDTH_HZ,
  )


def read_spectral(header: Header, layout: AxisLayout) -> Spectral:
  """Reads the spectral axis of the primary HDU header. Frequencies are
  in Hz. FREQ axes are linear; VOPT and VRAD axes are linear in velocity;
  FELO axes are linear in frequency"""
  if layout.frequency is None:
    return default_spectral()

  axis = layout.axes[layout.frequency]
  axis_type = axis.ctype[:4].upper()
  rest = rest_frequency(header)
  frame = spectral_frame(header, axis.ctype)
  doppler = doppler_type(header, axis_type)
  velocity = None

  if axis_type == "FREQ":
    to_hz = unit_scale(axis.cunit, u.Hz, axis.ctype)
    frequency = axis.linear_values()
    if to_hz != 1.0:
      frequency = frequency * to_hz
    reference_frequency = axis.crval * to_hz
    channel_width = abs(axis.cdelt * to_hz)
  else:
    if rest is None:
      raise UnsupportedFitsImage(
        f"Spectral axis {axis.ctype} in FITS header is velocity, but there "
        f"is no rest frequency (RESTFRQ, RESTFREQ or RESTWAV) so "
        f"converting to frequency is not possible"
      )

    to_ms = unit_scale(axis.cunit, u.m / u.s, axis.ctype)
    c = C.value

    if axis_type == "VOPT":
      unit = parse_unit(axis.cunit, axis.ctype)
      v0 = axis.crval - axis.cdelt * axis.crpix
      vel = [v0 + i * axis.cdelt for i in range(axis.naxis)] * unit
      rest_q = rest * u.Hz
      frequency = np.asarray(
        (rest_q / (np.array(vel.value) * vel.unit / C + 1)).to(u.Hz).value
      )
      velocity = (vel.value * unit).to(u.m / u.s).value
      reference = rest_q / (axis.crval * vel.unit / C + 1)
      reference_frequency = float(reference.to(u.Hz).value)
      # The frequency step from the reference channel to the next
      ref_velocity = axis.crval * to_ms
      channel_width = abs(
        rest / (1.0 + (ref_velocity + axis.cdelt * to_ms) / c)
        - rest / (1.0 + ref_velocity / c)
      )
    elif axis_type == "FELO":
      ref_velocity = axis.crval * to_ms
      reference_frequency = rest / (1.0 + ref_velocity / c)
      increment = -axis.cdelt * to_ms * reference_frequency / (c + ref_velocity)
      frequency = np.array(
        [reference_frequency + (i - axis.crpix) * increment for i in range(axis.naxis)]
      )
      channel_width = abs(increment)
    else:
      velocity = axis.linear_values() * to_ms
      frequency = rest * (1.0 - velocity / c)
      reference_frequency = rest * (1.0 - axis.crval * to_ms / c)
      channel_width = abs(rest * axis.cdelt * to_ms / c)

  if velocity is None and rest is not None:
    velocity = velocities(rest, frequency, doppler)

  return Spectral(
    frequency=frequency,
    velocity=velocity,
    doppler_type=doppler,
    frame=frame,
    rest_frequency=UNKNOWN_REST_FREQUENCY_HZ if rest is None else rest,
    reference_frequency=reference_frequency,
    channel_width=channel_width,
  )
