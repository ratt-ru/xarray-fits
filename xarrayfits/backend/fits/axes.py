from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Tuple

import numpy as np

from xarrayfits.errors import UnsupportedFitsImage

if TYPE_CHECKING:
  from astropy.io.fits import Header

#: FITS ``BITPIX`` -> numpy dtype of the pixels, in native byte order
BITPIX_DTYPES = {
  8: np.dtype(np.uint8),
  16: np.dtype(np.int16),
  32: np.dtype(np.int32),
  64: np.dtype(np.int64),
  -32: np.dtype(np.float32),
  -64: np.dtype(np.float64),
}

#: Spectral axis types (the first four ``CTYPE`` characters) that are
#: converted to frequency
SPECTRAL_AXIS_TYPES = ("FREQ", "VOPT", "VRAD", "FELO")

#: Spectral axis types that are not supported
OTHER_SPECTRAL_AXIS_TYPES = ("VELO", "WAVE", "AWAV", "WAVN", "ZOPT", "BETA", "ENER")

#: AIPS spectral frame suffix of ``CTYPEi`` -> casacore spectral frame
CTYPE_FRAME_TAGS = {
  "LSR": "LSRK",
  "LSRK": "LSRK",
  "HEL": "BARY",
  "OBS": "TOPO",
  "LSD": "LSRD",
  "GEO": "GEO",
  "SOU": "REST",
  "REST": "REST",
  "GAL": "GALACTO",
}


@dataclasses.dataclass(frozen=True, slots=True)
class Axis:
  """World coordinate description of a FITS axis"""

  ctype: str
  """``CTYPEi``"""
  naxis: int
  """``NAXISi``"""
  crval: float
  """``CRVALi``"""
  cdelt: float
  """``CDELTi``"""
  crpix: float
  """``CRPIXi``, 0-based"""
  cunit: str
  """``CUNITi``, or the default unit of the axis type"""

  def linear_values(self) -> np.ndarray:
    """World values of a linear axis"""
    return self.crval + (np.arange(self.naxis) - self.crpix) * self.cdelt


@dataclasses.dataclass(frozen=True, slots=True)
class AxisLayout:
  """The FITS axes of an image and the role of each"""

  axes: Tuple[Axis, ...]
  """Axes in FITS order"""
  lon: int
  """0-based FITS index of the longitude axis"""
  lat: int
  """0-based FITS index of the latitude axis"""
  frequency: int | None
  """0-based FITS index of the spectral axis"""
  polarization: int | None
  """0-based FITS index of the ``STOKES`` axis"""

  @property
  def shape(self) -> Tuple[int, ...]:
    """Shape of the pixels in numpy (reversed FITS) order"""
    return tuple(a.naxis for a in reversed(self.axes))

  def numpy_axis(self, fits_axis: int | None) -> int | None:
    """Returns the numpy axis of a 0-based FITS axis"""
    return None if fits_axis is None else len(self.axes) - 1 - fits_axis


def spectral_ctype_tag(ctype: str) -> str:
  """Returns the AIPS frame suffix of a spectral ``CTYPE``, such as
  ``"LSR"`` for ``"FREQ-LSR"``, or ``""``"""
  return ctype[4:].replace("-", "").strip().upper()


def is_spectral(ctype: str) -> bool:
  """Returns whether ``ctype`` is a supported spectral axis: one of
  :data:`SPECTRAL_AXIS_TYPES`, alone or with an AIPS frame suffix"""
  tag = spectral_ctype_tag(ctype)
  return ctype[:4].upper() in SPECTRAL_AXIS_TYPES and (
    tag == "" or tag in CTYPE_FRAME_TAGS
  )


def default_axis_unit(ctype: str) -> str:
  """Unit of an axis without ``CUNITi``"""
  if ctype[:4].upper() == "FREQ":
    return "Hz"
  if ctype[:4].upper() in SPECTRAL_AXIS_TYPES:
    return "m/s"
  if ctype.upper() == "STOKES":
    return ""
  return "deg"


def read_axes(header: Header) -> AxisLayout:
  """Reads the axes of the primary HDU header"""
  axes = []
  lon = lat = frequency = polarization = None

  for i in range(header["NAXIS"]):
    n = i + 1
    ctype = header[f"CTYPE{n}"]

    if ctype.startswith("RA-"):
      lon = i
    elif ctype.startswith("DEC-"):
      lat = i
    elif ctype == "STOKES":
      polarization = i
    elif is_spectral(ctype):
      frequency = i
    elif ctype[:4].upper() in SPECTRAL_AXIS_TYPES + OTHER_SPECTRAL_AXIS_TYPES:
      raise UnsupportedFitsImage(
        f"{ctype} is an unsupported spectral axis; supported spectral "
        f"axes are {', '.join(SPECTRAL_AXIS_TYPES)}, optionally with an "
        f"AIPS frame suffix such as '-LSR'"
      )
    else:
      raise UnsupportedFitsImage(f"{ctype} is an unsupported axis")

    if f"CDELT{n}" not in header and any(
      re.fullmatch(r"CD\d+_\d+", key) for key in header.keys()
    ):
      raise UnsupportedFitsImage(
        "FITS images whose coordinates are given by a CDi_j matrix "
        "instead of CDELTi (and PCi_j) are not supported"
      )

    unit = str(header.get(f"CUNIT{n}", "")).strip()
    axes.append(
      Axis(
        ctype=ctype,
        naxis=header[f"NAXIS{n}"],
        crval=header[f"CRVAL{n}"],
        cdelt=header[f"CDELT{n}"],
        crpix=header[f"CRPIX{n}"] - 1,
        cunit=unit or default_axis_unit(ctype),
      )
    )

  if lon is None or lat is None:
    raise UnsupportedFitsImage("Could not find both direction axes")

  return AxisLayout(tuple(axes), lon, lat, frequency, polarization)


def dtype_from_bitpix(header: Header) -> np.dtype:
  """Returns the native byte order dtype of the primary HDU pixels"""
  return BITPIX_DTYPES[header["BITPIX"]]
