"""Writes synthetic FITS Images for tests"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits

if TYPE_CHECKING:
  import numpy.typing as npt


@dataclasses.dataclass(frozen=True, slots=True)
class FitsAxis:
  """World coordinate description of a FITS image axis"""

  ctype: str
  """``CTYPEi``"""
  naxis: int
  """``NAXISi``"""
  crval: float
  """``CRVALi``"""
  cdelt: float
  """``CDELTi``"""
  crpix: float
  """``CRPIXi`` (1-based)"""
  cunit: str | None = None
  """``CUNITi``, which is omitted from the header if ``None``"""


#: Right ascension axis of 6 pixels in the SIN projection
RA = FitsAxis("RA---SIN", 6, 105.0, -1.0 / 60.0, 4.0, "deg")
#: Declination axis of 5 pixels in the SIN projection
DEC = FitsAxis("DEC--SIN", 5, -40.0, 1.0 / 60.0, 3.0, "deg")
#: Frequency axis of 3 channels
FREQ = FitsAxis("FREQ", 3, 1.415e9, 1.0e3, 2.0, "Hz")


def stokes_axis(
  crval: float = 1.0, cdelt: float = 1.0, n: int = 1, crpix: float = 1.0
) -> FitsAxis:
  """Returns a ``STOKES`` axis of ``n`` planes, Stokes I by default"""
  return FitsAxis("STOKES", n, crval, cdelt, crpix)


#: Axes of the default simulated image: RA, DEC, FREQ and Stokes I
DEFAULT_AXES = (RA, DEC, FREQ, stokes_axis())


def simulate_fits_image(
  path: str | os.PathLike,
  *,
  axes: Sequence[FitsAxis] = DEFAULT_AXES,
  cards: Mapping[str, Any] | None = None,
  remove: Iterable[str] = (),
  data: npt.NDArray | None = None,
  dtype: npt.DTypeLike = np.float32,
  beams: npt.NDArray | None = None,
  extra_hdus: Sequence[fits.hdu.base.ExtensionHDU] = (),
) -> str:
  """Writes a FITS Image whose primary HDU holds an image with the given axes.

  Args:
    path: Output file.
    axes: The FITS axes, in FITS order (``NAXIS1`` first).
    cards: Header cards to add or replace, after the axis cards.
    remove: Header cards to remove.
    data: Pixels in numpy (reversed FITS axis) order. By default every
      pixel holds its own flat index, so that reordering is detectable.
    dtype: Pixel type of the default data.
    beams: Per-plane beams of shape ``(nchan, npol, 3)``, holding the
      major and minor axes in arcsec and the position angle in degrees,
      indexed by FITS plane. They are written as a ``BEAMS`` binary table.
    extra_hdus: Further extension HDUs.

  Returns:
    The path of the FITS Image.
  """
  shape = tuple(axis.naxis for axis in reversed(axes))

  if data is None:
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)

  header = fits.Header()

  for i, axis in enumerate(axes, start=1):
    header[f"CTYPE{i}"] = axis.ctype
    header[f"CRVAL{i}"] = axis.crval
    header[f"CDELT{i}"] = axis.cdelt
    header[f"CRPIX{i}"] = axis.crpix
    if axis.cunit is not None:
      header[f"CUNIT{i}"] = axis.cunit

  if beams is not None:
    header["CASAMBM"] = True

  for key, value in (cards or {}).items():
    header[key] = value

  for key in remove:
    if key in header:
      del header[key]

  hdus: list[Any] = [fits.PrimaryHDU(data=data, header=header)]

  if beams is not None:
    hdus.append(_beams_table(beams))

  hdus.extend(extra_hdus)
  fits.HDUList(hdus).writeto(path, overwrite=True)
  return os.fspath(path)


def _beams_table(beams: npt.NDArray) -> fits.BinTableHDU:
  """Returns a CASA style ``BEAMS`` table of per-plane beams"""
  nchan, npol = beams.shape[:2]
  chans = np.repeat(np.arange(nchan), npol)
  pols = np.tile(np.arange(npol), nchan)
  rows = beams[chans, pols]
  table = fits.BinTableHDU.from_columns(
    [
      fits.Column(name="BMAJ", format="E", unit="arcsec", array=rows[:, 0]),
      fits.Column(name="BMIN", format="E", unit="arcsec", array=rows[:, 1]),
      fits.Column(name="BPA", format="E", unit="deg", array=rows[:, 2]),
      fits.Column(name="CHAN", format="J", array=chans),
      fits.Column(name="POL", format="J", array=pols),
    ],
    name="BEAMS",
  )
  table.header["NCHAN"] = nchan
  table.header["NPOL"] = npol
  return table
