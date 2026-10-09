from __future__ import annotations

from typing import TYPE_CHECKING, List

from xarrayfits.errors import UnsupportedFitsImage

if TYPE_CHECKING:
  from xarrayfits.backend.fits.axes import AxisLayout

#: FITS ``STOKES`` axis code -> polarization label
FITS_STOKES_LABELS = {
  1: "I",
  2: "Q",
  3: "U",
  4: "V",
  -1: "RR",
  -2: "LL",
  -3: "RL",
  -4: "LR",
  -5: "XX",
  -6: "YY",
  -7: "XY",
  -8: "YX",
}


def read_polarizations(layout: AxisLayout) -> List[str]:
  """Returns the polarization labels of the ``STOKES`` axis, in FITS order"""
  if layout.polarization is None:
    return ["I"]

  axis = layout.axes[layout.polarization]
  labels = []

  for pixel in range(axis.naxis):
    code = int(round(axis.crval + axis.cdelt * (pixel - axis.crpix)))
    try:
      labels.append(FITS_STOKES_LABELS[code])
    except KeyError:
      raise UnsupportedFitsImage(
        f"FITS STOKES axis value {code} (pixel {pixel + 1}) is not a "
        f"supported Stokes or correlation code ({sorted(FITS_STOKES_LABELS)})"
      ) from None

  return labels
