from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

from xarrayfits.errors import UnsupportedFitsImage
from xarrayfits.msv4_image_types import CANONICAL_POLARIZATION_ORDER

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


def canonical_order(labels: Sequence[str]) -> List[int]:
  """Returns the permutation that puts polarization labels in canonical
  order. Unknown labels keep their relative order after the known ones"""
  unknown = len(CANONICAL_POLARIZATION_ORDER)
  index = {label: i for i, label in enumerate(CANONICAL_POLARIZATION_ORDER)}
  return sorted(range(len(labels)), key=lambda i: (index.get(labels[i], unknown), i))
