from __future__ import annotations

import warnings

from astropy.io import fits

from xarrayfits.errors import IgnoredHduWarning, InvalidFitsImage, UnsupportedFitsImage

#: Name of the extension holding per-plane beams
BEAMS_HDU = "BEAMS"


def primary_header(hdu_list: fits.HDUList) -> fits.Header:
  """Validates the HDUs of a FITS Image and returns its primary HDU header.

  Raises:
    UnsupportedFitsImage: If an HDU is compressed or the primary HDU
      pixels are scaled.
    InvalidFitsImage: If the primary HDU holds no image.
  """
  for i, hdu in enumerate(hdu_list):
    if isinstance(hdu, fits.CompImageHDU):
      raise UnsupportedFitsImage(
        f"HDU {i} ({hdu.name}) is a compressed image, which is not supported. "
        f"Decompress the FITS file, for example with funpack"
      )

  primary = hdu_list[0]
  header = primary.header
  scale = header.get("BSCALE", 1.0)
  zero = header.get("BZERO", 0.0)

  if not (scale == 1.0 and zero == 0.0):
    raise UnsupportedFitsImage(
      f"Scaled FITS pixels (BSCALE/BZERO set) are not supported. "
      f"BZERO={zero}, BSCALE={scale}"
    )

  if not primary.is_image or header.get("NAXIS", 0) < 2:
    raise InvalidFitsImage(
      "The primary HDU of the FITS file holds no image (it has fewer than "
      "two axes, or holds random groups data such as UVFITS); only images "
      "in the primary HDU are supported"
    )

  for hdu in hdu_list[1:]:
    if hdu.name != BEAMS_HDU:
      warnings.warn(
        f"Ignoring the FITS extension HDU {hdu.name!r}: only the image in the "
        f"primary HDU and a {BEAMS_HDU} table of per-plane beams are read",
        IgnoredHduWarning,
        stacklevel=2,
      )

  return header
