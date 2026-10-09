from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from astropy import units as u

from xarrayfits.backend.fits.hdus import BEAMS_HDU
from xarrayfits.errors import InvalidFitsImage

if TYPE_CHECKING:
  import numpy.typing as npt
  from astropy.io import fits

#: Beam table columns and their default units
BEAM_COLUMNS = (("BMAJ", "arcsec"), ("BMIN", "arcsec"), ("BPA", "deg"))


def read_beam_table(
  hdu_list: fits.HDUList, nchan: int, npol: int
) -> npt.NDArray[np.float64]:
  """Reads the per-plane beams of a CASA style ``BEAMS`` table, in radians,
  with shape ``(nchan, npol, 3)`` in FITS plane order. A table of a single
  channel or polarization is broadcast over the image's planes"""
  try:
    table = hdu_list[BEAMS_HDU]
  except KeyError:
    raise InvalidFitsImage(
      f"The FITS header sets CASAMBM, but the file has no {BEAMS_HDU} table"
    ) from None

  table_nchan = int(table.header["NCHAN"])
  table_npol = int(table.header["NPOL"])
  params = []

  for name, default_unit in BEAM_COLUMNS:
    unit = u.Unit(table.columns[name].unit or default_unit)
    values = np.asarray(table.data[name], dtype=float)
    params.append((values * unit).to(u.rad).value)

  chans = np.asarray(table.data["CHAN"], dtype=int)
  pols = np.asarray(table.data["POL"], dtype=int)
  beams = np.zeros([table_nchan, table_npol, 3])
  beams[chans, pols] = np.stack(params, axis=-1)

  try:
    return np.broadcast_to(beams, (nchan, npol, 3))
  except ValueError:
    raise InvalidFitsImage(
      f"The {BEAMS_HDU} table describes {table_nchan} channels and {table_npol} "
      f"polarizations, but the image has {nchan} channels and {npol} polarizations"
    ) from None


def read_beams(
  hdu_list: fits.HDUList, polarization_order: Sequence[int], nchan: int
) -> npt.NDArray[np.float64] | None:
  """Reads the beams of a FITS Image, from ``BMAJ``, ``BMIN`` and ``BPA``
  or a ``BEAMS`` table, as ``(time, frequency, polarization, 3)`` radians
  in canonical polarization order. Returns ``None`` without beams"""
  header = hdu_list[0].header
  npol = len(polarization_order)

  if "BMAJ" in header:
    beam = [
      (float(header[card]) * u.deg).to(u.rad).value for card in ("BMAJ", "BMIN", "BPA")
    ]
    beams = np.zeros((1, nchan, npol, 3))
    beams[...] = beam
    return beams

  if header.get("CASAMBM", False):
    beams = read_beam_table(hdu_list, nchan, npol)
    return beams[np.newaxis][:, :, list(polarization_order)].copy()

  return None
