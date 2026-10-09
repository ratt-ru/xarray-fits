from __future__ import annotations

from typing import TYPE_CHECKING, List, TypeAlias

import numpy as np
from rarg_python_patterns.multiton import Multiton

from xarrayfits.backend.fits.axes import AxisLayout, dtype_from_bitpix, read_axes
from xarrayfits.backend.fits.coordinate_system import (
  CoordinateSystem,
  read_coordinate_system,
  to_radians,
)
from xarrayfits.backend.fits.file import FitsFile
from xarrayfits.backend.fits.hdus import primary_header
from xarrayfits.backend.fits.observation import Observation, read_observation
from xarrayfits.backend.fits.polarization import canonical_order, read_polarizations
from xarrayfits.backend.fits.spectral import Spectral, read_spectral

if TYPE_CHECKING:
  import numpy.typing as npt

FitsFileFactory: TypeAlias = Multiton[FitsFile]
"""Multiton producing an open FITS file"""

FitsImageStructureFactory: TypeAlias = Multiton["FitsImageStructure"]
"""Multiton producing the parsed structure of a FITS Image"""


class FitsImageStructure:
  """The parsed primary HDU header of a FITS Image. Polarizations are in
  canonical order, and ``polarization_order`` gives the FITS plane
  of each"""

  __slots__ = (
    "layout",
    "dtype",
    "coordinate_system",
    "polarizations",
    "polarization_order",
    "spectral",
    "observation",
  )

  layout: AxisLayout
  dtype: np.dtype
  coordinate_system: CoordinateSystem
  polarizations: List[str]
  polarization_order: List[int]
  spectral: Spectral
  observation: Observation

  def __init__(self, file_factory: FitsFileFactory):
    header = primary_header(file_factory.instance.hdu_list)
    self.layout = read_axes(header)
    self.dtype = dtype_from_bitpix(header)
    self.coordinate_system = read_coordinate_system(header, self.layout)
    fits_polarizations = read_polarizations(self.layout)
    self.polarization_order = canonical_order(fits_polarizations)
    self.polarizations = [fits_polarizations[i] for i in self.polarization_order]
    self.spectral = read_spectral(header, self.layout)
    self.observation = read_observation(header, self.layout, self.coordinate_system)

  def direction_values(self, fits_axis: int) -> npt.NDArray[np.float64]:
    """Returns the projection plane coordinates of a celestial axis in radians"""
    axis = self.layout.axes[fits_axis]
    cdelt = to_radians(axis.cdelt, axis.cunit)
    return 0.0 + (np.arange(axis.naxis) - axis.crpix) * cdelt
