from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from xarrayfits.errors import MissingSpectralFrameWarning
from xarrayfits.measures import quantity, spectral_coord

if TYPE_CHECKING:
  import numpy.typing as npt
  from astropy.io.fits import Header

  from xarrayfits.backend.fits.axes import AxisLayout

#: Rest frequency recorded when it is unknown
UNKNOWN_REST_FREQUENCY_HZ = 0.0

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


@dataclasses.dataclass(frozen=True, slots=True)
class Spectral:
  """The spectral axis of a FITS Image"""

  frequency: npt.NDArray[np.float64]
  """Channel frequencies in Hz"""
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


def spectral_frame(header: Header) -> str:
  """Returns the casacore spectral frame of the image"""
  warnings.warn(
    "The FITS header gives no spectral reference frame (SPECSYS, a "
    "frame suffix of the spectral CTYPE or VELREF); assuming LSRK",
    MissingSpectralFrameWarning,
    stacklevel=2,
  )
  return "LSRK"


def read_spectral(header: Header, layout: AxisLayout) -> Spectral:
  """Reads the spectral axis of the primary HDU header"""
  assert layout.frequency is not None
  axis = layout.axes[layout.frequency]

  return Spectral(
    frequency=axis.linear_values(),
    frame=spectral_frame(header),
    rest_frequency=UNKNOWN_REST_FREQUENCY_HZ,
    reference_frequency=float(axis.crval),
    channel_width=abs(float(axis.cdelt)),
  )
