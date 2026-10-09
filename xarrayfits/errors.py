class MissingMetadataWarning(UserWarning):
  """Warning raised when a FITS Image lacks metadata and a default is assumed"""


class MissingSpectralFrameWarning(MissingMetadataWarning):
  """Warning raised when a FITS Image has no spectral reference frame"""


class MissingObservationDateWarning(MissingMetadataWarning):
  """Warning raised when a FITS Image has no observation date"""


class InvalidFitsImage(ValueError):
  """Raised when a FITS file holds no image that can be read"""


class UnsupportedFitsImage(ValueError):
  """Raised when a FITS Image uses a feature that is not supported"""


class IgnoredHduWarning(UserWarning):
  """Warning raised when a FITS extension HDU is not read"""


class UnknownMetadataWarning(UserWarning):
  """Warning raised when FITS Image metadata cannot be interpreted
  and is ignored"""


class UnknownSpectralFrameWarning(UnknownMetadataWarning):
  """Warning raised when a FITS Image's spectral reference frame is unknown"""


class UnknownTimeScaleWarning(UnknownMetadataWarning):
  """Warning raised when a FITS Image's time scale is unknown"""


class InvalidObservationDateWarning(UnknownMetadataWarning):
  """Warning raised when a FITS Image's observation date cannot be interpreted"""
