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
