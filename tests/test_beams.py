import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from xarrayfits.errors import InvalidFitsImage
from xarrayfits.testing.simulator import (
  DEC,
  FREQ,
  RA,
  simulate_fits_image,
  stokes_axis,
)

ENGINE = "xarray-fits:fits"
ARCSEC = np.deg2rad(1.0 / 3600.0)
CIRCULAR = (RA, DEC, FREQ, stokes_axis(-1.0, -1.0, 4))

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingMetadataWarning"
)


def test_single_beams_cover_every_plane(tmp_path, assert_conforms):
  cards = {"BMAJ": 1.0 / 3600.0, "BMIN": 0.5 / 3600.0, "BPA": 30.0}
  path = simulate_fits_image(tmp_path / "image.fits", axes=CIRCULAR, cards=cards)
  ds = xr.open_dataset(path, engine=ENGINE)
  beams = ds.BEAM_FIT_PARAMS_SKY

  assert beams.dims == ("time", "frequency", "polarization", "beam_params_label")
  assert beams.shape == (1, 3, 4, 3)
  assert beams.dtype == np.float64
  assert beams.attrs == {"units": "rad", "type": "beam_fit_params_sky"}
  assert_allclose(beams.sel(beam_params_label="major"), ARCSEC)
  assert_allclose(beams.sel(beam_params_label="minor"), 0.5 * ARCSEC)
  assert_allclose(beams.sel(beam_params_label="pa"), np.deg2rad(30.0))
  assert ds.SKY.attrs["beam_fit_params"] == "BEAM_FIT_PARAMS_SKY"
  assert ds.attrs["data_groups"]["base"]["beam_fit_params_sky"] == (
    "BEAM_FIT_PARAMS_SKY"
  )
  assert_conforms(ds, path)


def test_beam_tables_give_per_plane_beams(tmp_path, assert_conforms):
  # (nchan, npol, [bmaj, bmin, bpa]) with FITS planes RR, LL, RL, LR
  beams = np.zeros((3, 4, 3))
  beams[..., 0] = np.arange(3)[:, None] + 10 * np.arange(4)[None, :] + 1
  beams[..., 1] = 0.5
  beams[..., 2] = 45.0
  path = simulate_fits_image(tmp_path / "image.fits", axes=CIRCULAR, beams=beams)
  ds = xr.open_dataset(path, engine=ENGINE)
  major = ds.BEAM_FIT_PARAMS_SKY.sel(beam_params_label="major")

  for label, plane in {"RR": 0, "LL": 1, "RL": 2, "LR": 3}.items():
    assert_allclose(
      major.sel(polarization=label)[0], (np.arange(3) + 10 * plane + 1) * ARCSEC
    )
  assert_conforms(ds, path)


def test_single_channel_beam_tables_broadcast(tmp_path, assert_conforms):
  beams = np.array([[[2.0, 1.0, 10.0]]])
  path = simulate_fits_image(tmp_path / "image.fits", beams=beams)
  ds = xr.open_dataset(path, engine=ENGINE)

  assert_allclose(ds.BEAM_FIT_PARAMS_SKY.sel(beam_params_label="major"), 2.0 * ARCSEC)
  assert_conforms(ds, path)


def test_mismatched_beam_tables_are_rejected(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits", beams=np.ones((2, 1, 3)))

  with pytest.raises(InvalidFitsImage, match="BEAMS table describes 2 channels"):
    xr.open_dataset(path, engine=ENGINE)


def test_images_without_beams_have_no_beam_variable(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits")
  ds = xr.open_dataset(path, engine=ENGINE)

  assert "BEAM_FIT_PARAMS_SKY" not in ds
  assert "beam_fit_params" not in ds.SKY.attrs
  assert "beam_params_label" in ds.coords
