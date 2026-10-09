import numpy as np
import pytest
import xarray as xr

from xarrayfits.errors import (
  InvalidObservationDateWarning,
  MissingObservationDateWarning,
  UnknownTimeScaleWarning,
)
from xarrayfits.testing.simulator import simulate_fits_image

ENGINE = "xarray-fits:fits"

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingSpectralFrameWarning"
)

#: Cards of a FITS Image exported by casacore
CASA_CARDS = {
  "BTYPE": "Intensity",
  "OBJECT": "TEST",
  "BUNIT": "Jy/beam",
  "EQUINOX": 2000.0,
  "RADESYS": "FK5",
  "LONPOLE": 180.0,
  "LATPOLE": -40.0,
  "PC1_1": 1.0,
  "PC2_2": 1.0,
  "RESTFRQ": 1.420405751786e9,
  "SPECSYS": "LSRK",
  "VELREF": 257,
  "TELESCOP": "ALMA",
  "OBSERVER": "Karl Jansky",
  "DATE-OBS": "2000-01-01T00:00:00.000",
  "TIMESYS": "UTC",
  "OBSRA": 105.5,
  "OBSDEC": -40.5,
  "OBSGEO-X": 2.225142180269e06,
  "OBSGEO-Y": -5.440307370349e06,
  "OBSGEO-Z": -2.481029851874e06,
  "ORIGIN": "casacore",
  "DATE": "2026-10-09T00:00:00",
  "INSTRUME": "ALMA band 3",
}


def open_image(tmp_path, cards=None, remove=()):
  path = simulate_fits_image(tmp_path / "image.fits", cards=cards, remove=remove)
  return xr.open_dataset(path, engine=ENGINE), path


def test_casa_exported_images_conform(tmp_path, assert_conforms):
  ds, path = open_image(tmp_path, cards=CASA_CARDS)
  assert_conforms(ds, path)


@pytest.mark.parametrize(
  "cards, mjd, scale",
  [
    ({"DATE-OBS": "2000-01-01T00:00:00.000"}, 51544.0, "utc"),
    ({"DATE-OBS": "2000-01-01T12:00:00", "TIMESYS": "TAI"}, 51544.5, "tai"),
    ({"DATE-OBS": "2000-01-01", "TIMESYS": "TDT"}, 51544.0, "tt"),
    ({"DATE-OBS": "02/01/95"}, 49719.0, "utc"),
    ({"MJD-OBS": 51544.25}, 51544.25, "utc"),
    ({"MJD-OBS": 51544.25, "TIMESYS": "GMT"}, 51544.25, "utc"),
  ],
)
def test_observation_date(tmp_path, assert_conforms, cards, mjd, scale):
  ds, path = open_image(tmp_path, cards=cards)
  attrs = {"units": "d", "scale": scale, "format": "mjd", "type": "time"}

  assert ds.time.values.tolist() == [mjd]
  assert ds.time.attrs == attrs
  assert ds.SKY.attrs["obsdate"] == {"attrs": attrs, "data": mjd, "dims": []}
  assert_conforms(ds, path)


@pytest.mark.parametrize("cards", [{}, {"MJD-OBS": 0.0}])
def test_unknown_observation_dates_are_mjd_zero(tmp_path, cards):
  with pytest.warns(MissingObservationDateWarning):
    ds, _ = open_image(tmp_path, cards=cards)

  assert ds.time.values.tolist() == [0.0]
  assert "obsdate" not in ds.SKY.attrs


def test_uninterpretable_dates_and_time_scales_warn(tmp_path):
  with pytest.warns(InvalidObservationDateWarning, match="tomorrow"):
    ds, _ = open_image(tmp_path, cards={"DATE-OBS": "tomorrow", "MJD-OBS": 51544.0})
  assert ds.time.values.tolist() == [51544.0]

  with pytest.warns(UnknownTimeScaleWarning, match="LAST"):
    ds, _ = open_image(tmp_path, cards={"MJD-OBS": 51544.0, "TIMESYS": "LAST"})
  assert ds.time.attrs["scale"] == "utc"


@pytest.mark.parametrize(
  "btype, sub_type",
  [
    ("Intensity", "Intensity"),
    ("spectral_index", "SpectralIndex"),
    ("Column Density", "ColumnDensity"),
    ("Undefined", None),
    ("Brightness", None),
    (None, None),
  ],
)
def test_sky_attributes(tmp_path, assert_conforms, btype, sub_type):
  cards = {"BUNIT": "Jy/beam", "OBJECT": "3C147", "OBSERVER": "Karl Jansky"}
  if btype is not None:
    cards["BTYPE"] = btype
  ds, path = open_image(tmp_path, cards={**cards, "MJD-OBS": 51544.0})
  attrs = ds.SKY.attrs

  assert attrs["type"] == "sky"
  assert attrs.get("sub_type") == sub_type
  assert attrs["units"] == "Jy/beam"
  assert attrs["object_name"] == "3C147"
  assert attrs["observer"] == "Karl Jansky"
  assert attrs["description"] is None
  assert_conforms(ds, path)


def test_pointing_center(tmp_path):
  with pytest.warns(MissingObservationDateWarning):
    ds, _ = open_image(tmp_path)
  assert np.allclose(ds.SKY.attrs["pointing_center"]["data"], np.deg2rad([105, -40]))

  ds, _ = open_image(
    tmp_path,
    cards={"OBSRA": 106.0, "OBSDEC": -41.0, "MJD-OBS": 1.0, "RADESYS": "FK5"},
  )
  center = ds.SKY.attrs["pointing_center"]
  assert np.allclose(center["data"], np.deg2rad([106, -41]))
  assert center["attrs"] == {"frame": "fk5", "type": "sky_coord", "units": "rad"}


def test_telescope(tmp_path, assert_conforms):
  ds, path = open_image(tmp_path, cards={"MJD-OBS": 1.0})
  assert ds.SKY.attrs["telescope"] == {"name": "UNKNOWN"}

  xyz = np.array([2.225142180269e06, -5.440307370349e06, -2.481029851874e06])
  cards = {"MJD-OBS": 1.0, "TELESCOP": "ALMA"}
  cards.update(zip(("OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z"), xyz))
  ds, path = open_image(tmp_path, cards=cards)
  telescope = ds.SKY.attrs["telescope"]
  r = np.linalg.norm(xyz)

  assert telescope["name"] == "ALMA"
  assert np.allclose(
    telescope["direction"]["data"], [np.arctan2(xyz[1], xyz[0]), np.arcsin(xyz[2] / r)]
  )
  assert np.allclose(telescope["distance"]["data"], [r])
  assert telescope["direction"]["attrs"]["frame"] == "ITRF"
  assert_conforms(ds, path)


def test_user_cards(tmp_path):
  cards = {"MJD-OBS": 1.0, "INSTRUME": "MeerKAT L", "ORIGIN": "x", "PV2_1": 0.0}
  ds, _ = open_image(tmp_path, cards=cards)
  user = ds.SKY.attrs["user"]

  assert user["instrume"] == "MeerKAT L"
  assert not {"origin", "pv2_1", "mjd-obs", "naxis1", "ctype1", "bitpix"} & set(user)
