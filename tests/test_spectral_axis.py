import warnings

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from xarrayfits.errors import (
  MissingSpectralFrameWarning,
  UnknownSpectralFrameWarning,
  UnsupportedFitsImage,
)
from xarrayfits.testing.simulator import (
  DEC,
  FREQ,
  RA,
  FitsAxis,
  simulate_fits_image,
  stokes_axis,
)

ENGINE = "xarray-fits:fits"
C = 299792458.0
HI = 1.420405751786e9

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingObservationDateWarning"
)


def open_image(tmp_path, axes=(RA, DEC, FREQ, stokes_axis()), cards=None):
  path = simulate_fits_image(tmp_path / "image.fits", axes=axes, cards=cards)
  return xr.open_dataset(path, engine=ENGINE), path


def test_images_without_a_spectral_axis_get_the_default_channel(
  tmp_path, assert_conforms
):
  with warnings.catch_warnings():
    warnings.simplefilter("error", MissingSpectralFrameWarning)
    ds, path = open_image(tmp_path, axes=(RA, DEC), cards={"SPECSYS": "BARYCENT"})

  attrs = ds.frequency.attrs
  assert ds.frequency.values.tolist() == [1.415e9]
  assert attrs["frame"] == "LSRK"
  assert attrs["rest_frequency"]["data"] == pytest.approx(HI)
  assert attrs["channel_width"]["data"] == 1000.0
  assert ds.velocity.attrs["doppler_type"] == "radio"
  assert_conforms(ds, path)


@pytest.mark.parametrize(
  "cards, rest",
  [
    ({"RESTFRQ": HI}, HI),
    ({"RESTFREQ": HI}, HI),
    ({"RESTWAV": C / HI}, HI),
    ({"RESTFRQ": 0.0}, 0.0),
    ({}, 0.0),
  ],
)
def test_rest_frequency(tmp_path, assert_conforms, cards, rest):
  ds, path = open_image(tmp_path, cards={"SPECSYS": "LSRK", **cards})

  assert ds.frequency.attrs["rest_frequency"]["data"] == pytest.approx(rest)
  assert ("velocity" in ds.coords) == (rest > 0)
  assert_conforms(ds, path)


@pytest.mark.parametrize(
  "axis_type, cards, frame, observer",
  [
    ("FREQ", {"SPECSYS": "LSRK"}, "LSRK", "lsrk"),
    ("FREQ", {"SPECSYS": "BARYCENT"}, "BARY", "BARY"),
    ("FREQ", {"SPECSYS": "HELIOCEN"}, "BARY", "BARY"),
    ("FREQ", {"SPECSYS": "GEOCENTR"}, "GEO", "gcrs"),
    ("FREQ", {"SPECSYS": "TOPOCENT"}, "TOPO", "TOPO"),
    ("FREQ", {"SPECSYS": "SOURCE"}, "REST", "REST"),
    ("FREQ-LSR", {}, "LSRK", "lsrk"),
    ("FREQ-HEL", {}, "BARY", "BARY"),
    ("FREQ", {"VELREF": 2}, "BARY", "BARY"),
    ("FREQ", {"VELREF": 257}, "LSRK", "lsrk"),
    ("FREQ-OBS", {"SPECSYS": "LSRD"}, "LSRD", "lsrd"),
  ],
)
def test_spectral_frame(tmp_path, assert_conforms, axis_type, cards, frame, observer):
  axes = (RA, DEC, FitsAxis(axis_type, 3, 1.415e9, 1.0e3, 2.0, "Hz"))
  ds, path = open_image(tmp_path, axes=axes, cards=cards)
  reference = ds.frequency.attrs["reference_frequency"]

  assert ds.frequency.attrs["frame"] == frame
  assert reference["attrs"]["observer"] == observer
  assert reference["data"] == 1.415e9
  assert_conforms(ds, path)


def test_missing_and_unknown_frames_warn(tmp_path):
  with pytest.warns(MissingSpectralFrameWarning):
    ds, _ = open_image(tmp_path)
  assert ds.frequency.attrs["frame"] == "LSRK"

  with pytest.warns(UnknownSpectralFrameWarning, match="NOWHERE"):
    ds, _ = open_image(tmp_path, cards={"SPECSYS": "NOWHERE", "VELREF": 3})
  assert ds.frequency.attrs["frame"] == "TOPO"


@pytest.mark.parametrize("velref, doppler", [(None, "radio"), (257, "radio"), (1, "z")])
def test_frequency_axis_doppler_type(tmp_path, assert_conforms, velref, doppler):
  cards = {"RESTFRQ": HI, "SPECSYS": "LSRK"}
  if velref is not None:
    cards["VELREF"] = velref
  ds, path = open_image(tmp_path, cards=cards)
  f = ds.frequency.values

  assert ds.velocity.attrs == {
    "units": "m/s",
    "doppler_type": doppler,
    "type": "doppler",
  }
  expected = (1 - f / HI) * C if doppler == "radio" else (HI / f - 1) * C
  assert_allclose(ds.velocity, expected)
  assert_conforms(ds, path)


@pytest.mark.parametrize("cunit", ["GHz", "GHZ", "MHz"])
def test_frequency_units_are_converted_to_hz(tmp_path, assert_conforms, cunit):
  scale = {"GHz": 1e9, "GHZ": 1e9, "MHz": 1e6}[cunit]
  axis = FitsAxis("FREQ", 3, 1.415e9 / scale, 1.0e3 / scale, 2.0, cunit)
  ds, path = open_image(tmp_path, axes=(RA, DEC, axis), cards={"SPECSYS": "LSRK"})

  assert_allclose(ds.frequency, [1.414999e9, 1.415e9, 1.415001e9])
  assert ds.frequency.attrs["channel_width"]["data"] == pytest.approx(1.0e3)
  assert_conforms(ds, path)


def test_optical_velocity_axes(tmp_path, assert_conforms):
  axis = FitsAxis("VOPT", 4, 1000.0, 2.0, 2.0, "km/s")
  ds, path = open_image(
    tmp_path, axes=(RA, DEC, axis), cards={"RESTFRQ": HI, "SPECSYS": "BARYCENT"}
  )
  v = (1000.0 + (np.arange(4) - 1) * 2.0) * 1e3

  assert_allclose(ds.velocity, v)
  assert_allclose(ds.frequency, HI / (1 + v / C))
  assert ds.velocity.attrs["doppler_type"] == "z"
  width = abs(HI / (1 + (1.0e6 + 2.0e3) / C) - HI / (1 + 1.0e6 / C))
  assert ds.frequency.attrs["channel_width"]["data"] == pytest.approx(width)
  assert_conforms(ds, path)


def test_aips_optical_velocity_axes_are_linear_in_frequency(tmp_path, assert_conforms):
  axis = FitsAxis("FELO-HEL", 4, 1.0e6, 2.0e3, 2.0, "m/s")
  ds, path = open_image(tmp_path, axes=(RA, DEC, axis), cards={"RESTFRQ": HI})
  reference = HI / (1 + 1.0e6 / C)
  increment = -2.0e3 * reference / (C + 1.0e6)

  assert ds.frequency.attrs["frame"] == "BARY"
  assert_allclose(ds.frequency, reference + (np.arange(4) - 1) * increment)
  assert ds.frequency.attrs["reference_frequency"]["data"] == pytest.approx(reference)
  assert_conforms(ds, path)


def test_radio_velocity_axes(tmp_path, assert_conforms):
  axis = FitsAxis("VRAD", 4, 1.0e5, -1.0e3, 1.0, "m/s")
  ds, path = open_image(
    tmp_path, axes=(RA, DEC, axis), cards={"RESTFRQ": HI, "SPECSYS": "LSRK"}
  )
  v = 1.0e5 - np.arange(4) * 1.0e3

  assert_allclose(ds.velocity, v)
  assert_allclose(ds.frequency, HI * (1 - v / C))
  assert ds.velocity.attrs["doppler_type"] == "radio"
  assert ds.frequency.attrs["channel_width"]["data"] == pytest.approx(HI * 1e3 / C)
  assert_conforms(ds, path)


def test_velocity_axes_need_a_rest_frequency(tmp_path):
  axis = FitsAxis("VRAD", 4, 1.0e5, -1.0e3, 1.0, "m/s")
  path = simulate_fits_image(
    tmp_path / "image.fits", axes=(RA, DEC, axis), cards={"SPECSYS": "LSRK"}
  )

  with pytest.raises(UnsupportedFitsImage, match="no rest frequency"):
    xr.open_dataset(path, engine=ENGINE)


def test_single_channel_width(tmp_path, assert_conforms):
  axis = FitsAxis("FREQ", 1, 1.4e9, -2.0e6, 1.0, "Hz")
  ds, path = open_image(tmp_path, axes=(RA, DEC, axis), cards={"SPECSYS": "TOPOCENT"})

  assert ds.frequency.values.tolist() == [1.4e9]
  assert ds.frequency.attrs["channel_width"]["data"] == 2.0e6
  assert_conforms(ds, path)
