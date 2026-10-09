import numpy as np
import pytest
import xarray as xr

from xarrayfits.errors import UnsupportedFitsImage
from xarrayfits.testing.simulator import (
  DEC,
  FREQ,
  RA,
  FitsAxis,
  simulate_fits_image,
  stokes_axis,
)

ENGINE = "xarray-fits:fits"

pytestmark = pytest.mark.filterwarnings(
  "ignore::xarrayfits.errors.MissingMetadataWarning"
)


def coordinate_system(tmp_path, **kwargs):
  path = simulate_fits_image(tmp_path / "image.fits", **kwargs)
  ds = xr.open_dataset(path, engine=ENGINE)
  return ds.attrs["coordinate_system_info"], ds, path


@pytest.mark.parametrize(
  "cards, frame, equinox",
  [
    ({}, "icrs", None),
    ({"RADESYS": "ICRS"}, "icrs", None),
    ({"RADESYS": "FK5", "EQUINOX": 2000.0}, "fk5", "j2000.0"),
    ({"RADESYS": "FK5"}, "fk5", "j2000.0"),
    ({"RADECSYS": "FK4"}, "fk4", "b1950.0"),
    ({"RADESYS": "FK4-NO-E", "EQUINOX": 1950.0}, "fk4noterms", "b1950.0"),
    ({"EQUINOX": 1950.0}, "fk4", "b1950.0"),
    ({"EQUINOX": "J2000"}, "fk5", "j2000.0"),
    ({"EPOCH": 2000.0}, "fk5", "j2000.0"),
  ],
)
def test_reference_frame_and_equinox(tmp_path, assert_conforms, cards, frame, equinox):
  info, ds, path = coordinate_system(tmp_path, cards=cards)
  direction = info["reference_direction"]

  assert direction["attrs"]["frame"] == frame
  assert direction["attrs"].get("equinox") == equinox
  assert direction["coords"]["sky_dir_label"]["data"] == ["ra", "dec"]
  assert np.allclose(direction["data"], np.deg2rad([105.0, -40.0]))
  assert_conforms(ds, path)


def test_unsupported_reference_systems_are_rejected(tmp_path):
  path = simulate_fits_image(tmp_path / "image.fits", cards={"RADESYS": "GAPPT"})

  with pytest.raises(UnsupportedFitsImage, match="GAPPT"):
    xr.open_dataset(path, engine=ENGINE)


def test_native_pole_comes_from_the_header(tmp_path, assert_conforms):
  info, ds, path = coordinate_system(
    tmp_path, cards={"LONPOLE": 180.0, "LATPOLE": -40.0}
  )
  pole = info["native_pole_direction"]

  assert pole["attrs"] == {
    "frame": "NATIVE_PROJECTION",
    "type": "location",
    "units": "rad",
  }
  assert np.allclose(pole["data"], np.deg2rad([180.0, -40.0]))
  assert_conforms(ds, path)


def test_native_pole_defaults_to_wcslib(tmp_path):
  info, _, _ = coordinate_system(tmp_path)
  assert np.allclose(info["native_pole_direction"]["data"], [np.pi, np.deg2rad(-40.0)])


@pytest.mark.parametrize(
  "cards, pc",
  [
    ({}, [[1.0, 0.0], [0.0, 1.0]]),
    (
      {"PC1_1": 0.5, "PC1_2": 0.25, "PC2_1": -0.25, "PC2_2": 0.5},
      [[0.5, 0.25], [-0.25, 0.5]],
    ),
    ({"PC001001": 0.5, "PC002002": 2.0}, [[0.5, 0.0], [0.0, 2.0]]),
    ({"PC01_02": 0.1}, [[1.0, 0.1], [0.0, 1.0]]),
  ],
)
def test_pixel_coordinate_transformation_matrix(tmp_path, assert_conforms, cards, pc):
  info, ds, path = coordinate_system(tmp_path, cards=cards)
  assert info["pixel_coordinate_transformation_matrix"] == pc
  assert_conforms(ds, path)


def test_crota_gives_a_rotation_matrix(tmp_path, assert_conforms):
  info, ds, path = coordinate_system(tmp_path, cards={"CROTA2": 30.0})
  rho = np.deg2rad(30.0)
  ratio = (1.0 / 60.0) / (-1.0 / 60.0)
  expected = [[np.cos(rho), -np.sin(rho) * ratio], [np.sin(rho) / ratio, np.cos(rho)]]

  assert np.allclose(info["pixel_coordinate_transformation_matrix"], expected)
  assert_conforms(ds, path)


@pytest.mark.parametrize(
  "axes, cards, parameters",
  [
    ((RA, DEC, FREQ, stokes_axis()), {}, [0.0, 0.0]),
    ((RA, DEC, FREQ, stokes_axis()), {"PV2_1": 0.0, "PV2_2": 0.5}, [0.0, 0.5]),
    ((RA, DEC, FREQ, stokes_axis()), {"PV2_2": 0.5}, [0.0, 0.5]),
    (
      (
        FitsAxis("RA---ZPN", 6, 105.0, -1.0 / 60.0, 4.0, "deg"),
        FitsAxis("DEC--ZPN", 5, 80.0, 1.0 / 60.0, 3.0, "deg"),
        FREQ,
      ),
      {"PV2_1": 1.0, "PV2_3": 0.05},
      [0.0, 1.0, 0.0, 0.05],
    ),
  ],
)
def test_projection_parameters(tmp_path, assert_conforms, axes, cards, parameters):
  info, ds, path = coordinate_system(tmp_path, axes=axes, cards=cards)
  assert info["projection_parameters"] == parameters
  assert_conforms(ds, path)


def test_projection_must_match(tmp_path):
  axes = (RA, FitsAxis("DEC--TAN", 5, -40.0, 1.0 / 60.0, 3.0, "deg"), FREQ)
  path = simulate_fits_image(tmp_path / "image.fits", axes=axes)

  with pytest.raises(UnsupportedFitsImage, match="Projections"):
    xr.open_dataset(path, engine=ENGINE)


def test_declination_first_images_conform(tmp_path, assert_conforms):
  info, ds, path = coordinate_system(tmp_path, axes=(DEC, RA, FREQ))
  assert info["projection"] == "SIN"
  assert np.all(np.diff(ds.l) < 0)
  assert_conforms(ds, path)
