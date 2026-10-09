import numpy as np
import pytest
import xarray as xr

from xarrayfits.errors import IncompatibleImages, UnknownRoleWarning
from xarrayfits.testing.simulator import (
  DEC,
  FREQ,
  RA,
  FitsAxis,
  simulate_fits_image,
  stokes_axis,
)

ENGINE = "xarray-fits:fits"
DATED = {"MJD-OBS": 51544.0, "SPECSYS": "LSRK", "RADESYS": "FK5"}
BEAM = {"BMAJ": 1.0 / 3600.0, "BMIN": 0.5 / 3600.0, "BPA": 30.0}
ONE_PIXEL = (
  FitsAxis("RA---SIN", 1, 105.0, -1.0 / 60.0, 1.0, "deg"),
  FitsAxis("DEC--SIN", 1, -40.0, 1.0 / 60.0, 1.0, "deg"),
  FREQ,
  stokes_axis(),
)


def tclean_products(directory, names=None, cards=None):
  names = names or ["image", "psf", "pb", "residual", "model", "sumwt", "mask"]
  paths = []
  for name in names:
    axes = ONE_PIXEL if name == "sumwt" else (RA, DEC, FREQ, stokes_axis())
    extra = BEAM if name in ("image", "psf") else {}
    paths.append(
      simulate_fits_image(
        directory / f"cube.{name}.fits",
        axes=axes,
        cards={**DATED, **extra, **(cards or {})},
      )
    )
  return paths


def test_tclean_products_open_as_one_image_dataset(tmp_path, assert_conforms):
  paths = tclean_products(tmp_path)
  ds = xr.open_dataset(paths, engine=ENGINE)

  assert set(ds.data_vars) >= {
    "SKY",
    "POINT_SPREAD_FUNCTION",
    "PRIMARY_BEAM",
    "SKY_RESIDUAL",
    "SKY_MODEL",
    "VISIBILITY_NORMALIZATION",
    "MASK",
    "BEAM_FIT_PARAMS_SKY",
    "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
  }
  assert ds.VISIBILITY_NORMALIZATION.dims == ("time", "frequency", "polarization")
  assert ds.SKY_RESIDUAL.attrs["type"] == "sky"
  assert ds.POINT_SPREAD_FUNCTION.attrs["type"] == "point_spread_function"

  shared = {
    "point_spread_function": "POINT_SPREAD_FUNCTION",
    "primary_beam": "PRIMARY_BEAM",
    "visibility_normalization": "VISIBILITY_NORMALIZATION",
    "mask": "MASK",
    "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
  }
  groups = ds.attrs["data_groups"]
  assert groups["base"] == {
    "sky": "SKY",
    "flag": "FLAG_SKY",
    "beam_fit_params_sky": "BEAM_FIT_PARAMS_SKY",
    **shared,
  }
  assert groups["residual"] == {
    "sky": "SKY_RESIDUAL",
    "flag": "FLAG_SKY_RESIDUAL",
    **shared,
  }
  assert groups["model"]["sky"] == "SKY_MODEL"
  assert_conforms(ds, paths)


def test_roles_can_be_given_explicitly(tmp_path, assert_conforms):
  a = simulate_fits_image(tmp_path / "a.fits", cards=DATED)
  b = simulate_fits_image(tmp_path / "b.fits", cards=DATED)
  store = {"sky": a, "psf": b}
  ds = xr.open_dataset(store, engine=ENGINE)

  assert {"SKY", "POINT_SPREAD_FUNCTION"} <= set(ds.data_vars)
  assert_conforms(ds, store)


@pytest.mark.parametrize(
  "name, role",
  [
    ("target.psf.fits", "POINT_SPREAD_FUNCTION"),
    ("target.residual", "SKY_RESIDUAL"),
    ("out.base.point_spread_function.fits", "POINT_SPREAD_FUNCTION"),
    ("ngc1234_pb.fits", "SKY"),
    ("ngc1234_pb", "PRIMARY_BEAM"),
    ("cube_residual.im", "SKY_RESIDUAL"),
    ("my_mask.im", "SKY"),
    ("simulation_mask", "MASK"),
    ("target.image.fits", "SKY"),
  ],
)
def test_roles_come_from_file_names(tmp_path, name, role):
  path = simulate_fits_image(tmp_path / name, cards=DATED)
  ds = xr.open_dataset([path], engine=ENGINE)
  assert role in ds.data_vars

  single = xr.open_dataset(path, engine=ENGINE)
  assert role in single.data_vars


def test_unknown_roles_are_sky_with_a_warning(tmp_path):
  path = simulate_fits_image(tmp_path / "cube.fts", cards=DATED)

  with pytest.warns(UnknownRoleWarning, match="names no Role"):
    ds = xr.open_dataset(path, engine=ENGINE)
  assert "SKY" in ds


def test_duplicate_roles_are_rejected(tmp_path):
  paths = [
    simulate_fits_image(tmp_path / "a.image.fits", cards=DATED),
    simulate_fits_image(tmp_path / "b.image.fits", cards=DATED),
  ]
  with pytest.raises(ValueError, match="Duplicate Role SKY"):
    xr.open_dataset(paths, engine=ENGINE)


def test_large_sum_of_weights_is_rejected(tmp_path):
  path = simulate_fits_image(tmp_path / "cube.sumwt.fits", cards=DATED)
  with pytest.raises(ValueError, match="direction axes of one pixel"):
    xr.open_dataset(path, engine=ENGINE)


@pytest.mark.filterwarnings("ignore::xarrayfits.errors.MissingObservationDateWarning")
def test_images_without_a_date_take_the_others(tmp_path, assert_conforms):
  sky = simulate_fits_image(tmp_path / "cube.image.fits", cards=DATED)
  psf = simulate_fits_image(
    tmp_path / "cube.psf.fits", cards={"SPECSYS": "LSRK", "RADESYS": "FK5"}
  )

  for store in ([sky, psf], [psf, sky]):
    ds = xr.open_dataset(store, engine=ENGINE)
    assert ds.time.values.tolist() == [51544.0]
    assert "obsdate" not in ds.POINT_SPREAD_FUNCTION.attrs
    assert_conforms(ds, store)


def test_coordinates_are_snapped_within_tolerance(tmp_path, assert_conforms):
  sky = simulate_fits_image(tmp_path / "cube.image.fits", cards=DATED)
  psf = simulate_fits_image(
    tmp_path / "cube.psf.fits", cards={**DATED, "CRVAL3": 1.415e9 + 1e-4}
  )
  store = [sky, psf]
  ds = xr.open_dataset(store, engine=ENGINE)

  assert ds.frequency.values.tolist() == [1.414999e9, 1.415e9, 1.415001e9]
  assert_conforms(ds, store)


def test_differing_coordinates_are_rejected(tmp_path):
  sky = simulate_fits_image(tmp_path / "cube.image.fits", cards=DATED)
  psf = simulate_fits_image(
    tmp_path / "cube.psf.fits", cards={**DATED, "CRVAL3": 1.416e9}
  )
  with pytest.raises(IncompatibleImages, match="frequency coordinate"):
    xr.open_dataset([sky, psf], engine=ENGINE)

  stokes = simulate_fits_image(
    tmp_path / "cube.pb.fits",
    axes=(RA, DEC, FREQ, stokes_axis(1.0, 1.0, 2)),
    cards=DATED,
  )
  with pytest.raises(IncompatibleImages, match="polarization coordinate"):
    xr.open_dataset([sky, stokes], engine=ENGINE)


def test_multi_file_image_datasets_read_lazily(tmp_path):
  paths = tclean_products(tmp_path, names=["image", "psf", "sumwt"])
  ds = xr.open_dataset(paths, engine=ENGINE, chunks={})
  eager = xr.open_dataset(paths, engine=ENGINE).load()

  assert ds.VISIBILITY_NORMALIZATION.chunks == ((1,), (1, 1, 1), (1,))
  xr.testing.assert_identical(ds.compute(), eager)
  assert np.isfinite(eager.POINT_SPREAD_FUNCTION).all()
