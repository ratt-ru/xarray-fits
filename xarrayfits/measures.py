"""Builders of the serialised measures held in Image Dataset attributes"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

MeasureT = Dict[str, Any]

#: Conversion factor from degrees to radians
DEG_TO_RAD = np.pi / 180.0

#: Spectral frames without an astropy equivalent, which keep their
#: casacore name as the ``reference_frequency`` observer
CASACORE_NAMED_OBSERVERS = ("TOPO", "BARY", "REST", "GALACTO", "LGROUP", "CMB")


def to_python(value: Any) -> Any:
  """Converts numpy scalars and arrays into python floats and lists"""
  if hasattr(value, "tolist"):
    return value.tolist()
  if isinstance(value, (list, tuple)):
    return [to_python(v) for v in value]
  return value


def quantity(value: Any, units: str) -> MeasureT:
  """Returns a quantity measure"""
  return {
    "data": to_python(value),
    "dims": [],
    "attrs": {"units": units, "type": "quantity"},
  }


def sky_coord(data: Sequence[float], units: str, frame: str) -> MeasureT:
  """Returns a sky coordinate measure on the ``sky_dir_label`` dimension"""
  labels = ["lon", "lat"] if frame.lower() == "galactic" else ["ra", "dec"]
  return {
    "attrs": {"frame": frame.lower(), "type": "sky_coord", "units": units},
    "data": to_python(data),
    "dims": "sky_dir_label",
    "coords": {"sky_dir_label": {"data": labels, "dims": "sky_dir_label"}},
  }


def direction_location(data: Sequence[float], units: str, frame: str) -> MeasureT:
  """Returns a location measure on the ``ellipsoid_dir_label`` dimension"""
  return {
    "attrs": {"frame": frame.upper(), "type": "location", "units": units},
    "data": to_python(data),
    "dims": "ellipsoid_dir_label",
    "coords": {
      "ellipsoid_dir_label": {"data": ["lon", "lat"], "dims": "ellipsoid_dir_label"}
    },
  }


def spectral_coord(value: float, units: str, observer: str) -> MeasureT:
  """Returns a spectral coordinate measure"""
  if observer not in CASACORE_NAMED_OBSERVERS:
    observer = observer.lower()

  return {
    "attrs": {"units": units, "observer": observer, "type": "spectral_coord"},
    "data": to_python(value),
    "dims": [],
  }


def time_attrs(scale: str) -> Dict[str, str]:
  """Returns the attributes of a time measure in MJD days"""
  return {"units": "d", "scale": scale.lower(), "format": "mjd", "type": "time"}


def time_measure(mjd: float, scale: str) -> MeasureT:
  """Returns a time measure in MJD days"""
  return {"attrs": time_attrs(scale), "data": to_python(mjd), "dims": []}
