"""Roles of the Images in an Image Dataset"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Mapping

from xarrayfits.errors import UnknownRoleWarning

#: Role of an Image named by the last dot separated token of its file name
#: (after removing a ``.fits`` extension)
ROLE_OF_NAME_TOKEN = {
  "image": "SKY",
  "im": "SKY",
  "sky": "SKY",
  "fits": "SKY",
  "aperture": "APERTURE",
  "psf": "POINT_SPREAD_FUNCTION",
  "point_spread_function": "POINT_SPREAD_FUNCTION",
  "pb": "PRIMARY_BEAM",
  "primary_beam": "PRIMARY_BEAM",
  "residual": "SKY_RESIDUAL",
  "model": "SKY_MODEL",
  "dirty": "SKY_DIRTY",
  "mask": "MASK",
  "sumwt": "VISIBILITY_NORMALIZATION",
  "visibility_normalization": "VISIBILITY_NORMALIZATION",
  "visibility": "VISIBILITY",
  "uv_sampling": "UV_SAMPLING",
  "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
  "aperture_normalization": "APERTURE_NORMALIZATION",
}

#: Last name tokens that name no particular Role
GENERIC_NAME_TOKENS = frozenset({"image", "im", "sky", "fits"})

#: Substrings of a file name and the Roles they give, in order of
#: precedence, for names whose last token names no particular Role
ROLE_OF_NAME_SUBSTRING = (
  ("fits", "SKY"),
  ("image", "SKY"),
  ("sky", "SKY"),
  ("point_spread_function", "POINT_SPREAD_FUNCTION"),
  ("psf", "POINT_SPREAD_FUNCTION"),
  ("model", "SKY_MODEL"),
  ("residual", "SKY_RESIDUAL"),
  ("dirty", "SKY_DIRTY"),
  ("primary_beam", "PRIMARY_BEAM"),
  ("pb", "PRIMARY_BEAM"),
  ("aperture_normalization", "APERTURE_NORMALIZATION"),
  ("aperture", "APERTURE"),
  ("visibility_normalization", "VISIBILITY_NORMALIZATION"),
  ("visibility", "VISIBILITY"),
  ("sumwt", "VISIBILITY_NORMALIZATION"),
  ("uv_sampling_normalization", "UV_SAMPLING_NORMALIZATION"),
  ("uv_sampling", "UV_SAMPLING"),
)

#: Other names of Roles, which may be given as keys of a mapping of Roles
ROLE_ALIASES = {
  "IMAGE": "SKY",
  "PSF": "POINT_SPREAD_FUNCTION",
  "PB": "PRIMARY_BEAM",
  "SUMWT": "VISIBILITY_NORMALIZATION",
  "MODEL": "SKY_MODEL",
  "RESIDUAL": "SKY_RESIDUAL",
  "DIRTY": "SKY_DIRTY",
  "MASK_DECONVOLVE": "MASK",
}

#: Role of an Image without directions, which has no l and m dimensions
VISIBILITY_NORMALIZATION = "VISIBILITY_NORMALIZATION"


def is_sky(role: str) -> bool:
  """Returns whether the Role is a version of the sky image"""
  return "sky" in role.lower()


def role_from_substrings(name: str) -> str | None:
  for substring, role in ROLE_OF_NAME_SUBSTRING:
    if substring in name:
      return role
  if "im" in name.split("."):
    return "SKY"
  if "mask" in name:
    return "MASK"
  return None


def role_from_name(url: str) -> str:
  """Returns the Role of an Image from its file name. Names that name
  no Role are sky images, with a warning"""
  full_name = os.path.basename(os.path.normpath(url)).lower()
  name = full_name.removesuffix(".fits")
  last_token = name.split(".")[-1]

  if last_token in ROLE_OF_NAME_TOKEN and last_token not in GENERIC_NAME_TOKENS:
    return ROLE_OF_NAME_TOKEN[last_token]

  if (role := role_from_substrings(full_name)) is not None:
    return role

  warnings.warn(
    f"The name of {url} names no image role: it is opened as a sky image "
    f"(SKY). To open it as another image, pass a dict such as "
    f"{{'point_spread_function': path}}",
    UnknownRoleWarning,
    stacklevel=2,
  )
  return "SKY"


def normalise_role(role: str) -> str:
  """Returns the Role of a Role name or alias"""
  role = role.upper()
  return ROLE_ALIASES.get(role, role)


def resolve_roles(filename_or_obj: Any) -> Dict[str, str]:
  """Returns the Role of each FITS Image, given a path, a list of
  paths or a mapping of Roles to paths"""
  if isinstance(filename_or_obj, (str, os.PathLike)):
    filename_or_obj = [filename_or_obj]

  if isinstance(filename_or_obj, Mapping):
    items = [(normalise_role(r), os.fspath(p)) for r, p in filename_or_obj.items()]
  elif isinstance(filename_or_obj, (list, tuple)):
    paths = [os.fspath(p) for p in filename_or_obj]
    items = [(role_from_name(p), p) for p in paths]
  else:
    raise TypeError(
      f"{type(filename_or_obj)} is not a path, a list of paths "
      f"or a mapping of roles to paths"
    )

  roles: Dict[str, str] = {}

  for role, path in items:
    if role in roles:
      raise ValueError(
        f"Duplicate Role {role} of {path} and {roles[role]}. Label the "
        f"FITS Images with their roles, for example "
        f"{{'sky': 'a.fits', 'sky_residual': 'b.fits'}}"
      )
    roles[role] = path

  return roles


def data_groups(roles: List[str]) -> Dict[str, Dict[str, str]]:
  """Returns a Data Group for each sky (or aperture) Image, holding it.
  The other Images are added to every Data Group"""
  groups: Dict[str, Dict[str, str]] = {}

  for role in roles:
    if is_sky(role):
      name = "base" if role == "SKY" else role.lower().replace("sky_", "")
      groups[name] = {"sky": role}
    if role == "APERTURE":
      groups["base"] = {"aperture": role}

  if not groups:
    groups["base"] = {}

  return groups
