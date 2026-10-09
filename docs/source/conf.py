# Configuration file for the Sphinx documentation builder.

# -- Project information

from datetime import datetime

project = "xarray-fits"
copyright = f"{datetime.now().year}, South African Radio Astronomy Observatory (SARAO)"
author = "South African Radio Astronomy Observatory (SARAO)"

release = "0.2.6"
version = release

# -- General configuration

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.autosummary",
  "sphinx.ext.extlinks",
  "sphinx_copybutton",
  "sphinx.ext.doctest",
  "sphinx.ext.napoleon",
  "sphinx.ext.intersphinx",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

extlinks = {
  "issue": ("https://github.com/ratt-ru/xarray-fits/issues/%s", "GH%s"),
  "pr": ("https://github.com/ratt-ru/xarray-fits/pull/%s", "PR%s"),
}

intersphinx_mapping = {
  "dask": ("https://dask.pydata.org/en/stable", None),
  "numpy": ("https://numpy.org/doc/stable/", None),
  "python": ("https://docs.python.org/3/", None),
  "xarray": ("https://docs.xarray.dev/en/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Options for HTML output

html_theme = "pydata_sphinx_theme"
html_sidebars: dict[str, list[str]] = {
  "**": [],
}
