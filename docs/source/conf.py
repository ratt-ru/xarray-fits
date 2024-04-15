# Configuration file for the Sphinx documentation builder.

# -- Project information

from datetime import datetime
import tomli

with open("../../pyproject.toml", "rb") as f:
    toml = tomli.load(f)
    pyproject = toml["tool"]["poetry"]

project = pyproject["name"]
copyright = f"{datetime.now().year}, South African Radio Astronomy Observatory (SARAO)"
author = "South African Radio Astronomy Observatory (SARAO)"

release = pyproject["version"]
version = pyproject["version"]

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "furo"

# -- Options for EPUB output
epub_show_urls = "footnote"
