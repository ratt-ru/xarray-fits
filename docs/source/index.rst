xarray-fits
===========

**xarray-fits** presents FITS Images as Image Datasets that conform to the
MSv4 Image Schema, through an xarray backend.

.. code-block:: python

  import xarray as xr

  ds = xr.open_dataset("cube.image.fits", engine="xarray-fits:fits")

Pixels are read lazily, and FITS Images may be local files or any
`fsspec <https://filesystem-spec.readthedocs.io>`_ URL.

.. toctree::
  :maxdepth: 2

  install
  tutorial
  image_dataset
  api
  changelog
