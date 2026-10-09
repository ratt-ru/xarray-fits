===========
xarray-fits
===========

xarray MSv4 Image Datasets over FITS Images.

.. code-block:: python

  import xarray as xr

  ds = xr.open_dataset("cube.image.fits", engine="xarray-fits:fits")

Documentation: https://xarray-fits.readthedocs.io
