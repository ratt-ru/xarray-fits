Usage
=====

.. _installation:

Installation
------------

To use xarray-fits, first install it using pip:

.. code-block:: console

   (.venv) $ pip install xarray-fits

Creating xarray datasets from FITS files
----------------------------------------

To retrieve a list of xarray datasets with
FITS file contents, you can use the
``xarrayfits.xds_from_fits()`` function:

.. autofunction:: xarrayfits.xds_from_fits
   :noindex:

For example:

>>> from xarrayfits import xds_from_fits
>>> xds_from_fits("3C147.fits")
