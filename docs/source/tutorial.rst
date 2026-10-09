Tutorial
========

Opening a FITS Image
--------------------

A FITS Image opens as an Image Dataset through xarray:

.. code-block:: python

  import xarray as xr

  ds = xr.open_dataset("cube.image.fits", engine="xarray-fits:fits")

The engine is detected from the FITS header, so ``engine`` may be omitted
when no other installed backend also claims FITS files.

The image becomes the ``SKY`` variable, with dimensions
``(time, frequency, polarization, l, m)``. Nothing is read from the pixels
until values are requested.

Opening several FITS Images together
------------------------------------

The products of an imager open as one Image Dataset, given a list of
FITS Images:

.. code-block:: python

  ds = xr.open_dataset(
    [
      "cube.image.fits",
      "cube.psf.fits",
      "cube.pb.fits",
      "cube.residual.fits",
      "cube.model.fits",
      "cube.sumwt.fits",
    ],
    engine="xarray-fits:fits",
  )

The Role of each FITS Image comes from its file name: ``cube.psf.fits``
holds the ``POINT_SPREAD_FUNCTION`` and ``cube.residual.fits`` the
``SKY_RESIDUAL``, for example. Give a mapping of Roles to FITS Images
when the names do not say:

.. code-block:: python

  ds = xr.open_dataset(
    {"sky": "a.fits", "psf": "b.fits"},
    engine="xarray-fits:fits",
  )

The FITS Images must share their coordinates, to within round-off.

Chunking
--------

``chunks={}`` gives dask arrays chunked by ``preferred_chunks``, one image
plane per chunk by default. Other chunk sizes may be preferred:

.. code-block:: python

  ds = xr.open_dataset(
    "cube.image.fits",
    engine="xarray-fits:fits",
    chunks={},
    preferred_chunks={"frequency": 4, "l": 1024, "m": 1024},
  )

Image Datasets pickle, so they can be computed on distributed clusters.

Dropping variables
------------------

The sky coordinates ``right_ascension`` and ``declination`` are computed
lazily, but writing them out doubles the size of a float64 image. The
flags of NaN pixels are similarly lazy. Omit either with
``drop_variables``:

.. code-block:: python

  ds = xr.open_dataset(
    "cube.image.fits",
    engine="xarray-fits:fits",
    drop_variables=["right_ascension", "declination", "FLAG_SKY"],
  )

Remote FITS Images
------------------

Any fsspec URL may be opened. Remote FITS Images are read in sections,
so only the requested pixels are transferred:

.. code-block:: python

  ds = xr.open_dataset("s3://bucket/cube.image.fits", engine="xarray-fits:fits")
