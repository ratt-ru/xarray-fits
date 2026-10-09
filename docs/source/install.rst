Installation
============

Install xarray-fits with pip:

.. code-block:: console

  $ pip install xarray-fits

Chunking Image Datasets with ``chunks=`` needs dask, and reading them on
a cluster needs distributed, which are installed separately:

.. code-block:: console

  $ pip install dask distributed

Remote FITS Images need the fsspec implementation of their protocol,
for example ``s3fs`` for ``s3://`` URLs or ``aiohttp`` for ``http://``.
