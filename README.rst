===========
xarray-fits
===========

Given some FITS files with matching HDUs:

.. code-block:: bash

  $ tree data
  /home/user/data
  ├── data-0.fits
  ├── data-1.fits
  └── data-2.fits

  0 directories, 3 files

.. code-block:: python

  >>> from xarrayfits import xds_from_fits
  >>> datasets = xds_from_fits("/home/user/data*", prefix="data")

The above returns a list of three xarray Datasets

.. code-block:: python

  >>> datasets
  [<xarray.Dataset> Size: 800B
   Dimensions:  (data0-0: 10, data0-1: 10)
   Dimensions without coordinates: data0-0, data0-1
   Data variables:
       data0    (data0-0, data0-1) float64 800B dask.array<chunksize=(10, 10), meta=np.ndarray>,
   <xarray.Dataset> Size: 800B
   Dimensions:  (data0-0: 10, data0-1: 10)
   Dimensions without coordinates: data0-0, data0-1
   Data variables:
       data0    (data0-0, data0-1) float64 800B dask.array<chunksize=(10, 10), meta=np.ndarray>,
   <xarray.Dataset> Size: 800B
   Dimensions:  (data0-0: 10, data0-1: 10)
   Dimensions without coordinates: data0-0, data0-1
   Data variables:
       data0    (data0-0, data0-1) float64 800B dask.array<chunksize=(10, 10), meta=np.ndarray>]

Using xarray these can be concatenated along a dimension:

  .. code-block:: python

    >>> import xarray
    >>> ds = xarray.concat(datasets, dim="data0-0")
    >>> ds
    <xarray.Dataset> Size: 2kB
    Dimensions:  (data0-0: 30, data0-1: 10)
    Dimensions without coordinates: data0-0, data0-1
    Data variables:
        data0    (data0-0, data0-1) float64 2kB dask.array<chunksize=(10, 10), meta=np.ndarray>
