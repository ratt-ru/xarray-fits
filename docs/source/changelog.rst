Changelog
=========

0.2.6 (26-02-2026)
------------------
* Update tested Python range to 3.11 - 3.13 inclusive (:pr:`48`)
* Update base supported Python to 3.11 (:pr:`48`)
* Remove dependency guards (:pr:`48`)
* Migrate to PEP 621 compliant pyproject.toml (:pr:`47`)

0.2.5 (20-05-2025)
------------------
* Include full range of 2024 calver dependencies (:pr:`42`)

0.2.4 (15-04-2024)
------------------
* Modernise documentation (:pr:`38`)
* Add basic Affine Grid coordinates to xarray datasets (:pr:`35`)
* Constrain dask versions (:pr:`34`)
* Specify dtype during chunk normalisation (:pr:`33`)
* Configure dependabot for github actions (:pr:`28`)

0.2.3 (22-03-2024)
------------------
* Move FITS header attributes into an xarray "header" attribute (:pr:`22`)

0.2.2 (21-03-2024)
------------------
* Open FITS files as memory-mapped on local file systems (:pr:`24`)
* Remove obsolete logger (:pr:`23`)
* Support lists of fits files (:pr:`21`)
* Test stacking in the globbing case (:pr:`20`)

0.2.1 (19-03-2024)
------------------
* Make distributed an optional package (:pr:`19`)

0.2.0 (19-03-2024)
------------------
* Update README (:pr:`18`)
* Convert from FITS big-endian to machine native (:pr:`17`)
* Test name prefix specification (:pr:`16`)
* Support globbing (:pr:`15`)
* Specify dimension chunking in C-order (:pr:`14`)
* Use fits section to selection portions of a FITS file on remote data (:pr:`13`)
* Add a weakref.finalize method to close HDUList objects on FitsProxy instances (:pr:`12`)
* Depend on fsspec (:pr:`11`)
* Improve dask array name determinism (:pr:`10`)
* Change license from GPL3 to BSD3 (:pr:`9``)
* Correct FITS Proxy Usage (:pr:`8`)
* Update ruff settings (:pr:`7`)
* Update Github Actions Deployment (:pr:`6`)
* Modernise xarray-fits (:pr:`5`)

0.1.0 (19-02-2018)
------------------

* First release on PyPI.
