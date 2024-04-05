from collections.abc import Mapping
import numpy as np


class UndefinedGridError(ValueError):
    pass


class AffineGrid:
    """Presents a C-ordered view over FITS Header grid attributes"""

    def __init__(self, header: Mapping):
        h = header

        # Read headers into C-order
        try:
            self._ndims = ndims = h["NAXIS"]
            axr = tuple(range(1, ndims + 1))
            self._naxis = list(reversed([h[f"NAXIS{n}"] for n in axr]))
        except KeyError as e:
            raise UndefinedGridError(f"{e} undefined") from e

        self._ctype = list(reversed([h.get(f"CTYPE{n}") for n in axr]))
        self._crpix = list(reversed([h.get(f"CRPIX{n}", 1) for n in axr]))
        self._crval = list(reversed([h.get(f"CRVAL{n}", 0.0) for n in axr]))
        self._cdelt = list(reversed([h.get(f"CDELT{n}", 1.0) for n in axr]))
        self._cunit = list(reversed([h.get(f"CUNIT{n}") for n in axr]))
        self._cname = list(reversed([h.get(f"CNAME{n}") for n in axr]))

        self._grid = []

        for na, rp, dt, rv in zip(self._naxis, self._crpix, self._cdelt, self._crval):
            pixels = np.arange(1, na + 1, dtype=np.float64)
            self._grid.append((pixels - rp) * dt + rv)

    @property
    def naxis(self):
        return self._naxis

    @property
    def ndims(self):
        return self._ndims

    @property
    def ctype(self):
        return self._ctype

    @property
    def crpix(self):
        return self._crpix

    @property
    def crval(self):
        return self._crval

    @property
    def cdelt(self):
        return self._cdelt

    @property
    def cunit(self):
        return self._cunit

    @property
    def cname(self):
        return self._cname

    def name(self, dim: int):
        """Return a name for dimension :code:`dim`"""
        if result := self.cname[dim]:
            return result
        elif result := self.ctype[dim]:
            return result
        else:
            return None

    def coords(self, dim: int):
        """Return the affine coordinates for dimension :code:`dim`"""
        return self._grid[dim]
