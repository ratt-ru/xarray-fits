from collections.abc import Mapping
import numpy as np

HEADER_PREFIXES = ["NAXIS", "CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT", "CNAME"]


class UndefinedGridError(ValueError):
    pass


def property_factory(prefix: str):
    def impl(self):
        return getattr(self, f"_{prefix}")

    return property(impl)


class AffineGridMetaclass(type):
    def __new__(cls, name, bases, dct):
        for prefix in (p.lower() for p in HEADER_PREFIXES):
            dct[prefix] = property_factory(prefix)
        return type.__new__(cls, name, bases, dct)


class AffineGrid(metaclass=AffineGridMetaclass):
    """Presents a C-ordered view over FITS Header grid attributes"""

    def __init__(self, header: Mapping):
        self._ndims = ndims = header["NAXIS"]
        axr = tuple(range(1, ndims + 1))
        h = header

        # Read headers into C-order
        for prefix in HEADER_PREFIXES:
            values = reversed([header.get(f"{prefix}{n}") for n in axr])
            values = [s.strip() if isinstance(s, str) else s for s in values]
            setattr(self, f"_{prefix.lower()}", values)

        # We must have all NAXIS
        for i, a in enumerate(self.naxis):
            if a is None:
                raise UndefinedGridError(f"NAXIS{ndims - i} undefined")

        # Fill in any missing CRVAL
        self._crval = [0.0 if v is None else v for v in self._crval]
        # Fill in any missing CRPIX
        self._crpix = [1 if p is None else p for p in self._crpix]
        # Fill in any missing CDELT
        self._cdelt = [1.0 if d is None else d for d in self._cdelt]

        self._grid = []

        for d in range(ndims):
            pixels = np.arange(1, self._naxis[d] + 1, dtype=np.float64)
            self._grid.append(
                (pixels - self._crpix[d]) * self._cdelt[d] + self._crval[d]
            )

    @property
    def ndims(self):
        return self._ndims

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
