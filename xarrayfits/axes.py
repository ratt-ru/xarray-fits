import numpy as np

HEADER_PREFIXES = ["NAXIS", "CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT", "CNAME"]


def property_factory(prefix):
    def impl(self):
        return getattr(self, f"_{prefix}")

    return property(impl)


class UndefinedGridError(ValueError):
    pass


class AxesMetaClass(type):
    def __new__(cls, name, bases, dct):
        for prefix in (p.lower() for p in HEADER_PREFIXES):
            dct[prefix] = property_factory(prefix)
        return type.__new__(cls, name, bases, dct)


class Axes(metaclass=AxesMetaClass):
    """Presents a C-ordered view over FITS Header grid attributes"""

    def __init__(self, header):
        self._ndims = ndims = header["NAXIS"]
        axr = tuple(range(1, ndims + 1))

        # Read headers into C-order
        for prefix in HEADER_PREFIXES:
            values = reversed([header.get(f"{prefix}{n}") for n in axr])
            values = [s.strip() if isinstance(s, str) else s for s in values]
            setattr(self, f"_{prefix.lower()}", values)

        # We must have all NAXIS
        for i, a in enumerate(self.naxis):
            if a is None:
                raise UndefinedGridError(f"NAXIS{ndims - i} undefined")

        # Fill in any None CRVAL
        self._crval = [0 if v is None else v for v in self._crval]
        # Fill in any None CRPIX
        self._crpix = [1 if p is None else p for p in self._crpix]
        # Fill in any None CDELT
        self._cdelt = [1 if d is None else d for d in self._cdelt]

        self._grid = [None] * ndims

    @property
    def ndims(self):
        return self._ndims

    def name(self, dim):
        """Return a name for dimension :code:`dim`"""
        if result := self.cname[dim]:
            return result
        elif result := self.ctype[dim]:
            return result
        else:
            return None

    def grid(self, dim):
        """Return the axis grid for dimension :code:`dim`"""
        if self._grid[dim] is None:
            # Create the grid
            pixels = np.arange(1, self.naxis[dim] + 1)
            self._grid[dim] = (pixels - self.crpix[dim]) * self.cdelt[dim] + self.crval[
                dim
            ]

        return self._grid[dim]
