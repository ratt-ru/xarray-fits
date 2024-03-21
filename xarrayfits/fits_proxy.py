from threading import Lock
import weakref

from astropy.io import fits

TABLE_CACHE_LOCK = Lock()
TABLE_CACHE = weakref.WeakValueDictionary()


class FitsProxyMetaClass(type):
    """https://en.wikipedia.org/wiki/Multiton_pattern"""

    def __call__(cls, *args, **kwargs):
        key = (cls,) + args + tuple(set(kwargs.items()))

        with TABLE_CACHE_LOCK:
            try:
                return TABLE_CACHE[key]
            except KeyError:
                instance = type.__call__(cls, *args, **kwargs)
                TABLE_CACHE[key] = instance
                return instance


class FitsProxy(metaclass=FitsProxyMetaClass):
    """
    Picklable object proxying a :class:`astropy.io.fits` class
    """

    def __init__(self, filename, **kwargs):
        """
        Parameters
        ----------
        filename : str
            fits filename
        **kwargs (optional) : dict
            Extra keywords arguments passed in to the
            :class:`astropy.iofits` constructor.
        """
        self._filename = filename
        self._kwargs = kwargs
        self._lock = Lock()

    @staticmethod
    def from_reduce_args(filename, kw):
        return FitsProxy(filename, **kw)

    @property
    def is_memory_mapped(self):
        return self._kwargs.get("memmap") in {None, True}

    @property
    def hdu_list(self):
        try:
            return self._hdul
        except AttributeError:
            with self._lock:
                try:
                    return self._hdul
                except AttributeError:
                    self._hdul = hdul = fits.open(self._filename, **self._kwargs)
                    weakref.finalize(self, hdul.close)
                    return hdul

    def __hash__(self):
        return hash((self._filename, tuple(set(self._kwargs.items()))))

    def __reduce__(self):
        return (FitsProxy.from_reduce_args, (self._filename, self._kwargs))
