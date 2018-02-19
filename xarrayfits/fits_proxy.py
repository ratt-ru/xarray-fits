from astropy.io import fits


class FitsProxy(object):
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
        self._fits_file = fits.open(filename, **kwargs)

    def __setstate__(self, state):
        self.__init__(*state)

    def __getstate__(self):
        return  (self._filename, self._kwargs)

    def __call__(self, fn, *args, **kwargs):
        return getattr(self._fits_file, fn)(*args, **kwargs)
