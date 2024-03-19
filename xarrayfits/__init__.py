# -*- coding: utf-8 -*-

"""Top-level package for xarray-fits."""

__author__ = """Simon Perkins"""
__email__ = "simon.perkins@gmail.com"
__version__ = "0.1.2"

import logging


def __create_logger():
    # Console formatter, mention name
    cfmt = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(cfmt)

    logger = logging.getLogger("xarray-fits")
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.propagate = False

    return logger


log = __create_logger()

from xarrayfits.fits import xds_from_fits  # noqa
