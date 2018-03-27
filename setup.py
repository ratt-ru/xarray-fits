#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "astropy >= 2.0.4",
    "dask >= 0.17.0",
    "six",
    "xarray >= 0.10.0",
    # TODO: Put package requirements here
]

setup_requirements = [
    'pytest-runner',
    # TODO(sjperkins): Put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: Put package test requirements here
]

setup(
    name='xarray-fits',
    version='0.1.1',
    description="xarray Datasets interacting with FITS files",
    long_description=readme + '\n\n' + history,
    author="Simon Perkins",
    author_email='sperkins@ska.ac.za',
    url='https://github.com/ska-sa/xarray-fits',
    packages=find_packages(include=['xarrayfits']),
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='xarrayfits',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
