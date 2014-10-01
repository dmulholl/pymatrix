#!/usr/bin/env python3
"""
Pymatrix
========

A lightweight matrix module in pure Python. Supports basic linear algebra 
operations.

Sample syntax::

    from pymatrix import matrix

    m = matrix([
        [1, 2],
        [3, 4]
    ])

    a = m + m * 2
    b = m * m
    c = m ** 3

    d = m.det()
    e = m.inv()

See the module's Github homepage (https://github.com/dmulholland/pymatrix) 
for further details.

"""

import os
import re
import io

from distutils.core import setup


filepath = os.path.join(os.path.dirname(__file__), 'pymatrix.py')
with io.open(filepath, encoding='utf-8') as metafile:
    regex = r'''^__([a-z]+)__ = ["'](.*)["']'''
    meta = dict(re.findall(regex, metafile.read(), flags=re.MULTILINE))


setup(
    name = 'pymatrix',
    version = meta['version'],
    py_modules = ['pymatrix'],
    author = 'Darren Mulholland',
    url = 'https://github.com/dmulholland/pymatrix',
    license = 'Public Domain',
    description = (
        'A lightweight matrix object with support for basic linear algebra '
        'operations.'
    ),
    long_description = __doc__,
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'License :: Public Domain',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
