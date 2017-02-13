#!/usr/bin/env python3
"""
Pymatrix
========

A lightweight, easy-to-use matrix module in pure Python. Supports a range of
basic linear algebra operations.

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

See the `package documentation <http://mulholland.xyz/docs/pymatrix/>`_ or
the library's `Github homepage <https://github.com/dmulholland/pymatrix>`_
for further details.

"""

import os
import re
import io

from setuptools import setup


filepath = os.path.join(os.path.dirname(__file__), 'pymatrix.py')
with io.open(filepath, encoding='utf-8') as metafile:
    regex = r'''^__([a-z]+)__ = ["'](.*)["']'''
    meta = dict(re.findall(regex, metafile.read(), flags=re.MULTILINE))


setup(
    name = 'pymatrix',
    version = meta['version'],
    py_modules = ['pymatrix'],
    entry_points = {
        'console_scripts': [
            'pymatrix = pymatrix:main',
        ],
    },
    author = 'Darren Mulholland',
    url = 'https://github.com/dmulholland/pymatrix',
    license = 'Public Domain',
    description = (
        'A lightweight, easy-to-use matrix module in pure Python.'
    ),
    long_description = __doc__,
    classifiers = [
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: Public Domain',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
