
# Pymatrix

A pure-Python matrix module with support for basic linear algebra operations:

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

See the [documentation][docs] for details.

[docs]: http://mulholland.xyz/docs/pymatrix/
