#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Unit tests for the pymatrix module.
# --------------------------------------------------------------------------

import unittest
from pymatrix import *


# --------------------------------------------------------------------------
# Test matrices.
# --------------------------------------------------------------------------


a = matrix('''
1 2 4
1 3 6
-1 0 1
''')

b = matrix('''
2 1 3
0 -1 1
1 2 0
''')

c = matrix('''
1
2
3
''')

d = matrix('''
2 0 0
0 2 0
0 0 2
''')

a_transpose = matrix('''
1 1 -1
2 3 0
4 6 1
''')

a_cofactors = matrix('''
3 -7 3
-2 5 -2
0 -2 1
''')

a_inverse = matrix('''
3 -2 0
-7 5 -2
3 -2 1
''')

a_negated = matrix('''
-1 -2 -4
-1 -3 -6
1 0 -1
''')

a_plus_b = matrix('''
3 3 7
1 2 7
0 2 1
''')

a_minus_b = matrix('''
-1 1 1
1 4 5
-2 -2 1
''')

a_mul_b = matrix('''
6 7 5
8 10 6
-1 1 -3
''')

a_mul_c = matrix('''
17
25
2
''')

a_mul_2 = matrix('''
2 4 8
2 6 12
-2 0 2
''')

d_pow_4 = matrix('''
16 0 0
0 16 0
0 0 16
''')

c_transpose = matrix('''1 2 3''')

i3 = Matrix.identity(3)
z3 = Matrix(3, 3)

i = matrix('1 0 0').trans()
j = matrix('0 1 0').trans()
k = matrix('0 0 1').trans()
u = matrix('1 2 3').trans()
v = matrix('4 5 6').trans()
w = matrix('-3 6 -3').trans()
x = matrix('0 3 4').trans()


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------


class MatrixInstantiationTests(unittest.TestCase):

    def test_instantiation(self):
        vals = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        m = Matrix(3, 3)
        for rownum, row in enumerate(vals):
            for colnum, element in enumerate(row):
                self.assertEqual(element, m[rownum][colnum])
        self.assertEqual(m.numrows, 3)
        self.assertEqual(m.numcols, 3)

    def test_instantiation_with_fill(self):
        vals = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        m = Matrix(3, 3, 1)
        for rownum, row in enumerate(vals):
            for colnum, element in enumerate(row):
                self.assertEqual(element, m[rownum][colnum])
        self.assertEqual(m.numrows, 3)
        self.assertEqual(m.numcols, 3)

    def test_instantiation_of_identity(self):
        vals = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        m = matrix(3)
        for rownum, row in enumerate(vals):
            for colnum, element in enumerate(row):
                self.assertEqual(element, m[rownum][colnum])
        self.assertEqual(m.numrows, 3)
        self.assertEqual(m.numcols, 3)

    def test_instantiation_from_list(self):
        vals = [[1, 2, 4], [1, 3, 6], [-1, 0, 1]]
        m = matrix(vals)
        for rownum, row in enumerate(vals):
            for colnum, element in enumerate(row):
                self.assertEqual(element, m[rownum][colnum])
        self.assertEqual(m.numrows, 3)
        self.assertEqual(m.numcols, 3)

    def test_instantiation_from_default_format_string(self):
        vals = [[1, 2, 4], [1, 3, 6], [-1, 0, 1]]
        for rownum, row in enumerate(vals):
            for colnum, element in enumerate(row):
                self.assertEqual(element, a[rownum][colnum])
        self.assertEqual(a.numrows, 3)
        self.assertEqual(a.numcols, 3)

    def test_instantiation_from_alt_format_string(self):
        vals = [[1, 2, 4], [1, 3, 6], [-1, 0, 1]]
        s = '1, 2, 4; 1, 3, 6; -1, 0, 1'
        m = matrix(s, ';', ',')
        for rownum, row in enumerate(vals):
            for colnum, element in enumerate(row):
                self.assertEqual(element, m[rownum][colnum])
        self.assertEqual(m.numrows, 3)
        self.assertEqual(m.numcols, 3)


class MatrixEqualityTests(unittest.TestCase):

    def test_equality_with_self(self):
        self.assertTrue(a == a)
        self.assertFalse(a != a)

    def test_equality_with_copy(self):
        self.assertTrue(a == a.copy())
        self.assertFalse(a != a.copy())

    def test_equality_with_other(self):
        self.assertTrue(a != b)
        self.assertFalse(a == b)


class MatrixAlgebraTests(unittest.TestCase):

    def test_unary_positive(self):
        self.assertEqual(+a, a)

    def test_unary_negative(self):
        self.assertEqual(-a, a_negated)

    def test_addition(self):
        self.assertEqual(a + z3, a)
        self.assertEqual(a + b, a_plus_b)

    def test_addition_invalid_types(self):
        self.assertRaises(MatrixError, a.__add__, 1)

    def test_addition_invalid_dimensions(self):
        self.assertRaises(MatrixError, a.__add__, c)

    def test_subtraction(self):
        self.assertEqual(a - z3, a)
        self.assertEqual(a - a, z3)
        self.assertEqual(a - b, a_minus_b)

    def test_subtraction_invalid_types(self):
        self.assertRaises(MatrixError, a.__sub__, 1)

    def test_subtraction_invalid_dimensions(self):
        self.assertRaises(MatrixError, a.__sub__, c)

    def test_multiplication(self):
        self.assertEqual(a * i3, a)
        self.assertEqual(i3 * a, a)
        self.assertEqual(a * b, a_mul_b)
        self.assertEqual(a * c, a_mul_c)

    def test_multiplication_invalid_dimensions(self):
        self.assertRaises(MatrixError, c.__mul__, a)

    def test_scalar_multiplication(self):
        self.assertEqual(a * 2, a_mul_2)
        self.assertEqual(2 * a, a_mul_2)

    def test_matrix_powers(self):
        self.assertEqual(d ** 4, d_pow_4)


class MatrixOperationTests(unittest.TestCase):

    def test_transpose(self):
        self.assertEqual(a.trans(), a_transpose)
        self.assertEqual(c.trans(), c_transpose)

    def test_determinant(self):
        self.assertEqual(a.det(), 1)
        self.assertEqual(b.det(), 0)

    def test_determinant_invalid_dimensions(self):
        self.assertRaises(MatrixError, c.det)

    def test_cofactors(self):
        self.assertEqual(a.cofactors(), a_cofactors)

    def test_inverse(self):
        self.assertEqual(a.inv(), a_inverse)

    def test_inverse_non_invertible(self):
        self.assertRaises(MatrixError, b.inv)


class VectorOperationTests(unittest.TestCase):

    def test_dot_product(self):
        self.assertEqual(dot(u, v), 32)
        self.assertEqual(dot(v, u), 32)

    def test_cross_product(self):
        self.assertEqual(cross(u, v), w)
        self.assertEqual(cross(v, u), -w)
        self.assertEqual(cross(i, j), k)
        self.assertEqual(cross(j, i), -k)

    def test_length(self):
        self.assertAlmostEqual(x.len(), 5)

    def test_direction(self):
        vals = [0, 0.6, 0.8]
        unitvector = x.dir()
        for i, element in enumerate(unitvector):
            self.assertAlmostEqual(vals[i], element)


if __name__ == '__main__':
    unittest.main()
