"""
Pymatrix: a lightweight matrix object with support for basic linear algebra
operations.

Note that matrix indices are zero-based in accordance with programming
convention rather than one-based in typical math style, i.e. the top-left
element in a matrix is m[0][0] rather than m[1][1].

Caution: Beware of rounding errors with floats. The algorithm used to
calculate the row echelon form (and hence the inverse, etc.) is sensitive
to small rounding errors near zero; i.e. if a matrix contains entries
which should be zero but aren't due to rounding errors then these errors
can be magnified by the algorithm.

An example of the kind of rounding error that can cause problems is:

    >>> 0.1 * 3 - 0.3
    5.551115123125783e-17

Author: Darren Mulholland <darren@mulholland.xyz>
License: Public Domain

"""

import fractions
import math
import operator
import functools


# Library version.
__version__ = "1.2.2"


def matrix(*pargs, **kwargs):
    """ Convenience function for instantiating Matrix objects. """
    if isinstance(pargs[0], int):
        return Matrix.Identity(pargs[0])
    elif isinstance(pargs[0], str):
        return Matrix.FromString(*pargs, **kwargs)
    elif isinstance(pargs[0], list):
        return Matrix.FromList(*pargs, **kwargs)
    else:
        raise NotImplementedError


def dot(u, v):
    """ Returns u . v - the dot product of vectors u and v. """
    return sum(map(operator.mul, u, v))


def cross(u, v):
    """ Returns u x v - the cross product of 3D column vectors u and v. """
    w = Matrix(3, 1)
    w[0][0] = u[1][0] * v[2][0] - u[2][0] * v[1][0]
    w[1][0] = u[2][0] * v[0][0] - u[0][0] * v[2][0]
    w[2][0] = u[0][0] * v[1][0] - u[1][0] * v[0][0]
    return w


class MatrixError(Exception):
    """ Invalid operation attempted on a Matrix object. """
    pass


class Matrix:

    """ Matrix object supporting basic linear algebra operations. """

    def __init__(self, rows, cols, fill=0):
        """ Initialize a `rows` x `cols` sized matrix filled with `fill`. """
        self.nrows = rows
        self.ncols = cols
        self.grid = [[fill for i in range(cols)] for j in range(rows)]

    def __str__(self):
        """ Returns a string representation of the matrix. """
        maxlen = max(len(str(e)) for e in self)
        return '\n'.join(
            ' '.join(str(e).rjust(maxlen) for e in row) for row in self.grid
        )

    def __repr__(self):
        """ Returns a string representation of the object. """
        return '<%s %sx%s 0x%x>' % (
            self.__class__.__name__, self.nrows, self.ncols, id(self)
        )

    def __getitem__(self, key):
        """ Enables `self[row][col]` indexing and assignment. """
        return self.grid[key]

    def __contains__(self, item):
        """ Containment: `item in self`. """
        for element in self:
            if element == item:
                return True
        return False

    def __neg__(self):
        """ Negative operator: `- self`. Returns a negated copy. """
        return self.map(lambda element: -element)

    def __pos__(self):
        """ Positive operator: `+ self`. Returns a copy. """
        return self.map(lambda element: element)

    def __eq__(self, other):
        """ Equality: `self == other`. """
        return self.equals(other)

    def __ne__(self, other):
        """ Inequality: `self != other`. """
        return not self.__eq__(other)

    def __add__(self, other):
        """ Addition: `self + other`. """
        if not isinstance(other, Matrix):
            raise MatrixError('cannot add %s to a matrix' % type(other))
        if self.nrows != other.nrows or self.ncols != other.ncols:
            raise MatrixError('cannot add matrices of different sizes')
        m = Matrix(self.nrows, self.ncols)
        for row, col, element in self.elements():
            m[row][col] = element + other[row][col]
        return m

    def __sub__(self, other):
        """ Subtraction: `self - other`. """
        if not isinstance(other, Matrix):
            raise MatrixError('cannot subtract %s from a matrix' % type(other))
        if self.nrows != other.nrows or self.ncols != other.ncols:
            raise MatrixError('cannot subtract matrices of different sizes')
        m = Matrix(self.nrows, self.ncols)
        for row, col, element in self.elements():
            m[row][col] = element - other[row][col]
        return m

    def __mul__(self, other):
        """ Multiplication: `self * other`. """
        if isinstance(other, Matrix):
            if self.ncols != other.nrows:
                raise MatrixError('incompatible sizes for multiplication')
            m = Matrix(self.nrows, other.ncols)
            for row, col, element in m.elements():
                for re, ce in zip(self.row(row), other.col(col)):
                    m[row][col] += re * ce
            return m
        else:
            return self.map(lambda element: element * other)

    def __rmul__(self, other):
        """ Multiplication: `other * self`. Note that this method is never
        called when `other` is a Matrix object - in that case the left
        matrix would handle the multiplication itself via its own __mul__
        method. This method is intended to handle multiplication on the left
        by simple numerical types. """
        return self * other

    def __pow__(self, other):
        """ Exponentiation: `self ** other`. """
        if not isinstance(other, int) or other < 1:
            raise MatrixError('only positive integer powers are supported')
        m = self.copy()
        for i in range(other - 1):
            m = m * self
        return m

    def __iter__(self):
        """ Iteration: `for i in self`. """
        for row in range(self.nrows):
            for col in range(self.ncols):
                yield self[row][col]

    def row(self, n):
        """ Returns an iterator over the specified row. """
        for col in range(self.ncols):
            yield self[n][col]

    def col(self, n):
        """ Returns an iterator over the specified column. """
        for row in range(self.nrows):
            yield self[row][n]

    def rows(self):
        """ Returns a row iterator for each row in the matrix. """
        for row in range(self.nrows):
            yield self.row(row)

    def cols(self):
        """ Returns a column iterator for each column in the matrix. """
        for col in range(self.ncols):
            yield self.col(col)

    def rowvec(self, n):
        """ Returns the specified row as a new row vector. """
        v = Matrix(1, self.ncols)
        for col in range(self.ncols):
            v[0][col] = self[n][col]
        return v

    def colvec(self, n):
        """ Returns the specified column as a new column vector. """
        v = Matrix(self.nrows, 1)
        for row in range(self.nrows):
            v[row][0] = self[row][n]
        return v

    def equals(self, other, delta=None):
        """ Returns true if `self` and `other` are identically-sized matrices
        and their corresponding elements agree to within `delta`. If `delta`
        is omitted, we perform a simple equality check (`==`) on corresponding
        elements instead. """
        if self.nrows != other.nrows or self.ncols != other.ncols:
            return False
        if delta:
            for row, col, element in self.elements():
                if abs(element - other[row][col]) > delta:
                    return False
        else:
            for row, col, element in self.elements():
                if element != other[row][col]:
                    return False
        return True

    def elements(self):
        """ Iterator returning the tuple (row, col, element). """
        for row in range(self.nrows):
            for col in range(self.ncols):
                yield row, col, self[row][col]

    def copy(self):
        """ Returns a copy of the matrix. """
        return self.map(lambda element: element)

    def transpose(self):
        """ Returns the transpose of the matrix as a new object. """
        m = Matrix(self.ncols, self.nrows)
        for row, col, element in self.elements():
            m[col][row] = element
        return m

    def trp(self):
        """ Alias of `.transpose()`. """
        return self.transpose()

    def det(self):
        """ Returns the determinant of the matrix. """
        if not self.is_square():
            raise MatrixError('non-square matrix does not have determinant')
        ref, _, multiplier = _get_row_echelon_form(self)
        ref_det = functools.reduce(
            operator.mul,
            (ref[i][i] for i in range(ref.nrows))
        )
        return ref_det / multiplier

    def minor(self, row, col):
        """ Returns the specified minor. """
        return self.del_row_col(row, col).det()

    def cofactor(self, row, col):
        """ Returns the specified cofactor. """
        return pow(-1, row + col) * self.minor(row, col)

    def cofactors(self):
        """ Returns the matrix of cofactors as a new object. """
        m = Matrix(self.nrows, self.ncols)
        for row, col, element in self.elements():
            m[row][col] = self.cofactor(row, col)
        return m

    def adjoint(self):
        """ Returns the adjoint matrix as a new object. """
        return self.cofactors().transpose()

    def inverse(self):
        """ Returns the inverse matrix if it exists or raises MatrixError. """
        if not self.is_square():
            raise MatrixError('non-square matrix cannot have an inverse')
        identity = Matrix.Identity(self.nrows)
        rref, inverse = _get_reduced_row_echelon_form(self, identity)
        if rref != identity:
            raise MatrixError('matrix is non-invertible')
        return inverse

    def inv(self):
        """ Alias of `.inverse()`. """
        return self.inverse()

    def del_row_col(self, row_to_delete, col_to_delete):
        """ Returns a new matrix with the specified row and column deleted. """
        return self.del_row(row_to_delete).del_col(col_to_delete)

    def del_row(self, row_to_delete):
        """ Returns a new matrix with the specified row deleted. """
        m = Matrix(self.nrows - 1, self.ncols)
        for row, col, element in self.elements():
            if row < row_to_delete:
                m[row][col] = element
            elif row > row_to_delete:
                m[row - 1][col] = element
        return m

    def del_col(self, col_to_delete):
        """ Returns a new matrix with the specified column deleted. """
        m = Matrix(self.nrows, self.ncols - 1)
        for row, col, element in self.elements():
            if col < col_to_delete:
                m[row][col] = element
            elif col > col_to_delete:
                m[row][col - 1] = element
        return m

    def map(self, func):
        """ Forms a new matrix by mapping `func` to each element. """
        m = Matrix(self.nrows, self.ncols)
        for row, col, element in self.elements():
            m[row][col] = func(element)
        return m

    def rowop_multiply(self, row, m):
        """ In-place row operation. Multiplies the specified row by `m`. """
        for col in range(self.ncols):
            self[row][col] = self[row][col] * m

    def rowop_swap(self, r1, r2):
        """ In-place row operation. Interchanges the two specified rows. """
        for col in range(self.ncols):
            self[r1][col], self[r2][col] = self[r2][col], self[r1][col]

    def rowop_add(self, r1, m, r2):
        """ In-place row operation. Adds `m` times row `r2` to row `r1`. """
        for col in range(self.ncols):
            self[r1][col] = self[r1][col] + m * self[r2][col]

    def ref(self):
        """ Returns the row echelon form of the matrix. """
        return _get_row_echelon_form(self)[0]

    def rref(self):
        """ Returns the reduced row echelon form of the matrix. """
        return _get_reduced_row_echelon_form(self)[0]

    def len(self):
        """ Vectors only. Returns the length of the vector. """
        return math.sqrt(sum(e ** 2 for e in self))

    def dir(self):
        """ Vectors only. Returns a unit vector in the same direction. """
        return (1 / self.len()) * self

    def is_square(self):
        """ True if the matrix is square. """
        return self.nrows == self.ncols

    def is_invertible(self):
        """ True if the matrix is invertible. """
        try:
            inverse = self.inverse()
            return True
        except MatrixError:
            return False

    def rank(self):
        """ Returns the rank of the matrix. """
        rank = 0
        for row in self.ref().rows():
            for element in row:
                if element != 0:
                    rank += 1
                    break
        return rank

    @staticmethod
    def FromList(l):
        """ Instantiates a new matrix object from a list of lists. """
        m = Matrix(len(l), len(l[0]))
        for rownum, row in enumerate(l):
            for colnum, element in enumerate(row):
                m[rownum][colnum] = element
        return m

    @staticmethod
    def FromString(s, rowsep=None, colsep=None, parser=fractions.Fraction):
        """ Instantiates a new matrix object from a string. """
        rows = s.strip().split(rowsep) if rowsep else s.strip().splitlines()
        m = Matrix(len(rows), len(rows[0].split(colsep)))
        for rownum, row in enumerate(rows):
            for colnum, element in enumerate(row.split(colsep)):
                m[rownum][colnum] = parser(element)
        return m

    @staticmethod
    def Identity(n):
        """ Instantiates a new n x n identity matrix. """
        m = Matrix(n, n)
        for i in range(n):
            m[i][i] = 1
        return m


# We determine the row echelon form of the matrix using the forward phase of
# the Gauss-Jordan elimination algorithm. If a `mirror` matrix is supplied,
# we apply the same sequence of row operations to it. Note that neither
# matrix is altered in-place; instead copies are returned.
def _get_row_echelon_form(matrix, mirror=None):
    m = matrix.copy()
    mirror = mirror.copy() if mirror else None
    det_multiplier = 1

    # Start with the top row and work downwards.
    for top_row in range(m.nrows):

        # Find the leftmost column that is not all zeros.
        # Note: this step is sensitive to small rounding errors around zero.
        found = False
        for col in range(m.ncols):
            for row in range(top_row, m.nrows):
                if m[row][col] != 0:
                    found = True
                    break
            if found:
                break
        if not found:
            break

        # Get a non-zero entry at the top of this column.
        if m[top_row][col] == 0:
            m.rowop_swap(top_row, row)
            det_multiplier *= -1
            if mirror:
                mirror.rowop_swap(top_row, row)

        # Make this entry '1'.
        if m[top_row][col] != 1:
            multiplier = 1 / m[top_row][col]
            m.rowop_multiply(top_row, multiplier)
            m[top_row][col] = 1 # assign directly in case of rounding errors
            det_multiplier *= multiplier
            if mirror:
                mirror.rowop_multiply(top_row, multiplier)

        # Make all entries below the leading '1' zero.
        for row in range(top_row + 1, m.nrows):
            if m[row][col] != 0:
                multiplier = -m[row][col]
                m.rowop_add(row, multiplier, top_row)
                if mirror:
                    mirror.rowop_add(row, multiplier, top_row)

    return m, mirror, det_multiplier


# Determine the reduced row echelon form of the matrix using the Gauss-Jordan
# elimination algorithm. If a `mirror` matrix is supplied, the same sequence of
# row operations will be applied to it. Note that neither matrix is altered
# in-place; instead copies are returned.
def _get_reduced_row_echelon_form(matrix, mirror=None):

    # Run the forward phase of the algorithm to determine the row echelon form.
    m, mirror, ignore = _get_row_echelon_form(matrix, mirror)

    # The backward phase of the algorithm. For each row, starting at the bottom
    # and working up, find the column containing the leading 1 and make all the
    # entries above it zero.
    for last_row in range(m.nrows - 1, 0, -1):
        for col in range(m.ncols):
            if m[last_row][col] == 1:
                for row in range(last_row):
                    if m[row][col] != 0:
                        multiplier = -m[row][col]
                        m.rowop_add(row, multiplier, last_row)
                        if mirror:
                            mirror.rowop_add(row, multiplier, last_row)
                break

    return m, mirror
