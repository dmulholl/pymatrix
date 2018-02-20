---
title: Guide
---

Pymatrix exports a lightweight, general purpose matrix class, `Matrix`. A matrix element can be any arbitrary object that supports the required arithmethic and comparison operators. All of Python's native numeric types --- integers, floats, complex numbers, and rational numbers --- are supported.

Note that Pymatrix has been built for comfort, not for speed. If you have heavy-duty computational needs you should turn to a C-based alternative like [NumPy](http://www.numpy.org) instead.



Instantiation
-------------

You can instantiate a matrix object directly, optionally specifying a fill value:

::: python

    m = Matrix(rows, cols, fill=0)

You can instantiate a matrix object from a list of lists using the `from_list()` static method:

::: python

    m = Matrix.from_list([
        [1, 2, 3],
        [4, 5, 6]
    ])

You can instantiate a matrix object from a string using the `from_string()` static method:

::: python

    string = '''
    1 2 3/7
    4/7 5 6
    '''

    m = Matrix.from_string(
        string,
        rowsep=None,
        colsep=None,
        parser=fractions.Fraction
    )

Row separators default to newlines, column separators default to spaces. Leading and trailing whitespace is stripped from the string. Elements are parsed as fractions (rational numbers) by default.

You can instantiate an n x n identity matrix using the `identity()` static method:

::: python

    m = Matrix.identity(n)

The shortcut `matrix()` function supports the syntax of all three static methods:

::: python

    m = matrix([[1, 2, 3]])
    m = matrix('1 2 3')
    m = matrix(3)



Iteration
---------

Matrix objects are iterable. Iteration proceeds left-to-right by column, then top-to-bottom by row; i.e. the top-left element will be returned first, the bottom-right element will be returned last.

The iterator returns a tuple containing the row number, the column number, and the element:

::: python

    for row, col, element in matrix:
        ...

Alternatively, the `elements()` method returns an iterator over just the matrix elements:

::: python

    for element in matrix.elements():
        ...



Indexing
--------

Matrices are indexed as two-dimensional arrays:

::: python

    matrix[row][col] = element

    element = matrix[row][col]

Note that indices are zero-based in accordance with programming convention rather than one-based in typical math style, i.e. the matrix's top-left element is `matrix[0][0]` rather than `matrix[1][1]`.



Matrix Methods
--------------

Matrix objects support the following methods:

||  `.adjoint()`  ||

    Returns the adjoint matrix as a new object.

||  `.cofactor(row, col)`  ||

    Returns the specified cofactor.

||  `.cofactors()`  ||

    Returns the matrix of cofactors as a new object.

||  `.col(n)`  ||

    Returns an iterator over the specified column.

||  `.cols()`  ||

    Iterator returning a column iterator for each column in the matrix.

||  `.colvec(n)`  ||

    Returns the specified column as a new column vector.

||  `.copy()`  ||

    Returns a copy of the matrix.

||  `.cross(other)`  ||

    Returns the cross/vector product of the matrix with `other` as a new matrix. The cross product is only defined for pairs of 3-dimensional column vectors.

||  `.del_col(col)`  ||

    Returns a new matrix with the specified column deleted.

||  `.del_row(row)`  ||

    Returns a new matrix with the specified row deleted.

||  `.det()`  ||

    Returns the determinant of the matrix.

||  `.dir()`  ||

    Vectors only. Returns the unit vector in the direction of the vector.

||  `.dot(other)`  ||

    Returns the dot/scalar product of the matrix with `other`. The dot product is only defined for pairs of vectors.

||  `.elements()`  ||

    Returns an iterator over the matrix's elements.

||  `.equals(other, delta=None)`  ||

    If `delta` is `None`, two matrices are equal if they are the same size and their corresponding elements are equal, i.e. `e1 == e2`.

    If `delta` is not `None`, two matrices are equal if they are the same size and their corresponding elements agree to within `delta`, i.e. `abs(e1 - e2) <= delta`.

||  `.inv()`  ||

    Returns the inverse matrix if it exists, otherwise raises `MatrixError`.

||  `.is_invertible()`  ||

    True if the matrix is invertible. Note that determining whether a matrix is invertible is as computationally expensive as actually calculating the inverse.

||  `.is_square()`  ||

    True if the matrix is square.

||  `.len()`  ||

    Vectors only. Returns the length of the vector.

||  `.map(func)`  ||

    Returns a new matrix formed by mapping `func` to each element.

||  `.minor(row, col)`  ||

    Returns the specified minor.

||  `.rank()`  ||

    Returns the rank of the matrix.

||  `.ref()`  ||

    Returns the row echelon form of the matrix.

||  `.row(n)`  ||

    Returns an iterator over the specified row.

||  `.rowop_add(r1, m, r2)`  ||

    In-place row operation. Adds `m` times row `r2` to row `r1`.

||  `.rowop_multiply(row, m)`  ||

    In-place row operation. Multiplies the specified row by the scalar `m`.

||  `.rowop_swap(r1, r2)`  ||

    In-place row operation. Interchanges the two specified rows.

||  `.rows()`  ||

    Iterator returning a row iterator for each row in the matrix.

||  `.rowvec(n)`  ||

    Returns the specified row as a new row vector.

||  `.rref()`  ||

    Returns the reduced row echelon form of the matrix.

||  `.trans()`  ||

    Returns the transpose of the matrix as a new object.



Module Functions
----------------

The `pymatrix` module exports the following functions:

|| `dot(u, v)`  ||

    Returns `u . v` - the inner/scalar/dot product of the vectors `u` and `v`.

|| `cross(u, v)`  ||

    Returns `u x v` - the vector/cross product of the 3D column vectors `u` and `v`.

|| `matrix()`  ||

    Shortcut function for instantiating `Matrix` objects; supports the syntax of the various static instantiation methods.



Exceptions
----------

An invalid operation on a matrix object will raise a `MatrixError` exception.
