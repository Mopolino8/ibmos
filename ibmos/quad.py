"""Quadrature operators."""

import numpy as np
import scipy.sparse as sp


def _fx(m, periodic=False, fmt='csr'):
    """Build operator for $$\int_{x_i}^{x_{i+1}} f_x dx$$

    Parameters
    ----------
    m : int
        Number of grid points.
    periodic : bool, optional
        periodic boundary conditions.
    fmt : str, optional
        Format of the returned sparse matrix.

    Returns
    -------
    Linear operator in sparse matrix form.

    """
    if periodic:
        return sp.diags((1, -1, 1), (-m + 2, 0, 1), (m - 1, m - 1), fmt)
    else:
        return sp.diags((-1, 1), (0, 1), (m - 1, m), fmt)


def _fxx(x, periodic=False, fmt='csr'):
    """Build operator for $$\int_{x_i}^{x_{i+1}} f_{xx} dx$$.

    Parameters
    ----------
    x : np.ndarray
        Numpy array with the list of grid points.
    periodic : bool, optional
        periodic boundary conditions.
    fmt : str, optional
        Format of the returned sparse matrix.

    Returns
    -------
    Linear operator in sparse matrix form.

    """

    m = x.size
    Dx = sp.diags(1 / np.diff(x), format=fmt) @ _fx(m, periodic, fmt)
    if periodic:
        ShiftR = sp.diags((1, 1), (-1, m - 2), (m - 1, m - 1), fmt)
        return _fx(m, periodic, fmt) @ ShiftR @ Dx
    else:
        return _fx(m - 1, periodic, fmt) @ Dx


def op(x, y, wrt, periodic=False, fmt='csr'):
    """Build quadrature operator for a two-dimensional field.

    Parameters
    ----------
    x : np.ndarray
        Numpy array with the list of grid points in the first dimension.
    y : np.ndarray
        Numpy array with the list of grid points in the second dimension.
    wrt : str
        Integration variable and order. Must be 'x', 'xx', 'y' or 'yy'.
    periodic : bool, optional
        Periodicity along the integration direction.
    fmt : str, optional
        Format of the returned sparse matrix.

    Returns
    -------
    Quadrature operator in sparse-matrix form.
    Lower boundary term in sparse-matrix form (only if periodic=false)
    Upper boundary term in sparse-matrix form (only if periodic=false)

    Raises
    ------
    ValueError
        wrt is not valid (see the ``Parameters`` section)

    """

    n, m, order = y.size, x.size, len(wrt)

    if wrt in ('x', 'xx'):
        In = sp.eye(n)
        D = _fx(m, periodic, fmt) if order is 1 else _fxx(x, periodic, fmt)
        if periodic:
            return sp.kron(In, D, fmt)
        else:
            D = D.tolil()
            return sp.kron(In, D[:, 1:-1], fmt), \
                   sp.kron(In, D[:, 0], fmt), sp.kron(In, D[:, -1], fmt)
    elif wrt in ('y', 'yy'):
        Im = sp.eye(m)
        D = _fx(n, periodic, fmt) if order is 1 else _fxx(y, periodic, fmt)
        if periodic:
            return sp.kron(D, Im, fmt)
        else:
            D = D.tolil()
            return sp.kron(D[:, 1:-1], Im, fmt), \
                   sp.kron(D[:, 0], Im, fmt), sp.kron(D[:, -1], Im, fmt)
    else:
        raise ValueError("wrt must be 'x', 'y', 'xx' or 'yy' (wrt = '", wrt, "')")
