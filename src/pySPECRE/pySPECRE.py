import itertools
from typing import Callable, Tuple, Union

import numpy as np
import scipy.linalg as scln
import scipy.sparse as sp
import scipy.sparse.linalg as spln
from multimethod import multimethod

"""
    TODO:

        To support sparse matrices and linear operators, the eigensolver needs
        to be replaced with some form of Jacobi-Davidson method which sorts
        based on the Cauchy-Riemann residue, since current (scipy) sparse
        eigensolvers cannot fully diagonalize a matrix.

"""

_DEFAULT_DR = 1e-6
_DEFAULT_DI = 1e-6


def _get_eigensystem(A, *args, **kwargs):
    """
    get unsorted eigenvalues and eigenvectors of any of
     - numpy dense arrays
     - scipy sparse matrices (not yet implemented)
     - scipy linear operators (not yet implemented)
    """

    if isinstance(A, np.ndarray):
        return scln.eig(A, *args, **kwargs)
    elif isinstance(A, spln.LinearOperator) or sp.issparse(A):
        raise NotImplementedError(
            "Sparse matrices and linear operators not yet supported"
        )
    else:
        raise TypeError(
            "Must be a numpy array, a scipy sparse matrix, or a scipy "
            "linear operator."
        )


def _is_2D_array(arr: np.ndarray) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim == 2


def _is_1D_array(arr: np.ndarray) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim == 1


"""
    All SPECRE multimethods convert the input form to a meshgrid to pass to the
    core function, then reshape the output to the input form.
"""


@multimethod
def SPECRE(
    A: Union[
        Callable,
        Callable[[complex], Union[np.ndarray, sp.spmatrix, spln.LinearOperator]],
    ],
    dr: Union[float, int],
    di: Union[float, int],
    rmin: Union[float, int],
    rmax: Union[float, int],
    imin: Union[float, int],
    imax: Union[float, int],
    horizontal: bool = True,
    *args,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SPECRE sorting algorithm on a matrix function A(c) for
    c = r + i * 1j, rmin <= r <= rmax, imin <= i <= imax
    dr, di are the increments in r and i respectively

    Parameters
    ----------
    A : Union[Callable, Callable[[complex], Union[np.ndarray, sp.spmatrix],
        spln.LinearOperator]] - the matrix function

    dr : float - the increment in the real part of c

    di : float - the increment in the imaginary part of c

    rmin : float - the minimum value of the real part of c

    rmax : float - the maximum value of the real part of c

    imin : float - the minimum value of the imaginary part of c

    imax : float - the maximum value of the imaginary part of c

    horizontal : bool - whether to sweep horizontally or vertically
        (Default: True)

    *args, **kwargs : additional arguments to pass to the eigensolver

    Returns
    -------

    C : 2D np.ndarray - the meshgrid of c values,
        indexed by C[real_idx, imag_idx]

    ws : 3D np.ndarray - the meshgrid of sorted eigenvalues of A(C), indexed by
        ws[real_idx, imag_idx, eigenvalue_idx]

    vs : 4D np.ndarray - the meshgrid of sorted eigenvectors of A(C), indexed
        by vs[real_idx, imag_idx, :, eigenvector_idx]
    """

    # can be passed as-is to core
    return _SPECRE_core(A, dr, di, rmin, rmax, imin, imax, horizontal, *args, **kwargs)


@multimethod
def SPECRE(
    A: Union[
        Callable,
        Callable[[complex], Union[np.ndarray, sp.spmatrix, spln.LinearOperator]],
    ],
    C: np.ndarray,
    horizontal: bool = True,
    *args,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SPECRE sorting algorithm on a matrix function A(c) for an array of complex
    values C

    Parameters
    ----------
    A : Union[Callable, Callable[[complex], Union[np.ndarray, sp.spmatrix],
        spln.LinearOperator]] - the matrix function

    C : np.ndarray - a 2D array of complex numbers, one axis for the real part
        and one axis for the imaginary part. Either axis can be the real or
        imaginary one.

    horizontal : bool - whether to sweep horizontally or vertically
        (Default: True)

    *args, **kwargs : additional arguments to pass to the eigensolver

    Returns
    -------

    C : 2D np.ndarray - the meshgrid of c values,
        indexed by C[real_idx, imag_idx]

    ws : 3D np.ndarray - the meshgrid of sorted eigenvalues of A(C), indexed by
        ws[real_idx, imag_idx, eigenvalue_idx]

    vs : 4D np.ndarray - the meshgrid of sorted eigenvectors of A(C), indexed
        by vs[real_idx, imag_idx, :, eigenvector_idx]
    """

    # assert that C is 2D
    if not _is_2D_array(C):
        raise ValueError("C must be a 2D array")

    # verify that the stepsizes along each axis are constant, but not
    # necessarily the same
    d1 = np.diff(C.real, axis=0)
    d2 = np.diff(C.imag, axis=1)
    if not np.allclose(d1, d1[0, 0]) or not np.allclose(d2, d2[0, 0]):
        raise ValueError("C must have constant stepsizes along each axis")

    d1, d2 = d1[0, 0], d2[0, 0]

    # figure out which one is the real axis and which one is the imaginary

    if d1.imag == 0 and d2.real == 0:
        transposed = False
        dr, di = d1.real, d2.imag

    elif d1.real == 0 and d2.imag == 0:
        transposed = True
        C = C.T
        dr, di = d2.real, d1.imag

    else:
        raise ValueError("C must have one purely real and one purely imaginary axis")

    dr, di = abs(dr), abs(di)

    # get the min and max values of the real and imaginary parts of C
    rmin, rmax = C.real.min(), C.real.max()
    imin, imax = C.imag.min(), C.imag.max()

    # pass to core
    C, ws, vs = _SPECRE_core(
        A, dr, di, rmin, rmax, imin, imax, horizontal, *args, **kwargs
    )

    # transpose back if necessary
    if transposed:
        C = C.T
        ws = ws.swapaxes(0, 1)
        vs = vs.swapaxes(0, 1)

    return C, ws, vs


"""
    Until PEP646 is fully supported by numpy, mypy, and multimethod, this
    overload will not work since there is no way to typehint the number of
    dimensions of an array.

    This should happen in Python 3.11
"""
# @multimethod
# def SPECRE(
#     A: Union[
#         Callable,
#         Callable[[complex], Union[np.ndarray, sp.spmatrix, spln.LinearOperator]],
#     ],
#     C: np.ndarray,
#     horizontal: bool = True,
#     *args,
#     **kwargs
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     SPECRE sorting algorithm on a matrix function A(c) for an array of complex
#     values C
#
#     Parameters
#     ----------
#     A : Union[Callable, Callable[[complex], Union[np.ndarray, sp.spmatrix],
#         spln.LinearOperator]] - the matrix function
#
#     C : np.ndarray - a 1D array of complex numbers, changing only along either
#         the real or imaginary axis, not both.
#
#     horizontal : bool - whether to sweep horizontally or vertically
#         (Default: True)
#
#     *args, **kwargs : additional arguments to pass to the eigensolver
#
#     Returns
#     -------
#
#     C : 2D np.ndarray - the meshgrid of c values,
#         indexed by C[real_idx, imag_idx]
#
#     ws : 3D np.ndarray - the meshgrid of sorted eigenvalues of A(C), indexed by
#         ws[real_idx, imag_idx, eigenvalue_idx]
#
#     vs : 4D np.ndarray - the meshgrid of sorted eigenvectors of A(C), indexed
#         by vs[real_idx, imag_idx, :, eigenvector_idx]
#     """
#
#     # assert that C is 1D
#     if not _is_1D_array(C):
#         raise ValueError("C must be a 1D array")
#
#     # verify that the stepsize is constant
#     d = np.diff(C)
#     if not np.allclose(d, d[0]):
#         raise ValueError("C must have constant stepsize")
#
#     d = d[0]
#
#     # verify that the stepsize is either purely real or purely imaginary
#     if d.real != 0 and d.imag != 0:
#         raise ValueError(
#             "1D C must change along only the real or imaginary axis, not both"
#         )
#
#     # figure out which axis C is changing along
#     if d.real == 0:
#         real_only = False
#         di = d.imag
#         dr = _DEFAULT_DR
#     else:
#         real_only = True
#         dr = d.real
#         di = _DEFAULT_DI
#
#     dr, di = abs(dr), abs(di)
#
#     # get the min and max values of the real and imaginary parts of C
#     rmin, rmax = C.real.min(), C.real.max()
#     imin, imax = C.imag.min(), C.imag.max()
#
#     # pass to core
#     C, ws, vs = _SPECRE_core(
#         A, dr, di, rmin, rmax, imin, imax, horizontal, *args, **kwargs
#     )
#
#     if real_only:
#         return C[:, 0], ws[:, 0, :], vs[:, 0, :, :]
#     else:
#         return C[0, :], ws[0, :, :], vs[0, :, :, :]


@multimethod
def SPECRE(
    A: Union[
        Callable,
        Callable[[complex], Union[np.ndarray, sp.spmatrix, spln.LinearOperator]],
    ],
    C_R: np.ndarray,
    C_I: np.ndarray,
    horizontal: bool = True,
    *args,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SPECRE sorting algorithm on a matrix function A(c) for an array of complex
    values C = C_R + C_I

    Parameters
    ----------
    A : Union[Callable, Callable[[complex], Union[np.ndarray, sp.spmatrix],
        spln.LinearOperator]] - the matrix function

    C_R : np.ndarray - a 2D meshgrid of the C values, changing along the real
        axis

    C_I : np.ndarray - a 2D meshgrid of the C values, changing along the
        imaginary axis

    horizontal : bool - whether to sweep horizontally or vertically
        (Default: True)

    *args, **kwargs : additional arguments to pass to the eigensolver

    Returns
    -------

    C : 2D np.ndarray - the meshgrid of c values,
        indexed by C[real_idx, imag_idx]

    ws : 3D np.ndarray - the meshgrid of sorted eigenvalues of A(C), indexed by
        ws[real_idx, imag_idx, eigenvalue_idx]

    vs : 4D np.ndarray - the meshgrid of sorted eigenvectors of A(C), indexed
        by vs[real_idx, imag_idx, :, eigenvector_idx]
    """

    # assert that C_R and C_I are 2D
    if not _is_2D_array(C_R) or not _is_2D_array(C_I):
        raise ValueError("C_R and C_I must be 2D arrays")

    # verify that the stepsizes along each axis are constant, but not
    # necessarily the same
    dr1, dr2 = np.diff(C_R, axis=0), np.diff(C_R, axis=1)
    di1, di2 = np.diff(C_I, axis=0), np.diff(C_I, axis=1)

    if (
        not np.allclose(dr1, dr1[0, 0])
        or not np.allclose(dr2, dr2[0, 0])
        or not np.allclose(di1, di1[0, 0])
        or not np.allclose(di2, di2[0, 0])
    ):
        raise ValueError("C must have constant stepsizes along each axis")

    dr1, dr2 = dr1[0, 0], dr2[0, 0]
    di1, di2 = di1[0, 0], di2[0, 0]

    # figure out which axis is the real axis and which is the imaginary axis
    if dr1 != 0 and di2 != 0 and dr1.imag == 0 and di2.real == 0:
        transposed = True
        dr, di = dr1.real, di2.imag
    elif dr2 != 0 and di1 != 0 and dr2.imag == 0 and di1.real == 0:
        transposed = False
        dr, di = dr2.real, di1.imag
    else:
        raise ValueError(
            "C_R must change along the real axis and C_I must change along the "
            "imaginary axis"
        )

    dr, di = abs(dr), abs(di)

    # get the min and max values of the real and imaginary parts of C
    rmin, rmax = C_R.real.min(), C_R.real.max()
    imin, imax = C_I.imag.min(), C_I.imag.max()

    # pass to core
    C, ws, vs = _SPECRE_core(
        A, dr, di, rmin, rmax, imin, imax, horizontal, *args, **kwargs
    )

    if transposed:
        C = C.T
        ws = ws.swapaxes(0, 1)
        vs = vs.swapaxes(0, 1)

    return C, ws, vs


@multimethod
def SPECRE(
    A: Union[
        Callable,
        Callable[[complex], Union[np.ndarray, sp.spmatrix, spln.LinearOperator]],
    ],
    c_R: np.ndarray,
    c_I: np.ndarray,
    horizontal: bool = True,
    *args,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SPECRE sorting algorithm on a matrix function A(c) for an array of complex
    values C = c_R[:, None] + c_I * 1j

    Parameters
    ----------
    A : Union[Callable, Callable[[complex], Union[np.ndarray, sp.spmatrix],
        spln.LinearOperator]] - the matrix function

    c_R : np.ndarray - a 1D array of the real part of the C values

    c_I : np.ndarray - a 1D array of the imaginary part of the C values

    horizontal : bool - whether to sweep horizontally or vertically
        (Default: True)

    *args, **kwargs : additional arguments to pass to the eigensolver

    Returns
    -------

    C : 2D np.ndarray - the meshgrid of c values,
        indexed by C[real_idx, imag_idx]

    ws : 3D np.ndarray - the meshgrid of sorted eigenvalues of A(C), indexed by
        ws[real_idx, imag_idx, eigenvalue_idx]

    vs : 4D np.ndarray - the meshgrid of sorted eigenvectors of A(C), indexed
        by vs[real_idx, imag_idx, :, eigenvector_idx]
    """

    # assert that c_R and c_I are 1D
    if not _is_1D_array(c_R) or not _is_1D_array(c_I):
        raise ValueError("c_R and c_I must be 1D arrays")

    # verify that the stepsize is constant
    dr = np.diff(c_R)
    di = np.diff(c_I)
    if not np.allclose(dr, dr[0]) or not np.allclose(di, di[0]):
        raise ValueError("c_R and c_I must each have constant stepsize")

    dr, di = dr[0], di[0]

    dr, di = abs(dr), abs(di)

    rmin, rmax = c_R.min(), c_R.max()
    imin, imax = c_I.min(), c_I.max()

    # pass to core
    return _SPECRE_core(A, dr, di, rmin, rmax, imin, imax, horizontal, *args, **kwargs)


def _SPECRE_core(
    A: Union[
        Callable,
        Callable[[complex], Union[np.ndarray, sp.spmatrix, spln.LinearOperator]],
    ],
    dr: float,
    di: float,
    rmin: float,
    rmax: float,
    imin: float,
    imax: float,
    horizontal: bool = True,
    *args,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # verify that dr and di are greater than zero
    if dr <= 0 or di <= 0:
        raise ValueError("dr and di must be greater than zero")

    # verify that rmin <= rmax and imin <= imax
    if rmin > rmax or imin > imax:
        raise ValueError("rmin <= rmax and imin <= imax must be true")

    N = A(0).shape[0]

    # initialize

    # -dr to include extra point for central finite difference
    # +2dr/di to include extra points for finite differences
    c_r = np.arange(rmin - dr, rmax + 2 * dr, dr)
    c_i = np.arange(imin, imax + 2 * di, di)

    c = c_r[:, None] + 1j * c_i
    # c is indexed as c[i, j] = c_r[i] + 1j * c_i[j]

    n = c_r.size
    m = c_i.size

    ws = [[None for _ in range(m)] for _ in range(n)]
    # ws is indexed as ws[i][j] = w(c_r[i] + 1j * c_i[j])

    vs = [[None for _ in range(m)] for _ in range(n)]
    # vs is indexed as vs[i][j] = v(c_r[i] + 1j * c_i[j])

    # compute eigenvalues at every c, sorted only by imaginary part
    for i in range(n):
        for j in range(m):
            ws[i][j], vs[i][j] = _get_eigensystem(A(c[i, j]))
            sort_idx = np.argsort(ws[i][j].imag)
            ws[i][j] = np.array(ws[i][j])[sort_idx]
            vs[i][j] = np.array(vs[i][j])[:, sort_idx]

    ws = np.array(ws)
    vs = np.array(vs)

    def cost(i, j, k, l, p):
        # cost at point c[i,j] between eigenvalues E_k and E_l with E_p being
        # the reference eigenvalue

        # central finite difference for the real part
        dwdcr = (ws[i + 1, j, l] - ws[i - 1, j, p]) / (2 * dr)
        # forward finite difference for the imaginary part
        dwdci = (ws[i, j + 1, k] - ws[i, j, p]) / (di)

        CR_residue = np.abs(np.real(dwdcr) - np.imag(dwdci)) + np.abs(
            np.real(dwdci) + np.imag(dwdcr)
        )

        return CR_residue

    i_range = range(1, n - 1)
    j_range = range(0, m - 1)

    # sweep direction
    if horizontal:
        ij_iter = itertools.product(j_range, i_range)
    else:
        ij_iter = itertools.product(i_range, j_range)

    # sort eigenvalues by SPECRE
    for ij in ij_iter:
        if horizontal:
            j, i = ij
        else:
            i, j = ij
        # sorting in the real direction
        sort_idx_l = [None for _ in range(N)]
        # sorting in the imaginary direction
        sort_idx_k = [None for _ in range(N)]
        for p in range(N):
            min_cost = np.inf
            min_k = None
            min_l = None
            for k in range(N):
                for l in range(N):
                    c_ = cost(i, j, k, l, p)
                    if c_ < min_cost and k not in sort_idx_k and l not in sort_idx_l:
                        min_cost = c_
                        min_k = k
                        min_l = l
            sort_idx_k[p] = min_k
            sort_idx_l[p] = min_l
        ws[i + 1, j, :] = ws[i + 1, j, sort_idx_l]
        ws[i, j + 1, :] = ws[i, j + 1, sort_idx_k]
        vs[i + 1, j, :, :] = vs[i + 1, j, :, sort_idx_l]
        vs[i, j + 1, :, :] = vs[i, j + 1, :, sort_idx_k]

    # remove extra points used for central finite difference
    c = c[1:~0, :~0]
    ws = ws[1:~0, :~0, :]
    vs = vs[1:~0, :~0, :, :]

    return c, ws, vs
