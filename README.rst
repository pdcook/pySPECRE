pySPEC-RE
=========

This is a Python 3 package (may be a ``pip`` package in the future) for computing and sorting complex eigenvalues of matrix functions. This package is based on the SPEC-RE (Sorting Procedure for Eigenvalues based on Cauchy-Riemann Equations) algorithm described in:

    Srinivasan, Usha, and Rangachari Kidambi.
    "A sorting algorithm for complex eigenvalues."
    arXiv preprint arXiv:2006.14254 (2020).

Installation
============

To install this package, simply clone this repository and run::

    pip3 install .

If you plan to edit the source code of the repository, instead install the package with::

    pip3 install -e .

Usage
=====

Usage examples can be found in the ``./examples`` directory.

``black``, ``flake8``, and ``pre-commit``
=========================================

To safeguard against poorly formatted code, the use of ``pre-commit`` with ``black`` and ``flake8`` is required. These will automatically format and lint all Python code before committing changes. The configuration for these tools is already defined in ``.pre-commit-config.yaml``, ``.toml`` and ``.flake8`` respectively. To install and enable ``pre-commit`` simply run::

    pip3 install pre-commit
    pre-commit install

in the project's main directory.

Contributing
============

To install the pre-commit hooks, simply install pre-commit with ``pip3 install pre-commit`` (or use ``pip``), then in this repository ``pre-commit install`` followed by ``pre-commit run --all-files``.

Documentation
=============

The ``pySPECRE`` module provides a sorting algorithm for a matrix function ``A(c)`` where ``c`` is a complex number. The algorithm calculates the sorted eigenvalues and eigenvectors of ``A(c)`` for a range of ``c`` values.

Functions (all overloads of one ``SPECRE`` function)

1::

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
        ...

2::

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
        ...


3::

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
        ...

4::

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
        ...


This function is overloaded and provides different ways to call the sorting algorithm. It takes a matrix function ``A(c)`` as the first argument, where ``c`` is a complex number. The function calculates the sorted eigenvalues and eigenvectors of ``A(c)`` for a range of ``c`` values.

The different versions of the function allow different ways to specify the ``c`` values:

- Version 1:
    - ``dr``: Increment in the real part of ``c``.
    - ``di``: Increment in the imaginary part of ``c``.
    - ``rmin``: Minimum value of the real part of ``c``.
    - ``rmax``: Maximum value of the real part of ``c``.
    - ``imin``: Minimum value of the imaginary part of ``c``.
    - ``imax``: Maximum value of the imaginary part of ``c``.
    - ``horizontal``: Whether to sweep horizontally or vertically (default is ``True``).
    - ``*args``, ``**kwargs``: Additional arguments to pass to the eigensolver.
    - Returns a tuple (C, ws, vs)``:
        - ``C``: 2D numpy array representing the meshgrid of ``c`` values, indexed by ``C[real_idx, imag_idx]``.
        - ``ws``: 3D numpy array representing the meshgrid of sorted eigenvalues of ``A(c)``, indexed by ``ws[real_idx, imag_idx, eigenvalue_idx]``.
        - ``vs``: 4D numpy array representing the meshgrid of sorted eigenvectors of ``A(c)``, indexed by ``vs[real_idx, imag_idx, :, eigenvector_idx]``.

- Version 2:
    - ``C``: 2D numpy array representing the meshgrid of ``c`` values, indexed by ``C[real_idx, imag_idx]``.
    - ``horizontal``: Whether to sweep horizontally or vertically (default is ``True``).
    - ``*args``, ``**kwargs``: Additional arguments to pass to the eigensolver.
    - Returns a tuple ``(C, ws, vs)``:
        - ``C``: 2D numpy array representing the meshgrid of ``c`` values, indexed by ``C[real_idx, imag_idx]``.
        - ``ws``: 3D numpy array representing the meshgrid of sorted eigenvalues of ``A(c)``, indexed by ``ws[real_idx, imag_idx, eigenvalue_idx]``.
        - ``vs``: 4D numpy array representing the meshgrid of sorted eigenvectors of ``A(c)``, indexed by ``vs[real_idx, imag_idx, :, eigenvector_idx]``.

- Version 3:
    - ``C_R``: np.ndarray - a 2D meshgrid of the ``c`` values, changing along the real
        axis

    - ``C_I``: np.ndarray - a 2D meshgrid of the ``c`` values, changing along the
        imaginary axis
    - ``horizontal``: Whether to sweep horizontally or vertically (default is ``True``).
    - ``*args``, ``**kwargs``: Additional arguments to pass to the eigensolver.
    - Returns a tuple ``(C, ws, vs)``:
        - ``C``: 2D numpy array representing the meshgrid of ``c`` values, indexed by ``C[real_idx, imag_idx]``.
        - ``ws``: 3D numpy array representing the meshgrid of sorted eigenvalues of ``A(c)``, indexed by ``ws[real_idx, imag_idx, eigenvalue_idx]``.
        - ``vs``: 4D numpy array representing the meshgrid of sorted eigenvectors of ``A(c)``, indexed by ``vs[real_idx, imag_idx, :, eigenvector_idx]``.

- Version 4:
    - ``c_R``: 1D numpy array representing the real part of ``c`` values.
    - ``c_I``: 1D numpy array representing the imaginary part of ``c`` values.
    - ``horizontal``: Whether to sweep horizontally or vertically (default is ``True``).
    - ``*args``, ``**kwargs``: Additional arguments to pass to the eigensolver.
    - Returns a tuple ``(C, ws, vs)``:
        - ``C``: 2D numpy array representing the meshgrid of ``c`` values, indexed by ``C[real_idx, imag_idx]``.
        - ``ws``: 3D numpy array representing the meshgrid of sorted eigenvalues of ``A(c)``, indexed by ``ws[real_idx, imag_idx, eigenvalue_idx]``.
        - ``vs``: 4D numpy array representing the meshgrid of sorted eigenvectors of ``A(c)``, indexed by ``vs[real_idx, imag_idx, :, eigenvector_idx]``.

Usage
=====
::

    from pySPECRE import pySPECRE

    pySPECRE.SPECRE(...)


