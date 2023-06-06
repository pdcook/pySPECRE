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
