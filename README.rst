
=============
`array_split`
=============

.. Start of sphinx doc include.
.. start long description.
.. start badges.

.. image:: https://img.shields.io/pypi/v/array_split.svg
   :target: https://pypi.python.org/pypi/array_split/
   :alt: array_split python package
.. image:: https://github.com/array-split/array_split/actions/workflows/python-test.yml/badge.svg
   :target: https://github.com/array-split/array_split/actions/workflows/python-test.yml
   :alt: array_split python package
.. image:: https://readthedocs.org/projects/array-split/badge/?version=stable
   :target: http://array-split.readthedocs.io/en/stable
   :alt: Documentation Status
.. image:: https://coveralls.io/repos/github/array-split/array_split/badge.svg
   :target: https://coveralls.io/github/array-split/array_split
   :alt: Coveralls Status
.. image:: https://img.shields.io/pypi/l/array_split.svg
   :target: https://pypi.python.org/pypi/array_split/
   :alt: MIT License
.. image:: https://img.shields.io/pypi/pyversions/array_split.svg
   :target: https://pypi.python.org/pypi/array_split/
   :alt: array_split python package
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.889078.svg
   :target: https://doi.org/10.5281/zenodo.889078
.. image:: http://joss.theoj.org/papers/10.21105/joss.00373/status.svg
   :target: http://joss.theoj.org/papers/4b59c7430176ef78c80c6a1100031eb6

.. end badges.

The `array_split <http://array-split.readthedocs.io/en/latest>`_ python package is
an enhancement to existing
`numpy.ndarray  <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_ functions,
such as
`numpy.array_split <http://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html>`_,
`skimage.util.view_as_blocks <http://scikit-image.org/docs/0.13.x/api/skimage.util.html#view-as-blocks>`_
and
`skimage.util.view_as_windows <http://scikit-image.org/docs/0.13.x/api/skimage.util.html#view-as-windows>`_,
which sub-divide a multi-dimensional array into a number of multi-dimensional sub-arrays (slices).
Example application areas include:

**Parallel Processing**
   A large (dense) array is partitioned into smaller sub-arrays which can be
   processed concurrently by multiple processes
   (`multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_
   or `mpi4py <http://pythonhosted.org/mpi4py/>`_) or other memory-limited hardware
   (e.g. GPGPU using `pyopencl <https://mathema.tician.de/software/pyopencl/>`_,
   `pycuda <https://mathema.tician.de/software/pycuda/>`_, etc).
   For GPGPU, it is necessary for sub-array not to exceed the GPU memory and
   desirable for the sub-array shape to be a multiple of the *work-group*
   (`OpenCL <https://en.wikipedia.org/wiki/OpenCL>`_)
   or *thread-block* (`CUDA <https://en.wikipedia.org/wiki/CUDA>`_) size.

**File I/O**
   A large (dense) array is partitioned into smaller sub-arrays which can be
   written to individual files
   (as, for example, a
   `HDF5 Virtual Dataset <https://support.hdfgroup.org/HDF5/docNewFeatures/NewFeaturesVirtualDatasetDocs.html>`_).
   It is often desirable for the individual files not to exceed a specified number
   of (Giga) bytes and, for `HDF5 <https://support.hdfgroup.org/HDF5/>`_, it is desirable
   to have the individual file sub-array shape a multiple of
   the `chunk shape <https://support.hdfgroup.org/HDF5/doc1.8/Advanced/Chunking/index.html>`_.
   Similarly, `out of core <https://en.wikipedia.org/wiki/Out-of-core_algorithm>`_
   algorithms for large dense arrays often involve processing the entire data-set as
   a series of *in-core* sub-arrays. Again, it is desirable for the individual sub-array shape
   to be a multiple of the
   `chunk shape <https://support.hdfgroup.org/HDF5/doc1.8/Advanced/Chunking/index.html>`_.  


The `array_split <http://array-split.readthedocs.io/en/latest>`_ package provides the
means to partition an array (or array shape) using any of the following criteria:

- Per-axis indices indicating the *cut* positions.
- Per-axis number of sub-arrays.
- Total number of sub-arrays (with optional per-axis *number of sections* constraints).
- Specific sub-array shape.
- Specification of *halo* (*ghost*) elements for sub-arrays.
- Arbitrary *start index* for the shape to be partitioned.
- Maximum number of bytes for a sub-array with constraints:

   - sub-arrays are an even multiple of a specified sub-tile shape
   - upper limit on the per-axis sub-array shape


Quick Start Example
===================


   >>> from array_split import array_split, shape_split
   >>> import numpy as np
   >>>
   >>> ary = np.arange(0, 4*9)
   >>> 
   >>> array_split(ary, 4) # 1D split into 4 sections (like numpy.array_split)
   [array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    array([ 9, 10, 11, 12, 13, 14, 15, 16, 17]),
    array([18, 19, 20, 21, 22, 23, 24, 25, 26]),
    array([27, 28, 29, 30, 31, 32, 33, 34, 35])]
   >>> 
   >>> shape_split(ary.shape, 4) # 1D split into 4 parts, returns slice objects 
   array([(slice(0, 9, None),), (slice(9, 18, None),), (slice(18, 27, None),), (slice(27, 36, None),)], 
         dtype=[('0', 'O')])
   >>> 
   >>> ary = ary.reshape(4, 9) # Make ary 2D
   >>> split = shape_split(ary.shape, axis=(2, 3)) # 2D split into 2*3=6 sections
   >>> split.shape
   (2, 3)
   >>> split
   array([[(slice(0, 2, None), slice(0, 3, None)),
           (slice(0, 2, None), slice(3, 6, None)),
           (slice(0, 2, None), slice(6, 9, None))],
          [(slice(2, 4, None), slice(0, 3, None)),
           (slice(2, 4, None), slice(3, 6, None)),
           (slice(2, 4, None), slice(6, 9, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])
   >>> sub_arys = [ary[tup] for tup in split.flatten()] # Create sub-array views from slice tuples.
   >>> sub_arys
   [array([[ 0,  1,  2], [ 9, 10, 11]]),
    array([[ 3,  4,  5], [12, 13, 14]]),
    array([[ 6,  7,  8], [15, 16, 17]]),
    array([[18, 19, 20], [27, 28, 29]]),
    array([[21, 22, 23], [30, 31, 32]]),
    array([[24, 25, 26], [33, 34, 35]])]


Latest sphinx documentation (including more examples)
at http://array-split.readthedocs.io/en/latest/.

.. end long description.

Installation
============

Using ``pip`` (root access required):

   ``pip install array_split``
   
or local user install (no root access required):
   
   ``pip install --user array_split``

or local user install from latest github source:

   ``pip install --user git+git://github.com/array-split/array_split.git#egg=array_split``


Requirements
============

Requires `numpy <http://docs.scipy.org/doc/numpy/>`_ version `>= 1.6`,
python-2 version `>= 2.6` or python-3 version `>= 3.2`.

Testing
=======

Run tests (unit-tests and doctest module docstring tests) using::

   python -m array_split.tests

or, from the source tree, run::

   python setup.py test


Travis CI at:

   https://travis-ci.org/array-split/array_split/

and AppVeyor at:

   https://ci.appveyor.com/project/array-split/array-split

Documentation
=============

Latest sphinx generated documentation is at:

    http://array-split.readthedocs.io/en/latest

and at github *gh-pages*:

    https://array-split.github.io/array_split/

Sphinx documentation can be built from the source::

   python setup.py build_sphinx

with the HTML generated in ``docs/_build/html``.


Latest source code
==================

Source at github:

   https://github.com/array-split/array_split


Bug Reports
===========

To search for bugs or report them, please use the bug tracker at:

   https://github.com/array-split/array_split/issues


Contributing
============

Check out the `CONTRIBUTING doc <https://github.com/array-split/array_split/blob/dev/CONTRIBUTING.rst>`_.


License information
===================

See the file `LICENSE.txt <https://github.com/array-split/array_split/blob/dev/LICENSE.txt>`_
for terms & conditions, for usage and a DISCLAIMER OF ALL WARRANTIES.
