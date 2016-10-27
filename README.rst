
=============
`array_split`
=============

.. Start of sphinx doc include.
.. start long description.

.. image:: https://img.shields.io/pypi/v/array_split.svg
   :target: https://pypi.python.org/pypi/array_split/
   :alt: array_split python package
.. image:: https://travis-ci.org/array-split/array_split.svg?branch=dev
   :target: https://travis-ci.org/array-split/array_split
   :alt: Build Status
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


The `array_split <http://array-split.readthedocs.io/en/latest>`_ python package is
a modest enhancement to the
`numpy.array_split <http://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html>`_
function for sub-dividing multi-dimensional arrays into sub-arrays (slices). The main motivation
comes from parallel processing where one desires to split (decompose) a large array
(or multiple arrays) into smaller sub-arrays which can be processed concurrently by
other processes (`multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_ or
`mpi4py <http://pythonhosted.org/mpi4py/>`_) or other memory-limited hardware
(e.g. GPGPU using `pyopencl <https://mathema.tician.de/software/pyopencl/>`_,
`pycuda <https://mathema.tician.de/software/pycuda/>`_, etc).


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
   >>> shape_split(ary.shape, 4) # 1D split into 4 sections, slice objects instead of numpy.ndarray views 
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
   >>> sub_arys = [ary[tup] for tup in split.flatten()] # Split ary into sub-array views using the slice tuples.
   >>> sub_arys
   [array([[ 0,  1,  2], [ 9, 10, 11]]),
    array([[ 3,  4,  5], [12, 13, 14]]),
    array([[ 6,  7,  8], [15, 16, 17]]),
    array([[18, 19, 20], [27, 28, 29]]),
    array([[21, 22, 23], [30, 31, 32]]),
    array([[24, 25, 26], [33, 34, 35]])]


Latest sphinx documentation examples at http://array-split.readthedocs.io/en/latest/examples/.

.. end long description.

Installation
============

Using ``pip``::

   pip install array_split # with root access
   
or::
   
   pip install --user array_split # no root/sudo permissions required

From latest github source::

    git clone https://github.com/array-split/array_split.git
    cd array_split
    python setup.py install --user

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


Documentation
=============

Latest sphinx generated documentation is at:

    http://array-split.readthedocs.io/en/latest

Latest source code
==================

Source at github:

    https://github.com/array-split/array_split


License information
===================

See the file `LICENSE.txt <https://github.com/array-split/array_split/blob/dev/LICENSE.txt>`_
for terms & conditions, for usage and a DISCLAIMER OF ALL WARRANTIES.
