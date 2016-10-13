
.. image:: https://travis-ci.org/array-split/array_split.svg?branch=dev
    :target: https://travis-ci.org/array-split/array_split

=============
`array_split`
=============

The `array_split <https://array-split.github.io/array_split>`_ python package is
an enhanced version of
`numpy.array_split <http://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html>`_
for sub-dividing multi-dimensional arrays into sub-arrays (slices).

Examples
========


   >>> from array_split import array_split, shape_split
   >>> import numpy as np
   >>>
   >>> ary = np.arange(0, 4*9)
   >>> 
   >>> array_split(ary, 4) # 1D split into 4 sections (like numpy.array_split)
   [array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([ 9, 10, 11, 12, 13, 14, 15, 16, 17]),
    array([18, 19, 20, 21, 22, 23, 24, 25, 26]), array([27, 28, 29, 30, 31, 32, 33, 34, 35])]
   >>> 
   >>> shape_split(ary.shape, 4) # 1D split into 4, slice objects instead of numpy.ndarray views 
   array([(slice(0, 9, None),), (slice(9, 18, None),), (slice(18, 27, None),),
          (slice(27, 36, None),)], 
         dtype=[('0', 'O')])
   >>> 
   >>> ary = ary.reshape(4, 9) # Make ary 2D
   >>> shape_split(ary.shape, axis=(2, 3)) # 2D split into 2*3=6 sections
   array([[(slice(0, 2, None), slice(0, 3, None)),
           (slice(0, 2, None), slice(3, 6, None)),
           (slice(0, 2, None), slice(6, 9, None))],
          [(slice(2, 4, None), slice(0, 3, None)),
           (slice(2, 4, None), slice(3, 6, None)),
           (slice(2, 4, None), slice(6, 9, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])



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


Testing
=======

Run unit tests using::

   python -m array_split.tests

Travis CI at::

   https://travis-ci.org/array-split/array_split/

Documentation
=============

Latest sphinx generated documentation is at:

    https://array-split.github.io/array_split

Latest source code
==================

Source at github:

    https://github.com/array-split/array_split


License information
===================

See the file `LICENSE.txt <https://github.com/array-split/array_split/blob/dev/LICENSE.txt>`_
for terms & conditions, for usage and a DISCLAIMER OF ALL WARRANTIES.
