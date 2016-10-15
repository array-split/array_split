.. _`array_split examples`:

+++++++++++++++++++++++++++
:mod:`array_split` Examples
+++++++++++++++++++++++++++

.. sectionauthor:: Shane J. Latham

.. toctree::
   :maxdepth: 4

In the following, we assume that the following statement has been
issued to import the relevant functions::

   >>> from array_split import array_split, shape_split, ShapeSplitter


===========
Terminology
===========

Definitions:

   *tiles*
      The multi-dimensional *sub-arrays* of an array decomposition.
   *slice*
      Akin to a :obj:`slice` object, a contiguous range of elements along an axis.
   *splitting*
      An array is *cut* along one or more axes to form tiles.
   *halo*
      An expansion of a tile (along one or more axes) to form an
      *overlap* with neighbouring tiles. Also often referred to as
      *ghost cells* or *ghost elements*.
   *sub-tile*
      The sub-array formed by spliting a tile.


====================================================
Splitting along a single axis into a number of tiles
====================================================

Number of tiles and axis are provided as input parameters::

   >>> split = shape_split([20,], 4)  # 1D, array_shape=[20,], number of tiles=4, default axis=0
   >>> split.shape
   (4,)
   >>> split
   array([(slice(0, 5, None),), (slice(5, 10, None),), (slice(10, 15, None),),
          (slice(15, 20, None),)], 
         dtype=[('0', 'O')])

For 2D shape::

   >>> split = shape_split([20,10], 4, axis=1)  # Split along axis=1
   >>> split.shape
   (1, 4)
   >>> split
   array([[(slice(0, 20, None), slice(0, 3, None)),
           (slice(0, 20, None), slice(3, 6, None)),
           (slice(0, 20, None), slice(6, 8, None)),
           (slice(0, 20, None), slice(8, 10, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])


====================================================
Splitting along multiple axes into a number of tiles
====================================================

Number of tiles per-axis is provided as input parameter::

   >>> split = shape_split([20, 10], axis=[3, 2])  # Split into 3*2=6 tiles
   >>> split.shape
   (3, 2)
   >>> split
   array([[(slice(0, 7, None), slice(0, 5, None)),
           (slice(0, 7, None), slice(5, 10, None))],
          [(slice(7, 14, None), slice(0, 5, None)),
           (slice(7, 14, None), slice(5, 10, None))],
          [(slice(14, 20, None), slice(0, 5, None)),
           (slice(14, 20, None), slice(5, 10, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])


In 3D, split into 8 tiles, but only split the :samp:`axis=1` and :samp:`axis=2` axes::

   >>> split = shape_split([20, 10, 15], 8, axis=[1, 0, 0])  # Split into 1*?*?=8 tiles
   >>> split.shape
   (1, 4, 2)
   >>> split
   array([[[(slice(0, 20, None), slice(0, 3, None), slice(0, 8, None)),
            (slice(0, 20, None), slice(0, 3, None), slice(8, 15, None))],
           [(slice(0, 20, None), slice(3, 6, None), slice(0, 8, None)),
            (slice(0, 20, None), slice(3, 6, None), slice(8, 15, None))],
           [(slice(0, 20, None), slice(6, 8, None), slice(0, 8, None)),
            (slice(0, 20, None), slice(6, 8, None), slice(8, 15, None))],
           [(slice(0, 20, None), slice(8, 10, None), slice(0, 8, None)),
            (slice(0, 20, None), slice(8, 10, None), slice(8, 15, None))]]], 
         dtype=[('0', 'O'), ('1', 'O'), ('2', 'O')])

In the above, non-positive elements of :samp:`axis` are replaced
so that :samp:`numpy.product(axis)` equals the number of requested tiles (:samp:`= 8` above).
Raises :obj:`ValueError` if the impossible is attempted::

   >>> try:
   ...     split = shape_split([20, 10, 15], 8, axis=[1, 3, 0])  # Impossible to split into 1*3*?=8 tiles
   ... except (ValueError,) as e:
   ...     e
   ...
   ValueError('Unable to construct grid of num_slices=8 elements from num_slices_per_axis=[1, 3, 0] (with max_slices_per_axis=[20 10 15])',)


==================================
Splitting with specific tile shape
==================================

