.. _`array_split examples`:

+++++++++++++++++++++++++++
:mod:`array_split` Examples
+++++++++++++++++++++++++++

.. sectionauthor:: Shane J. Latham

.. toctree::
   :maxdepth: 4


===========
Terminology
===========

Definitions:

   *tiles*
      The multi-dimensional *sub-arrays* of an array decomposition.
   *slice*
      Equivalent to a tile, a :obj:`tuple` of :obj:`slice` elements indicating the extents
      of a tile/sub-array.
   *split*
      A *cut* along one or more (array) axes to form tiles.
   *halo*
      An expansion of a tile (along one or more axes) to form an
      *overlap* with neighbouring tiles. Also often referred to as
      *ghost cells* or *ghost elements*.
   *sub-tile*
      The sub-array formed by splitting a tile.

====================
Parameter Categories
====================

There are four categories of parameters for specifying a split:

   **Number of tiles**
      The total number of tiles and/or the number of slices per axis.
      The :samp:`{indices_or_sections}` parameter can specify the
      number of tiles in the resulting split (as an :obj:`int`).
     
   **Per-axis split indices**
      The per-axis indices where the array (shape) is to be split.
      The :samp:`{indices_or_sections}` parameter doubles up to indicate
      the indices at which splits are to occur.
   
   **Tile shape**
      Explicitly specify the shape of the tile in the split.
      The :samp:`{tile_shape}` parameter (typically as a lone
      *keyword argument*) indicates the tile shape.
   
   **Tile maximum number of bytes**
      Given the number of bytes per array element, a tile shape
      is calculated such that all tiles of the split do not exceed a specified
      (maximum) number of bytes. The :samp:`{array_itemsize}` parameter
      gives the number of bytes per array element and the :samp:`{max_tile_bytes}`
      parameter constrains the maximum number of bytes per tile.

The subsequent sections provides examples from each of these categories.

In the examples, we assume that the following statement has been
issued to import the relevant functions::

   >>> import numpy
   >>> from array_split import array_split, shape_split, ShapeSplitter


============================
Splitting by number of tiles
============================


Splitting along a single axis into a number of tiles
====================================================

Number of tiles is provided as input parameter (default :samp:`{axis}=0`)::

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


Splitting along multiple axes into a number of tiles
====================================================

Number of slices per-axis is provided as input parameter::

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
with positive values such that :samp:`numpy.product(axis)` equals
the number of requested tiles (:samp:`= 8` above).
Raises :obj:`ValueError` if the impossible is attempted::

   >>> try:
   ...     split = shape_split([20, 10, 15], 8, axis=[1, 3, 0])  # Impossible to split into 1*3*?=8 tiles
   ... except (ValueError,) as e:
   ...     e
   ...
   ValueError('Unable to construct grid of num_slices=8 elements from num_slices_per_axis=[1, 3, 0] (with max_slices_per_axis=[20 10 15])',)

===================================
Splitting by per-axis split indices
===================================

Splitting along a single axis with per-axis split indices
=========================================================

Indices of splits provided as input parameter::

   >>> split = shape_split([20,], [5, 7, 9])  # 1D, split into 4 tiles, default cut axis=0
   >>> split.shape
   (4,)
   >>> split
   array([(slice(0, 5, None),), (slice(5, 7, None),), (slice(7, 9, None),),
          (slice(9, 20, None),)], 
         dtype=[('0', 'O')])

In 2D, split :samp:`axis=1` only::

   >>> split = shape_split([20, 13], [5, 7, 9], axis=1)  # 2D, split into 4 tiles, cut axis=1
   >>> split.shape
   (1, 4)
   >>> split
   array([[(slice(0, 20, None), slice(0, 5, None)),
           (slice(0, 20, None), slice(5, 7, None)),
           (slice(0, 20, None), slice(7, 9, None)),
           (slice(0, 20, None), slice(9, 13, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])

Splitting along multiple axes with per-axis split indices
=========================================================

In 3D, split along :samp:`axis=1` and :samp:`axis=2` only::

   >>> split = shape_split([20, 13, 64], [[], [7], [15, 30, 45]])  # 3D, split into 8 tiles, no cuts on axis=0
   >>> split.shape
   (1, 2, 4)
   >>> split
   array([[[(slice(0, 20, None), slice(0, 7, None), slice(0, 15, None)),
            (slice(0, 20, None), slice(0, 7, None), slice(15, 30, None)),
            (slice(0, 20, None), slice(0, 7, None), slice(30, 45, None)),
            (slice(0, 20, None), slice(0, 7, None), slice(45, 64, None))],
           [(slice(0, 20, None), slice(7, 13, None), slice(0, 15, None)),
            (slice(0, 20, None), slice(7, 13, None), slice(15, 30, None)),
            (slice(0, 20, None), slice(7, 13, None), slice(30, 45, None)),
            (slice(0, 20, None), slice(7, 13, None), slice(45, 64, None))]]], 
         dtype=[('0', 'O'), ('1', 'O'), ('2', 'O')])

The :samp:`{indices_or_sections}=[[], [7], [15, 30, 45]]` parameter indicates
that the cut indices for :samp:`axis=0` are :samp:`[]` (i.e. no splits), the
cut indices for :samp:`axis=1` are :samp:`[7]` (a single split at index :samp:`7`)
and the cut indices for :samp:`axis=2` are :samp:`[15, 30, 45]` (three splits).
 
=======================
Splitting by tile shape
=======================

Explicitly set the tile shape, 1D::

   >>> split = shape_split([20,], tile_shape=[6,])  # Split (6,) shaped tiles
   >>> split.shape
   (4,)
   >>> split
   array([(slice(0, 6, None),), (slice(6, 12, None),), (slice(12, 18, None),),
          (slice(18, 20, None),)], 
         dtype=[('0', 'O')])

and 2D::

   >>> split = shape_split([20, 32], tile_shape=[6, 16])  # Split into (6, 16) shaped tiles
   >>> split.shape
   (4, 2)
   >>> split
   array([[(slice(0, 6, None), slice(0, 16, None)),
           (slice(0, 6, None), slice(16, 32, None))],
          [(slice(6, 12, None), slice(0, 16, None)),
           (slice(6, 12, None), slice(16, 32, None))],
          [(slice(12, 18, None), slice(0, 16, None)),
           (slice(12, 18, None), slice(16, 32, None))],
          [(slice(18, 20, None), slice(0, 16, None)),
           (slice(18, 20, None), slice(16, 32, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])


===================================
Splitting by maximum bytes per tile
===================================

1D split, tile shape 

   >>> split = shape_split(
   ...   array_shape=[512,],
   ...   array_itemsize=1,
   ...   max_tile_bytes=512 # Equals number of array bytes
   ... )
   ...
   >>> split.shape
   (1,)
   >>> split
   array([(slice(0, 512, None),)], 
         dtype=[('0', 'O')])


Double the array per-element number of bytes::

   >>> split = shape_split(
   ...   array_shape=[512,],
   ...   array_itemsize=2,
   ...   max_tile_bytes=512 # Equals half the number of array bytes
   ... )
   ...
   >>> split.shape
   (2,)
   >>> split
   array([(slice(0, 256, None),), (slice(256, 512, None),)], 
         dtype=[('0', 'O')])


Decrement :samp:`{max_tile_bytes}` to :samp:`511` to split into 3 tiles::

   >>> split = shape_split(
   ...   array_shape=[512,],
   ...   array_itemsize=2,
   ...   max_tile_bytes=511 # Less than half the number of array bytes
   ... )
   ...
   >>> split.shape
   (3,)
   >>> split
   array([(slice(0, 171, None),), (slice(171, 342, None),),
          (slice(342, 512, None),)], 
         dtype=[('0', 'O')])

Note that the split is calculated so that tiles are approximately equal in size.

In 2D::

   >>> split = shape_split(
   ...   array_shape=[512, 1024],
   ...   array_itemsize=1,
   ...   max_tile_bytes=512*512
   ... )
   ...
   >>> split.shape
   (2, 1)
   >>> split
   array([[(slice(0, 256, None), slice(0, 1024, None))],
          [(slice(256, 512, None), slice(0, 1024, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])

and increasing :samp:`{array_itemsize}` to :samp:`4`::

   >>> split = shape_split(
   ...   array_shape=[512, 1024],
   ...   array_itemsize=4,
   ...   max_tile_bytes=512*512
   ... )
   ...
   >>> split.shape
   (8, 1)
   >>> split
   array([[(slice(0, 64, None), slice(0, 1024, None))],
          [(slice(64, 128, None), slice(0, 1024, None))],
          [(slice(128, 192, None), slice(0, 1024, None))],
          [(slice(192, 256, None), slice(0, 1024, None))],
          [(slice(256, 320, None), slice(0, 1024, None))],
          [(slice(320, 384, None), slice(0, 1024, None))],
          [(slice(384, 448, None), slice(0, 1024, None))],
          [(slice(448, 512, None), slice(0, 1024, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])

The preference is to split into (:samp:`'C'` order) contiguous memory tiles.


Constraining split with tile shape upper bound
==============================================

The split can be influenced by specifying the :samp:`{max_tile_shape}`
parameter. For the previous 2D example, we can force splitting
along :samp:`axis=1` by constraining the tile shape::

   >>> split = shape_split(
   ...   array_shape=[512, 1024],
   ...   array_itemsize=4,
   ...   max_tile_bytes=512*512,
   ...   max_tile_shape=[numpy.inf, 256]
   ... )
   ...
   >>> split.shape
   (2, 4)
   >>> split
   array([[(slice(0, 256, None), slice(0, 256, None)),
           (slice(0, 256, None), slice(256, 512, None)),
           (slice(0, 256, None), slice(512, 768, None)),
           (slice(0, 256, None), slice(768, 1024, None))],
          [(slice(256, 512, None), slice(0, 256, None)),
           (slice(256, 512, None), slice(256, 512, None)),
           (slice(256, 512, None), slice(512, 768, None)),
           (slice(256, 512, None), slice(768, 1024, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])


Constraining split with shape with sub-tiling
=============================================

The split can also be influenced by specifying the :samp:`{sub_tile_shape}`
parameter which forces the tile shape to be an even multiple of
the  :samp:`{sub_tile_shape}`::

   >>> split = shape_split(
   ...   array_shape=[512, 1024],
   ...   array_itemsize=4,
   ...   max_tile_bytes=512*512,
   ...   max_tile_shape=[numpy.inf, 256],
   ...   sub_tile_shape=(15, 10)
   ... )
   ...
   >>> split.shape
   (3, 5)
   >>> split
   array([[(slice(0, 180, None), slice(0, 210, None)),
           (slice(0, 180, None), slice(210, 420, None)),
           (slice(0, 180, None), slice(420, 630, None)),
           (slice(0, 180, None), slice(630, 840, None)),
           (slice(0, 180, None), slice(840, 1024, None))],
          [(slice(180, 360, None), slice(0, 210, None)),
           (slice(180, 360, None), slice(210, 420, None)),
           (slice(180, 360, None), slice(420, 630, None)),
           (slice(180, 360, None), slice(630, 840, None)),
           (slice(180, 360, None), slice(840, 1024, None))],
          [(slice(360, 512, None), slice(0, 210, None)),
           (slice(360, 512, None), slice(210, 420, None)),
           (slice(360, 512, None), slice(420, 630, None)),
           (slice(360, 512, None), slice(630, 840, None)),
           (slice(360, 512, None), slice(840, 1024, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])

