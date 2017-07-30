
===========
Terminology
===========

Definitions:

   *tile*
      A multi-dimensional *sub-array* of a :obj:`numpy.ndarray`.
   *slice*
      A :obj:`tuple` of :obj:`slice` elements defining the extents
      of a tile/sub-array.
   *cut*
      A *division* along an axis to form tiles or slices.
   *split*
      The sub-division (tiling) of an array (or an array shape) resulting from cuts.  
   *halo*
      Per-axis number of elements which specifies the expansion of a tile
      (in the negative and positive axis directions) to form an
      *overlap* of elements with neighbouring tiles. The *overlaps* are often
      referred to as *ghost cells* or *ghost elements*.
   *sub-tile*
      A sub-array of a tile.

====================
Parameter Categories
====================

There are four categories of parameters for specifying a split:

   **Number of tiles**
      The total number of tiles and/or the number of slices per axis.
      The :samp:`{indices_or_sections}` parameter can specify the
      number of tiles in the resulting split (as an :obj:`int`).
     
   **Per-axis split indices**
      The per-axis indices specifying where the array (shape) is to be cut.
      The :samp:`{indices_or_sections}` parameter doubles up to indicate
      the indices at which cuts are to occur.
   
   **Tile shape**
      Explicitly specify the shape of the tile in a split.
      The :samp:`{tile_shape}` parameter (typically as a lone
      *keyword argument*) indicates the tile shape.
   
   **Tile maximum number of bytes**
      Given the number of bytes per array element, a tile shape
      is calculated such that all tiles (including halo extension) of the
      resulting split do not exceed a specified (maximum) number of bytes.
      The :samp:`{array_itemsize}` parameter gives the number of bytes
      per array element and the :samp:`{max_tile_bytes}`
      parameter constrains the maximum number of bytes per tile.

The subsequent sections provides examples from each of these categories.

==================================
Import statements for the examples
==================================

In the examples of the following sections, we assume that the following statement
has been issued to :obj:`import` the relevant functions::

   >>> import numpy
   >>> from array_split import array_split, shape_split, ShapeSplitter

==================================================================
:samp:`array_split`, :samp:`shape_split` and :samp:`ShapeSplitter`
==================================================================

The :func:`array_split.array_split` function is analogous to
the :func:`numpy.array_split` function. It takes a :obj:`numpy.ndarray`
object as an argument and returns a :obj:`list` of tile (:obj:`numpy.ndarray` sub-array
objects) elements::

   >>> numpy.array_split(numpy.arange(0, 10), 3)
   [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
   >>> array_split(numpy.arange(0, 10), 3) # array_split.array_split
   [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]

The :func:`array_split.shape_split` function takes an array *shape* as an
argument instead of an actual :obj:`numpy.ndarray` object, and returns
a :mod:`numpy` `structured array`_
of :obj:`tuple` elements. The tuple elements can then be used to generate
the tiles from a :obj:`numpy.ndarray` of an equivalent shape::

   >>> ary = numpy.arange(0, 10)
   >>> split = shape_split(ary.shape, 3) # returns array of tuples
   >>> split
   array([(slice(0, 4, None),), (slice(4, 7, None),), (slice(7, 10, None),)], 
         dtype=[('0', 'O')])
   >>> [ary[slyce] for slyce in split.flatten()] # generates tile views of ary
   [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]

Each :obj:`tuple` element, of the returned split, has length
equal to the  dimension of the multi-dimensional shape,
i.e. :samp:`N = len({array_shape})`. Each :obj:`tuple`
indicates the indexing extent of a tile.

The :obj:`array_split.ShapeSplitter` class contains the bulk of the split implementation
for the :func:`array_split.shape_split`. The :meth:`array_split.ShapeSplitter.__init__`
constructor takes the same arguments as the :func:`array_split.shape_split` function and
the :meth:`array_split.ShapeSplitter.calculate_split` method computes the split. After
the split computation, some state information is preserved in the
:obj:`array_split.ShapeSplitter` data attributes::

   >>> ary = numpy.arange(0, 10)
   >>> splitter = ShapeSplitter(ary.shape, 3)
   >>> split = splitter.calculate_split()
   >>> split.shape
   (3,)
   >>> split
   array([(slice(0, 4, None),), (slice(4, 7, None),), (slice(7, 10, None),)], 
         dtype=[('0', 'O')])
   >>> [ary[slyce] for slyce in split.flatten()]
   [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
   >>> 
   >>> splitter.split_shape # equivalent to split.shape above
   array([3])
   >>> splitter.split_begs  # start indices for tile extents
   [array([0, 4, 7])]
   >>> splitter.split_ends  # stop indices for tile extents
   [array([ 4,  7, 10])]

Methods of the :obj:`array_split.ShapeSplitter` class can be over-ridden
in sub-classes in order to customise the splitting behaviour.

The examples of the following section explicitly illustrate the behaviour for
the :func:`array_split.shape_split` function, but with minor modifications,
the examples are also relevant for the :func:`array_split.array_split` function
and for instances of the :obj:`array_split.ShapeSplitter` class.

.. _structured array: http://docs.scipy.org/doc/numpy/user/basics.rec.html

.. _splitting-by-number-of-tiles-examples:

============================
Splitting by number of tiles
============================

Splitting an array is performed by specifying:
*total number of tiles* in the final split
and *per-axis number of slices*.

Single axis number of tiles
===========================

When the :samp:`{indices_or_sections}` parameter is specified as an
integer (scalar), it specifies the number of tiles in the returned split::

   >>> split = shape_split([20,], 4)  # 1D, array_shape=[20,], number of tiles=4, default axis=0
   >>> split.shape
   (4,)
   >>> split
   array([(slice(0, 5, None),), (slice(5, 10, None),), (slice(10, 15, None),),
          (slice(15, 20, None),)], 
         dtype=[('0', 'O')])

By default, cuts are made along the :samp:`{axis} = 0` axis. In the multi-dimensional
case, one can over-ride the axis using the :samp:`{axis}` parameter, e.g. for a 2D shape::

   >>> split = shape_split([20,10], 4, axis=1)  # Split along axis=1
   >>> split.shape
   (1, 4)
   >>> split
   array([[(slice(0, 20, None), slice(0, 3, None)),
           (slice(0, 20, None), slice(3, 6, None)),
           (slice(0, 20, None), slice(6, 8, None)),
           (slice(0, 20, None), slice(8, 10, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])


Multiple axes number of tiles
=============================

The :samp:`{axis}` parameter can also be used to specify the number of slices (sections)
per-axis::

   >>> split = shape_split([20, 10], axis=[3, 2])  # Cut into 3*2=6 tiles
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

The array axis 0 has been cut into three sections and axis 1 has been cut into two
sections for a total of :samp:`3*2 = 6` tiles. In general, if :samp:`{axis}` is an
integer (scalar) it indicates the single axis which is to be cut to form slices.
When :samp:`{axis}` is a sequence, then :samp:`{axis}[i]` indicates the number of
sections into which axis :samp:`i` is to be cut. 

In addition, one can also specify a total number of tiles and use the :samp:`{axis}`
parameter to limit which axes are to be cut by specifying non-positive values for
elements of the :samp:`{axis}` sequence. For example, in 3D, cut into 8 tiles, but
only cut the :samp:`axis=1` and :samp:`axis=2` axes::

   >>> split = shape_split([20, 10, 15], 8, axis=[1, 0, 0])  # Cut into 1*?*?=8 tiles
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
   ...     split = shape_split([20, 10, 15], 8, axis=[1, 3, 0])  # Impossible to cut into 1*3*?=8 tiles
   ... except (ValueError,) as e:
   ...     e
   ...
   ValueError('Unable to construct grid of num_slices=8 elements from num_slices_per_axis=[1, 3, 0] (with max_slices_per_axis=[20 10 15])',)

.. _splitting-by-per-axis-split-indices-examples:

=================================
Splitting by per-axis cut indices
=================================

Array splitting is performed by explicitly specifying the
indices at which cuts are performed.

Single axis cut indices
=======================

The :samp:`{indices_or_sections}` parameter can also be used to
specify the location (index values) of cuts::

   >>> split = shape_split([20,], [5, 7, 9])  # 1D, split into 4 tiles, default cut axis=0
   >>> split.shape
   (4,)
   >>> split
   array([(slice(0, 5, None),), (slice(5, 7, None),), (slice(7, 9, None),),
          (slice(9, 20, None),)], 
         dtype=[('0', 'O')])

Here, three cuts have been made to form :samp:`4` slices, cuts at index :samp:`5`, index :samp:`7`
and index :samp:`9`.

Similarly, in 2D, the :samp:`{indices_or_sections}` cut indices can made
along :samp:`{axis} = 1` only::

   >>> split = shape_split([20, 13], [5, 7, 9], axis=1)  # 2D, cut into 4 tiles, cut axis=1
   >>> split.shape
   (1, 4)
   >>> split
   array([[(slice(0, 20, None), slice(0, 5, None)),
           (slice(0, 20, None), slice(5, 7, None)),
           (slice(0, 20, None), slice(7, 9, None)),
           (slice(0, 20, None), slice(9, 13, None))]], 
         dtype=[('0', 'O'), ('1', 'O')])

Multiple axes cut indices
=========================

The :samp:`{indices_or_sections}` parameter can also be used to cut
along multiple axes. In this case, the :samp:`{indices_or_sections}`
parameter is specified as a *sequence of sequence*,
so that :samp:`{indices_or_sections}[i]` specifies the cut
indices along axis :samp:`i`.
For example, in 3D, cut along :samp:`axis=1` and :samp:`axis=2` only::

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
that the cut indices for :samp:`axis=0` are :samp:`[]` (i.e. no cuts), the
cut indices for :samp:`axis=1` are :samp:`[7]` (a single cut at index :samp:`7`)
and the cut indices for :samp:`axis=2` are :samp:`[15, 30, 45]` (three cuts).

.. _splitting-by-tile-shape-examples:

=======================
Splitting by tile shape
=======================

The tile shape can be explicitly set with the :samp:`{tile_shape}` parameter,
e.g. in 1D::

   >>> split = shape_split([20,], tile_shape=[6,])  # Cut into (6,) shaped tiles
   >>> split.shape
   (4,)
   >>> split
   array([(slice(0, 6, None),), (slice(6, 12, None),), (slice(12, 18, None),),
          (slice(18, 20, None),)], 
         dtype=[('0', 'O')])

and 2D::

   >>> split = shape_split([20, 32], tile_shape=[6, 16])  # Cut into (6, 16) shaped tiles
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

.. _splitting-by-maximum-bytes-per-tile-examples:

===================================
Splitting by maximum bytes per tile
===================================

Tile shape can constrained by specifying a maximum number of bytes
per tile by specifying the :samp:`array_itemsize` and
the :samp:`max_tile_bytes` parameters. In 1D:: 

   >>> split = shape_split(
   ...   array_shape=[512,],
   ...   array_itemsize=1,  # Default value
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

The preference is to cut into (:samp:`'C'` order) contiguous memory tiles.


Tile shape upper bound constraint
=================================

The split can be influenced by specifying the :samp:`{max_tile_shape}`
parameter. For the previous 2D example, cuts can for forced
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


Sub-tile shape constraint
=========================

The split can also be influenced by specifying the :samp:`{sub_tile_shape}`
parameter which forces the tile shape to be an even multiple of
the :samp:`{sub_tile_shape}`::

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

.. _the-array_start-parameter-examples:

===================================
The :samp:`{array_start}` parameter
===================================

The :samp:`{array_start}` argument to the :func:`array_split.shape_split` function
and the :meth:`array_split.ShapeSplitter.__init__` constructor specifies
an index offset for the slices in the returned :obj:`tuple` of :obj:`slice` objects::

   >>> split = shape_split((15,), 3)
   >>> split 
   array([(slice(0, 5, None),), (slice(5, 10, None),), (slice(10, 15, None),)], 
         dtype=[('0', 'O')])
   >>> split = shape_split((15,), 3, array_start=(20,))
   >>> split
   array([(slice(20, 25, None),), (slice(25, 30, None),),
          (slice(30, 35, None),)], 
         dtype=[('0', 'O')])

.. _the-halo-parameter-examples:

============================
The :samp:`{halo}` parameter
============================

The :samp:`{halo}` parameter can be used to generate tiles
which overlap with neighbouring tiles by a specified number
of array elements (in each axis direction)::

   >>> from array_split import ARRAY_BOUNDS, NO_BOUNDS
   >>> split = shape_split([16,], 4) # No halo
   >>> split.shape
   (4,)
   >>> split
   array([(slice(0, 4, None),), (slice(4, 8, None),), (slice(8, 12, None),),
          (slice(12, 16, None),)], 
         dtype=[('0', 'O')])
   >>> split = shape_split([16,], 4, halo=2, tile_bounds_policy=ARRAY_BOUNDS) # halo width = 2
   >>> split.shape
   (4,)
   >>> split
   array([(slice(0, 6, None),), (slice(2, 10, None),), (slice(6, 14, None),),
          (slice(10, 16, None),)], 
         dtype=[('0', 'O')])
   >>> split = shape_split(
   ... [16,],
   ... 4,
   ... halo=2,
   ... tile_bounds_policy=NO_BOUNDS  # halo width = 2 and tile halos extend outside array_shape bounds
   ... )
   >>> split.shape
   (4,)
   >>> split
   array([(slice(-2, 6, None),), (slice(2, 10, None),), (slice(6, 14, None),),
          (slice(10, 18, None),)], 
         dtype=[('0', 'O')])

The :samp:`tile_bounds_policy` parameter specifies whether the :samp:`{halo}`
extended tiles can extend beyond the bounding box defined by the *start*
index :samp:`{array_start}` and the *stop* index :samp:`{array_start} + {array_shape}`.

Asymmetric halo extensions can also be specified::
   
   >>> split = shape_split(
   ... [16,],
   ... 4,
   ... halo=((1,2),),
   ... tile_bounds_policy=NO_BOUNDS
   ... )
   >>> split.shape
   (4,)
   >>> split
   array([(slice(-1, 6, None),), (slice(3, 10, None),), (slice(7, 14, None),),
          (slice(11, 18, None),)], 
         dtype=[('0', 'O')])


For an :samp:`N` dimensional split (i.e. :samp:`N = len(array_shape)`), the :samp:`{halo}`
parameter can be either a

   scalar
      Tiles are extended by :samp:`{halo}` elements in the negative and positive
      directions for all axes.

   1D sequence
      Tiles are extended by :samp:`{halo[a]}` elements in the negative and positive
      directions for axis :samp:`a`.

   2D sequence
      Tiles are extended by :samp:`{halo[a][0]}` elements in the negative direction
      and :samp:`{halo[a][1]}` in the positive direction for axis :samp:`a`.

For example, in 3D:
 
   >>> split = shape_split(
   ... [16, 8, 8],
   ... 2,
   ... halo=1,  # halo=1 in +ve and -ve directions for all axes
   ... tile_bounds_policy=NO_BOUNDS
   ... )
   >>> split.shape
   (2, 1, 1)
   >>> split
   array([[[(slice(-1, 9, None), slice(-1, 9, None), slice(-1, 9, None))]],
   <BLANKLINE>
          [[(slice(7, 17, None), slice(-1, 9, None), slice(-1, 9, None))]]], 
         dtype=[('0', 'O'), ('1', 'O'), ('2', 'O')])
   >>> split = shape_split(
   ... [16, 8, 8],
   ... 2,
   ... halo=(1, 2, 3),  # halo=1 for axis 0, halo=2 for axis 1, halo=3 for axis=2
   ... tile_bounds_policy=NO_BOUNDS
   ... )
   >>> split.shape
   (2, 1, 1)
   >>> split
   array([[[(slice(-1, 9, None), slice(-2, 10, None), slice(-3, 11, None))]],
   <BLANKLINE>
          [[(slice(7, 17, None), slice(-2, 10, None), slice(-3, 11, None))]]], 
         dtype=[('0', 'O'), ('1', 'O'), ('2', 'O')])
   >>> split = shape_split(
   ... [16, 8, 8],
   ... 2,
   ... halo=((1, 2), (3, 4), (5, 6)),  # halo=1 for -ve axis 0, halo=2 for +ve axis 0
   ...                                 # halo=3 for -ve axis 1, halo=4 for +ve axis 1
   ...                                 # halo=5 for -ve axis 2, halo=6 for +ve axis 2
   ... tile_bounds_policy=NO_BOUNDS
   ... )
   >>> split.shape
   (2, 1, 1)
   >>> split
   array([[[(slice(-1, 10, None), slice(-3, 12, None), slice(-5, 14, None))]],
   <BLANKLINE>
          [[(slice(7, 18, None), slice(-3, 12, None), slice(-5, 14, None))]]], 
         dtype=[('0', 'O'), ('1', 'O'), ('2', 'O')])


