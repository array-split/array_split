"""
===================================
The :mod:`array_split.split` Module
===================================

Defines array splitting functions and classes.

.. currentmodule:: array_split.split

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   shape_factors - Compute *largest* factors of a given integer.
   calculate_num_slices_per_axis - Computes per-axis divisions for a multi-dimensional shape.
   calculate_tile_shape_for_max_bytes - Calculate a tile shape subject to max bytes restriction.
   ArraySplitter - Array shape splitting akin to :func:`numpy.array_split`.
   array_split - Equivalent to :func:`numpy.array_split`.

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright

import array_split as _array_split
import array_split.logging as _logging
import numpy as _np


__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _array_split.__version__


def is_scalar(obj):
    """
    Returns :samp:`True` if argument :samp:`{obj}` is
    a numeric type.
    """
    return hasattr(obj, "__int__") or hasattr(obj, "__long__")


def is_sequence(obj):
    """
    Returns :samp:`True` if argument :samp:`{obj}` is
    a sequence (e.g. a :obj:`list` or :obj:`tuple`, etc).
    """
    return hasattr(obj, "__len__") or hasattr(obj, "__getitem__")


def is_indices(indices_or_sections):
    """
    Returns :samp:`True` if argument :samp:`{indices_or_sections}` is
    a sequence (e.g. a :obj:`list` or :obj:`tuple`, etc).
    """
    return is_sequence(indices_or_sections)


def pad_with_object(sequence, new_length, obj=None):
    """
    Returns :samp:`sequence` :obj:`list` end-padded with :samp:`{obj}`
    elements so that the length of the returned list equals :samp:`{new_length}`.

    """
    if len(sequence) < new_length:
        sequence = \
            list(sequence) + [obj, ] * (new_length - len(sequence))
    elif len(sequence) > new_length:
        raise ValueError(
            "Got len(sequence)=%s which exceeds new_length=%s"
            %
            (len(sequence), new_length)
        )

    return sequence


def pad_with_none(sequence, new_length):
    """
    Returns :samp:`sequence` :obj:`list` end-padded with :samp:`None`
    elements so that the length of the returned list equals :samp:`{new_length}`.

    """
    return pad_with_object(sequence, new_length, obj=None)


def shape_factors(n, dim=2):
    """
    Returns a :obj:`numpy.ndarray` of factors :samp:`f` such
    that :samp:`(len(f) == {dim}) and (numpy.product(f) == {n})`.
    The returned factors are as *square* (*cubic*, etc) as possible.
    For example::

       >>> shape_factors(24, 1)
       array([24])
       >>> shape_factors(24, 2)
       array([4,6])
       >>> shape_factors(24, 3)
       array([2,3,4])
       >>> shape_factors(24, 4)
       array([2,2,2,3])
       >>> shape_factors(24, 5)
       array([1,2,2,2,3])
       >>> shape_factors(24, 6)
       array([1,1,2,2,2,3])

    :type n: :obj:`int`
    :param n: Integer which is factored into :samp:`{dim}` factors.
    :type dim: :obj:`int`
    :param dim: Number of factors.
    :rtype: :obj:`numpy.ndarray`
    :return: A :samp:`({dim},)` shaped array of integers which are factors of :samp:`{n}`.
    """
    if dim <= 1:
        factors = [n, ]
    else:
        for f in range(int(n**(1.0 / float(dim))) + 1, 0, -1):
            if ((n % f) == 0):
                factors = [f, ] + list(shape_factors(n // f, dim=dim - 1))
                break

    factors.sort()
    return _np.array(factors)


def calculate_tile_shape_for_max_bytes(
    array_shape,
    array_itemsize,
    max_tile_bytes,
    max_shape=None,
    sub_tile_shape=None,
    halo=None
):
    """
    Returns a tile shape :samp:`tile_shape`
    such that :samp:`numpy.product(tile_shape)*numpy.sum({array_itemsize}) <= {max_tile_bytes}`.
    Also, if :samp:`{max_shape} is not None`
    then :samp:`numpy.all(tile_shape <= {max_shape}) is True` and
    if :samp:`{sub_tile_shape} is not None`
    the :samp:`numpy.all((tile_shape % {sub_tile_shape}) == 0) is True`.

    :type array_shape: sequence of integers
    :param array_shape: Shape of the array which is to be split into tiles.
    :type array_itemsize: :obj:`int`
    :param array_itemsize: The number of bytes per element of the array to be tiled.
    :type max_tile_bytes: :obj:`int`
    :param max_tile_bytes: The maximum number of bytes for the returned :samp:`tile_shape`.
    :type max_shape: sequence of integers
    :param max_shape: Per axis maximum shapes for the returned :samp:`tile_shape`.
    :type sub_tile_shape: sequence of integers
    :param sub_tile_shape: The returned :samp:`tile_shape` will be an even multiple
       of this sub-tile shape.
    :type halo: sequence of integers
    :param halo: Width of halo voxels in each axis direction.
    """
    array_shape = _np.array(array_shape, dtype="int64")
    array_itemsize = _np.sum(array_itemsize, dtype="int64")

    if max_shape is None:
        max_shape = _np.array(array_shape, copy=True)

    if sub_tile_shape is None:
        sub_tile_shape = _np.ones((len(array_shape),), dtype="int64")

    sub_tile_shape = _np.array(sub_tile_shape, dtype="int64")

    if halo is None:
        halo = _np.zeros((len(array_shape), 2), dtype="int64")
    elif is_scalar(halo):
        halo = _np.zeros((len(array_shape), 2), dtype="int64") + halo
    elif len(_np.array(halo).shape) == 1:
        halo = _np.array([halo, halo]).T.copy()

    if _np.any(array_shape < sub_tile_shape):
        raise ValueError(
            "Got array_shape=%s element less than corresponding sub_tile_shape=%s element."
            %
            (
                array_shape,
                sub_tile_shape
            )
        )

    logger = _logging.getLogger(__name__ + ".calculate_tile_shape_for_max_bytes")
    logger.debug("max_shape=%s", max_shape)
    logger.debug("sub_tile_shape=%s", sub_tile_shape)
    logger.debug("halo=%s", halo)
    array_sub_tile_split_shape = ((array_shape - 1) // sub_tile_shape) + 1
    num_sub_tiles_per_max_bytes = max_tile_bytes // (_np.product(sub_tile_shape) * array_itemsize)
    tile_sub_tile_split_shape = array_sub_tile_split_shape.copy()
    logger.debug("tile_sub_tile_split_shape=%s", tile_sub_tile_split_shape)
    tile_sub_tile_split_shape = \
        _np.minimum(
            tile_sub_tile_split_shape,
            max_shape // sub_tile_shape
        )
    current_axis = 0
    while (
        (current_axis < len(tile_sub_tile_split_shape.shape))
        and
        (
            (
                _np.product(tile_sub_tile_split_shape * sub_tile_shape + _np.sum(halo, axis=1))
                *
                array_itemsize
            )
            >
            max_tile_bytes
        )
    ):
        if current_axis > 1:
            tile_sub_tile_split_shape[current_axis] = 1
            tile_sub_tile_split_shape[current_axis] = \
                (
                    max_tile_bytes
                    //
                    (
                        _np.product(
                            tile_sub_tile_split_shape *
                            sub_tile_shape +
                            _np.sum(
                                halo,
                                axis=1))
                        *
                        array_itemsize
                    )
            )
            if tile_sub_tile_split_shape[current_axis] <= 0:
                tile_sub_tile_split_shape[current_axis] = 1
            current_axis += 1
        else:
            tile_sub_tile_split_shape[current_axis] //= 2
            if tile_sub_tile_split_shape[current_axis] <= 0:
                tile_sub_tile_split_shape[current_axis] = 1
                current_axis += 1
    tile_shape = _np.minimum(array_shape, tile_sub_tile_split_shape * sub_tile_shape)

    tile_split_shape = ((array_shape - 1) // tile_shape) + 1

    tile_shape = (array_sub_tile_split_shape // tile_split_shape) * sub_tile_shape

    return tile_shape


def calculate_num_slices_per_axis(num_slices_per_axis, num_slices, max_slices_per_axis=None):
    """
    Returns :obj:`numpy.ndarray` where non-positive elements of
    the :samp:`num_slices_per_axis` sequence have been replaced with
    positive integer values such that :samp:`numpy.product(return_array) == num_slices`
    and::

       numpy.all(
           return_array[numpy.where(num_slices_per_axis <= 0)]
           <=
           max_slices_per_axis[numpy.where(num_slices_per_axis <= 0)]
       ) is True


    :rtype: :obj:`numpy.ndarray`
    :return: An array :samp:`return_array`
       such that :samp:`numpy.product(return_array) == num_slices`.
    """
    logger = _logging.getLogger(__name__)

    ret_array = _np.array(num_slices_per_axis, copy=True)
    if max_slices_per_axis is None:
        max_slices_per_axis = _np.array([_np.inf, ] * len(num_slices_per_axis))

    max_slices_per_axis = _np.array(max_slices_per_axis)

    if _np.any(max_slices_per_axis <= 0):
        raise ValueError("Got non-positive value in max_slices_per_axis=%s" % max_slices_per_axis)

    while _np.any(ret_array <= 0):
        prd = _np.product(ret_array[_np.where(ret_array > 0)])  # returns 1 for zero-length array
        if (num_slices < prd) or ((num_slices % prd) > 0):
            raise ValueError(
                (
                    "Unable to construct grid of num_slices=%s elements from "
                    +
                    "num_slices_per_axis=%s (with max_slices_per_axis=%s)"
                )
                %
                (num_slices, num_slices_per_axis, max_slices_per_axis)
            )
        ridx = _np.where(ret_array <= 0)
        f = shape_factors(num_slices // prd, ridx[0].shape[0])[::-1]
        if _np.all(f < max_slices_per_axis[ridx]):
            ret_array[ridx] = f
        else:
            for i in range(ridx[0].shape[0]):
                if f[i] < max_slices_per_axis[ridx[0][i]]:
                    ret_array[ridx[0][i]] = f[i]
                else:
                    ret_array[ridx[0][i]] = max_slices_per_axis[ridx[0][i]]
                    prd = _np.product(ret_array[_np.where(ret_array > 0)])
                    while (num_slices % prd) > 0:
                        ret_array[ridx[0][i]] -= 1
                        prd = _np.product(ret_array[_np.where(ret_array > 0)])
                    break
        logger.debug("ridx=%s, f=%s, ret_array=%s", ridx, f, ret_array)
    return ret_array


class ArraySplitter(object):
    """
    Implements array splitting akin to :func:`numpy.array_split` but
    with more options for specifying the sub-array (slice) shape.


    """

    #: Class attribute for :obj:`logging.Logger` logging.
    logger = _logging.getLogger(__name__ + ".ArraySplitter")

    def __init__(
        self,
        array_shape,
        indices_or_sections=None,
        axis=None,
        array_start=None,
        array_itemsize=1
    ):
        """
        Initialise.

        :type array_shape: sequence of :obj:`int`
        :param array_shape: The shape which is to be *split*.
        :type indices_or_sections: :obj:`int` or sequence of :obj:`int`
        :param indices_or_sections: If an integer, indicates the number of
            elements in the calculated *split* array. If a sequence indicates
            the indicies (per axis) at which the splits occur.
        :type axis: :obj:`int` or sequence of :obj:`int`
        :param axis: If an integer, indicates the axis which is to be split.
           Sequence integers indicates the number of slices per axis,
           i.e. if :samp:`{axis} == [3, 5]` then axis :samp:`0` is split into
           3 slices and axis :samp:`1` is split into 5 slices for a total
           of 15 (:samp:`3*5`) rectangular slices in the returned :samp:`(3, 5)`
           shaped slice array.

        """
        #: The shape of the array which is to be split
        self.array_shape = _np.array(array_shape)
        if array_start is None:
            array_start = _np.zeros_like(self.array_shape)
        #: The start index (i.e. assume array indexing starts at self.array_start),
        #: defaults to :samp:`numpy.zeros_like(array_shape)`.
        self.array_start = array_start
        #: The number of bytes per array element
        self.array_itemsize = array_itemsize

        indices_per_axis = None
        if is_indices(indices_or_sections):
            num_subarrays = None
            indices_per_axis = indices_or_sections
            if is_scalar(axis) and (not _np.any([is_sequence(_e) for _e in indices_or_sections])):
                indices_per_axis = [indices_or_sections, ]
        else:
            indices_per_axis = None
            num_subarrays = indices_or_sections

        #: Specifies the indices (per axis) where the splits occur.
        self.indices_per_axis = indices_per_axis

        #: Specifies the size (total number of structure elements)
        #: of the returned split.
        self.split_size = num_subarrays
        split_num_slices_per_axis = None
        if (self.split_size is not None) or (axis is not None):
            if axis is None:
                axis = 0
            if is_sequence(axis):
                split_num_slices_per_axis = pad_with_object(axis, len(self.array_shape), 1)
            else:
                split_num_slices_per_axis = pad_with_object([1, ], len(self.array_shape), 1)
                split_num_slices_per_axis[axis] = self.split_size
        #: Defines number of slices per axis
        self.split_num_slices_per_axis = split_num_slices_per_axis

        self.split_shape = None
        self.split_begs = None
        self.split_ends = None

    def set_split_extents_by_indices_per_axis(self):
        """
        Sets split shape :samp:`{self}.split_shape` and
        split extents (:samp:`{self}.split_begs` and :samp:`{self}.split_ends`)
        from values in :samp:`{self}.indices_per_axis`.
        """
        if self.indices_per_axis is None:
            raise ValueError("Got None for self.indices_per_axis")

        self.indices_per_axis = \
            pad_with_none(self.indices_per_axis, len(self.array_shape))

        # Define the start and stop indices (extents) for each axis slice
        self.split_shape = _np.ones(len(self.array_shape), dtype="int64")
        self.split_begs = [[], ] * len(self.array_shape)
        self.split_ends = [[], ] * len(self.array_shape)
        for i in range(len(self.indices_per_axis)):
            indices = self.indices_per_axis[i]
            if (indices is not None) and (len(indices) > 0):
                self.split_shape[i] = len(indices) + 1
                self.split_begs[i] = _np.zeros((len(indices) + 1,), dtype="int64")
                self.split_begs[i][1:] = indices
                self.split_ends[i] = _np.zeros((len(self.split_begs[i]),), dtype="int64")
                self.split_ends[i][0:-1] = self.split_begs[i][1:]
                self.split_ends[i][-1] = self.array_shape[i]
            else:
                # start and stop is the full width of the axis
                self.split_begs[i] = [0, ]
                self.split_ends[i] = [self.array_shape[i], ]

        self.logger.debug("self.indices_per_axis=%s", self.indices_per_axis)

    def calculate_split_from_extents(self):
        """
        Returns split calculated using extents obtained
        from :samp:`{self}.split_begs` and :samp:`{self}.split_ends`.

        :rtype: :obj:`numpy.ndarray`
        :return:
           A :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           where each element is a :obj:`tuple` of :obj:`slice` objects.
        """
        self.logger.debug("self.split_shape=%s", self.split_shape)
        self.logger.debug("self.split_begs=%s", self.split_begs)
        self.logger.debug("self.split_ends=%s", self.split_ends)

        ret = \
            _np.array(
                [
                    tuple(
                        [
                            slice(
                                self.split_begs[d][idx[d]],
                                self.split_ends[d][idx[d]]
                            )
                            for d in range(len(self.split_shape))
                        ]
                    )
                    for idx in
                    _np.array(
                        _np.unravel_index(
                            _np.arange(0, _np.product(self.split_shape)),
                            self.split_shape
                        )
                    ).T
                ],
                dtype=[("%d" % d, "object") for d in range(len(self.split_shape))]
            ).reshape(self.split_shape)

        return ret

    def calculate_split_by_indices_per_axis(self):
        """
        Returns split calculated using extents obtained
        from :samp:`{self}.indices_per_axis`.

        :rtype: :obj:`numpy.ndarray`
        :return: :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           where each element is a :obj:`tuple` of :obj:`slice` objects.
        """
        self.set_split_extents_by_indices_per_axis()
        return self.calculate_split_from_extents()

    def calculate_axis_split_extents(self, num_sections, size):
        """
        Divides :samp:`range(0, {size})` into (approximately) equal sized
        intervals. Returns :samp:`(begs, ends)` where :samp:`slice(begs[i], ends[i])`
        define the intervals for :samp:`i in range(0, {num_sections})`.

        :type num_sections: :obj:`int`
        :param num_sections: Divide  :samp:`range(0, {size})` into this
           many intervals (approximately) equal sized intervals.
        :type size: :obj:`int`
        :param size: Range for the subdivision.
        :rtype: :obj:`tuple`
        :return: Two element tuple :samp:`(begs, ends)`
           such that :samp:`slice(begs[i], ends[i])` define the
           intervals for :samp:`i in range(0, {num_sections})`.

        """
        section_size = size // num_sections
        if section_size >= 1:
            begs = _np.arange(0, section_size * num_sections, section_size)
            rem = size - section_size * num_sections
            if rem > 0:
                for i in range(rem):
                    begs[i + 1:] += 1
            ends = _np.zeros_like(begs)
            ends[0:-1] = begs[1:]
            ends[-1] = size
        else:
            begs = _np.arange(0, num_sections)
            begs[size:] = size
            ends = begs.copy()
            ends[0:-1] = begs[1:]

        return begs, ends

    def set_split_extents_by_split_size(self):
        """
        Sets split shape :samp:`{self}.split_shape` and
        split extents (:samp:`{self}.split_begs` and :samp:`{self}.split_ends`)
        from values in :samp:`{self}.split_size` and :samp:`{self}.axis`.
        """

        if self.split_size is None:
            if (
                _np.all([s is not None for s in self.split_num_slices_per_axis])
                and
                _np.all([s > 0 for s in self.split_num_slices_per_axis])
            ):
                self.split_size = _np.product(self.split_num_slices_per_axis)
            else:
                raise ValueError(
                    (
                        "Got invalid self.split_num_slices_per_axis=%s, all elements "
                        +
                        "need to be integers greater than zero when self.split_size is None."
                    )
                    %
                    self.split_num_slices_per_axis
                )
        self.logger.debug(
            "Pre  cannonicalise: self.split_num_slices_per_axis=%s",
            self.split_num_slices_per_axis)
        self.split_num_slices_per_axis = \
            calculate_num_slices_per_axis(
                self.split_num_slices_per_axis,
                self.split_size,
                self.array_shape
            )
        self.logger.debug(
            "Post cannonicalise: self.split_num_slices_per_axis=%s",
            self.split_num_slices_per_axis)
        # Define the start and stop indices (extents) for each axis slice
        self.split_shape = self.split_num_slices_per_axis.copy()
        self.split_begs = [[], ] * len(self.array_shape)
        self.split_ends = [[], ] * len(self.array_shape)
        for i in range(len(self.array_shape)):
            self.split_begs[i], self.split_ends[i] = \
                self.calculate_axis_split_extents(
                    self.split_shape[i],
                    self.array_shape[i]
            )

    def calculate_split_by_split_size(self):
        """
        Returns split calculated using extents obtained
        from :samp:`{self}.split_size` and :samp:`{self}.axis`.

        :rtype: :obj:`numpy.ndarray`
        :return: :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           where each element is a :obj:`tuple` of :obj:`slice` objects.
        """
        self.set_split_extents_by_split_size()
        return self.calculate_split_from_extents()

    def calculate_split(self):
        """
        Computes the split.

        :rtype: :obj:`numpy.ndarray`
        :return: :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
            of dimension :samp:`len(self.array_shape)`.
            Each element of the returned array is a :obj:`tuple`
            containing :samp:`len(self.array_shape)` elements, with each element
            being a :obj:`slice` object. Each :obj:`tuple` defines a slice within
            the original bounds of :samp:`self.array_start`
            to :samp:`self.array_start + self.array_shape`.
        """
        split = None

        if self.indices_per_axis is not None:
            split = self.calculate_split_by_indices_per_axis()
        elif (self.split_size is not None) or (self.split_num_slices_per_axis):
            split = self.calculate_split_by_split_size()

        return split


def array_split(ary, indices_or_sections, axis=0):
    """
        Equivalent of :func:`numpy.array_split`, split an array into multiple sub-arrays.

        :type ary: :obj:`numpy.ndarray`
        :param ary: Array to be divided into sub-arrays.
        :type indices_or_sections: :obj:`int` or 1-D array
        :param indices_or_sections: If :samp:`{indices_or_sections}`
           is an integer, :samp:`N`, the array will be divided into :samp:`N`
           (approximately) equal arrays along :samp:`{axis}`.
           If :samp:`{indices_or_sections}` is a 1-D array of sorted integers,
           the entries indicate where along axis the array is split.
           For example, :samp:`indices_or_sections=[2, 3]` would, for :samp:`{axis}=0`,
           result in::

              ary[:2]
              ary[2:3]
              ary[3:]

           If an index exceeds the dimension of the array along axis, an empty sub-array
           is returned correspondingly.
        :type axis: :obj:`int`
        :param axis: The axis along which to split, default is :samp:`0`.
        :rtype: :obj:`list` of :obj:`numpy.ndarray`
        :return: A list of sub-arrays.

        Examples::

           >>> import numpy as np
           >>> from array_split import array_split
           >>> x = np.arange(9.0)
           >>> array_split(x, 3)
           [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]
           >>> array_split(x, 4)
           [array([ 0.,  1.,  2.]), array([ 3.,  4.]), array([ 5.,  6.]), array([ 7.,  8.])]
           >>> array_split(x, [3, 5, 6, 10])
           [array([ 0.,  1.,  2.]),
            array([ 3.,  4.]),
            array([ 5.]),
            array([ 6.,  7.]),
            array([], dtype=float64)]


    """
    return [
        ary[slyce]
        for slyce in
        ArraySplitter(
            array_shape=ary.shape,
            indices_or_sections=indices_or_sections,
            axis=axis
        ).calculate_split().flatten()
    ]

__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest.main()
