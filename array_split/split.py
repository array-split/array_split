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
   ArraySplitter - Implements array splitting akin to :func:`numpy.array_split`.


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
        begs = _np.arange(0, size, section_size)
        rem = size % section_size
        if rem > 0:
            begs = begs[0:-1]
            for i in range(rem):
                begs[i + 1:] += 1
        ends = _np.zeros_like(begs)
        ends[0:-1] = begs[1:]
        ends[-1] = size

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

__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest.main()
