"""
===================================
The :mod:`array_split.split` Module
===================================

.. currentmodule:: array_split.split

Defines array splitting functions and classes.

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   shape_factors - Compute *largest* factors of a given integer.
   calculate_num_slices_per_axis - Computes per-axis divisions for a multi-dimensional shape.
   calculate_tile_shape_for_max_bytes - Calculate a tile shape subject to max bytes restriction.
   convert_halo_to_array_form - converts halo argument to :samp:`(ndim, 2)` shaped array.
   ShapeSplitter - Splits a given shape into slices.
   shape_split - Splits a specified shape and returns :obj:`numpy.ndarray` of :obj:`slice` elements.
   array_split - Equivalent to :func:`numpy.array_split`.

Attributes
==========

.. autodata:: ARRAY_BOUNDS
.. autodata:: NO_BOUNDS

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
       array([4, 6])
       >>> shape_factors(24, 3)
       array([2, 3, 4])
       >>> shape_factors(24, 4)
       array([2, 2, 2, 3])
       >>> shape_factors(24, 5)
       array([1, 2, 2, 2, 3])
       >>> shape_factors(24, 6)
       array([1, 1, 2, 2, 2, 3])

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
    max_tile_shape=None,
    sub_tile_shape=None,
    halo=None
):
    """
    Returns a tile shape :samp:`tile_shape`
    such that :samp:`numpy.product(tile_shape)*numpy.sum({array_itemsize}) <= {max_tile_bytes}`.
    Also, if :samp:`{max_tile_shape} is not None`
    then :samp:`numpy.all(tile_shape <= {max_tile_shape}) is True` and
    if :samp:`{sub_tile_shape} is not None`
    the :samp:`numpy.all((tile_shape % {sub_tile_shape}) == 0) is True`.

    :type array_shape: sequence of :obj:`int`
    :param array_shape: Shape of the array which is to be split into tiles.
    :type array_itemsize: :obj:`int`
    :param array_itemsize: The number of bytes per element of the array to be tiled.
    :type max_tile_bytes: :obj:`int`
    :param max_tile_bytes: The maximum number of bytes for the returned :samp:`tile_shape`.
    :type max_tile_shape: sequence of :obj:`int`
    :param max_tile_shape: Per axis maximum shapes for the returned :samp:`tile_shape`.
    :type sub_tile_shape: sequence of :obj:`int`
    :param sub_tile_shape: The returned :samp:`tile_shape` will be an even multiple
       of this sub-tile shape.
    :type halo: :obj:`int`, sequence of :obj:`int`, or :samp:`(len({array_shape}), 2)`
       shaped :obj:`numpy.ndarray`
    :param halo: How tiles are extended in each axis direction with *halo*
       elements. See :ref:`the-halo-parameter-examples` for meaning of :samp:`{halo}` values.
    :rtype: :obj:`numpy.ndarray`
    :return: A 1D array of shape :samp:`(len(array_shape),)` indicating a *tile shape*
       which will (approximately) uniformly divide the given :samp:`{array_shape}` into
       tiles (sub-arrays).

    Examples::

       >>> from array_split.split import calculate_tile_shape_for_max_bytes
       >>> calculate_tile_shape_for_max_bytes(
       ... array_shape=[512,],
       ... array_itemsize=1,
       ... max_tile_bytes=512
       ... )
       array([512])
       >>> calculate_tile_shape_for_max_bytes(
       ... array_shape=[512,],
       ... array_itemsize=2,  # Doubling the itemsize halves the tile size.
       ... max_tile_bytes=512
       ... )
       array([256])
       >>> calculate_tile_shape_for_max_bytes(
       ... array_shape=[512,],
       ... array_itemsize=1,
       ... max_tile_bytes=512-1  # tile shape will now be halved
       ... )
       array([256])


    """

    logger = _logging.getLogger(__name__ + ".calculate_tile_shape_for_max_bytes")
    logger.debug("calculate_tile_shape_for_max_bytes: enter:")
    logger.debug("array_shape=%s", array_shape)
    logger.debug("array_itemsize=%s", array_itemsize)
    logger.debug("max_tile_bytes=%s", max_tile_bytes)
    logger.debug("max_tile_shape=%s", max_tile_shape)
    logger.debug("sub_tile_shape=%s", sub_tile_shape)
    logger.debug("halo=%s", halo)

    array_shape = _np.array(array_shape, dtype="int64")
    array_itemsize = _np.sum(array_itemsize, dtype="int64")

    if max_tile_shape is None:
        max_tile_shape = _np.array(array_shape, copy=True)
    max_tile_shape = \
        _np.array(_np.minimum(max_tile_shape, array_shape), copy=True, dtype=array_shape.dtype)

    if sub_tile_shape is None:
        sub_tile_shape = _np.ones((len(array_shape),), dtype="int64")

    sub_tile_shape = _np.array(sub_tile_shape, dtype="int64")

    halo = convert_halo_to_array_form(halo=halo, ndim=len(array_shape))

    if _np.any(array_shape < sub_tile_shape):
        raise ValueError(
            "Got array_shape=%s element less than corresponding sub_tile_shape=%s element."
            %
            (
                array_shape,
                sub_tile_shape
            )
        )

    logger.debug("max_tile_shape=%s", max_tile_shape)
    logger.debug("sub_tile_shape=%s", sub_tile_shape)
    logger.debug("halo=%s", halo)
    array_sub_tile_split_shape = ((array_shape - 1) // sub_tile_shape) + 1
    tile_sub_tile_split_shape = array_shape // sub_tile_shape
    if len(tile_sub_tile_split_shape) <= 1:
        tile_sub_tile_split_shape[0] = \
            int(_np.floor(
                (
                    (max_tile_bytes / float(array_itemsize))
                    -
                    _np.sum(halo)
                )
                /
                float(sub_tile_shape[0])
            ))

    tile_sub_tile_split_shape = \
        _np.minimum(
            tile_sub_tile_split_shape,
            max_tile_shape // sub_tile_shape
        )
    logger.debug("tile_sub_tile_split_shape=%s", tile_sub_tile_split_shape)

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
        if current_axis < (len(tile_sub_tile_split_shape) - 1):
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
            sub_tile_shape_h = sub_tile_shape.copy()
            sub_tile_shape_h[0:current_axis] += _np.sum(halo[0:current_axis, :], axis=1)
            tile_sub_tile_split_shape[current_axis] = \
                int(_np.floor(
                    (
                        (max_tile_bytes / float(array_itemsize))
                        -
                        _np.sum(halo[current_axis]) * _np.product(sub_tile_shape_h[0:current_axis])
                    )
                    /
                    float(_np.product(sub_tile_shape_h))
                ))

    logger.debug("tile_sub_tile_split_shape=%s", tile_sub_tile_split_shape)
    tile_shape = _np.minimum(array_shape, tile_sub_tile_split_shape * sub_tile_shape)
    logger.debug("pre cannonicalise tile_shape=%s", tile_shape)

    tile_split_shape = ((array_shape - 1) // tile_shape) + 1
    logger.debug("tile_split_shape=%s", tile_split_shape)

    tile_shape = (((array_sub_tile_split_shape - 1) // tile_split_shape) + 1) * sub_tile_shape
    logger.debug("post cannonicalise tile_shape=%s", tile_shape)

    return tile_shape


def calculate_num_slices_per_axis(num_slices_per_axis, num_slices, max_slices_per_axis=None):
    """
    Returns a :obj:`numpy.ndarray` (:samp:`return_array` say) where non-positive elements of
    the :samp:`{num_slices_per_axis}` sequence have been replaced with
    positive integer values such that :samp:`numpy.product(return_array) == num_slices`
    and::

       numpy.all(
           return_array[numpy.where(num_slices_per_axis <= 0)]
           <=
           max_slices_per_axis[numpy.where(num_slices_per_axis <= 0)]
       ) is True


    :type num_slices_per_axis: sequence of :obj:`int`
    :param num_slices_per_axis: Constraint for per-axis sub-divisions.
       Non-positive elements indicate values to be replaced in the
       returned array. Positive values are identical to the corresponding
       element in the returned array.
    :type num_slices: integer
    :param num_slices: Indicates the number of slices (rectangular sub-arrays)
       formed by performing sub-divisions per axis. The returned array :samp:`return_array`
       has elements assigned such that :samp:`numpy.product(return_array) == {num_slices}`.
    :type max_slices_per_axis: sequence of :obj:`int` (or :samp:`None`)
    :param max_slices_per_axis: Constraint specifying maximum number of per-axis sub-divisions.
       If :samp:`None` defaults to :samp:`numpy.array([numpy.inf,]*len({num_slices_per_axis}))`.
    :rtype: :obj:`numpy.ndarray`
    :return: An array :samp:`return_array`
       such that :samp:`numpy.product(return_array) == num_slices`.


    Examples::

       >>> from array_split.split import calculate_num_slices_per_axis
       >>>
       >>> calculate_num_slices_per_axis([0, 0, 0], 16)
       array([4, 2, 2])
       >>> calculate_num_slices_per_axis([1, 0, 0], 16)
       array([1, 4, 4])
       >>> calculate_num_slices_per_axis([1, 0, 0], 16, [2, 2, 16])
       array([1, 2, 8])


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


_array_shape_param_doc =\
    """
:type array_shape: sequence of :obj:`int`
:param array_shape: The shape to be *split*.
"""

_array_start_param_doc =\
    """
:type array_start: :samp:`None` or sequence of :obj:`int`
:param array_start: The start index. Defaults to :samp:`[0,]*len(array_shape)`.
   The array indexing extents are assumed to range from :samp:`{array_start}`
   to :samp:`{array_start} + {array_shape}`.
   See :ref:`the-array_start-parameter-examples` examples.
"""
_array_itemsize_param_doc =\
    """
:type array_itemsize: int or sequence of :obj:`int`
:param array_itemsize: Number of bytes per array element.
   Only relevant when :samp:`{max_tile_bytes}` is specified.
   See :ref:`splitting-by-maximum-bytes-per-tile-examples` examples.
"""

_array_tile_bounds_policy_param_doc =\
    """
:type tile_bounds_policy: :obj:`str`
:param tile_bounds_policy: Specifies whether tiles can extend beyond the array boundaries.
   Only relevant for halo values greater than one. If :samp:`{tile_bounds_policy}`
   is :data:`ARRAY_BOUNDS`
   then the calculated tiles will not extend beyond the array
   extents :samp:`{array_start}` and :samp:`{array_start} + {array_shape}`.
   If :samp:`{tile_bounds_policy}` is :data:`NO_BOUNDS`
   then the returned tiles will extend beyond
   the :samp:`{array_start}` and :samp:`{array_start} + {array_shape}` extend
   for positive :samp:`{halo}` values. See :ref:`the-halo-parameter-examples` examples.
"""

_ShapeSplitter__init__params_doc =\
    """
:type indices_or_sections: :samp:`None`, :obj:`int` or sequence of :obj:`int`
:param indices_or_sections: If an integer, indicates the number of
    elements in the calculated *split* array. If a sequence, indicates
    the indicies (per axis) at which the splits occur.
    See :ref:`splitting-by-number-of-tiles-examples` examples.
:type axis: :samp:`None`, :obj:`int` or sequence of :obj:`int`
:param axis: If an integer, indicates the axis which is to be split.
   If a sequence integers, indicates the number of slices per axis,
   i.e. if :samp:`{axis} = [3, 5]` then axis :samp:`0` is cut into
   3 slices and axis :samp:`1` is cut into 5 slices for a total
   of 15 (:samp:`3*5`) rectangular slices in the returned :samp:`(3, 5)`
   shaped split.
   See :ref:`splitting-by-number-of-tiles-examples` examples
   and :ref:`splitting-by-per-axis-split-indices-examples` examples.
%s%s
:type tile_shape: :samp:`None` or sequence of :obj:`int`
:param tile_shape: When not :samp:`None`, specifies explicit shape for tiles.
   Should be same length as :samp:`{array_shape}`.
   See :ref:`splitting-by-tile-shape-examples` examples.
:type max_tile_bytes: :samp:`None` or :obj:`int`
:param max_tile_bytes: The maximum number of bytes for calculated :samp:`tile_shape`.
   See :ref:`splitting-by-maximum-bytes-per-tile-examples` examples.
:type max_tile_shape: :samp:`None` or sequence of :obj:`int`
:param max_tile_shape: Per axis maximum shapes for the calculated :samp:`tile_shape`.
   Only relevant when :samp:`{max_tile_bytes}` is specified. Should be same length
   as :samp:`{array_shape}`.
   See :ref:`splitting-by-maximum-bytes-per-tile-examples` examples.
:type sub_tile_shape: :samp:`None` or sequence of :obj:`int`
:param sub_tile_shape: When not :samp:`None`, the calculated :samp:`tile_shape` will
    be an even multiple of this sub-tile shape. Only relevant when :samp:`{max_tile_bytes}`
    is specified. Should be same length as :samp:`{array_shape}`.
    See :ref:`splitting-by-maximum-bytes-per-tile-examples` examples.%s%s
"""
_halo_param_doc =\
    """
:type halo: :samp:`None`, :obj:`int`, sequence of :obj:`int`, or :samp:`(len({array_shape}), 2)`
   shaped :obj:`numpy.ndarray`
:param halo: How tiles are extended per axis in -ve and +ve directions with *halo*
   elements. See :ref:`the-halo-parameter-examples` examples.
"""

#: Indicates that tiles are always within the array bounds.
#: See :ref:`the-halo-parameter-examples` examples.
__ARRAY_BOUNDS = "array_bounds"


@property
def ARRAY_BOUNDS():
    """
    Indicates that tiles are always within the array bounds,
    resulting in tiles which have truncated halos.
    See :ref:`the-halo-parameter-examples` examples.
    """
    return __ARRAY_BOUNDS


#: Indicates that tiles may extend beyond the array bounds.
#: See :ref:`the-halo-parameter-examples` examples.
__NO_BOUNDS = "no_bounds"


@property
def NO_BOUNDS():
    """
    Indicates that tiles may have halos which extend beyond the array bounds.
    See :ref:`the-halo-parameter-examples` examples.
    """
    return __NO_BOUNDS


def convert_halo_to_array_form(halo, ndim):
    """
    Converts the :samp:`{halo}` argument to a :samp:`(ndim, 2)`
    shaped array.

    :type halo: :samp:`None`, :obj:`int`, an :samp:`{ndim}` length sequence
        of :samp:`int` or :samp:`({ndim}, 2)` shaped array
        of :samp:`int`
    :param halo: Halo to be converted to :samp:`({ndim}, 2)` shaped array form.
    :type ndim: :obj:`int`
    :param ndim: Number of dimensions.
    :rtype: :obj:`numpy.ndarray`
    :return: A :samp:`({ndim}, 2)` shaped array of :obj:`numpy.int64` elements.
    """
    dtyp = _np.int64
    if halo is None:
        halo = _np.zeros((ndim, 2), dtype=dtyp)
    elif is_scalar(halo):
        halo = _np.zeros((ndim, 2), dtype=dtyp) + halo
    elif (ndim == 1) and (_np.array(halo).shape == (2,)):
        halo = _np.array([halo, ], copy=True, dtype=dtyp)
    elif len(_np.array(halo).shape) == 1:
        halo = _np.array([halo, halo], dtype=dtyp).T.copy()
    else:
        halo = _np.array(halo, copy=True, dtype=dtyp)

    if halo.shape[0] != ndim:
        raise ValueError(
            "Got halo.shape=%s, expecting halo.shape=(%s, 2)"
            %
            (halo.shape, ndim)
        )

    return halo


class ShapeSplitter(object):
    """
    Implements array shape splitting. There are three main (top-level) methods:

       :meth:`__init__`
          Initialisation of parameters which define the split.
       :meth:`set_split_extents`
          Calculates the per-axis indices for the cuts. Sets
          the :attr:`split_shape`, :attr:`split_begs`
          and :attr:`split_ends` attributes.
       :meth:`calculate_split`
          Calls :meth:`set_split_extents` followed
          by :meth:`calculate_split_from_extents` to
          return the :obj:`numpy.ndarray` of :obj:`tuple` elements (slices).

    """

    #: Class attribute for :obj:`logging.Logger` logging.
    logger = _logging.getLogger(__name__ + ".ShapeSplitter")

    #: Class attribute indicating list of valid values for :attr:`tile_bound_policy`.
    #: See :data:`ARRAY_BOUNDS` and :data:`NO_BOUNDS`.
    valid_tile_bounds_policies = [ARRAY_BOUNDS, NO_BOUNDS]

    def __init__(
        self,
        array_shape,
        indices_or_sections=None,
        axis=None,
        array_start=None,
        array_itemsize=1,
        tile_shape=None,
        max_tile_bytes=None,
        max_tile_shape=None,
        sub_tile_shape=None,
        halo=None,
        tile_bounds_policy=ARRAY_BOUNDS
    ):
        self.array_shape = _np.array(array_shape)

        if array_start is None:
            array_start = _np.zeros_like(self.array_shape)

        self.array_start = array_start

        self.array_itemsize = array_itemsize

        indices_per_axis = None
        if is_indices(indices_or_sections):
            num_subarrays = None
            indices_per_axis = indices_or_sections
            if (
                ((axis is None) or is_scalar(axis))
                and
                (not _np.any([is_sequence(_e) for _e in indices_or_sections]))
            ):
                if axis is None:
                    axis = 0
                # Make indices_per_axis a list of lists, so that
                # element 0 is a list of indices for axis 0
                indices_per_axis = [None, ] * len(array_shape)
                indices_per_axis[axis] = indices_or_sections
        else:
            indices_per_axis = None
            num_subarrays = indices_or_sections

        self.indices_per_axis = indices_per_axis

        self.split_size = num_subarrays
        split_num_slices_per_axis = None
        if (self.split_size is not None) or (axis is not None):
            if axis is None:
                axis = 0
            if is_sequence(axis):
                split_num_slices_per_axis = pad_with_object(axis, len(self.array_shape), 1)
            elif self.split_size is not None:
                split_num_slices_per_axis = pad_with_object([], len(self.array_shape), 1)
                split_num_slices_per_axis[axis] = self.split_size

        self.split_num_slices_per_axis = split_num_slices_per_axis

        self.tile_shape = tile_shape

        self.max_tile_bytes = max_tile_bytes

        self.max_tile_shape = max_tile_shape

        self.sub_tile_shape = sub_tile_shape

        halo = self.convert_halo_to_array_form(halo)
        self.halo = halo

        if tile_bounds_policy is None:
            tile_bounds_policy = ARRAY_BOUNDS

        self.tile_bounds_policy = tile_bounds_policy

        self.tile_beg_min = self.array_start

        self.tile_end_max = self.array_start + self.array_shape

        self.split_shape = None

        self.split_begs = None

        self.split_ends = None

    def convert_halo_to_array_form(self, halo):
        """
        Converts the :samp:`{halo}` argument to a :samp:`({self}.array_shape.size, 2)`
        shaped array.

        :type halo: :samp:`None`, :obj:`int`, :samp:`self.array_shape.size` length sequence
            of :samp:`int` or :samp:`(self.array_shape.size, 2)` shaped array
            of :samp:`int`
        :param halo: Halo to be converted to :samp:`(len(self.array_shape), 2)` shaped array form.
        :rtype: :obj:`numpy.ndarray`
        :return: A :samp:`(len(self.array_shape), 2)` shaped array of :obj:`numpy.int64` elements.
        """
        return convert_halo_to_array_form(halo=halo, ndim=len(self.array_shape))

    @property
    def array_shape(self):
        """
        The shape of the array which is to be split. A sequence of :obj:`int` indicating the
        per-axis sizes which are to be split.
        """
        return self.__array_shape

    @array_shape.setter
    def array_shape(self, array_shape):
        self.__array_shape = array_shape

    @property
    def array_start(self):
        """
        The start index. A sequence of :obj:`int` indicating the start of indexing for
        the tile slices. Defaults to :samp:`numpy.zeros_like({self}.array_shape)`.
        """
        return self.__array_start

    @array_start.setter
    def array_start(self, array_start):
        self.__array_start = array_start

    @property
    def array_itemsize(self):
        """
        The number of bytes per array element, see :attr:`max_tile_bytes`.
        """
        return self.__array_itemsize

    @array_itemsize.setter
    def array_itemsize(self, array_itemsize):
        self.__array_itemsize = array_itemsize

    @property
    def indices_per_axis(self):
        """
        The per-axis indices indicating the cuts for the split.
        A :obj:`list` of 1D :obj:`numpy.ndarray` objects such
        that :samp:`{self}.indices_per_axis[i]` indicates the
        cut positions for axis :samp:`i`.
        """
        return self.__indices_per_axis

    @indices_per_axis.setter
    def indices_per_axis(self, indices_per_axis):
        self.__indices_per_axis = indices_per_axis

    @property
    def split_size(self):
        """
        An :obj:`int` indicating the number of tiles in the calculated split.
        """
        return self.__split_size

    @split_size.setter
    def split_size(self, split_size):
        self.__split_size = split_size

    @property
    def split_num_slices_per_axis(self):
        """
        Number of slices per axis.
        A 1D :obj:`numpy.ndarray` of :obj:`int` indicating the number of slices (sections)
        per axis, so that :samp:`{self}.split_num_slices_per_axis[i]` is an integer
        indicating the number of sections along axis :samp:`i` in the calculated split.
        """
        return self.__split_num_slices_per_axis

    @split_num_slices_per_axis.setter
    def split_num_slices_per_axis(self, split_num_slices_per_axis):
        self.__split_num_slices_per_axis = split_num_slices_per_axis

    @property
    def tile_shape(self):
        """
        The shape of all tiles in the calculated split.
        A 1D :samp:`numpy.ndarray` of :obj:`int` indicating the per-axis
        number of elements for tiles in the calculated split.
        """
        return self.__tile_shape

    @tile_shape.setter
    def tile_shape(self, tile_shape):
        self.__tile_shape = tile_shape

    @property
    def max_tile_bytes(self):
        """
        The maximum number of bytes for any tile (including :attr:`halo`) in the returned split.
        An :obj:`int` which constrains the tile shape such that any tile
        from the computed split is no bigger than :samp:`{max_tile_bytes}`.
        """
        return self.__max_tile_bytes

    @max_tile_bytes.setter
    def max_tile_bytes(self, max_tile_bytes):
        self.__max_tile_bytes = max_tile_bytes

    @property
    def max_tile_shape(self):
        """
        Per-axis maximum sizes for calculated tiles.
        A 1D :samp:`numpy.ndarray` of :obj:`int` indicating the per-axis
        maximum number of elements for tiles in the calculated split.
        """
        return self.__max_tile_shape

    @max_tile_shape.setter
    def max_tile_shape(self, max_tile_shape):
        self.__max_tile_shape = max_tile_shape

    @property
    def sub_tile_shape(self):
        """
        Calculated tile shape will be an integer multiple of this sub-tile shape.
        i.e. :samp:`(self.tile_shape[i] % self.sub_tile_shape[i]) == 0`,
        for :samp:`i in range(0, len(self.tile_shape))`.
        A 1D :samp:`numpy.ndarray` of :obj:`int` indicating sub-tile shape.
        """
        return self.__sub_tile_shape

    @sub_tile_shape.setter
    def sub_tile_shape(self, sub_tile_shape):
        self.__sub_tile_shape = sub_tile_shape

    @property
    def halo(self):
        """
        Per-axis -ve and +ve halo sizes for extending tiles to overlap with neighbouring tiles.
        A :samp:`(N, 2)` shaped array indicating the
        """
        return self.__halo

    @halo.setter
    def halo(self, halo):
        self.__halo = halo

    @property
    def tile_bounds_policy(self):
        """
        A string indicating whether tile halo extents can extend beyond the array domain.
        Valid values are indicated by :attr:`valid_tile_bounds_policies`.
        """
        return self.__tile_bounds_policy

    @tile_bounds_policy.setter
    def tile_bounds_policy(self, tile_bounds_policy):
        self.__tile_bounds_policy = tile_bounds_policy

    @property
    def tile_beg_min(self):
        """
        The per-axis minimum index for :attr:`slice.start`. The per-axis lower bound for
        tile start indices. A 1D :obj:`numpy.ndarray`.
        """
        return self.__tile_beg_min

    @tile_beg_min.setter
    def tile_beg_min(self, tile_beg_min):
        self.__tile_beg_min = tile_beg_min

    @property
    def tile_end_max(self):
        """
        The per-axis maximum index for :attr:`slice.stop`. The per-axis upper bound for
        tile stop indices. A 1D :obj:`numpy.ndarray`.
        """
        return self.__tile_end_max

    @tile_end_max.setter
    def tile_end_max(self, tile_end_max):
        self.__tile_end_max = tile_end_max

    @property
    def split_shape(self):
        """
        The shape of the calculated split array. Indicates the per-axis number
        of sections in the calculated split. A 1D :obj:`numpy.ndarray`.
        """
        return self.__split_shape

    @split_shape.setter
    def split_shape(self, split_shape):
        self.__split_shape = split_shape

    @property
    def split_begs(self):
        """
        The list of per-axis start indices for :obj:`slice` objects.
        A :obj:`list` of 1D :obj:`numpy.ndarray` objects indicating
        the :attr:`slice.start` index for for tiles.
        """
        return self.__split_begs

    @split_begs.setter
    def split_begs(self, split_begs):
        self.__split_begs = split_begs

    @property
    def split_ends(self):
        """
        The list of per-axis stop indices for :obj:`slice` objects.
        A :obj:`list` of 1D :obj:`numpy.ndarray` objects indicating
        the :attr:`slice.stop` index for for tiles.
        """
        return self.__split_ends

    @split_ends.setter
    def split_ends(self, split_ends):
        self.__split_ends = split_ends

    def check_halo(self):
        """
        Raises :obj:`ValueError` if there is an inconsistency
        between shapes of :attr:`array_shape` and :attr:`halo`.
        """
        if (
            (len(self.halo.shape) != 2)
            or
            (self.halo.shape[0] != len(self.array_shape))
            or
            (self.halo.shape[1] != 2)
        ):
            raise ValueError(
                "Got halo.shape=%s, expecting halo.shape=(%s, 2)"
                %
                (self.halo.shape, self.array_shape.shape[0])
            )

    def check_tile_bounds_policy(self):
        """
        Raises :obj:`ValueError` if :attr:`tile_bounds_policy`
        is not in :samp:`[{self}.ARRAY_BOUNDS, {self}.NO_BOUNDS]`.
        """
        if not (self.tile_bounds_policy in self.valid_tile_bounds_policies):
            raise ValueError(
                "Got self.tile_bounds_policy=%s, which is not in %s."
                %
                (self.tile_bounds_policy, self.valid_tile_bounds_policies)
            )

    def check_consistent_parameter_dimensions(self):
        """
        Ensure that all parameter dimensions are consistent with
        the :attr:`array_shape` dimension.

        :raises ValueError: For inconsistent parameter dimensions.
        """
        if self.indices_per_axis is not None:
            if len(self.indices_per_axis) > len(self.array_shape):
                raise ValueError(
                    "Got len(self.indices_per_axis)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.indices_per_axis), len(self.array_shape))
                )
        if self.split_num_slices_per_axis is not None:
            if len(self.split_num_slices_per_axis) > len(self.array_shape):
                raise ValueError(
                    (
                        "Got len(self.split_num_slices_per_axis)=%s > len(self.array_shape)=%s,"
                        +
                        " should be equal."
                    )
                    %
                    (len(self.split_num_slices_per_axis), len(self.array_shape))
                )
        if self.tile_shape is not None:
            if len(self.tile_shape) != len(self.array_shape):
                raise ValueError(
                    "Got len(self.tile_shape)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.tile_shape), len(self.array_shape))
                )

        if self.sub_tile_shape is not None:
            if len(self.sub_tile_shape) != len(self.array_shape):
                raise ValueError(
                    "Got len(self.sub_tile_shape)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.sub_tile_shape), len(self.array_shape))
                )

        if self.max_tile_shape is not None:
            if len(self.max_tile_shape) != len(self.array_shape):
                raise ValueError(
                    "Got len(self.max_tile_shape)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.max_tile_shape), len(self.array_shape))
                )

        if self.array_start is not None:
            if len(self.array_start) != len(self.array_shape):
                raise ValueError(
                    "Got len(self.array_start)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.array_start), len(self.array_shape))
                )

    def check_consistent_parameter_grouping(self):
        """
        Ensures this object does not have conflicting groups of parameters.

        :raises ValueError: For conflicting or absent parameters.
        """
        parameter_groups = {}
        if self.indices_per_axis is not None:
            parameter_groups["indices_per_axis"] = \
                {"self.indices_per_axis": self.indices_per_axis}
        if (self.split_size is not None) or (self.split_num_slices_per_axis is not None):
            parameter_groups["split_size"] = \
                {
                    "self.split_size": self.split_size,
                    "self.split_num_slices_per_axis": self.split_num_slices_per_axis,
            }
        if self.tile_shape is not None:
            parameter_groups["tile_shape"] = \
                {"self.tile_shape": self.tile_shape}
        if self.max_tile_bytes is not None:
            parameter_groups["max_tile_bytes"] = \
                {"self.max_tile_bytes": self.max_tile_bytes}
        if self.max_tile_shape is not None:
            if "max_tile_bytes" not in parameter_groups.keys():
                parameter_groups["max_tile_bytes"] = {}
            parameter_groups["max_tile_bytes"]["self.max_tile_shape"] = self.max_tile_shape
        if self.sub_tile_shape is not None:
            if "max_tile_bytes" not in parameter_groups.keys():
                parameter_groups["max_tile_bytes"] = {}
            parameter_groups["max_tile_bytes"]["self.sub_tile_shape"] = self.sub_tile_shape

        if (len(parameter_groups.keys()) > 1):
            group_keys = sorted(parameter_groups.keys())
            raise ValueError(
                "Got conflicting parameter groups specified, "
                +
                "should only specify one group to define the split:\n"
                +
                (
                    "\n".join(
                        [
                            (
                                ("Group %18s: " % ("'%s'" % group_key))
                                +
                                str(parameter_groups[group_key])
                            )
                            for group_key in group_keys
                        ]
                    )
                )
            )
        if (len(parameter_groups.keys()) <= 0):
            raise ValueError(
                "No split parameters specified, need parameters from one of the groups: "
                +
                "'indices_per_axis', 'split_size', 'tile_shape' or 'max_tile_bytes'"
            )

    def check_split_parameters(self):
        """
        Ensures this object has a state consistent with evaluating a split.

        :raises ValueError: For conflicting or absent parameters.
        """

        self.check_halo()
        self.check_tile_bounds_policy()
        self.check_consistent_parameter_dimensions()
        self.check_consistent_parameter_grouping()

    def update_tile_extent_bounds(self):
        """
        Updates the :attr:`tile_beg_min` and :attr:`tile_end_max`
        data members according to :attr:`tile_bounds_policy`.
        """

        if self.tile_bounds_policy == NO_BOUNDS:
            self.tile_beg_min = self.array_start - self.halo[:, 0]
            self.tile_end_max = self.array_start + self.array_shape + self.halo[:, 1]
        elif self.tile_bounds_policy == ARRAY_BOUNDS:
            self.tile_beg_min = self.array_start
            self.tile_end_max = self.array_start + self.array_shape

    def set_split_extents_by_indices_per_axis(self):
        """
        Sets split shape :attr:`split_shape` and
        split extents (:attr:`split_begs` and :attr:`split_ends`)
        from values in :attr:`indices_per_axis`.
        """
        if self.indices_per_axis is None:
            raise ValueError("Got None for self.indices_per_axis")

        self.logger.debug("self.array_shape=%s", self.array_shape)
        self.logger.debug("self.indices_per_axis=%s", self.indices_per_axis)
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
        from :attr:`split_begs` and :attr:`split_ends`.
        All calls to calculate the split end up here to produce
        the :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
        of :obj:`tuple`-of-:obj:`slice` elements.

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
                                max([
                                    self.split_begs[d][idx[d]]
                                    + self.array_start[d]
                                    - self.halo[d, 0]
                                    * (self.split_ends[d][idx[d]] > self.split_begs[d][idx[d]]),
                                    self.tile_beg_min[d]
                                ]),
                                min([
                                    self.split_ends[d][idx[d]]
                                    + self.array_start[d]
                                    + self.halo[d, 1]
                                    * (self.split_ends[d][idx[d]] > self.split_begs[d][idx[d]]),
                                    self.tile_end_max[d]
                                ])
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

    def calculate_split_halos_from_extents(self):
        """
        Returns :samp:`(self.ndim, 2)` shaped halo array elements indicating
        the halo for each split. Tiles on the boundary may have the halo trimmed
        to account for the :attr:`tile_bounds_policy`.

        :rtype: :obj:`numpy.ndarray`
        :return:
           A :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           where each element is a :samp:`(self.ndim, 2)` shaped :obj:`numpy.ndarray`
           indicating the per-axis and per-direction number of halo elements for each tile
           in the split.
        """
        self.logger.debug("self.split_shape=%s", self.split_shape)
        self.logger.debug("self.split_begs=%s", self.split_begs)
        self.logger.debug("self.split_ends=%s", self.split_ends)

        ret = \
            _np.array(
                [
                    (
                        tuple(
                            (
                                min([
                                    self.split_begs[d][idx[d]] - self.tile_beg_min[d],
                                    self.halo[d, 0]
                                    *
                                    (self.split_ends[d][idx[d]] > self.split_begs[d][idx[d]])
                                ]),
                                min([
                                    self.tile_end_max[d] - self.split_ends[d][idx[d]],
                                    self.halo[d, 1]
                                    *
                                    (self.split_ends[d][idx[d]] > self.split_begs[d][idx[d]])
                                ])
                            )
                            for d in range(len(self.split_shape))
                        )
                    )
                    for idx in
                    _np.array(
                        _np.unravel_index(
                            _np.arange(0, _np.product(self.split_shape)),
                            self.split_shape
                        )
                    ).T
                ],
                dtype=[("%d" % d, "2int64") for d in range(len(self.split_shape))]
            ).reshape(self.split_shape)

        return ret

    def calculate_split_by_indices_per_axis(self):
        """
        Returns split calculated using extents obtained
        from :attr:`indices_per_axis`.

        :rtype: :obj:`numpy.ndarray`
        :return:
           A :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
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
        Sets split shape :attr:`split_shape` and
        split extents (:attr:`split_begs` and :attr:`split_ends`)
        from values in :attr:`split_size` and :attr:`split_num_slices_per_axis`.
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
        from :attr:`split_size` and :attr:`split_num_slices_per_axis`.

        :rtype: :obj:`numpy.ndarray`
        :return:
           A :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           where each element is a :obj:`tuple` of :obj:`slice` objects.
        """
        self.set_split_extents_by_split_size()
        return self.calculate_split_from_extents()

    def set_split_extents_by_tile_shape(self):
        """
        Sets split shape :attr:`split_shape` and
        split extents (:attr:`split_begs` and :attr:`split_ends`)
        from value of :attr:`tile_shape`.
        """
        self.split_shape = ((self.array_shape - 1) // self.tile_shape) + 1
        self.split_begs = [[], ] * len(self.array_shape)
        self.split_ends = [[], ] * len(self.array_shape)
        for i in range(len(self.array_shape)):
            self.split_begs[i] = _np.arange(0, self.array_shape[i], self.tile_shape[i])
            self.split_ends[i] = _np.zeros_like(self.split_begs[i])
            self.split_ends[i][0:-1] = self.split_begs[i][1:]
            self.split_ends[i][-1] = self.array_shape[i]

    def calculate_split_by_tile_shape(self):
        """
        Returns split calculated using extents obtained
        from :attr:`tile_shape`.

        :rtype: :obj:`numpy.ndarray`
        :return:
           A :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           where each element is a :obj:`tuple` of :obj:`slice` objects.
        """
        self.set_split_extents_by_tile_shape()
        return self.calculate_split_from_extents()

    def set_split_extents_by_tile_max_bytes(self):
        """
        Sets split extents (:attr:`split_begs`
        and :attr:`split_ends`) calculated using
        from :attr:`max_tile_bytes`
        (and :attr:`max_tile_shape`, :attr:`sub_tile_shape`, :attr:`halo`).

        """
        self.tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=self.array_shape,
                array_itemsize=self.array_itemsize,
                max_tile_bytes=self.max_tile_bytes,
                max_tile_shape=self.max_tile_shape,
                sub_tile_shape=self.sub_tile_shape,
                halo=self.halo
            )
        self.set_split_extents_by_tile_shape()

    def calculate_split_by_tile_max_bytes(self):
        """
        Returns split calculated using extents obtained
        from :attr:`max_tile_bytes`
        (and :attr:`max_tile_shape`, :attr:`sub_tile_shape`, :attr:`halo`).

        :rtype: :obj:`numpy.ndarray`
        :return:
           A :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           where each element is a :obj:`tuple` of :obj:`slice` objects.
        """

        self.set_split_extents_by_tile_max_bytes()
        return self.calculate_split_from_extents()

    def set_split_extents(self):
        """
        Sets split extents (:attr:`split_begs`
        and :attr:`split_ends`) calculated using
        selected attributes set from :meth:`__init__`.
        """

        self.check_split_parameters()
        self.update_tile_extent_bounds()

        if self.indices_per_axis is not None:
            self.set_split_extents_by_indices_per_axis()
        elif (self.split_size is not None) or (self.split_num_slices_per_axis is not None):
            self.set_split_extents_by_split_size()
        elif self.tile_shape is not None:
            self.set_split_extents_by_tile_shape()
        elif self.max_tile_bytes is not None:
            self.set_split_extents_by_tile_max_bytes()

    def calculate_split(self):
        """
        Computes the split.

        :rtype: :obj:`numpy.ndarray`
        :return:
           A :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           of dimension :samp:`len({self}.array_shape)`.
           Each element of the returned array is a :obj:`tuple`
           containing :samp:`len({self}.array_shape)` elements, with each element
           being a :obj:`slice` object. Each :obj:`tuple` defines a slice within
           the bounds :samp:`{self}.array_start - {self}.halo[:, 0]`
           to :samp:`{self}.array_start + {self}.array_shape + {self}.halo[:, 1]`.
        """

        self.set_split_extents()
        return self.calculate_split_from_extents()


ShapeSplitter([0, ]).__init__.__func__.__doc__ = \
    """
Initialises parameters which define a split.


%s
%s

.. seealso:: :ref:`array_split-examples`

""" % (
    _array_shape_param_doc,
    (
        _ShapeSplitter__init__params_doc
        %
        (
            _array_start_param_doc,
            "\n" + _array_itemsize_param_doc,
            _halo_param_doc,
            _array_tile_bounds_policy_param_doc,
        )
    )
)


def shape_split(array_shape, *args, **kwargs):
    return \
        ShapeSplitter(
            array_shape,
            *args,
            **kwargs
        ).calculate_split()


shape_split.__doc__ =\
    """
Splits specified :samp:`{array_shape}` in tiles, returns array of :obj:`slice` tuples.

%s
%s
:rtype: :obj:`numpy.ndarray`
:return: Array of :obj:`tuple` objects. Each :obj:`tuple` element
   is a :obj:`slice` object so that each :obj:`tuple` defines
   a multi-dimensional slice of an array of shape :samp:`{array_shape}`.

.. seealso:: :func:`array_split.array_split`, :meth:`array_split.ShapeSplitter`,
   :ref:`array_split-examples`


""" % (
        _array_shape_param_doc,
        (
            _ShapeSplitter__init__params_doc
            %
            (
                _array_start_param_doc,
                "\n" + _array_itemsize_param_doc,
                _halo_param_doc,
                _array_tile_bounds_policy_param_doc,
            )
        )
    )


def array_split(
    ary,
    indices_or_sections=None,
    axis=None,
    tile_shape=None,
    max_tile_bytes=None,
    max_tile_shape=None,
    sub_tile_shape=None,
    halo=None
):
    return [
        ary[slyce]
        for slyce in
        shape_split(
            array_shape=ary.shape,
            indices_or_sections=indices_or_sections,
            axis=axis,
            array_start=None,
            array_itemsize=ary.itemsize,
            tile_shape=tile_shape,
            max_tile_bytes=max_tile_bytes,
            max_tile_shape=max_tile_shape,
            sub_tile_shape=sub_tile_shape,
            halo=halo,
            tile_bounds_policy=ARRAY_BOUNDS
        ).flatten()
    ]


array_split.__doc__ =\
    """
Splits the specified array :samp:`{ary}` into sub-arrays, returns list of :obj:`numpy.ndarray`.

:type ary: :obj:`numpy.ndarray`
:param ary: Array which is split into sub-arrays.
%s
:rtype: :obj:`list`
:return: List of :obj:`numpy.ndarray` elements, where each element is
   a *slice* from :samp:`{ary}` (potentially an empty slice).

.. seealso:: :func:`array_split.shape_split`, :meth:`array_split.ShapeSplitter`,
   :ref:`array_split-examples`


""" % (
        _ShapeSplitter__init__params_doc
        %
        (
            "",
            "",
            _halo_param_doc.replace("len({array_shape})", "len({ary}.shape)"),
            ""
        )
    )

__all__ = [s for s in dir() if not s.startswith('_')]
