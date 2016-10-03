"""
===================================
The :mod:`array_split.split` Module
===================================

Defines array splitting functions and classes.

.. autosummary::
   :toctree: generated/

   ArraySplitter - Implements array splitting akin to :func:`numpy.array_split`.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright

import array_split as _array_split
import array_split.logging as _logging
import numpy as _np
from SCons.Util import IDX

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _array_split.__version__


def is_indices(indices_or_sections):
    """
    Returns :samp:`True` if argument :samp:`{indices_or_sections}` is
    a sequence (e.g. a :obj:`list` or :obj:`tuple`, etc).
    """
    return hasattr(indices_or_sections, "__len__") or hasattr(indices_or_sections, "__getitem__")


def pad_with_none(sequence, new_length):
    """
    Returns :samp:`sequence` :obj:`list` end-padded with :samp:`None`
    elements so that the length of the returned list equals :samp:`{new_length}`.

    """
    if len(sequence) < new_length:
        sequence = \
            list(sequence) + [None, ] * (new_length - len(sequence))
    elif len(sequence) > new_length:
        raise ValueError(
            "Got len(sequence)=%s which exceeds new_length=%s"
            %
            (len(sequence), new_length)
        )

    return sequence


class ArraySplitter(object):
    """
    Implements array splitting akin to :func:`numpy.array_split` but
    with a plethora of options for specifying the sub-array (slice)
    sizing.
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
           sub-arrays

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
        if is_indices(indices_or_sections):
            num_subarrays = None
        else:
            indices_per_axis = None
            num_subarrays = indices_or_sections

        #: Specifies the indices (per axis) where the splits occur.
        self.indices_per_axis = indices_or_sections

        #: Specifies the size (total number of structure elements)
        #: of the returned split.
        self.split_size = num_subarrays

        self.axis = axis

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
        :return: :mod:`numpy` `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
           where each element is a :obj:`tuple` of :obj:`slice` objects.
        """
        self.logger.debug("self.split_shape=%s", self.split_shape)
        self.logger.debug("self.split_begs=%s", self.split_begs)
        self.logger.debug("self.split_ends=%s", self.split_ends)
        self.logger.debug("_np.arange=%s", _np.arange(0, _np.product(self.split_shape)))

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
        return self.calculate_split_by_indices_per_axis()

__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest.main()
