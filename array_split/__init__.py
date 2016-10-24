"""
==============================
The :mod:`array_split` Package
==============================

.. currentmodule:: array_split

Small python package for splitting a :obj:`numpy.ndarray` (or just an array shape)
into a number of sub-arrays.

The two main functions are:

   :func:`array_split.array_split`
       Similar to :func:`numpy.array_split`, returns a list of
       *views* of sub-arrays of the input :obj:`numpy.ndarray`.
       Can split along multiple axes and has more splitting
       criteria (parameters) than :func:`numpy.array_split`.

   :func:`array_split.shape_split`
      Instead taking an :obj:`numpy.ndarray` as an argument, it
      takes the array *shape* and returns tuples of :obj:`slice`
      objects which indicate the extents of the sub-arrays.

These two functions use an instance of the :obj:`array_split.ShapeSplitter` class
which contains the bulk of the *split* implementation and maintains
some state related to the computed split.

Splitting of multi-dimensional arrays can be performed according to several criteria:

   * Per-axis indicies indicating the *cut* positions.
   * Per-axis number of sub-arrays.
   * Total number of sub-arrays (with optional per-axis *number of sections* constraints).
   * Specific sub-array shape.
   * Maximum number of bytes for a sub-array with constraints:

        - sub-arrays are an even multiple of a specified sub-tile shape
        - upper limit on the per-axis sub-array shape

The usage documentation is given in the :ref:`array_split-examples` section.


Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   shape_split - Splits a shape and returns :obj:`numpy.ndarray` of :obj:`slice` elements.
   array_split - Equivalent to :func:`numpy.array_split`.
   ShapeSplitter - Array shape splitting class.

Attributes
==========

.. autodata:: ARRAY_BOUNDS
.. autodata:: NO_BOUNDS


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("array_split", "version.txt").decode()

from . import split  # noqa: E402,F401
from .split import array_split, shape_split, ShapeSplitter  # noqa: E402,F401

#: See :data:`array_split.split.ARRAY_BOUNDS`
ARRAY_BOUNDS = split.ARRAY_BOUNDS

#: See :data:`array_split.split.NO_BOUNDS`
NO_BOUNDS = split.NO_BOUNDS

__all__ = [s for s in dir() if not s.startswith('_')]
