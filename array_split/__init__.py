"""
==============================
The :mod:`array_split` Package
==============================

.. currentmodule:: array_split

Small python package for splitting a :obj:`numpy.ndarray` (or just an array shape)
into a number of sub-arrays.

The two main functions are:

   :func:`array_split`
       Similar to :func:`numpy.array_split`, returns a list of
       *views* of sub-arrays of the input :obj:`numpy.ndarray`.

   :func:`shape_split`
      Instead taking an :obj:`numpy.ndarray` as an argument, it
      takes the array *shape* and returns tuples of :obj:`slice`
      objects which indicate the extents of the sub-arrays.

These two functions use an instance of the :obj:`ShapeSplitter` class
which contains the bulk of the *split* implementation and maintains
some state related to the computed split.

Splitting of multi-dimensional arrays can be performed according to several criteria:

   * Per-axis indicies indicating the *cut* positions.
   * Per-axis number of sub-arrays.
   * Total number of sub-arrays (with optional per-axis number of sub-array constraints).
   * Specific sub-array shape.
   * Maximum number of bytes for a sub-array with constraints:

        - sub-arrays are an even multiple of a specified sub-tile shape
        - upper limit on the per-axis sub-array shape

A variety of use-cases are given in the :ref:`array_split examples` section.


Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   shape_split - Splits a shape and returns :obj:`numpy.ndarray` of :obj:`slice` elements.
   array_split - Equivalent to :func:`numpy.array_split`.
   ShapeSplitter - Array shape splitting class.

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("array_split", "version.txt").decode()

from .split import array_split, shape_split, ShapeSplitter  # noqa: E402,F401

__all__ = [s for s in dir() if not s.startswith('_')]
