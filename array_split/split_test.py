"""
========================================
The :mod:`array_split.split_test` Module
========================================

Module defining :mod:`array_split.split` unit-tests.
Execute as::

   python -m array_split.split_tests

.. autosummary::
   :toctree: generated/

   SplitTest - :obj:`unittest.TestCase` for :mod:`array_split.split` functions.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import array_split as _array_split
import unittest as _unittest

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _array_split.__version__


class SplitTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :mod:`array_split.split` functions.
    """
    def test_split(self):
        pass

__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest.main()
