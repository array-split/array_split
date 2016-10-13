"""
===================================
The :mod:`array_split.tests` Module
===================================

Module for running all :mod:`array_split` unit-tests.
Execute as::

   python -m array_split.tests

.. currentmodule:: array_split.tests

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import unittest as _unittest
import array_split as _array_split
from .split_test import *  # noqa: F401,F403

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()

__version__ = _array_split.__version__

__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest.main()
