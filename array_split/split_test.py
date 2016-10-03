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
import array_split.unittest as _unittest
import array_split.logging as _logging
import array_split as _array_split

import numpy as _np

from .split import ArraySplitter

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _array_split.__version__


class SplitTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :mod:`array_split.split` functions.
    """

    #: Class attribute for :obj:`logging.Logger` logging.
    logger = _logging.getLogger(__name__ + ".SplitTest")

    def test_split_by_per_axis_indices(self):
        """
        Test for case for splitting by specified indices::

           ArraySplitter(array_shape=(10, 4), indices_or_sections=[[2, 6, 8], ])
        """
        splitter = ArraySplitter((10, 4), [[2, 6, 8], ])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [4, 1]))
        self.assertEqual(slice(0, 2), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 4), split[0, 0][1])  # axis 1 slice

__all__ = [s for s in dir() if not s.startswith('_')]

_unittest.main(__name__)
