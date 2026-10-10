"""
======================================
The :mod:`array_split.unittest` Module
======================================

Some simple wrappers of python built-in :mod:`unittest` module
for :mod:`array_split` unit-tests.

.. currentmodule:: array_split.unittest

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   main - Convenience command-line test-case *search and run* function.
   TestCase - Extends :obj:`unittest.TestCase` with :obj:`TestCase.assertArraySplitEqual`.

"""

import unittest as _builtin_unittest

import numpy as _np

from . import logging as _logging
from .license import copyright as _copyright, license as _license, version as _version

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

# pylint: disable=invalid-name
# pylint: disable=arguments-differ
# pylint: disable=trailing-whitespace
# pylint: disable=no-member
# pylint: disable=deprecated-method
# pylint: disable=broad-except
# pylint: disable=too-many-locals
# pylint: disable=protected-access
# pylint: disable=too-many-branches


def main(module_name, log_level=_logging.DEBUG, init_logger_names=None):
    """
    Small wrapper for :func:`unittest.main` which initialises :mod:`logging.Logger` objects.
    Loads a set of tests from module and runs them;
    this is primarily for making test modules conveniently executable.
    The simplest use for this function is to include the following line at
    the end of a test module::

       array_split.unittest.main(__name__)

    If :samp:`__name__ == "__main__"`, then *discoverable* :obj:`unittest.TestCase`
    test cases are executed.
    Logging level can be explicitly set for a group of modules using::

       import logging

       array_split.unittest.main(
           __name__,
           logging.DEBUG,
           [__name__, "module_name_0", "module_name_1", "package.module_name_2"]
       )


    :type module_name: :obj:`str`
    :param module_name: If :samp:`{module_name} == "__main__"` then unit-tests
       are *discovered* and run.
    :type log_level: :obj:`int`
    :param log_level: The default logging level for all
       :obj:`array_split.logging.Logger` objects.
    :type init_logger_names: sequence of :obj:`str`
    :param init_logger_names: List of logger names to initialise
       (using :func:`array_split.logging.initialise_loggers`). If :samp:`None`,
       then the list defaults to :samp:`[{module_name}, "array_split"]`. If list
       is empty no loggers are initialised.

    """
    if module_name == "__main__":
        if init_logger_names is None:
            init_logger_names = [module_name, "array_split"]

        if len(init_logger_names) > 0:
            _logging.initialise_loggers(
                init_logger_names, log_level=log_level)

        _builtin_unittest.main()


def _fix_docstring_for_sphinx(docstr):
    """
    Remove 8-space indentation from lines of specified :samp:`{docstr}` string.
    """
    lines = docstr.split("\n")
    for i in range(len(lines)):
        if lines[i].find(" " * 8) == 0:
            lines[i] = lines[i][8:]
    return "\n".join(lines)


class TestCase(_builtin_unittest.TestCase):

    """
    Extends :obj:`unittest.TestCase` with the :meth:`assertArraySplitEqual`.
    """

    def assertArraySplitEqual(self, splt1, splt2):
        """
        Compares :obj:`list` of :obj:`numpy.ndarray` results returned by :func:`numpy.array_split`
        and :func:`array_split.split.array_split` functions.

        :type splt1: :obj:`list` of :obj:`numpy.ndarray`
        :param splt1: First object in equality comparison.
        :type splt2: :obj:`list` of :obj:`numpy.ndarray`
        :param splt2: Second object in equality comparison.
        :raises unittest.AssertionError: If any element of :samp:`{splt1}` is not equal to
            the corresponding element of :samp:`splt2`.
        """
        self.assertEqual(len(splt1), len(splt2))
        for i in range(len(splt1)):
            self.assertTrue(
                (
                    _np.all(_np.array(splt1[i]) == _np.array(splt2[i]))
                    or
                    ((_np.array(splt1[i]).size == 0) and (_np.array(splt2[i]).size == 0))
                ),
                msg=(
                    f"element {i:d} of split is not equal"
                    f" {_np.array(splt1[i])} != {_np.array(splt2[i])}"
                )
            )

    #
    # Method over-rides below are just to avoid sphinx warnings
    #
    def assertItemsEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertItemsEqual`.
        """
        _builtin_unittest.TestCase.assertItemsEqual(self, *args, **kwargs)

    def assertListEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertListEqual`.
        """
        _builtin_unittest.TestCase.assertListEqual(self, *args, **kwargs)

    def assertRaisesRegexp(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertRaisesRegexp`.
        """
        _builtin_unittest.TestCase.assertRaisesRegexp(self, *args, **kwargs)

    def assertRaisesRegex(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertRaisesRegex`.
        """
        _builtin_unittest.TestCase.assertRaisesRegex(self, *args, **kwargs)

    def assertSetEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertSetEqual`.
        """
        _builtin_unittest.TestCase.assertSetEqual(self, *args, **kwargs)

    def assertTupleEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertTupleEqual`.
        """
        _builtin_unittest.TestCase.assertTupleEqual(self, *args, **kwargs)

    def assertWarnsRegex(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertWarnsRegex`.
        """
        _builtin_unittest.TestCase.assertWarnsRegex(self, *args, **kwargs)

    def assertSequenceEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertSequenceEqual`.
        """
        _builtin_unittest.TestCase.assertSequenceEqual(self, *args, **kwargs)
