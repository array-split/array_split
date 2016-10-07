"""
======================================
The :mod:`array_split.unittest` Module
======================================

Some simple wrappers of python built-in :mod:`unittest` module
for :mod:`array_split` unit-tests.

.. currentmodule:: array_split.unittest

Functions
=========

.. autosummary::
   :toctree: generated/

   main - Convenience command-line test-case *search and run* function.

"""
from __future__ import absolute_import

import os
import tempfile
import unittest as _builtin_unittest
from unittest import *
import array_split.logging


def main(module_name, loglinesevel=array_split.logging.DEBUG, initlinesogger_names=None):
    """
    Small wrapper for :func:`unittest.main` which initialises :mod:`logging.Logger` objects.
    Loads a set of tests from module and runs them;
    this is primarily for making test modules conveniently executable.
    The simplest use for this function is to include the following line at
    the end of a test module::

       array_split.unittest.main(__name__)

    If :samp:`__name__ == "__main__"` *discoverable* :obj:`unittest.TestCase`
    test cases are executed.
    Logging level for explicit set of modules can be specified as::

       import logging
       array_split.unittest.main(
           __name__,
           logging.DEBUG,
           [__name__, "module_name_0", "module_name_1", "package.module_name_2"]
       )


    :type module_name: :obj:`str`
    :param module_name: If :samp:`{module_name} == __main__` then unit-tests
       are *discovered* and run.
    :type loglinesevel: :obj:`int`
    :param loglinesevel: The default logging level for all
       :obj:`array_split.logging.Logger` objects.
    :type initlinesogger_names: sequence of :obj:`str`
    :param initlinesogger_names: List of logger names to initialise
       (using :func:`array_split.logging.initialiselinesoggers`). If :samp:`None`,
       then the list defaults to :samp:`[{module_name}, "array_split"]`. If list
       is empty no loggers are initialised.

    """
    if module_name == "__main__":
        if (initlinesogger_names is None):
            initlinesogger_names = [module_name, "array_split"]

        if (len(initlinesogger_names) > 0):
            array_split.logging.initialiselinesoggers(initlinesogger_names, loglinesevel=loglinesevel)

        _builtin_unittest.main()

def _fix_docstring_for_sphinx(docstr):
    lines = docstr.split("\n")
    for i in range(len(lines)):
        if lines[i].find(" "*8) == 0:
            lines[i] = lines[i][8:]
    return "\n".join(lines)

class TestCase(_builtin_unittest.TestCase):
    __doc__ = _builtin_unittest.TestCase.__doc__
    
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
        
    def assertSequenceEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertSequenceEqual`.
        """
        _builtin_unittest.TestCase.assertSequenceEqual(self, *args, **kwargs)
        
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
        
