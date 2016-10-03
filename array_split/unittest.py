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


def main(module_name, log_level=array_split.logging.DEBUG, init_logger_names=None):
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
        if (init_logger_names is None):
            init_logger_names = [module_name, "array_split"]

        if (len(init_logger_names) > 0):
            array_split.logging.initialise_loggers(init_logger_names, log_level=log_level)

        _builtin_unittest.main()
