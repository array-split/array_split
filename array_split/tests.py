"""
===================================
The :mod:`array_split.tests` Module
===================================

Module for running all :mod:`array_split` unit-tests, including :mod:`unittest` test-cases
and :mod:`doctest` tests for module doc-strings and sphinx (RST) documentation.
Execute as::

   python -m array_split.tests

.. currentmodule:: array_split.tests

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import unittest as _unittest
import doctest as _doctest
import array_split as _array_split
from array_split import split as _split

from .split_test import *  # noqa: F401,F403
import os.path

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _array_split.__version__


class DocTestTestSuite(_unittest.TestSuite):

    def __init__(self):
        readme_file_name = \
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "README.rst")
            )
        examples_rst_file_name = \
            os.path.realpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "docs",
                    "source",
                    "examples",
                    "index.rst"
                )
            )
        suite = _unittest.TestSuite()
        if os.path.exists(readme_file_name):
            suite.addTests(
                _doctest.DocFileSuite(
                    readme_file_name,
                    module_relative=False,
                    optionflags=_doctest.NORMALIZE_WHITESPACE
                )
            )
        if os.path.exists(examples_rst_file_name):
            suite.addTests(
                _doctest.DocFileSuite(
                    examples_rst_file_name,
                    module_relative=False,
                    optionflags=_doctest.NORMALIZE_WHITESPACE
                )
            )
        suite.addTests(
            _doctest.DocTestSuite(
                _array_split,
                optionflags=_doctest.NORMALIZE_WHITESPACE
            )
        )
        suite.addTests(
            _doctest.DocTestSuite(
                _split,
                optionflags=_doctest.NORMALIZE_WHITESPACE
            )
        )

        _unittest.TestSuite.__init__(self, suite)


def load_tests(loader, tests, pattern):
    suite = loader.loadTestsFromNames(["array_split.split_test", ])
    suite.addTests(DocTestTestSuite())
    return suite


__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest.main()
