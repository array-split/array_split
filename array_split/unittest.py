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
from __future__ import absolute_import

import unittest as _builtin_unittest
import array_split.logging
import numpy as _np


def main(module_name, log_level=array_split.logging.DEBUG, init_logger_names=None):
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
        if (init_logger_names is None):
            init_logger_names = [module_name, "array_split"]

        if (len(init_logger_names) > 0):
            array_split.logging.initialise_loggers(
                init_logger_names, log_level=log_level)

        _builtin_unittest.main()


def _fix_docstring_for_sphinx(docstr):
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
                    "element %d of split is not equal %s != %s"
                    %
                    (i, _np.array(splt1[i]), _np.array(splt2[i]))
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


if not hasattr(TestCase, "assertSequenceEqual"):
    # code from python-2.7 unitest.case.TestCase
    _MAX_LENGTH = 80

    def safe_repr(obj, short=False):
        try:
            result = repr(obj)
        except Exception:
            result = object.__repr__(obj)
        if not short or len(result) < _MAX_LENGTH:
            return result
        return result[:_MAX_LENGTH] + ' [truncated]...'

    def strclass(cls):
        return "%s.%s" % (cls.__module__, cls.__name__)

    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        """An equality assertion for ordered sequences (like lists and tuples).

        For the purposes of this function, a valid ordered sequence type is one
        which can be indexed, has a length, and has an equality operator.

        :param seq1: The first sequence to compare.
        :param seq2: The second sequence to compare.
        :param seq_type: The expected datatype of the sequences, or None if no
                    datatype should be enforced.
        :param msg: Optional message to use on failure instead of a list of
                    differences.
        """

        import pprint
        import difflib
        if seq_type is not None:
            seq_type_name = seq_type.__name__
            if not isinstance(seq1, seq_type):
                raise self.failureException('First sequence is not a %s: %s'
                                            % (seq_type_name, safe_repr(seq1)))
            if not isinstance(seq2, seq_type):
                raise self.failureException('Second sequence is not a %s: %s'
                                            % (seq_type_name, safe_repr(seq2)))
        else:
            seq_type_name = "sequence"

        differing = None
        try:
            len1 = len(seq1)
        except (TypeError, NotImplementedError):
            differing = 'First %s has no length.    Non-sequence?' % (
                seq_type_name)

        if differing is None:
            try:
                len2 = len(seq2)
            except (TypeError, NotImplementedError):
                differing = 'Second %s has no length.    Non-sequence?' % (
                    seq_type_name)

        if differing is None:
            if seq1 == seq2:
                return

            seq1_repr = safe_repr(seq1)
            seq2_repr = safe_repr(seq2)

            if len(seq1_repr) > 30:
                seq1_repr = seq1_repr[:30] + '...'
            if len(seq2_repr) > 30:
                seq2_repr = seq2_repr[:30] + '...'
            elements = (seq_type_name.capitalize(), seq1_repr, seq2_repr)
            differing = '%ss differ: %s != %s\n' % elements

            for i in range(min(len1, len2)):
                try:
                    item1 = seq1[i]
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('\nUnable to index element %d of first %s\n' %
                                  (i, seq_type_name))
                    break

                try:
                    item2 = seq2[i]
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('\nUnable to index element %d of second %s\n' %
                                  (i, seq_type_name))
                    break

                if item1 != item2:
                    differing += ('\nFirst differing element %d:\n%s\n%s\n' %
                                  (i, item1, item2))
                    break
            else:
                if (len1 == len2 and seq_type is None and
                        not isinstance(seq1, type(seq2))):
                    # The sequences are the same, but have differing types.
                    return

            if len1 > len2:
                differing += ('\nFirst %s contains %d additional '
                              'elements.\n' % (seq_type_name, len1 - len2))
                try:
                    differing += ('First extra element %d:\n%s\n' %
                                  (len2, seq1[len2]))
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('Unable to index element %d '
                                  'of first %s\n' % (len2, seq_type_name))
            elif len1 < len2:
                differing += ('\nSecond %s contains %d additional '
                              'elements.\n' % (seq_type_name, len2 - len1))
                try:
                    differing += ('First extra element %d:\n%s\n' %
                                  (len1, seq2[len1]))
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('Unable to index element %d '
                                  'of second %s\n' % (len1, seq_type_name))
        standardMsg = differing
        diffMsg = '\n' + '\n'.join(
            difflib.ndiff(pprint.pformat(seq1).splitlines(),
                          pprint.pformat(seq2).splitlines()))
        standardMsg = self._truncateMessage(standardMsg, diffMsg)
        msg = self._formatMessage(msg, standardMsg)
        self.fail(msg)

    def _formatMessage(self, msg, standardMsg):
        """Honour the longMessage attribute when generating failure messages.
        If longMessage is False this means:
        * Use only an explicit message if it is provided
        * Otherwise use the standard message for the assert

        If longMessage is True:
        * Use the standard message
        * If an explicit message is provided, plus ' : ' and the explicit message
        """
        if not self.longMessage:
            return msg or standardMsg
        if msg is None:
            return standardMsg
        try:
            # don't switch to '{}' formatting in Python 2.X
            # it changes the way unicode input is handled
            return '%s : %s' % (standardMsg, msg)
        except UnicodeDecodeError:
            return '%s : %s' % (safe_repr(standardMsg), safe_repr(msg))

    def _truncateMessage(self, message, diff):
        DIFF_OMITTED = ('\nDiff is %s characters long. '
                        'Set self.maxDiff to None to see it.')

        max_diff = self.maxDiff
        if max_diff is None or len(diff) <= max_diff:
            return message + diff
        return message + (DIFF_OMITTED % len(diff))

    _maxDiff = 80 * 8
    setattr(TestCase, "maxDiff", _maxDiff)
    setattr(TestCase, "_truncateMessage", _truncateMessage)
    setattr(TestCase, "_formatMessage", _formatMessage)
    setattr(TestCase, "assertSequenceEqual", assertSequenceEqual)
else:

    def assertSequenceEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertSequenceEqual`.
        """
        _builtin_unittest.TestCase.assertSequenceEqual(self, *args, **kwargs)

    setattr(TestCase, "assertSequenceEqual", assertSequenceEqual)
