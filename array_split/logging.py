"""
=====================================
The :mod:`array_split.logging` Module
=====================================

Default initialisation of python logging.


Some simple wrappers of python built-in :mod:`logging` module
for :mod:`array_split` logging.

.. currentmodule:: array_split.logging

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   SplitStreamHandler - A :obj:`logging.StreamHandler` which splits errors and warnings to *stderr*.
   initialise_loggers - Initialises handlers and formatters for loggers.
   get_formatter - "Returns :obj:`logging.Formatter` with time prefix string.
"""

from __future__ import absolute_import

import sys
import logging as _builtin_logging
from logging import *  # noqa: F401,F403


class _Python2SplitStreamHandler(_builtin_logging.Handler):
    """
    A python :obj:`logging.handlers` :samp:`Handler` class for
    splitting logging messages to different streams depending on
    the logging-level.
    """

    def __init__(self, outstr=sys.stdout, errstr=sys.stderr, splitlevel=_builtin_logging.WARNING):
        """
        Initialise with a pair of streams and a threshold level which determines
        the stream where the messages are writting.

        :type outstr: file-like
        :param outstr: Logging messages are written to this stream if
           the message level is less than :samp:`self.splitLevel`.
        :type errstr: stream
        :param errstr: Logging messages are written to this stream if
           the message level is greater-than-or-equal-to :samp:`self.splitLevel`.
        :type splitlevel: int
        :param splitlevel: Logging level threshold determining split streams for log messages.
        """
        self.outStream = outstr
        self.errStream = errstr
        self.splitLevel = splitlevel
        _builtin_logging.Handler.__init__(self)

    def emit(self, record):
        """
        Mostly copy-paste from :obj:`logging.StreamHandler`.
        """
        try:
            msg = self.format(record)
            if record.levelno < self.splitLevel:
                stream = self.outStream
            else:
                stream = self.errStream
            fs = "%s\n"

            try:
                if (isinstance(msg, unicode) and  # noqa: F405
                        getattr(stream, 'encoding', None)):
                    ufs = fs.decode(stream.encoding)
                    try:
                        stream.write(ufs % msg)
                    except UnicodeEncodeError:
                        stream.write((ufs % msg).encode(stream.encoding))
                else:
                    stream.write(fs % msg)
            except UnicodeError:
                stream.write(fs % msg.encode("UTF-8"))

            stream.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class _Python3SplitStreamHandler(_builtin_logging.Handler):
    """
    A python :obj:`logging.handlers` :samp:`Handler` class for
    splitting logging messages to different streams depending on
    the logging-level.
    """

    terminator = '\n'

    def __init__(self, outstr=sys.stdout, errstr=sys.stderr, splitlevel=_builtin_logging.WARNING):
        """
        Initialise with a pair of streams and a threshold level which determines
        the stream where the messages are writting.

        :type outstr: file-like
        :param outstr: Logging messages are written to this stream if
           the message level is less than :samp:`self.splitLevel`.
        :type errstr: stream
        :param errstr: Logging messages are written to this stream if
           the message level is greater-than-or-equal-to :samp:`self.splitLevel`.
        :type splitlevel: int
        :param splitlevel: Logging level threshold determining split streams for log messages.
        """
        self.outStream = outstr
        self.errStream = errstr
        self.splitLevel = splitlevel
        _builtin_logging.Handler.__init__(self)

    def flush(self):
        """
        Flushes the stream.
        """
        self.acquire()
        try:
            if self.outStream and hasattr(self.outStream, "flush"):
                self.outStream.flush()
            if self.errStream and hasattr(self.errStream, "flush"):
                self.errStream.flush()
        finally:
            self.release()

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            if record.levelno < self.splitLevel:
                stream = self.outStream
            else:
                stream = self.errStream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except:
            self.handleError(record)


if (sys.version_info[0] <= 2):
    class SplitStreamHandler(_Python2SplitStreamHandler):
        __doc__ = _Python2SplitStreamHandler.__doc__
        pass
else:
    class SplitStreamHandler(_Python3SplitStreamHandler):
        __doc__ = _Python3SplitStreamHandler.__doc__
        pass


def get_formatter(prefix_string="ARRSPLT| "):
    """
    Returns :obj:`logging.Formatter` object which produces messages
    with *time* and :samp:`prefix_string` prefix.

    :type prefix_string: :obj:`str` or :samp:`None`
    :param prefix_string: Prefix for all logging messages.
    :rtype: :obj:`logging.Formatter`
    :return: Regular formatter for logging.
    """
    if (prefix_string is None):
        prefix_string = ""
    formatter = \
        _builtin_logging.Formatter(
            "%(asctime)s|" + prefix_string + "%(message)s",
            "%H:%M:%S"
        )

    return formatter


def initialise_loggers(names, log_level=_builtin_logging.WARNING, handler_class=SplitStreamHandler):
    """
    Initialises specified loggers to generate output at the
    specified logging level. If the specified named loggers do not exist,
    they are created.

    :type names: :obj:`list` of :obj:`str`
    :param names: List of logger names.
    :type log_level: :obj:`int`
    :param log_level: Log level for messages, typically
       one of :obj:`logging.DEBUG`, :obj:`logging.INFO`, :obj:`logging.WARN`, :obj:`logging.ERROR`
       or :obj:`logging.CRITICAL`.
       See :ref:`levels`.
    :type handler_class: One of the :obj:`logging.handlers` classes.
    :param handler_class: The handler class for output of log messages,
       for example :obj:`SplitStreamHandler` or :obj:`logging.StreamHandler`.

    """
    frmttr = get_formatter()
    for name in names:
        logr = _builtin_logging.getLogger(name)
        handler = handler_class()
        handler.setFormatter(frmttr)
        logr.addHandler(handler)
        logr.setLevel(log_level)
