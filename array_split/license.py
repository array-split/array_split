"""
To be replaced.
"""
# pylint: disable=redefined-builtin
from __future__ import absolute_import
import pkg_resources as _pkg_resources

__copyright__ = _pkg_resources.resource_string("array_split", "copyright.txt").decode()
__license__ = (
    __copyright__
    +
    "\n\n"
    +
    _pkg_resources.resource_string("array_split", "license.txt").decode()
)
__author__ = "Shane J. Latham"
__version__ = _pkg_resources.resource_string("array_split", "version.txt").decode().strip()


def version():
    """
    Returns :mod:`array_split` version string.

    :rtype: :obj:`str`
    :return: Version string.
    """
    return __version__


def license():
    """
    Returns the :mod:`array_split` license string.

    :rtype: :obj:`str`
    :return: License string.
    """
    return __license__


def copyright():
    """
    Returns the :mod:`array_split` copyright string.

    :rtype: :obj:`str`
    :return: Copyright string.
    """
    return __copyright__


__doc__ = \
    """
=====================================
The :mod:`array_split.license` Module
=====================================

License and copyright info.

.. currentmodule:: array_split.license

License
=======

%s

Copyright
=========

%s

Functions
=========

.. autosummary::
   :toctree: generated/

   license - Returns :mod:`array_split` license string.
   copyright - Returns :mod:`array_split` copyright string.
   version - Returns :mod:`array_split` version string.

""" % (license(), copyright())

__all__ = [s for s in dir() if not s.startswith('_')]
