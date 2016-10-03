from __future__ import absolute_import
import os as _os

__copyright__ = file(_os.path.join(_os.path.split(__file__)[0], "copyright.txt"), "rt").read()
__license__ = (
    __copyright__
    +
    "\n\n"
    +
    file(_os.path.join(_os.path.split(__file__)[0], "license.txt"), "rt").read()
)
__author__ = "Shane J. Latham"


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

   license - Function which returns :mod:`array_split` license string.
   copyright - Function which returns :mod:`array_split` copyright string.


""" % (license(), copyright())

__all__ = [s for s in dir() if not s.startswith('_')]
