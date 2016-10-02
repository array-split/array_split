"""
===================================
The :mod:`array_split.split` Module
===================================

Defines array splitting functions and classes.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright

import array_split as _array_split

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _array_split.__version__


__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest.main()
