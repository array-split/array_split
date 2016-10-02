"""
==============================
The :mod:`array_split` Package
==============================


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()

import os as _os

__version__ = file(_os.path.join(_os.path.split(__file__)[0], "version.txt"), "rt").read()

__all__ = [s for s in dir() if not s.startswith('_')]
