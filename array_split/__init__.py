"""
==============================
The :mod:`array_split` Package
==============================


"""
from __future__ import absolute_import
import os as _os

__version__ = file(_os.path.join(_os.path.split(__file__)[0], "version.txt"), "rt").read()

__all__ = [s for s in dir() if not s.startswith('_')]
