"""
==============================
The :mod:`array_split` Package
==============================

.. currentmodule:: array_split

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()

import os as _os

__version__ = _pkg_resources.resource_string("array_split", "version.txt")

__all__ = [s for s in dir() if not s.startswith('_')]
