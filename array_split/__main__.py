#!/usr/bin/env python
from __future__ import absolute_import
import array_split
import array_split.split
from array_split.license import license as _license
from array_split.license import copyright as _copyright

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = array_split.__version__


if __name__ == "__main__":
    import doctest
    doctest.testmod(array_split)
    doctest.testmod(array_split.split)
