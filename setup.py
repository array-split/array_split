#!/usr/bin/env python
from setuptools import setup, find_packages
import os

(sin, sout) = os.popen2("git describe")
file("array_split/git_describe.txt", "wt").write(sout.read())

setup(
    name = "array_split",
    version = file("array_split/version.txt", "rt").read().strip(),
    packages = find_packages(),
    # metadata for upload to PyPI
    author = "Shane J. Latham",
    author_email = "array_split@gmail.com",
    description =  (
        "Python for splitting arrays into sub-arrays "
        +
        "(i.e. rectangular-tiling, rectangular-domain-decomposition)."
    ),
    license = "MIT",
    keywords = (
        "subarray tile tiling splitting split array "
        +
        "ndarray domain-decomposition array-decomposition"
    ),
    url = "http://github.com/array_split/array_split",   # project home page

    # could also include long_description, download_url, classifiers, etc.
)
