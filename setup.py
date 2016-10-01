#!/usr/bin/env python
from setuptools import setup, find_packages
import os

(sin, sout) = os.popen2("git describe")
file("array_split/git_describe.txt", "wt").write(sout.read())

setup(
    name="array_split",
    version=file("array_split/version.txt", "rt").read().strip(),
    packages=find_packages(),
    # metadata for upload to PyPI
    author="Shane J. Latham",
    author_email="array_split@gmail.com",
    description=(
        "Python for splitting arrays into sub-arrays "
        +
        "(i.e. rectangular-tiling and rectangular-domain-decomposition)."
    ),
    license="MIT",
    keywords=(
        "subarray tile tiling splitting split array "
        +
        "ndarray domain-decomposition array-decomposition"
    ),
    url="http://github.com/array_split/array_split",   # project home page
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
    
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
    
        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: MIT License',
    
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    # could also include long_description, download_url, classifiers, etc.
)
