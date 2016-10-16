#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import subprocess


def create_git_describe():
    try:
        cmd = ["/usr/bin/env", "git", "describe"]
        p = \
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        p.wait()
        if p.returncode != 0:
            e = \
                subprocess.CalledProcessError(
                    returncode=p.returncode,
                    cmd=cmd
                )
            setattr(e, "output", " ".join([i.decode() for i in p.communicate()]))

            raise e
        # Write the git describe to text file
        open("array_split/git_describe.txt", "wt").write(p.communicate()[0].decode())
    except (Exception,) as e:
        # Try and make up a git-describe like string.
        print("Problem with '%s': %s: %s" % (" ".join(cmd), e, e.output))
        version_str = open("array_split/version.txt", "rt").read().strip()
        if ("TRAVIS_TAG" in os.environ.keys()) and (len(os.environ["TRAVIS_TAG"]) > 0):
            version_str = os.environ["TRAVIS_TAG"]
        else:
            if ("TRAVIS_BRANCH" in os.environ.keys()) and (len(os.environ["TRAVIS_BRANCH"]) > 0):
                version_str += os.environ["TRAVIS_BRANCH"]
            if ("TRAVIS_COMMIT" in os.environ.keys()) and (len(os.environ["TRAVIS_COMMIT"]) > 0):
                version_str += "-" + \
                    os.environ["TRAVIS_COMMIT"][0:min([7, len(os.environ["TRAVIS_COMMIT"])])]
        open("array_split/git_describe.txt", "wt").write(version_str)

create_git_describe()

_long_description =\
    """
The `array_split <https://array-split.github.io/array_split>`_ python package is
a modest enhancement to the
`numpy.array_split <http://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html>`_
function for sub-dividing multi-dimensional arrays into sub-arrays (slices). The main motivation
comes from parallel processing where one desires to split (decompose) a large array
(or multiple arrays) into smaller sub-arrays which can be processed concurrently by
other processes (`multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_ or
`mpi4py <http://pythonhosted.org/mpi4py/>`_) or other memory-limited hardware
(e.g. GPGPU using `pyopencl <https://mathema.tician.de/software/pyopencl/>`_,
`pycuda <https://mathema.tician.de/software/pycuda/>`_, etc).


Examples
========


   >>> from array_split import array_split, shape_split
   >>> import numpy as np
   >>>
   >>> ary = np.arange(0, 4*9)
   >>>
   >>> array_split(ary, 4) # 1D split into 4 sections (like numpy.array_split)
   [array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([ 9, 10, 11, 12, 13, 14, 15, 16, 17]),
    array([18, 19, 20, 21, 22, 23, 24, 25, 26]), array([27, 28, 29, 30, 31, 32, 33, 34, 35])]
   >>>
   >>> shape_split(ary.shape, 4) # 1D split into 4, slice objects instead of numpy.ndarray views
   array([(slice(0, 9, None),), (slice(9, 18, None),), (slice(18, 27, None),),
          (slice(27, 36, None),)],
         dtype=[('0', 'O')])
   >>>
   >>> ary = ary.reshape(4, 9) # Make ary 2D
   >>> split = shape_split(ary.shape, axis=(2, 3)) # 2D split into 2*3=6 sections
   array([[(slice(0, 2, None), slice(0, 3, None)),
           (slice(0, 2, None), slice(3, 6, None)),
           (slice(0, 2, None), slice(6, 9, None))],
          [(slice(2, 4, None), slice(0, 3, None)),
           (slice(2, 4, None), slice(3, 6, None)),
           (slice(2, 4, None), slice(6, 9, None))]],
         dtype=[('0', 'O'), ('1', 'O')])
   >>> sub_arys = [ary[tup] for tup in split.flatten()] # Split ary in sub-array views using the slice tuples.


Further examples at https://array-split.github.io/array_split/examples/.


"""

setup(
    name="array_split",
    version=open("array_split/version.txt", "rt").read().strip(),
    packages=find_packages(),
    # metadata for upload to PyPI
    author="Shane J. Latham",
    author_email="array.split@gmail.com",
    description=(
        "Python package for splitting arrays into sub-arrays "
        +
        "(i.e. rectangular-tiling and rectangular-domain-decomposition), "
        +
        "similar to ``numpy.array_split``."
    ),
    long_description=_long_description,
    license="MIT",
    keywords=(
        "sub-array tile tiling splitting split array "
        +
        "scipy numpy ndarray domain-decomposition array-decomposition"
    ),
    url="http://github.com/array-split/array_split",   # project home page
    classifiers=[
        # How mature is this project? Common values are
        #   2 - Pre-Alpha
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Utilities',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    install_requires=["numpy>=1.6", ],
    package_data={
        "array_split": ["version.txt", "git_describe.txt", "copyright.txt", "license.txt"]
    },
    # could also include download_url, etc.
)
