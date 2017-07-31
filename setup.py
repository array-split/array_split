#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
import os
import os.path
import subprocess


def read_readme():
    """
    Reads part of the README.rst for use as long_description in setup().
    """
    text = open("README.rst", "rt").read()
    text_lines = text.split("\n")
    ld_i_beg = 0
    while text_lines[ld_i_beg].find("start long description") < 0:
        ld_i_beg += 1
    ld_i_beg += 1
    ld_i_end = ld_i_beg
    while text_lines[ld_i_end].find("end long description") < 0:
        ld_i_end += 1

    ld_text = "\n".join(text_lines[ld_i_beg:ld_i_end])

    return ld_text


class CalledProcessError(subprocess.CalledProcessError):

    """
    Adds :samp:`output` attribute to :obj:`subprocess.CalledProcessError`.
    """

    def __init__(self, *args, **kwargs):
        """
        Adds :samp:`output` attribute to :samp:`self` if it doesn't exist.
        """
        subprocess.CalledProcessError.__init__(self, *args, **kwargs)
        if not hasattr(self, "output"):
            self.output = None


def create_git_describe():
    try:
        cmd = ["git", "describe"]
        p = \
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        p.wait()
        if p.returncode != 0:
            e = \
                CalledProcessError(
                    returncode=p.returncode,
                    cmd=cmd
                )
            setattr(e, "output", " ".join([i.decode() for i in p.communicate()]))

            raise e
        # Write the git describe to text file
        open(os.path.join("array_split", "git_describe.txt"), "wt").write(
            p.communicate()[0].decode()
        )
    except (Exception,) as e:
        # Try and make up a git-describe like string.
        output = ""
        if hasattr(e, "output"):
            output = e.output
        print("Problem with '%s': %s: %s" % (" ".join(cmd), e, output))
        version_str = open(os.path.join("array_split", "version.txt"), "rt").read().strip()
        if ("TRAVIS_TAG" in os.environ.keys()) and (len(os.environ["TRAVIS_TAG"]) > 0):
            version_str = os.environ["TRAVIS_TAG"]
        else:
            if ("TRAVIS_BRANCH" in os.environ.keys()) and (len(os.environ["TRAVIS_BRANCH"]) > 0):
                version_str += os.environ["TRAVIS_BRANCH"]
            if ("TRAVIS_COMMIT" in os.environ.keys()) and (len(os.environ["TRAVIS_COMMIT"]) > 0):
                version_str += "-" + \
                    os.environ["TRAVIS_COMMIT"][0:min([7, len(os.environ["TRAVIS_COMMIT"])])]
        open(os.path.join("array_split", "git_describe.txt"), "wt").write(version_str)


create_git_describe()

_long_description = read_readme()

sphinx_requires = []

# Only require sphinx for CI and readthedocs.org. 
if (
    (os.environ.get('READTHEDOCS', None) is not None)
    or
    (os.environ.get('CI', None) is not None)
    or
    (os.environ.get('TRAVIS', None) is not None)
    or
    (os.environ.get('APPVEYOR', None) is not None)
):
    sphinx_requires = ["sphinx>=1.4,<1.6", "sphinx_rtd_theme", ]

if (
    (int(sys.version_info[0]) < 2)
    or
    ((int(sys.version_info[0]) == 2) and (int(sys.version_info[1]) <= 6))
    or
    ((int(sys.version_info[0]) == 3) and (int(sys.version_info[1]) <= 3))
):
    sphinx_requires = []

setup(
    name="array_split",
    version=open(os.path.join("array_split", "version.txt"), "rt").read().strip(),
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
        "multi-dimendional-array  array sub-array tile tiling splitting split partition"
        +
        "partitioning scipy numpy ndarray domain-decomposition array-decomposition"
    ),
    url="http://github.com/array-split/array_split",   # project home page
    classifiers=[
        # How mature is this project? Common values are
        #   2 - Pre-Alpha
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
    ],
    install_requires=["numpy>=1.6", ] + sphinx_requires,
    package_data={
        "array_split": ["version.txt", "git_describe.txt", "copyright.txt", "license.txt"]
    },
    test_suite="array_split.tests",
    # could also include download_url, etc.
)
