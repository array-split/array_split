#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import subprocess

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
        setattr(e, "output", " ".join([i.encode() for i in p.communicate()]))
        
        raise e
    # Write the git describe to text file
    open("array_split/git_describe.txt", "wt").write(p.communicate()[0].encode())
except (Exception ,) as e:
    print("Problem with '%s': %s" % (" ".join(cmd), e))
    open("array_split/git_describe.txt", "wt").write(
        open("array_split/version.txt", "rt").read().strip()
    )

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
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    install_requires=["numpy>=1.6", ],
    package_data={
        "array_split": ["version.txt", "git_describe.txt", "copyright.txt", "license.txt"]
    },
    # could also include long_description, download_url, etc.
)
