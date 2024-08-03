#!/usr/bin/env python
from setuptools import setup, find_packages
import versioneer


setup(
    name="array_split",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    packages=find_packages(),
    package_data={'array_split': ["*.txt"]},
)

