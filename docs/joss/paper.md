---
title: 'array_split: Multi-dimensional array partitioning'
tags:
  - multi-dimensional array
  - partitioning
  - tile
  - tiling
  - domain decomposition
  - python
  - ndarray
  - numpy
authors:
 - name: Shane J Latham
   orcid: 0000-0001-8033-4665
   affiliation: 1
affiliations:
 - name: Department of Applied Mathematics, Research School of Physics and Engineering, The Australian National University
   index: 1
date: 12 September 2017
bibliography: paper.bib
---

# Summary

The `array_split` [@latham2017arraysplit] Python package extends existing dense
array partitioning capabilities found in the `numpy` [@walt2011numpy] (`numpy.array_split`)
and `skimage` [@van2014scikit] (`skimage.util.view_as_blocks`) Python packages.
In particular, it provides the means for partitioning
based on *array shape* (rather than requiring an actual `numpy.ndarray` object)
and can partition into *sub-arrays* based on a variety of criteria including:
per-axis number of partitions, total number of sub-arrays (with per-axis
number of partition constraints), explicit sub-array shape and constraining a
partitioning with an upper bound on the resulting sub-array number of bytes.

Application areas include:

Parallel Processing
:   Data parallelism by partitioning array for multi-process concurrency
    (e.g. `multiprocessing` [@pythonmultiprocessingmodule] or `mpi4py` [@dalcin2011parallel])
    based on number of cores,
    or partitioning for accelerator hardware concurrency
    (e.g. `pyopencl` or `pycuda` [kloeckner_pycuda_2012]) based on hardware memory limits.

File I/O
:   Partitioning large arrays for output to separate files
    (e.g. as part of a
    [virtual dataset](https://support.hdfgroup.org/HDF5/docNewFeatures/NewFeaturesVirtualDatasetDocs.html "HDF5 Virtual Dataset (VDS) Documentation")
    [@hdf5, @collette_python_hdf5_2014]) based on maximum file size, or out-of-core partitioning
    based on in-core memory limits.

# References
