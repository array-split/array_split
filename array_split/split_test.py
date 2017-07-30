"""
========================================
The :mod:`array_split.split_test` Module
========================================

.. currentmodule:: array_split.split_test

Module defining :mod:`array_split.split` unit-tests.
Execute as::

   python -m array_split.split_tests



Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   SplitTest - :obj:`unittest.TestCase` for :mod:`array_split.split` functions.


"""
from __future__ import absolute_import
import array_split as _array_split
import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging

from .split import ShapeSplitter, array_split, shape_split
from .split import calculate_num_slices_per_axis, shape_factors
from .split import calculate_tile_shape_for_max_bytes, pad_with_object, convert_halo_to_array_form
from .split import ARRAY_BOUNDS, NO_BOUNDS

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class SplitTest(_unittest.TestCase):

    """
    Tests for :mod:`array_split.split` module.
    """

    #: Class attribute for :obj:`logging.Logger` logging.
    logger = _logging.getLogger(__name__ + ".SplitTest")

    def test_properties(self):
        """
        Test :attr:`array_split.split.ARRAY_BOUNDS`
        and :attr:`array_split.split.NO_BOUNDS`.
        """
        self.assertNotEqual(None, _array_split.split.ARRAY_BOUNDS)
        self.assertNotEqual(None, _array_split.split.NO_BOUNDS)

    def test_pad_with_object(self):
        """
        Tests :func:`array_split.split.pad_with_object`.
        """
        lst = pad_with_object([1, 3, 4, ], 5, obj=1)
        self.assertSequenceEqual([1, 3, 4, 1, 1], lst)

        self.assertRaises(ValueError, pad_with_object, [1, 2, 3, 4], 3)

    def test_convert_halo_to_array_form(self):
        """
        Tests :func:`array_split.split.convert_halo_to_array_form`.
        """
        self.assertRaises(
            ValueError,
            convert_halo_to_array_form,
            halo=[0, 2, 4],
            ndim=2
        )
        self.assertRaises(
            ValueError,
            convert_halo_to_array_form,
            halo=[0, 2, 4],
            ndim=4
        )
        self.assertTrue(
            _np.all(
                convert_halo_to_array_form(1, 4)
                ==
                [[1, 1], [1, 1], [1, 1], [1, 1]]
            )
        )

        self.assertTrue(
            _np.all(
                convert_halo_to_array_form([1, 2, 3, 4], 4)
                ==
                [[1, 1], [2, 2], [3, 3], [4, 4]]
            )
        )

        self.assertTrue(
            _np.all(
                convert_halo_to_array_form([[1, 2], [3, 4], [5, 6], [7, 8]], 4)
                ==
                [[1, 2], [3, 4], [5, 6], [7, 8]]
            )
        )

    def test_shape_factors(self):
        """
        Tests for :func:`array_split.split.shape_factors`.
        """
        f = shape_factors(4, 2)
        self.assertTrue(_np.all(f == 2))

        f = shape_factors(4, 1)
        self.assertTrue(_np.all(f == 4))

        f = shape_factors(5, 2)
        self.assertTrue(_np.all(f == [1, 5]))

        f = shape_factors(6, 2)
        self.assertTrue(_np.all(f == [2, 3]))

        f = shape_factors(6, 3)
        self.assertTrue(_np.all(f == [1, 2, 3]))

    def test_calculate_num_slices_per_axis(self):
        """
        Tests for :func:`array_split.split.calculate_num_slices_per_axis`.
        """

        self.assertRaises(
            ValueError,
            calculate_num_slices_per_axis,
            [0, 1, 0],
            15,
            [1, 0, 1024]
        )

        spa = calculate_num_slices_per_axis([0, ], 5)
        self.assertEqual(1, len(spa))
        self.assertTrue(_np.all(spa == 5))

        spa = calculate_num_slices_per_axis([2, 0], 4)
        self.assertEqual(2, len(spa))
        self.assertTrue(_np.all(spa == 2))

        spa = calculate_num_slices_per_axis([0, 2], 4)
        self.assertEqual(2, len(spa))
        self.assertTrue(_np.all(spa == 2))

        spa = calculate_num_slices_per_axis([0, 0], 4)
        self.assertEqual(2, len(spa))
        self.assertTrue(_np.all(spa == 2))

        spa = calculate_num_slices_per_axis([0, 0], 16)
        self.assertEqual(2, len(spa))
        self.assertTrue(_np.all(spa == 4))

        spa = calculate_num_slices_per_axis([0, 0, 0], 8)
        self.assertEqual(3, len(spa))
        self.assertTrue(_np.all(spa == 2))

        spa = calculate_num_slices_per_axis([0, 1, 0], 8)
        self.assertEqual(3, len(spa))
        self.assertTrue(_np.all(spa == [4, 1, 2]))

        spa = calculate_num_slices_per_axis([0, 1, 0], 17)
        self.assertEqual(3, len(spa))
        self.assertTrue(_np.all(spa == [17, 1, 1]))

        spa = calculate_num_slices_per_axis([0, 1, 0], 15, [1, _np.inf, _np.inf])
        self.assertEqual(3, len(spa))
        self.assertTrue(_np.all(spa == [1, 1, 15]))

        spa = calculate_num_slices_per_axis([0, 1, 0], 16, [1, _np.inf, _np.inf])
        self.assertEqual(3, len(spa))
        self.assertTrue(_np.all(spa == [1, 1, 16]))

        spa = calculate_num_slices_per_axis([0, 0, 0], 64, [1, 2, _np.inf])
        self.assertEqual(3, len(spa))
        self.assertTrue(_np.all(spa == [1, 2, 32]))

        spa = calculate_num_slices_per_axis([0, 0, 0], 64, [_np.inf, 1, 2])
        self.assertEqual(3, len(spa))
        self.assertSequenceEqual([32, 1, 2], spa.tolist())

        spa = calculate_num_slices_per_axis([0, 0, 0], 27, [_np.inf, 2, 2])
        self.assertEqual(3, len(spa))
        self.assertSequenceEqual([27, 1, 1], spa.tolist())

    def test_calculate_tile_shape_for_max_bytes_1d(self):
        """
        Test case for :func:`array_split.split.calculate_tile_shape_for_max_bytes`,
        where :samp:`array_shape` parameter is 1D, i.e. of the form :samp:`(N,)`.
        """
        self.assertRaises(
            ValueError,
            calculate_tile_shape_for_max_bytes,
            array_shape=(512, 1024, 1024),
            array_itemsize=1,
            max_tile_bytes=2**20,
            sub_tile_shape=(1024, 128, 128)
        )
        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=1,
                max_tile_bytes=1024
            )
        self.assertSequenceEqual((512,), tile_shape)

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=1,
                max_tile_bytes=1024,
                sub_tile_shape=[64, ]
            )
        self.assertSequenceEqual((512,), tile_shape)

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=1,
                max_tile_bytes=1024,
                sub_tile_shape=[26, ]
            )
        self.assertSequenceEqual((260,), tile_shape)

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=1,
                max_tile_bytes=512
            )
        self.assertSequenceEqual((512,), tile_shape)

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=2,
                max_tile_bytes=512,
            )
        self.assertSequenceEqual((256,), tile_shape)

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=2,
                max_tile_bytes=512,
                sub_tile_shape=[32, ]
            )
        self.assertSequenceEqual((256,), tile_shape)

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=2,
                max_tile_bytes=512,
                sub_tile_shape=[60, ]
            )
        self.assertSequenceEqual((180,), tile_shape)

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=1,
                max_tile_bytes=512,
                halo=1
            )
        self.assertSequenceEqual((256,), tile_shape)

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=1,
                max_tile_bytes=514,
                halo=1
            )
        self.assertSequenceEqual((512,), tile_shape)

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512,),
                array_itemsize=2,
                max_tile_bytes=511
            )
        self.assertSequenceEqual((171,), tile_shape)

    def test_calculate_tile_shape_for_max_bytes_2d(self):
        """
        Test case for :func:`array_split.split.calculate_tile_shape_for_max_bytes`,
        where :samp:`array_shape` parameter is 2D, i.e. of the form :samp:`(H,W)`.
        """
        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512, 512),
                array_itemsize=1,
                max_tile_bytes=512 ** 2
            )
        self.assertSequenceEqual((512, 512), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512, 512),
                array_itemsize=1,
                max_tile_bytes=512 ** 2 - 1
            )
        self.assertSequenceEqual((256, 512), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(513, 512),
                array_itemsize=1,
                max_tile_bytes=512 ** 2 - 1
            )
        self.assertSequenceEqual((257, 512), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512, 512),
                array_itemsize=1,
                max_tile_bytes=512 ** 2 // 2
            )
        self.assertSequenceEqual((256, 512), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512, 512),
                array_itemsize=2,
                max_tile_bytes=512 ** 2 // 2
            )
        self.assertSequenceEqual((128, 512), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512, 512),
                array_itemsize=2,
                max_tile_bytes=512 ** 2 // 2,
                sub_tile_shape=(32, 64)
            )
        self.assertSequenceEqual((128, 512), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512, 512),
                array_itemsize=1,
                max_tile_bytes=512 ** 2 // 2,
                sub_tile_shape=(30, 64)
            )
        self.assertSequenceEqual((180, 512), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512, 512),
                array_itemsize=2,
                max_tile_bytes=512 ** 2 // 2,
                sub_tile_shape=(30, 64)
            )
        self.assertSequenceEqual((120, 512), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512, 1024),
                array_itemsize=1,
                max_tile_bytes=512 ** 2,
                sub_tile_shape=(30, 60)
            )
        self.assertSequenceEqual((180, 540), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(512, 32),
                array_itemsize=1,
                max_tile_bytes=2 * 32 * 32,
                sub_tile_shape=(32, 32)
            )
        self.assertSequenceEqual((64, 32), tile_shape.tolist())

        tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=(32, 512),
                array_itemsize=1,
                max_tile_bytes=2 * 32 * 32,
                sub_tile_shape=(32, 32)
            )
        self.assertSequenceEqual((32, 64), tile_shape.tolist())

    def test_multiple_parameter_groups_error(self):
        """
        Test for case for inconsistent parameter group arguments.
        """

        splitter = ShapeSplitter((100, ), axis=(3,), max_tile_bytes=30)
        self.assertRaises(
            ValueError,
            splitter.calculate_split
        )

        splitter = ShapeSplitter((100, ))
        self.assertRaises(
            ValueError,
            splitter.calculate_split
        )

        self.assertRaises(
            ValueError,
            splitter.set_split_extents_by_indices_per_axis
        )

    def test_array_split(self):
        """
        Test for case for :func:`array_split.split.array_split`.
        """
        x = _np.arange(9.0)
        self.assertArraySplitEqual(
            _np.array_split(x, 3),
            array_split(x, 3)
        )
        self.assertArraySplitEqual(
            _np.array_split(x, 4),
            array_split(x, 4)
        )
        idx = [2, 3, 5, ]
        self.assertArraySplitEqual(
            _np.array_split(x, idx),
            array_split(x, idx)
        )

        x = _np.arange(32)
        x = x.reshape((4, 8))
        self.logger.info("_np.array_split(x, 3, axis=0) = \n%s", _np.array_split(x, 3, axis=0))
        self.logger.info(
            "array_split.split.array_split(x, 3, axis=0) = \n%s", array_split(x, 3, axis=0)
        )
        self.assertArraySplitEqual(
            _np.array_split(x, 3, axis=0),
            array_split(x, 3, axis=0)
        )

        self.logger.info("_np.array_split(x, 3, axis=1) = \n%s", _np.array_split(x, 3, axis=1))
        self.logger.info(
            "array_split.split.array_split(x, 3, axis=1) = \n%s", array_split(x, 3, axis=1)
        )
        self.assertArraySplitEqual(
            _np.array_split(x, 3, axis=1),
            array_split(x, 3, axis=1)
        )

        self.logger.info("_np.array_split(x, 8, axis=0) = \n%s", _np.array_split(x, 8, axis=0))
        self.assertArraySplitEqual(
            _np.array_split(x, 8, axis=0),
            array_split(x, 8, axis=0)
        )

        x = _np.arange(0, 64)
        x = x.reshape((4, 16))
        self.assertArraySplitEqual(
            _np.array_split(x, [3, 8, 12], axis=1),
            array_split(x, [3, 8, 12], axis=1)
        )

        x = _np.arange(0, 512, dtype="int16")
        self.assertArraySplitEqual(
            [_np.arange(0, 256), _np.arange(256, 512)],
            array_split(x, max_tile_bytes=512)
        )

    def test_split_by_per_axis_indices(self):
        """
        Test for case for splitting by specified
        indices. For example::

           ShapeSplitter(array_shape=(10, 4), indices_or_sections=[[2, 6, 8], ]).calculate_split()

        """
        splitter = ShapeSplitter((10, 4), [[2, 6, 8], [1, ], [8, 4]])
        self.assertRaises(
            ValueError,
            splitter.calculate_split
        )
        splitter = ShapeSplitter((10, 4), [[2, 6, 8], ])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [4, 1]))
        self.assertEqual(slice(0, 2), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(2, 6), split[1, 0][0])  # axis 0 slice
        self.assertEqual(slice(6, 8), split[2, 0][0])  # axis 0 slice
        self.assertEqual(slice(8, 10), split[3, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 4), split[0, 0][1])  # axis 1 slice
        self.assertEqual(slice(0, 4), split[1, 0][1])  # axis 1 slice
        self.assertEqual(slice(0, 4), split[2, 0][1])  # axis 1 slice
        self.assertEqual(slice(0, 4), split[3, 0][1])  # axis 1 slice

        split1 = splitter.calculate_split_by_indices_per_axis()
        self.assertTrue(_np.all(split == split1))

        splitter = ShapeSplitter((10, 13), [None, [2, 5, 8], ])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [1, 4]))
        self.assertEqual(slice(0, 10), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 10), split[0, 1][0])  # axis 0 slice
        self.assertEqual(slice(0, 10), split[0, 2][0])  # axis 0 slice
        self.assertEqual(slice(0, 10), split[0, 3][0])  # axis 0 slice
        self.assertEqual(slice(0, 2), split[0, 0][1])  # axis 1 slice
        self.assertEqual(slice(2, 5), split[0, 1][1])  # axis 1 slice
        self.assertEqual(slice(5, 8), split[0, 2][1])  # axis 1 slice
        self.assertEqual(slice(8, 13), split[0, 3][1])  # axis 1 slice

        splitter = ShapeSplitter((10, 4), [[2, 6], [2, ]])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [3, 2]))
        self.assertEqual(slice(0, 2), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(2, 6), split[1, 0][0])  # axis 0 slice
        self.assertEqual(slice(6, 10), split[2, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 2), split[0, 1][0])  # axis 0 slice
        self.assertEqual(slice(2, 6), split[1, 1][0])  # axis 0 slice
        self.assertEqual(slice(6, 10), split[2, 1][0])  # axis 0 slice

        self.assertEqual(slice(0, 2), split[0, 0][1])  # axis 1 slice
        self.assertEqual(slice(0, 2), split[1, 0][1])  # axis 1 slice
        self.assertEqual(slice(0, 2), split[2, 0][1])  # axis 1 slice
        self.assertEqual(slice(2, 4), split[0, 1][1])  # axis 1 slice
        self.assertEqual(slice(2, 4), split[1, 1][1])  # axis 1 slice
        self.assertEqual(slice(2, 4), split[2, 1][1])  # axis 1 slice

        splitter = ShapeSplitter((10,), [[2, 6, 8], ])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [4, ]))
        self.assertEqual(slice(0, 2), split[0][0])  # axis 0 slice
        self.assertEqual(slice(2, 6), split[1][0])  # axis 0 slice
        self.assertEqual(slice(6, 8), split[2][0])  # axis 0 slice
        self.assertEqual(slice(8, 10), split[3][0])  # axis 0 slice

    def test_split_by_num_slices_1d(self):
        """
        Test for case for splitting by number of
        slice elements. For example::

           ShapeSplitter(array_shape=(10, ), indices_or_sections=3).calculate_split()
           ShapeSplitter(array_shape=(10, ), axis=[2, ]).calculate_split()

        """

        self.assertRaises(
            ValueError,
            ShapeSplitter,
            (10,),
            axis=[3, 4]
        )

        splitter = ShapeSplitter((10,), axis=[0, ])
        self.assertRaises(
            ValueError,
            splitter.set_split_extents_by_split_size
        )

        splitter = ShapeSplitter((10,), axis=[2, ])
        splitter.split_num_slices_per_axis = [2, 2]
        self.assertRaises(
            ValueError,
            splitter.check_consistent_parameter_dimensions
        )

        splitter = ShapeSplitter((10,), 3)
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [3, ]))
        self.assertEqual(slice(0, 4), split[0][0])  # axis 0 slice
        self.assertEqual(slice(4, 7), split[1][0])  # axis 0 slice
        self.assertEqual(slice(7, 10), split[2][0])  # axis 0 slice

        split1 = splitter.calculate_split_by_split_size()
        self.assertTrue(_np.all(split == split1))

        splitter = ShapeSplitter((10,), axis=[3, ])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [3, ]))
        self.assertEqual(slice(0, 4), split[0][0])  # axis 0 slice
        self.assertEqual(slice(4, 7), split[1][0])  # axis 0 slice
        self.assertEqual(slice(7, 10), split[2][0])  # axis 0 slice

        splitter = ShapeSplitter((10,), 3, axis=[3, ])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [3, ]))
        self.assertEqual(slice(0, 4), split[0][0])  # axis 0 slice
        self.assertEqual(slice(4, 7), split[1][0])  # axis 0 slice
        self.assertEqual(slice(7, 10), split[2][0])  # axis 0 slice

        splitter = ShapeSplitter((10,), 3, axis=[0, ])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [3, ]))
        self.assertEqual(slice(0, 4), split[0][0])  # axis 0 slice
        self.assertEqual(slice(4, 7), split[1][0])  # axis 0 slice
        self.assertEqual(slice(7, 10), split[2][0])  # axis 0 slice

        splitter = ShapeSplitter((10,), 2, axis=[0, ])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [2, ]))
        self.assertEqual(slice(0, 5), split[0][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1][0])  # axis 0 slice

    def test_split_by_num_slices_2d_non_0_axis_elems(self):
        """
        Test for case for splitting by number of
        slice elements. For example::

           ShapeSplitter(array_shape=(10, 13), axis=[2, 3]).calculate_split()

        """

        splitter = ShapeSplitter((10, 13), axis=[2, 2])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [2, 2]))
        self.assertEqual(slice(0, 5), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 5), split[0, 1][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 0][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 1][0])  # axis 0 slice
        self.assertEqual(slice(0, 7), split[0, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[0, 1][1])  # axis 1 slice
        self.assertEqual(slice(0, 7), split[1, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[1, 1][1])  # axis 1 slice

        splitter = ShapeSplitter((10, 13), 4, axis=[2, 2])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [2, 2]))
        self.assertEqual(slice(0, 5), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 5), split[0, 1][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 0][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 1][0])  # axis 0 slice
        self.assertEqual(slice(0, 7), split[0, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[0, 1][1])  # axis 1 slice
        self.assertEqual(slice(0, 7), split[1, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[1, 1][1])  # axis 1 slice

    def test_split_by_num_slices_2d_0_axis_elems(self):
        """
        Test for case for splitting by number of
        slice elements. For example::

           ShapeSplitter(array_shape=(10, 13), 4, axis=[2, 0]).calculate_split()

        """

        splitter = ShapeSplitter((10, 13), 4, axis=[1, 0])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [1, 4]))
        self.assertEqual(slice(0, 10), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 10), split[0, 1][0])  # axis 0 slice
        self.assertEqual(slice(0, 10), split[0, 2][0])  # axis 0 slice
        self.assertEqual(slice(0, 10), split[0, 3][0])  # axis 0 slice
        self.assertEqual(slice(0, 4), split[0, 0][1])  # axis 1 slice
        self.assertEqual(slice(4, 7), split[0, 1][1])  # axis 1 slice
        self.assertEqual(slice(7, 10), split[0, 2][1])  # axis 1 slice
        self.assertEqual(slice(10, 13), split[0, 3][1])  # axis 1 slice

        splitter = ShapeSplitter((10, 13), 4, axis=[0, 2])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [2, 2]))
        self.assertEqual(slice(0, 5), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 5), split[0, 1][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 0][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 1][0])  # axis 0 slice
        self.assertEqual(slice(0, 7), split[0, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[0, 1][1])  # axis 1 slice
        self.assertEqual(slice(0, 7), split[1, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[1, 1][1])  # axis 1 slice

        splitter = ShapeSplitter((10, 13), 4, axis=[2, 0])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [2, 2]))
        self.assertEqual(slice(0, 5), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 5), split[0, 1][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 0][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 1][0])  # axis 0 slice
        self.assertEqual(slice(0, 7), split[0, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[0, 1][1])  # axis 1 slice
        self.assertEqual(slice(0, 7), split[1, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[1, 1][1])  # axis 1 slice

        splitter = ShapeSplitter((10, 13), 4, axis=[0, 0])
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertTrue(_np.all(_np.array(split.shape) == [2, 2]))
        self.assertEqual(slice(0, 5), split[0, 0][0])  # axis 0 slice
        self.assertEqual(slice(0, 5), split[0, 1][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 0][0])  # axis 0 slice
        self.assertEqual(slice(5, 10), split[1, 1][0])  # axis 0 slice
        self.assertEqual(slice(0, 7), split[0, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[0, 1][1])  # axis 1 slice
        self.assertEqual(slice(0, 7), split[1, 0][1])  # axis 1 slice
        self.assertEqual(slice(7, 13), split[1, 1][1])  # axis 1 slice

    def test_calculate_split_by_tile_shape_1d(self):
        """
        Test for case for splitting by explicit tile shape. For example::

           ShapeSplitter(array_shape=(10, ), tile_shape=(3, )).calculate_split()

        """

        splitter = ShapeSplitter((10, ), tile_shape=(3, 3))
        self.assertRaises(
            ValueError,
            splitter.calculate_split
        )

        splitter = ShapeSplitter((100, ), max_tile_bytes=25, sub_tile_shape=(5, 5))
        self.assertRaises(
            ValueError,
            splitter.calculate_split
        )

        splitter = \
            ShapeSplitter((100, ), max_tile_bytes=25, sub_tile_shape=(5, ), max_tile_shape=(10, 10))
        self.assertRaises(
            ValueError,
            splitter.calculate_split
        )

        splitter = \
            ShapeSplitter((100, ), max_tile_shape=(10, ))
        self.assertRaises(
            ValueError,
            splitter.calculate_split
        )

        splitter = \
            ShapeSplitter((100, ), sub_tile_shape=(10, ))
        self.assertRaises(
            ValueError,
            splitter.calculate_split
        )

        splitter = ShapeSplitter((10, ), tile_shape=(3,))
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((4,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 3),), (slice(3, 6),), (slice(6, 9),), (slice(9, 10),)],
            split.tolist()
        )

        split1 = splitter.calculate_split_by_tile_shape()
        self.assertTrue(_np.all(split == split1))

        splitter = ShapeSplitter((10, ), tile_shape=(4,))
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((3,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 4),), (slice(4, 8),), (slice(8, 10),)],
            split.tolist()
        )

        splitter = ShapeSplitter((10, ), tile_shape=(5,))
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((2,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 5),), (slice(5, 10),)],
            split.tolist()
        )

        splitter = ShapeSplitter((10, ), tile_shape=(10,))
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((1,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 10),)],
            split.tolist()
        )

        splitter = ShapeSplitter((10, ), tile_shape=(11,))
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((1,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 10),)],
            split.tolist()
        )

    def test_calculate_split_by_tile_shape_2d(self):
        """
        Test for case for splitting by explicit tile shape. For example::

           ShapeSplitter(array_shape=(10, 17), tile_shape=(3, 8)).calculate_split()

        """

        splitter = ShapeSplitter((10, 17), tile_shape=(3, 8))
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((4, 3), split.shape)
        self.assertSequenceEqual(
            shape_split(splitter.array_shape, [[3, 6, 9], [8, 16]]).flatten().tolist(),
            split.flatten().tolist()
        )

        splitter = ShapeSplitter((10, 17), tile_shape=(2, 9))
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((5, 2), split.shape)
        self.assertSequenceEqual(
            shape_split(splitter.array_shape, [[2, 4, 6, 8], [9, ]]).flatten().tolist(),
            split.flatten().tolist()
        )

    def test_calculate_split_by_tile_max_bytes_1d(self):
        """
        Test for case for splitting with maximum number of tile bytes constraint.
        For example::

           ShapeSplitter((512, ), max_tile_bytes=256, array_itemsize=1).calculate_split()

        """

        splitter = ShapeSplitter((512, ), max_tile_bytes=256, array_itemsize=1)
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((2,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 256),), (slice(256, 512),)],
            split.tolist()
        )

        splitter = ShapeSplitter((512, ), max_tile_bytes=256, array_itemsize=2)
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((4,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 128),), (slice(128, 256),), (slice(256, 384),), (slice(384, 512),)],
            split.tolist()
        )

        split1 = splitter.calculate_split_by_tile_max_bytes()
        self.assertTrue(_np.all(split == split1))

        splitter = ShapeSplitter((512, ), max_tile_bytes=511, array_itemsize=2)
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((3,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 171),), (slice(171, 342),), (slice(342, 512),)],
            split.tolist()
        )

        splitter = ShapeSplitter((512, ), max_tile_bytes=256, array_itemsize=1, halo=1)
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((3,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 172),), (slice(170, 343),), (slice(341, 512),)],
            split.tolist()
        )

        splitter = \
            ShapeSplitter((512, ), max_tile_bytes=256, array_itemsize=1, max_tile_shape=(128,))
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((4,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 128),), (slice(128, 256),), (slice(256, 384),), (slice(384, 512),)],
            split.tolist()
        )

        splitter = \
            ShapeSplitter((512, ), max_tile_bytes=256, array_itemsize=1, sub_tile_shape=(130,))
        split = splitter.calculate_split()
        self.logger.info("split.shape = %s", split.shape)
        self.logger.info("split =\n%s", split)
        self.assertSequenceEqual((4,), split.shape)
        self.assertSequenceEqual(
            [(slice(0, 130),), (slice(130, 260),), (slice(260, 390),), (slice(390, 512),)],
            split.tolist()
        )

    def test_calculate_split_with_array_start_1d(self):
        """
        Test for case for splitting with explicit array start multi-index. For example::

           shape_split((10,), 2, array_start=(32,))

        """

        self.assertRaises(
            ValueError,
            shape_split,
            (10,),
            2,
            array_start=(2, 3)
        )
        split = shape_split((10,), 2, array_start=(0,))
        self.assertSequenceEqual(
            [(slice(0, 5),), (slice(5, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 2, array_start=(32,))
        self.assertSequenceEqual(
            [(slice(32, 37),), (slice(37, 42),)],
            split.tolist()
        )

    def test_calculate_split_with_array_start_2d(self):
        """
        Test for case for splitting with explicit array start multi-index. For example::

           shape_split((10, 12), axis=(2, 2), array_start=(32, 16))

        """

        split = shape_split((10, 12), axis=(2, 2), array_start=(0, 0))
        self.assertSequenceEqual(
            [
                [(slice(0, 5), slice(0, 6)), (slice(0, 5), slice(6, 12))],
                [(slice(5, 10), slice(0, 6)), (slice(5, 10), slice(6, 12))]
            ],
            split.tolist()
        )

        split = shape_split((10, 12), axis=(2, 2), array_start=(32, 16))
        self.assertSequenceEqual(
            [
                [(slice(32, 37), slice(16, 22)), (slice(32, 37), slice(22, 28))],
                [(slice(37, 42), slice(16, 22)), (slice(37, 42), slice(22, 28))]
            ],
            split.tolist()
        )

    def test_calculate_split_with_halo_1d(self):
        """
        Test for case for splitting with explicit halo. For example::

           shape_split((10,), 3, halo=[(1, 2), ])

        """

        split = shape_split((10,), 3, halo=(0,))
        self.assertSequenceEqual(
            [(slice(0, 4),), (slice(4, 7),), (slice(7, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=(0, 0))
        self.assertSequenceEqual(
            [(slice(0, 4),), (slice(4, 7),), (slice(7, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=(1, 0))
        self.assertSequenceEqual(
            [(slice(0, 4),), (slice(3, 7),), (slice(6, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=(0, 1))
        self.assertSequenceEqual(
            [(slice(0, 5),), (slice(4, 8),), (slice(7, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=(1, 1))
        self.assertSequenceEqual(
            [(slice(0, 5),), (slice(3, 8),), (slice(6, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=[(1, 2), ])
        self.assertSequenceEqual(
            [(slice(0, 6),), (slice(3, 9),), (slice(6, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=1)
        self.assertSequenceEqual(
            [(slice(0, 5),), (slice(3, 8),), (slice(6, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=1, tile_bounds_policy=ARRAY_BOUNDS)
        self.assertSequenceEqual(
            [(slice(0, 5),), (slice(3, 8),), (slice(6, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=1, tile_bounds_policy=None)
        self.assertSequenceEqual(
            [(slice(0, 5),), (slice(3, 8),), (slice(6, 10),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=1, tile_bounds_policy=NO_BOUNDS)
        self.assertSequenceEqual(
            [(slice(-1, 5),), (slice(3, 8),), (slice(6, 11),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=((2, 3),), tile_bounds_policy=NO_BOUNDS)
        self.assertSequenceEqual(
            [(slice(-2, 7),), (slice(2, 10),), (slice(5, 13),)],
            split.tolist()
        )

        split = shape_split((10,), 3, halo=(2, 3), tile_bounds_policy=NO_BOUNDS)
        self.assertSequenceEqual(
            [(slice(-2, 7),), (slice(2, 10),), (slice(5, 13),)],
            split.tolist()
        )

    def test_calculate_split_with_halo_2d(self):
        """
        Test for case for splitting with explicit halo. For example::

            shape_split(
                (15, 13),
                axis=[3, 3],
                halo=[[1, 2], [2, 3]],
                tile_bounds_policy=ARRAY_BOUNDS
            )

        """

        self.assertRaises(
            ValueError,
            shape_split,
            (15, 13),
            axis=[3, 3],
            halo=[0, 1, 2]
        )

        self.assertRaises(
            ValueError,
            shape_split,
            (15, 13),
            axis=[3, 3],
            halo=[0, 1],
            tile_bounds_policy="bogus"
        )

        split = shape_split((15, 13), axis=[3, 3], halo=0)
        self.assertSequenceEqual(
            [
                [
                    (slice(0, 5), slice(0, 5)),
                    (slice(0, 5), slice(5, 9)),
                    (slice(0, 5), slice(9, 13))
                ],
                [
                    (slice(5, 10), slice(0, 5)),
                    (slice(5, 10), slice(5, 9)),
                    (slice(5, 10), slice(9, 13))
                ],
                [
                    (slice(10, 15), slice(0, 5)),
                    (slice(10, 15), slice(5, 9)),
                    (slice(10, 15), slice(9, 13))
                ],
            ],
            split.tolist()
        )

        split = shape_split((15, 13), axis=[3, 3], halo=(0, 0))
        self.assertSequenceEqual(
            [
                [
                    (slice(0, 5), slice(0, 5)),
                    (slice(0, 5), slice(5, 9)),
                    (slice(0, 5), slice(9, 13))
                ],
                [
                    (slice(5, 10), slice(0, 5)),
                    (slice(5, 10), slice(5, 9)),
                    (slice(5, 10), slice(9, 13))
                ],
                [
                    (slice(10, 15), slice(0, 5)),
                    (slice(10, 15), slice(5, 9)),
                    (slice(10, 15), slice(9, 13))
                ],
            ],
            split.tolist()
        )

        split = shape_split((15, 13), axis=[3, 3], halo=[[0, 0], [0, 0]])
        self.assertSequenceEqual(
            [
                [
                    (slice(0, 5), slice(0, 5)),
                    (slice(0, 5), slice(5, 9)),
                    (slice(0, 5), slice(9, 13))
                ],
                [
                    (slice(5, 10), slice(0, 5)),
                    (slice(5, 10), slice(5, 9)),
                    (slice(5, 10), slice(9, 13))
                ],
                [
                    (slice(10, 15), slice(0, 5)),
                    (slice(10, 15), slice(5, 9)),
                    (slice(10, 15), slice(9, 13))
                ],
            ],
            split.tolist()
        )

        split = \
            shape_split(
                (15, 13),
                axis=[3, 3],
                halo=[[0, 0], [0, 0]],
                tile_bounds_policy=ARRAY_BOUNDS
            )
        self.assertSequenceEqual(
            [
                [
                    (slice(0, 5), slice(0, 5)),
                    (slice(0, 5), slice(5, 9)),
                    (slice(0, 5), slice(9, 13))
                ],
                [
                    (slice(5, 10), slice(0, 5)),
                    (slice(5, 10), slice(5, 9)),
                    (slice(5, 10), slice(9, 13))
                ],
                [
                    (slice(10, 15), slice(0, 5)),
                    (slice(10, 15), slice(5, 9)),
                    (slice(10, 15), slice(9, 13))
                ],
            ],
            split.tolist()
        )

        split = \
            shape_split(
                (15, 13),
                axis=[3, 3],
                halo=[[0, 0], [0, 0]],
                tile_bounds_policy=NO_BOUNDS
            )
        self.assertSequenceEqual(
            [
                [
                    (slice(0, 5), slice(0, 5)),
                    (slice(0, 5), slice(5, 9)),
                    (slice(0, 5), slice(9, 13))
                ],
                [
                    (slice(5, 10), slice(0, 5)),
                    (slice(5, 10), slice(5, 9)),
                    (slice(5, 10), slice(9, 13))
                ],
                [
                    (slice(10, 15), slice(0, 5)),
                    (slice(10, 15), slice(5, 9)),
                    (slice(10, 15), slice(9, 13))
                ],
            ],
            split.tolist()
        )

        split = \
            shape_split(
                (15, 13),
                axis=[3, 3],
                halo=1,
                tile_bounds_policy=ARRAY_BOUNDS
            )
        self.assertSequenceEqual(
            [
                [
                    (slice(0, 6), slice(0, 6)),
                    (slice(0, 6), slice(4, 10)),
                    (slice(0, 6), slice(8, 13))
                ],
                [
                    (slice(4, 11), slice(0, 6)),
                    (slice(4, 11), slice(4, 10)),
                    (slice(4, 11), slice(8, 13))
                ],
                [
                    (slice(9, 15), slice(0, 6)),
                    (slice(9, 15), slice(4, 10)),
                    (slice(9, 15), slice(8, 13))
                ],
            ],
            split.tolist()
        )

        split = \
            shape_split(
                (15, 13),
                axis=[3, 3],
                halo=(2, 3),
                tile_bounds_policy=ARRAY_BOUNDS
            )
        self.assertSequenceEqual(
            [
                [
                    (slice(0, 7), slice(0, 8)),
                    (slice(0, 7), slice(2, 12)),
                    (slice(0, 7), slice(6, 13))
                ],
                [
                    (slice(3, 12), slice(0, 8)),
                    (slice(3, 12), slice(2, 12)),
                    (slice(3, 12), slice(6, 13))
                ],
                [
                    (slice(8, 15), slice(0, 8)),
                    (slice(8, 15), slice(2, 12)),
                    (slice(8, 15), slice(6, 13))
                ],
            ],
            split.tolist()
        )

        split = \
            shape_split(
                (15, 13),
                axis=[3, 3],
                halo=[[1, 2], [2, 3]],
                tile_bounds_policy=ARRAY_BOUNDS
            )
        self.assertSequenceEqual(
            [
                [
                    (slice(0, 7), slice(0, 8)),
                    (slice(0, 7), slice(3, 12)),
                    (slice(0, 7), slice(7, 13))
                ],
                [
                    (slice(4, 12), slice(0, 8)),
                    (slice(4, 12), slice(3, 12)),
                    (slice(4, 12), slice(7, 13))
                ],
                [
                    (slice(9, 15), slice(0, 8)),
                    (slice(9, 15), slice(3, 12)),
                    (slice(9, 15), slice(7, 13))
                ],
            ],
            split.tolist()
        )

        # NO_BOUNDS

        split = \
            shape_split(
                (15, 13),
                axis=[3, 3],
                halo=1,
                tile_bounds_policy=NO_BOUNDS
            )
        self.assertSequenceEqual(
            [
                [
                    (slice(-1, 6), slice(-1, 6)),
                    (slice(-1, 6), slice(4, 10)),
                    (slice(-1, 6), slice(8, 14))
                ],
                [
                    (slice(4, 11), slice(-1, 6)),
                    (slice(4, 11), slice(4, 10)),
                    (slice(4, 11), slice(8, 14))
                ],
                [
                    (slice(9, 16), slice(-1, 6)),
                    (slice(9, 16), slice(4, 10)),
                    (slice(9, 16), slice(8, 14))
                ],
            ],
            split.tolist()
        )

        split = \
            shape_split(
                (15, 13),
                axis=[3, 3],
                halo=(2, 3),
                tile_bounds_policy=NO_BOUNDS
            )
        self.assertSequenceEqual(
            [
                [
                    (slice(-2, 7), slice(-3, 8)),
                    (slice(-2, 7), slice(2, 12)),
                    (slice(-2, 7), slice(6, 16))
                ],
                [
                    (slice(3, 12), slice(-3, 8)),
                    (slice(3, 12), slice(2, 12)),
                    (slice(3, 12), slice(6, 16))
                ],
                [
                    (slice(8, 17), slice(-3, 8)),
                    (slice(8, 17), slice(2, 12)),
                    (slice(8, 17), slice(6, 16))
                ],
            ],
            split.tolist()
        )

        split = \
            shape_split(
                (15, 13),
                axis=[3, 3],
                halo=[[1, 2], [2, 3]],
                tile_bounds_policy=NO_BOUNDS
            )
        self.assertSequenceEqual(
            [
                [
                    (slice(-1, 7), slice(-2, 8)),
                    (slice(-1, 7), slice(3, 12)),
                    (slice(-1, 7), slice(7, 16))
                ],
                [
                    (slice(4, 12), slice(-2, 8)),
                    (slice(4, 12), slice(3, 12)),
                    (slice(4, 12), slice(7, 16))
                ],
                [
                    (slice(9, 17), slice(-2, 8)),
                    (slice(9, 17), slice(3, 12)),
                    (slice(9, 17), slice(7, 16))
                ],
            ],
            split.tolist()
        )

    def test_calculate_split_with_halo_for_empty_tiles(self):
        """
        Tests :func:`array_split.shape_split` for case of
        empty tiles and non-zero halo to ensure halo elements
        are not added to empty tiles.
        """
        # Zero halo, empty tiles.
        split = shape_split((5, 12), axis=[8, 1], halo=0)
        self.assertSequenceEqual(
            (
                slice(4, 5, None),
                slice(0, 12, None)
            ),
            split[4, 0].tolist()
        )
        for i in range(5, 8):
            self.assertSequenceEqual(
                (
                    slice(5, 5, None),
                    slice(0, 12, None)
                ),
                split[i, 0].tolist()
            )

        # Now ensure that empty tiles remain empty despite halo=1
        split = shape_split((5, 12), axis=[8, 1], halo=1)
        self.assertSequenceEqual(
            (
                slice(3, 5, None),
                slice(0, 12, None)
            ),
            split[4, 0].tolist()
        )
        for i in range(5, 8):
            self.assertSequenceEqual(
                (
                    slice(5, 5, None),
                    slice(0, 12, None)
                ),
                split[i, 0].tolist()
            )

        split = shape_split((5, 12), axis=[8, 15], halo=1)
        self.assertSequenceEqual(
            (
                slice(3, 5, None),
                slice(0, 2, None)
            ),
            split[4, 0].tolist()
        )
        for i in range(5, 8):
            for j in range(0, 12):
                self.assertEqual(
                    slice(5, 5, None),
                    split[i, j].tolist()[0]
                )
            for j in range(12, 15):
                self.assertSequenceEqual(
                    (
                        slice(5, 5, None),
                        slice(12, 12, None)
                    ),
                    split[i, j].tolist()
                )
        for i in range(0, 5):
            for j in range(12, 15):
                self.assertEqual(
                    slice(12, 12, None),
                    split[i, j].tolist()[1]
                )

    def test_calculate_split_halos_from_extents(self):
        """
        Tests the :meth:`array_split.split.ShapeSplitter.calculate_split_halos_from_extents`
        method.
        """

        # Tiles wider than halo width
        splitter = ShapeSplitter((15, 13), axis=[3, 3], halo=0)
        splt = splitter.calculate_split()
        splt_halos = splitter.calculate_split_halos_from_extents()
        self.assertSequenceEqual(splt.shape, splt_halos.shape)
        self.assertTrue(_np.all(splt_halos.astype(_np.int64) == 0))

        # Some tiles narrower than halo width
        splitter = ShapeSplitter((15, 13), axis=[3, 3], halo=5, tile_bounds_policy=ARRAY_BOUNDS)
        splt = splitter.calculate_split()
        splt_halos = splitter.calculate_split_halos_from_extents()
        self.assertSequenceEqual(splt.shape, splt_halos.shape)
        for i in range(3):
            self.assertSequenceEqual([0, 5], tuple(splt_halos[0, i][0]))
            self.assertSequenceEqual([5, 5], tuple(splt_halos[1, i][0]))
            self.assertSequenceEqual([5, 0], tuple(splt_halos[2, i][0]))
            self.assertSequenceEqual([0, 5], tuple(splt_halos[i, 0][1]))
            self.assertSequenceEqual([5, 4], tuple(splt_halos[i, 1][1]))
            self.assertSequenceEqual([5, 0], tuple(splt_halos[i, 2][1]))

        splitter = ShapeSplitter((15, 13), axis=[3, 3], halo=0)
        splt = splitter.calculate_split()
        splt_halos = splitter.calculate_split_halos_from_extents()
        self.assertSequenceEqual(splt.shape, splt_halos.shape)
        self.assertTrue(_np.all(splt_halos.astype(_np.int64) == 0))

        # Tiles narrower than halo width
        splitter = ShapeSplitter((5, 13), axis=[5, 3], halo=5, tile_bounds_policy=ARRAY_BOUNDS)
        splt = splitter.calculate_split()
        splt_halos = splitter.calculate_split_halos_from_extents()
        self.assertSequenceEqual(splt.shape, splt_halos.shape)
        for i in range(3):
            self.assertSequenceEqual([0, 4], tuple(splt_halos[0, i][0]))
            self.assertSequenceEqual([1, 3], tuple(splt_halos[1, i][0]))
            self.assertSequenceEqual([2, 2], tuple(splt_halos[2, i][0]))
            self.assertSequenceEqual([3, 1], tuple(splt_halos[3, i][0]))
            self.assertSequenceEqual([4, 0], tuple(splt_halos[4, i][0]))
        for i in range(5):
            self.assertSequenceEqual([0, 5], tuple(splt_halos[i, 0][1]))
            self.assertSequenceEqual([5, 4], tuple(splt_halos[i, 1][1]))
            self.assertSequenceEqual([5, 0], tuple(splt_halos[i, 2][1]))

        # Zero sized tiles
        # Tiles narrower than halo width
        splitter = ShapeSplitter((5, 13), axis=[7, 3], halo=5, tile_bounds_policy=ARRAY_BOUNDS)
        splt = splitter.calculate_split()
        splt_halos = splitter.calculate_split_halos_from_extents()
        self.assertSequenceEqual(splt.shape, splt_halos.shape)
        for i in range(3):
            self.assertSequenceEqual([0, 4], tuple(splt_halos[0, i][0]))
            self.assertSequenceEqual([1, 3], tuple(splt_halos[1, i][0]))
            self.assertSequenceEqual([2, 2], tuple(splt_halos[2, i][0]))
            self.assertSequenceEqual([3, 1], tuple(splt_halos[3, i][0]))
            self.assertSequenceEqual([4, 0], tuple(splt_halos[4, i][0]))
            self.assertSequenceEqual([0, 0], tuple(splt_halos[5, i][0]))
            self.assertSequenceEqual([0, 0], tuple(splt_halos[6, i][0]))
        for i in range(5):
            self.assertSequenceEqual([0, 5], tuple(splt_halos[i, 0][1]))
            self.assertSequenceEqual([5, 4], tuple(splt_halos[i, 1][1]))
            self.assertSequenceEqual([5, 0], tuple(splt_halos[i, 2][1]))


__all__ = [s for s in dir() if not s.startswith('_')]

_unittest.main(__name__)
