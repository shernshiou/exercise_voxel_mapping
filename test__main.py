import unittest
import numpy as np
from main import map_voxel_index, get_affine


class TestVoxelMapping(unittest.TestCase):
    def test_voxel_mapping(self):
        s = [0, 0, 0]
        d = [1, 1, 5]
        x = [1, 0, 5]
        y = [0, 1, 0]
        affine = get_affine(s, d, x, y)
        voxel_index = np.array([10, 2, 12])
        new_voxel_index = map_voxel_index(voxel_index, affine, affine)
        np.testing.assert_array_equal(voxel_index, new_voxel_index)


if __name__ == "__main__":
    unittest.main()
