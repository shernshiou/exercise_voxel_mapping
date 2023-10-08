import numpy as np
from numpy.typing import NDArray
from typing import List


def map_voxel_index(
    voxel_index_a: NDArray, affine_a: NDArray, affine_b: NDArray
) -> NDArray:
    '''
    Map voxel index from volume A to volume B

    Parameters
    ----------
    voxel_index_a : NDArray
        Voxel index in volume A
    affine_a : NDArray
        Affine matrix of volume A
    affine_b : NDArray
        Affine matrix of volume B

    Returns
    -------
    NDArray : Voxel index in volume B
    '''
    # Convert voxel index to homogeneous coordinate
    coord = np.append(voxel_index_a, 1)
    # Inverse affine_a and apply affine_b
    new_coord = affine_b.dot(np.linalg.inv(affine_a).dot(coord))
    return new_coord[:3]


def get_affine(s: List, d: List, x: List, y: List, z: List = [0, 0, 1]) -> NDArray:
    '''
    Get affine matrix from s, d, x, y, z

    Parameters
    ----------
    s : List
        Origin of the volume
    d : List
        Dimensions of a single voxel in x, y, z.
    x : List
        Direction of the x-axis
    y : List
        Direction of the y-axis
    z : List, optional
        Direction of the z-axis, by default [0, 0, 1]
    
    Returns
    -------
    NDArray : Affine matrix of the volume
    '''
    return np.array(
        [
            [x[0] * d[0], y[0] * d[1], z[0] * d[2], s[0]],
            [x[1] * d[0], y[1] * d[1], z[1] * d[2], s[1]],
            [x[2] * d[0], y[2] * d[1], z[2] * d[2], s[2]],
            [0, 0, 0, 1],
        ]
    )


if __name__ == "__main__":
    # Volume A
    s_a = [0, 0, 0]
    d_a = [1, 1, 5]
    x_a = [1, 0, 5]
    y_a = [0, 1, 0]

    # Volume B
    s_b = [20, 10, 5]
    d_b = [1, 1, 5]
    x_b = [-1, 0, 0]
    y_b = [0, 0, 1]

    # Affine matrices
    affine_a = get_affine(s_a, d_a, x_a, y_a)
    affine_b = get_affine(s_b, d_b, x_b, y_b)

    voxel_index_a = np.array([10, 2, 12])
    voxel_index_b = map_voxel_index(voxel_index_a, affine_a, affine_b)
    print(voxel_index_b)
