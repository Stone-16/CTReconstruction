import numpy as np


def siddon(src, dst, grid):
    """
    get the intersections of the line and grids
    Siddon, Robert, and L. "Fast calculation of the exact radiological path for a three-dimensional CT array." Medical Physics (1985).
    :param src: source point
    :param dst: detector point
    :param grid: image grid
    :return: the index and length of intersections
    """
    eps = 1e-6
    intersection_x = np.zeros(1)
    intersection_y = np.zeros(1)
    if (abs(src[0] - dst[0]) < eps) & (abs(src[1] - dst[1]) < eps):
        assert False, f"The source {src} and the detector {dst} are too close"
    elif abs(src[0] - dst[0]) < eps:
        if (src[0] > np.min(grid.x_grid)) & (src[0] < np.max(grid.x_grid)):
            intersection_x = src[0] * np.ones(grid.x_grid.shape[0])
            intersection_y = grid.y_grid
    elif abs(src[1] - dst[1]) < eps:
        if (src[1] > np.min(grid.y_grid)) & (src[1] < np.max(grid.y_grid)):
            intersection_x = grid.x_grid
            intersection_y = src[1] * np.ones(grid.y_grid.shape[0])
    else:
        x_grid_intersection_y = (grid.x_grid - src[0]) * (dst[1] - src[1]) / (dst[0] - src[0]) + src[1]
        y_grid_intersection_x = (grid.y_grid - src[1]) * (dst[0] - src[0]) / (dst[1] - src[1]) + src[0]
        intersection_x = np.concatenate((grid.x_grid, y_grid_intersection_x))
        intersection_y = np.concatenate((x_grid_intersection_y, grid.y_grid))

        in_boundary_index = (intersection_y >= np.min(grid.y_grid)) & (intersection_y <= np.max(grid.y_grid))
        intersection_x = intersection_x[in_boundary_index]
        intersection_y = intersection_y[in_boundary_index]
        in_boundary_index = (intersection_x >= np.min(grid.x_grid)) & (intersection_x <= np.max(grid.x_grid))
        intersection_y = intersection_y[in_boundary_index]
        intersection_x = intersection_x[in_boundary_index]

        intersection_x, index = np.unique(intersection_x, return_index=True)
        intersection_y = intersection_y[index]

    intersection_length = np.sqrt(np.power(np.diff(intersection_x), 2) + np.power(np.diff(intersection_y), 2))

    intersection_index_x = (intersection_x[:-1] + intersection_x[1:]) / 2
    intersection_index_y = (intersection_y[:-1] + intersection_y[1:]) / 2

    intersection_index_x = np.floor((intersection_index_x - np.min(grid.x_grid)) / grid.voxel_spacing[0])
    intersection_index_y = grid.voxel_shape[1] - np.floor((intersection_index_y - np.min(grid.y_grid)) / grid.voxel_spacing[1]) - 1

    intersection_index_x = intersection_index_x.astype(int)
    intersection_index_y = intersection_index_y.astype(int)

    return intersection_index_x, intersection_index_y, intersection_length