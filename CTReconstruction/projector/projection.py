import numpy as np
from .utils import siddon

# You can implement a faster projection function following the template of "function projection_numpy"

# 构建正弦图

'''
geometry Geometry2d类型的
sinogram_shape （角度个数，探测器个数）
'''
def projection_siddon_numpy(img, geometry):
    sinogram_shape = geometry.sinogram_shape
    sinogram = np.zeros(sinogram_shape)
    source_coordinates = geometry.source_coordinates
    detector_coordinates = geometry.detector_coordinates
    angles = geometry.angles
    for theta_i in range(sinogram_shape[0]):
        theta = angles[theta_i]
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        source_coordinates_temp = np.dot(source_coordinates, rotation_matrix)
        detector_coordinates_temp = np.dot(detector_coordinates, rotation_matrix)
        for detector in range(sinogram_shape[1]):
            index_x, index_y, length = siddon(source_coordinates_temp[detector], detector_coordinates_temp[detector], geometry.grid)
            #穿过的格子的横纵坐标和长度
            sinogram[theta_i, detector] = np.sum(img[index_y, index_x] * length)
    return sinogram
