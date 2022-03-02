import numpy as np
from .utils import siddon

def easy_backprojection(sinogram, geometry):
    voxel_shape = geometry.voxel_shape
    voxel_spacing = geometry.voxel_spacing
    voxel = np.zeros(voxel_shape)
    source_coordinates = geometry.source_coordinates
    detector_coordinates = geometry.detector_coordinates
    angles = geometry.angles
    for theta_i in range(sinogram.shape[0]):
        theta = angles[theta_i]
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        source_coordinates_temp = np.dot(source_coordinates, rotation_matrix)
        detector_coordinates_temp = np.dot(detector_coordinates, rotation_matrix)
        for detector in range(sinogram.shape[1]):
            index_x, index_y, length = siddon(source_coordinates_temp[detector], detector_coordinates_temp[detector], geometry.grid)
            for i in range(index_x.shape[0]) :
                voxel[index_y[i], index_x[i]] += sinogram[theta_i, detector]
    return voxel

def easy_backprojection2(sinogram, geometry):
    voxel_shape = geometry.voxel_shape
    voxel_spacing = geometry.voxel_spacing
    voxel = np.zeros(voxel_shape)
    source_coordinates = geometry.source_coordinates
    detector_coordinates = geometry.detector_coordinates
    angles = geometry.angles
    for theta_i in range(sinogram.shape[0]):
        theta = angles[theta_i]
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        source_coordinates_temp = np.dot(source_coordinates, rotation_matrix)
        detector_coordinates_temp = np.dot(detector_coordinates, rotation_matrix)
        for detector in range(sinogram.shape[1]):
            index_x, index_y, length = siddon(source_coordinates_temp[detector], detector_coordinates_temp[detector], geometry.grid)
            for i in range(index_x.shape[0]) :
                voxel[index_y[i], index_x[i]] += sinogram[theta_i, detector] * length[i]
    
    return voxel

