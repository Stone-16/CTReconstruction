import numpy as np

'''
addfafsadfsss
mode三类["parallel2d", "fanbeam2d_equiangular", "fanbeam2d_equispace"]
voxel_shape：像素 512，512  大小为2的array，分别是长宽，格子数
voxel_spacing：1，1  大小为2的array，每格长宽
angles：把2pi等分成720份 大小为720的array
detector_shape：1024  探测器个数 int
detector_spacing： 2 探测器间隔 float
distance_source_center 1024
distance_source_detector 2048


self.grid 每个格点的坐标
self.sinogram_shape = (self.angles.shape[0], self.detector_shape) （角度个数，探测器个数）
'''

class Geometry2d:
    def __init__(self, mode, voxel_shape, voxel_spacing, angles, detector_shape, detector_spacing,
                 distance_source_center=None, distance_source_detector=None):
        """
        :param mode: ["parallel2d", "fanbeam2d_equiangular", "fanbeam2d_equispace"]
        :param voxel_shape: image shape, 512 or [512, 512]
        :param voxel_spacing: pixel size, 1 or [1.0, 1.0]
        :param angles: radian, int or list or np.ndarray
        :param detector_shape: detector length, int
        :param detector_spacing: detector spacing, int or float
        :param distance_source_center: distance from source to rotation center
        :param distance_source_detector: distance from source to detector

        examples:
        # example1
        geometry = Geometry2d('parallel2d', [512, 512], 1, 720, 1024, 2)
        # example2
        geometry = Geometry2d('parallel2d', 512, [1, 1], 720, 1024, 2)
        # example3
        angles = np.arange(720, dtype=np.float32) * 2 * np.pi / 720
        # geometry = Geometry2d('fanbeam2d_equispace', 512, [1, 1], angles, 1024, 2, 1024, 2048)
        """

        # whether mode is valid
        mode_supported = ["parallel2d", "fanbeam2d_equiangular", "fanbeam2d_equispace"]
        assert mode in mode_supported, f"Input mode must in {mode_supported}, but got mode '{mode}'."

        # whether voxel_shape is valid
        if isinstance(voxel_shape, (int, float)):
            self.voxel_shape = np.array([voxel_shape, voxel_shape])
        elif isinstance(voxel_shape, list):
            self.voxel_shape = np.array(voxel_shape)
        elif isinstance(voxel_shape, np.ndarray):
            self.voxel_shape = voxel_shape
        else:
            assert False, f"The type of voxel shape must be int, float, list or np.ndarray, but got '{type(voxel_shape)}'"

        # whether voxel_spacing is valid
        if isinstance(voxel_spacing, (int, float)):
            self.voxel_spacing = np.array([voxel_spacing, voxel_spacing])
        elif isinstance(voxel_spacing, list):
            self.voxel_spacing = np.array(voxel_spacing)
        elif isinstance(voxel_spacing, np.ndarray):
            self.voxel_spacing = voxel_spacing
        else:
            assert False, f"The type of voxel spacing must be int, float, list or np.ndarray, but got '{type(voxel_spacing)}'"

        # whether angles is valid
        if isinstance(angles, int):
            self.angles = np.arange(angles, dtype=np.float32) * 2 * np.pi / angles
        elif isinstance(angles, list):
            self.angles = np.array(angles)
        elif isinstance(angles, np.ndarray):
            self.angles = angles
        else:
            assert False, f"The type of angles must be int, list or np.ndarray, but got '{type(angles)}'"

        # whether detector_shape is valid
        if isinstance(detector_shape, int):
            self.detector_shape = detector_shape
        else:
            assert False, f"The type of detector shape must be int, but got '{type(detector_shape)}'"

        # whether detector_spacing is valid
        if isinstance(detector_spacing, (int, float)):
            self.detector_spacing = detector_spacing
        else:
            assert False, f"The type of detector spacing must be int, but got '{type(detector_spacing)}'"

        # whether distance_source_center and distance_source_detector are valid
        if mode == "parallel2d":
            self.distance_source_center = np.max(self.voxel_shape * self.voxel_spacing)
            self.distance_source_detector = 2 * self.distance_source_center
        else:
            if distance_source_center is not None and distance_source_detector is not None:
                if isinstance(distance_source_center, (int, float)):
                    self.distance_source_center = distance_source_center
                else:
                    assert False, f"The type of distance_source_center must be int or float, but got '{type(distance_source_center)}'"
                if isinstance(distance_source_detector, (int, float)):
                    self.distance_source_detector = distance_source_detector
                else:
                    assert False, f"The type of distance_source_detector must be int or float, but got '{type(distance_source_detector)}'"
            else:
                assert False, f"distance_source_center and distance_source_detector must both be given in nonparallel mode."

        # define grid
        self.grid = Grid(self.voxel_shape, self.voxel_spacing)
        # define sinogram's shape
        self.sinogram_shape = (self.angles.shape[0], self.detector_shape)

        if mode == "parallel2d":
            source_x = np.array([[-self.distance_source_center]] * self.detector_shape)
            '二维数组，detector_shape行，1列，每行都是-self.distance_source_center'
            detector_x = np.array([[self.distance_source_center]] * self.detector_shape)
            '二维数组，detector_shape行，1列，每行都是self.distance_source_center'
            detector_y = (np.arange(self.detector_shape) - (self.detector_shape - 1) / 2.0) * self.detector_spacing
            detector_y = np.expand_dims(detector_y, axis=1)
            '二维数组，detector_shape行，1列，每行的数关于0对称，共detector_shape个，每个间隔detector_spacing'
            self.detector_coordinates = np.concatenate((detector_x, detector_y), axis=1)
            self.source_coordinates = np.concatenate((source_x, detector_y), axis=1)
        if mode == "fanbeam2d_equiangular":
            # TODO
            self.source_coordinates = np.repeat([[-self.distance_source_center, 0]], self.detector_shape, axis=0)
            detector_r = np.array([self.distance_source_detector] * self.detector_shape)
            detector_alpha = (np.arange(self.detector_shape) - (self.detector_shape - 1) / 2.0) * self.detector_spacing
            self.detector_coordinates = np.zeros([self.detector_shape, 2])
            for i in range(detector_shape) :
                self.detector_coordinates[i][0] = detector_r[i] * np.cos(detector_alpha[i]) - self.distance_source_center
                self.detector_coordinates[i][1] = detector_r[i] * np.sin(detector_alpha[i])
        if mode == "fanbeam2d_equispace":
            self.source_coordinates = np.repeat([[-self.distance_source_center, 0]], self.detector_shape, axis=0)
            #焦点复制detector_shape份
            detector_x = np.array([[self.distance_source_detector - self.distance_source_center]] * self.detector_shape)
            detector_y = (np.arange(self.detector_shape) - (self.detector_shape - 1) / 2.0) * self.detector_spacing
            detector_y = np.expand_dims(detector_y, axis=1)
            self.detector_coordinates = np.concatenate((detector_x, detector_y), axis=1)

class Grid:
    def __init__(self, voxel_shape, voxel_spacing):
        self.voxel_shape = voxel_shape
        self.voxel_spacing = voxel_spacing
        self.x_grid = (np.arange(voxel_shape[0] + 1) - voxel_shape[0] / 2.0) * voxel_spacing[0]
        self.y_grid = (np.arange(voxel_shape[1] + 1) - voxel_shape[1] / 2.0) * voxel_spacing[1]
