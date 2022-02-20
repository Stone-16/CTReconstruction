import numpy as np
from .projection import projection_siddon_numpy
from .backprojection import irandon_transform


class Projector:
    def __init__(self, geometry, device='cuda'):
        assert device in ['cuda', 'cpu'], f"Input device must be 'cpu' or 'cuda', but got device '{device}'."

        self.geometry = geometry
        self.device = device

    def projection(self, img):
        sinogram = projection_siddon_numpy(img, self.geometry)
        return sinogram

    def filter(self, sinogram, filter_type="ram-lak"):
        # TODO
        filtered_sinogram = None
        return filtered_sinogram

    def backprojection(self, sinogram, steps):
        img = irandon_transform(sinogram, steps)
        return img
