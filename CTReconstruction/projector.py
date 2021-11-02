class Projector:
    def __init__(self, geometry, device='cuda'):
        assert device in ['cuda', 'cpu'], f"Input device must be 'cpu' or 'cuda', but got device '{device}'."

        self.geometry = geometry
        self.device = device

    def projection(self, img):
        # TODO
        sinogram = None
        return sinogram

    def filter(self, sinogram, filter_type):
        # TODO
        filtered_sinogram = None
        return filtered_sinogram

    def backprojection(self, sinogram):
        # TODO
        img = None
        return None


if __name__ == '__main__':
    from geometry import Geometry2d

    geometry = Geometry2d('parallel2d', 512, 1, 720, 1024, 2)
