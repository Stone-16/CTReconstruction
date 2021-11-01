from geometry import Geometry2d

class Projector:
    def __init__(self, geometry, device='cuda'):

        assert device in ['cuda', 'cpu'], f"Input device must be 'cpu' or 'cuda', but got device '{device}'."

        self.geometry = geometry
        self.device = device

    def projection(self):
        # TODO
        pass

    def backprojection(self):
        # TODO
        pass
