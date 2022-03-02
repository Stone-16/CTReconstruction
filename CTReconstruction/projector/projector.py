import numpy as np
from scipy.signal import convolve
from scipy.fft import fft, ifft, fftfreq, fftshift

from .projection import projection_siddon_numpy
# from .backprojection import easy_backprojection
from .backprojection import easy_backprojection2

# import sys
# sys.path.append("..")

# from .. import geometry
from ..geometry import Geometry2d
# from ..geometry.geometry_2d import Geometry2d


class Projector:
    def __init__(self, geometry , device='cuda'):
        assert device in ['cuda', 'cpu'], f"Input device must be 'cpu' or 'cuda', but got device '{device}'."

        self.geometry = geometry
        self.device = device

    def projection(self, img):
        sinogram = projection_siddon_numpy(img, self.geometry)
        return sinogram
    '''
    def derivate(self, sinogram):
        print("der")
        dp = np.zeros(sinogram.shape)
        for theta_i in range(sinogram.shape[0]) :
            for i in range(sinogram.shape[1] - 1) :
                dp[theta_i][i] = (sinogram[theta_i][i + 1] - sinogram[theta_i][i]) / self.geometry.detector_spacing
            dp[theta_i][sinogram.shape[1] - 1] = 0
        return dp
    '''
    def filter(self, sinogram : np.array, filter_type="ram-lak"):
        # TODO
        # d = self.geometry.detector_spacing
        d = 1
        N = sinogram.shape[1]
        # N = sinogram.shape[0]
        fil = np.zeros(N)
        for i in range(N) :
            if i == N / 2 :
                fil[i] = 1.0 / (4.0 * (d ** 2))
            elif (i - N / 2) % 2 != 0 :
                fil[i] = - 1.0 / (((i - N / 2) * np.pi * d) ** 2)
        print("-----------fil---------------")
        print(fil)
        # fil = fft(fil)
        # print(fil)
        return abs(fil)
        return fil

    def backprojection(self, sinogram):
        # TODO
        print("backprojection")
        fil = self.filter(sinogram)
        
        for theta_i in range(sinogram.shape[0]):
            pro = sinogram[theta_i, :]
            prof = convolve(fil, pro, "same")
            # prof = fft(pro) * fil
            # prof = abs(np.real(ifft(prof)))
            sinogram[theta_i, :] = prof
        
        sinogram = sinogram * np.pi / (sinogram.shape[0] * 2.0)
        img = easy_backprojection2(sinogram, self.geometry)
        print(" --------------- img ----------------")
        print(img)
        return img


if __name__ == '__main__':
    print("asd")
