import numpy as np
from scipy import ndimage
from scipy.signal import convolve

# 两种滤波器的实现
# def RLFilter(N, d):
#     filterRL = np.zeros((N,))
#     for i in range(N):
#         filterRL[i] = - 1.0 / np.power((i - N / 2) * np.pi * d, 2.0)
#         if np.mod(i - N / 2, 2) == 0:
#             filterRL[i] = 0
#     filterRL[int(N/2)] = 1 / (4 * np.power(d, 2.0))
#     return filterRL

# def SLFilter(N, d):
#     filterSL = np.zeros((N,))
#     for i in range(N):
#         filterSL[i] = - 2 / (np.pi**2.0 * d**2.0 * (4 * (i - N / 2)**2.0 - 1))
#     return filterSL

def irandon_transform(image, steps):
    # 定义用于存储重建后的图像的数组
    channels = len(image[0])
    origin = np.zeros((steps, channels, channels))
    # filter = RLFilter(channels, 1)
    # filter = SLFilter(channels, 1)
    for i in range(steps):
        Value = image[i, :]
        # ValueFiltered = convolve(filter, Value, "same")
        # ValueExpandDim = np.expand_dims(ValueFiltered, axis=0)
        ValueExpandDim = np.expand_dims(Value, axis=0)
        ValueRepeat = ValueExpandDim.repeat(channels, axis=0)
        origin[i] = ndimage.rotate(ValueRepeat, i*180/steps, reshape=False).astype(np.float64)
    iradon = np.sum(origin, axis=0)
    return iradon