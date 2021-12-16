import numpy as np
from scipy import ndimage

def IRandonTransform(image, steps):
    channels = len(image[0])
    origin = np.zeros((steps, channels, channels))
    for i in range(steps):
        projectionValue = image[i, :]
        projectionValueExpandDim = np.expand_dims(projectionValue, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)
        origin[i] = ndimage.rotate(projectionValueRepeat, i*180/steps, reshape=False).astype(np.float64)
    iradon = np.sum(origin, axis=0)
    return iradon
