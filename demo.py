import CTReconstruction
import matplotlib.pyplot as plt
import numpy as np

from CTReconstruction.geometry.geometry_2d import Geometry2d
from CTReconstruction.projector.projector import Projector

# define phantom
# img = CTReconstruction.phantom.shepp_logan_2d(512)
img = CTReconstruction.phantom.shepp_logan_2d(128)
# show phantom
plt.figure(1)
plt.title("Phantom")
plt.imshow(img, cmap='gray')

# define geometry
# geometry = CTReconstruction.geometry.Geometry2d("fanbeam2d_equiangular", 50, 1, 120, 120, 0.001, 120, 240)
# geometry = CTReconstruction.geometry.Geometry2d("fanbeam2d_equiangular", 512, 1, 720, 1024, 0.001, 1024, 2048)
geometry : Geometry2d = CTReconstruction.geometry.Geometry2d("parallel2d", 128, 2, 180, 360, 2)
# geometry = CTReconstruction.geometry.Geometry2d("parallel2d", 512, 1, 720, 1024, 2, 1024, 2048)
# define projector


my_projector : Projector = CTReconstruction.projector.Projector(geometry)

# projeciton
sinogram : np.array = my_projector.projection(img)

# show sinogram
plt.figure(2)
plt.title("Sinogram")
plt.imshow(sinogram, cmap='gray')

print("--------- sinogram ---------------")
print(sinogram)

# TODO how to reconstruct(FBP or ART)


voxel : np.array = my_projector.backprojection(sinogram)

plt.figure(3)
plt.title("voxel")
plt.imshow(voxel, cmap='gray')

plt.show()

