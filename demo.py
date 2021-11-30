import CTReconstruction
import matplotlib.pyplot as plt

# define phantom
img = CTReconstruction.phantom.shepp_logan_2d(512)
# show phantom
plt.figure(1)
plt.title("Phantom")
plt.imshow(img, cmap='gray')
# define geometry
# geometry = CTReconstruction.geometry.Geometry2d("parallel2d", 512, 1, 360, 512, 1.5)
geometry = CTReconstruction.geometry.Geometry2d("fanbeam2d_equispace", 512, 1, 720, 1024, 2, 1024, 2048)
# define projector
projector = CTReconstruction.projector.Projector(geometry)
# projeciton
sinogram = projector.projection(img)
# show sinogram
plt.figure(2)
plt.title("Sinogram")
plt.imshow(sinogram, cmap='gray')

plt.show()

# TODO how to reconstruct(FBP or ART)




