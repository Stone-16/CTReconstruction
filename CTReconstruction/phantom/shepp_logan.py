import numpy as np


def shepp_logan_2d(phantom_size: int = 512, modified: bool = True) -> np.ndarray:
    """
    :param phantom_size: the size of phantom, default is 512
    :param modified: if true, return the modified phantom which is higher contrast, default is true
    :return: 2d shepp logan phantom
    """
    if modified:
        # gray level   major axis   minor axis   center x  center y  theta
        shep = np.array([[1.0, 0.69, 0.92, 0, 0, 0],
                         [-0.8, 0.6624, 0.874, 0, -0.0184, 0],
                         [-0.2, 0.11, 0.31, 0.22, 0, -18],
                         [-0.2, 0.16, 0.41, -0.22, 0, 18],
                         [0.1, 0.21, 0.25, 0, 0.35, 0],
                         [0.1, 0.046, 0.046, 0, 0.1, 0],
                         [0.1, 0.046, 0.046, 0, -0.1, 0],
                         [0.1, 0.046, 0.023, -0.08, -0.605, 0],
                         [0.1, 0.023, 0.023, 0, -0.606, 0],
                         [0.1, 0.023, 0.046, 0.06, -0.605, 0]
                         ])
    else:
        # gray level   major axis   minor axis   center x  center y  theta
        shep = np.array([[2.0, 0.69, 0.92, 0, 0, 0],
                         [-0.98, 0.6624, 0.874, 0, -0.0184, 0],
                         [-0.02, 0.11, 0.31, 0.22, 0, -18],
                         [-0.02, 0.16, 0.41, -0.22, 0, 18],
                         [0.01, 0.21, 0.25, 0, 0.35, 0],
                         [0.01, 0.046, 0.046, 0, 0.1, 0],
                         [0.01, 0.046, 0.046, 0, -0.1, 0],
                         [0.01, 0.046, 0.023, -0.08, -0.605, 0],
                         [0.01, 0.023, 0.023, 0, -0.606, 0],
                         [0.01, 0.023, 0.046, 0.06, -0.605, 0]
                         ])

    p = np.zeros((phantom_size, phantom_size))

    yy, xx = np.mgrid[:phantom_size, :phantom_size]

    yy = -(yy - (phantom_size - 1) / 2.0) / ((phantom_size - 1) / 2)
    xx = (xx - (phantom_size - 1) / 2.0) / ((phantom_size - 1) / 2)

    for i in range(shep.shape[0]):
        gray = shep[i][0]
        a_2 = shep[i][1] ** 2
        b_2 = shep[i][2] ** 2
        x_ellipse = shep[i][3]
        y_ellipse = shep[i][4]
        theta = np.deg2rad(shep[i][5])

        x = xx - x_ellipse
        y = yy - y_ellipse

        x_r = x * np.cos(theta) + y * np.sin(theta)
        y_r = -x * np.sin(theta) + y * np.cos(theta)

        p[np.where((x_r ** 2 / a_2 + y_r ** 2 / b_2) <= 1)] += gray
            
    return p


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    p = shepp_logan_2d(128)
    plt.imshow(p, cmap='gray')
    plt.show()
