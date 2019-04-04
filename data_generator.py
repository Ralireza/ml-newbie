import numpy as np
import matplotlib.pyplot as plt


def data_generator(mean, cov):
    x, y = np.random.multivariate_normal(mean, cov, 7).T
    # plt.plot(x, y, 'x')
    # plt.axis('equal')
    # plt.show()
    return [x, y]


