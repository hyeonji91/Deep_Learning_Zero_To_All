import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm


def compute_linear_regression(theta, point):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # complete the blanks
    #

    value = theta.np.matmul(point)
    # hint: you may need to use something like np.sum( , axis=1)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    print(value)
    return value