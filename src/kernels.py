#!/usr/local/bin/python

"""kernels.py: definition of the kernel functions used for the Relevance Vector
Machine (RVM) for regression and classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

import numpy as np
from scipy.spatial import distance
import math


def linearKernel(x, y, *args):
    """Kernel for linear separation

    Args:
    x (numpy.ndarray): a vector of datapoints
    y (numpy.ndarray): a vector of datapoints
    *args (none)

    """
    return np.dot(x.T, y)


def polynomialKernel(x, y, *args):
    """Polynomial kernel for curved decision boundaries

    Args:
    x (numpy.ndarray): a vector of datapoints
    y (numpy.ndarray): a vector of datapoints
    *args (int): the degree of the polynomial

    """
    return math.pow((np.dot(x.T, y) + 1), args[0])


def RBFKernel(x, y, *args):
    """RBF kernel that uses explicit euclidian distance between x and y,

    Args:
    x (numpy.ndarray): a vector of datapoints
    y (numpy.ndarray): a vector of datapoints
    *args (float): sigma; controls the smoothness of the boundary

    """
    num = distance.euclidean(x, y)
    denom = 2 * args[0] ** 2

    return math.exp(- num / denom)

