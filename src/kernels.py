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
    x (float): a datapoint
    y (float): a datapoint
    *args (none)

    """
    return np.dot(x, y)


def linearSplineKernel(x, y, *args):
    """Univariate linear spline kernel

    Args:
    x (float): a datapoint
    y (float): a datapoint
    *args (none)

    """
    return 1 \
           + x * y \
           + x * y * min(x, y) \
           - (x + y) / 2 * min(x, y)**2 \
           + min(x, y)**3 / 3


def polynomialKernel(x, y, *args):
    """Polynomial kernel for curved decision boundaries

    Args:
    x (float): a datapoint
    y (float): a datapoint
    *args (int): the degree of the polynomial

    """
    return math.pow(np.dot(x,y) + 1, args[0])


def RBFKernel(x, y, *args):
    """RBF kernel that uses explicit euclidian distance between x and y,

    Args:
    x (float): a datapoint
    y (float): a datapoint
    *args (float): sigma; controls the smoothness of the boundary

    """
    num = distance.euclidean(x, y)
    denom = 2 * args[0] ** 2

    return math.exp(- num / denom)


def cosineKernel(x, y, *args):
    return (np.pi / 4.0) * np.cos(np.pi * 0.5 * np.linalg.norm(x - y))
