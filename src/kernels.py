#!/usr/local/bin/python

import numpy as np
from scipy.spatial import distance
import math


def linearKernel(x, y, *args):
    """For linear separation."""
    return np.dot(x.T, y)


def polynomialKernel(x, y, p):
    """For curved decision boundaries, where p controls polynomials degree"""
    return math.pow((np.dot(x.T, y) + 1), p)


def RBFKernel(x, y, sig):
    """Uses explicit euclidian distance between x and y.
    sig controls the smoothness of the boundary."""
    num = distance.euclidean(x, y)
    denom = 2 * sig ** 2

    return math.exp(- num / denom)

