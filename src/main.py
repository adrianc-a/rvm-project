#!/usr/bin/env python3

"""rvm.py: Relevance Vector Machine (RVM) for regression and classification."""

__author__      = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
                   and Leo Zeitler"

from data import sincNoiseFree, sincGaussianNoise
from kernels import linearKernel, polynomialKernel, RBFKernel

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.optimize import minimize


# Set a random seed
np.random.seed(0)

### Relevance Vector Regression


# Generating data based on 100 uniformly-spaced noise-free samples in [-10, 10]
# from the sinc function: sinc(x) = |x|^-1 sin(x)
X, T = sincNoiseFree(100)

plt.scatter(X,T)
plt.title("The sinc function without noise")
plt.xlabel("x")
plt.ylabel("t")
plt.show()

# Generating data based on 100 uniformly-spaced samples in [-10, 10] from the
# sinc function: sinc(x) = |x|^-1 sin(x) with added Gaussisan noise of
# sigma = 0.2
X, T = sincGaussianNoise(100, 0.2)

plt.scatter(X,T)
plt.title("The sinc function with added Gaussian noise")
plt.xlabel("x")
plt.ylabel("t")
plt.show()
