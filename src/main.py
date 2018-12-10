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

#### DATA
# Generating data based on 100 uniformly-spaced noise-free samples in [-10, 10]
# from the sinc function: sinc(x) = |x|^-1 sin(x)
N = 3
X, T = sincNoiseFree(N)

#plt.scatter(X,T)
#plt.title("The sinc function without noise")
#plt.xlabel("x")
#plt.ylabel("t")
#plt.show()


#### Relevance Vector Regression

# Design matrix PHI (N x (N+1))
# PHI = [phi(x_1), phi(x_2), ..., phi(x_n)]
# phi(x_n) = [1, K(x_n, x_1), K(x_n, x_2), ..., K(x_n, x_n)].T


designM = np.ones(N * (N+1)).reshape((N, N+1))
for num, x_n in enumerate(X):
    phi_x_n = [linearKernel(x_n, x_i) for x_i in X]
    phi_x_n = np.insert(phi_x_n, 0, 1)

    designM[num] = phi_x_n



# Prior distribution over w: p(w | alpha)

# Hyper parameters
a = b = c = d = 10**-4

# Prior over the hyperparameter alpha: p(alpha)
# p(alpha) = prod_{i=0}^{N} Gamma(alpa_i | a, b)

# Prior over the hyperparameter beta: p(beta)
# p(beta) = prod_{i=0}^{N} Gamma(beta_i | c, d)

alpha = 1**-6 * np.ones(N+1)
beta = 1**-6 # beta = sigma^{-2}



### POSTERIOR
# For p(w | t, alpha, sigma^2)
covPosterior = np.linalg.inv(np.dot(np.dot(designM.T, (beta * np.eye(N))), designM) + np.diag(alpha))

muPosterior = np.dot(np.dot(np.dot(covPosterior, designM.T), (beta * np.eye(N))), T)

### UPDATING THE HYPER PARAMETERS



# Generating data based on 100 uniformly-spaced samples in [-10, 10] from the
# sinc function: sinc(x) = |x|^-1 sin(x) with added Gaussisan noise of
# sigma = 0.2
#X, T = sincGaussianNoise(100, 0.2)

#plt.scatter(X,T)
#plt.title("The sinc function with added Gaussian noise")
#plt.xlabel("x")
#plt.ylabel("t")
#plt.show()
