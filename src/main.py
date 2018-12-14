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
N = 300
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


# Starting values for alpha and beta
alpha = 10**-6 * np.ones(N+1)
beta = 10**-6 # beta = sigma^{-2}


# Thresholds
# For estimation of alpha
convergenceThresh = 10**-3
# For pruning alphaI: i.e. threshold to determine that alphaI is going to inf
alphaThresh = 10**9

# Calculating the mean and covariance matrix of
# the posterior p(w | t, alpha, sigma^2)
covPosterior = np.linalg.inv(np.diag(alpha) + beta * np.dot(designM.T, designM))

muPosterior = beta * np.dot(np.dot(covPosterior, designM.T), T)

# To not divide by zero
eps = 10**-30


j = 0
alphaOld = -1 * np.ones(N+1)
while abs(sum(alpha) - sum(alphaOld)) >= convergenceThresh:
    #print(j)
    # Updating alpha_i = gamma_i / m_i^2
    # where gamma_i = 1 - alpha_i * Sigma_ii
    alphaOld = alpha

    gamma = []
    for i, alphaI in enumerate(alpha):
        gammaI = (1 - alphaI * covPosterior[i][i])
        alpha[i] = gammaI / (muPosterior[i]**2 + eps)
        gamma.append(gammaI)


    # Updating beta = (N - sum_from_i=1_to_M gamma_i) / ||t - designM * muPosterior||^2
    beta = (N - np.sum(gamma)) / np.linalg.norm(T - np.dot(designM, muPosterior))**2
    #print(beta)


    # Prune the design matrix
    keepAlpha = alpha < alphaThresh
    alpha = alpha[keepAlpha]
    designM = designM[:, keepAlpha]

    # Use eps to avoid a singular matrix
    covPosterior = np.linalg.inv(np.diag(alpha) + beta * np.dot(designM.T, designM) + eps)
    muPosterior = beta * np.dot(np.dot(covPosterior, designM.T), T)

    #print(alpha)
    #print(alphaOld)
    j += 1


### DATA
# Generating data based on 100 uniformly-spaced samples in [-10, 10] from the
# sinc function: sinc(x) = |x|^-1 sin(x) with added Gaussisan noise of
# sigma = 0.2
#X, T = sincGaussianNoise(100, 0.2)

#plt.scatter(X,T)
#plt.title("The sinc function with added Gaussian noise")
#plt.xlabel("x")
#plt.ylabel("t")
#plt.show()
