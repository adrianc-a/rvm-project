#!/usr/bin/env python3

"""rvm.py: Relevance Vector Machine (RVM) for regression and classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"


from kernels import linearKernel, polynomialKernel, RBFKernel
from data import sincNoiseFree, sincGaussianNoise

import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import time

# Set a random seed
np.random.seed(0)


class RVM():
    """Relevance Vector Machine class implementation based on Mike Tipping's
    The Relevance Vector Machine (2000)

    """

    def __init__(
            self,
            kernelName='RBFKernel',
            p=3,
            sigma=5,
            alpha=10**-6,
            beta=10**-6,
            convergenceThresh=10**-3,
            alphaThresh=10**9,
            maxIter = 25
            ):
        """RVM parameters initialization

        Args:
        kernelName (string): the kernel function
        p (int): polynomial kernel function parameter
        sigma (int): RBF kernel function parameter
        alpha (float): inital alpha value
        beta (float): inital beta value
        convergenceThresh (float): threshold for convergence in fit() function
        alphaThresh (float): threshold for pruning alpha values
        maxIter (int): maximum number of iterations for the posterior mode finder
                       in RVM classification

        """
        self.kernelName = kernelName
        self.p = p
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.convergenceThresh = convergenceThresh
        self.alphaThresh = alphaThresh


    def _initPhi(self, X, kernelName):
        """Initialize the design matrix based on the specified kernel function.

        Args:
        X (numpy.ndarray): the training data
        kernelName (str): the kernel function

        Returns:
        PHI (numpy.ndarray): the design matrix

        """
        N = np.shape(X)[0]
        PHI = np.ones(N * (N+1)).reshape((N, N+1))

        if self.kernelName == 'linearKernel':
            for num, x_n in enumerate(X):
                phi_x_n = [linearKernel(x_n, x_i) for x_i in X]
                phi_x_n = np.insert(phi_x_n, 0, 1)
                PHI[num] = phi_x_n

        if self.kernelName == 'RBFKernel':
            for num, x_n in enumerate(X):
                phi_x_n = [RBFKernel(x_n, x_i, self.sigma) for x_i in X]
                phi_x_n = np.insert(phi_x_n, 0, 1)
                PHI[num] = phi_x_n

        if self.kernelName == 'polynomialKernel':
            for num, x_n in enumerate(X):
                phi_x_n = [polynomialKernel(x_n, x_i, self.p) for x_i in X]
                phi_x_n = np.insert(phi_x_n, 0, 1)
                PHI[num] = phi_x_n

        return PHI


    def _prune(self):
        """Prunes alpha such that only relevant weights are kept"""
        useful = self.alpha < self.alphaThresh
        self.alpha = self.alpha[useful]
        self.phi = self.phi[:, useful]
        self.alphaOld = self.alphaOld[useful]
        self.gamma = self.gamma[useful]
        self.covPosterior = self.covPosterior[np.ix_(useful, useful)]
        self.muPosterior = self.muPosterior[useful]
        self.relevanceVectors = self.relevanceVectors[useful]


    def fit(self, X, T, kernelName):
        """Fit the training data

        Args:
        X (numpy.ndarray): the training data
        T (numpy.ndarray): the targets
        kernelName (str): the kernel function

        Returns:
        self (object): the RVM model with all its properties

        """

        N = np.shape(X)[0]
        self.phi = self._initPhi(X, kernelName)
        self.relevanceVectors = np.append(1, X)
        self.T = T

        self.alpha = self.alpha * np.ones(N+1)

        self.covPosterior = np.linalg.inv(
                np.diag(self.alpha) + self.beta * np.dot(self.phi.T, self.phi))
        self.muPosterior = self.beta * \
                np.dot(np.dot(self.covPosterior, self.phi.T), self.T)

        self.alphaOld = 0 * np.ones(N+1)
        self.alpha = -1 * np.ones(N+1)

        while abs(sum(self.alpha) - sum(self.alphaOld)) >= self.convergenceThresh:
            self.alphaOld = np.array(self.alpha)

            self.gamma = 1 - self.alpha * np.diag(self.covPosterior)
            self.alpha = self.gamma / self.muPosterior**2

            self.beta = (N - np.sum(self.gamma)) / \
                    np.linalg.norm(T - np.dot(self.phi, self.muPosterior))**2


            self._prune()

            self._posterior()

        return self


class RVR(RVM):
    """Relevance Vector Machine regression"""
    def _posterior(self):
        """Compute the posterior distribution over the weights"""
        self.covPosterior = np.linalg.inv(
                np.diag(self.alpha) + self.beta * np.dot(self.phi.T, self.phi))
        self.muPosterior = self.beta * \
                np.dot(np.dot(self.covPosterior, self.phi.T), self.T)


    def predict(self, X):
        """Predict the value of a new data point

        Args:
        X (numpy.ndarray): training data

        Returns:
        T (numpy.ndarray): predicted target values

        """
        phi = self.initPhi(X)
        T = np.dot(phi, self.muPosterior)

        return T


class RVC(RVM):
    """Relevance Vector Machine classification"""

    def _classify(self, weights, PHI):
        """Classify a data point

        Args:
        weights (numpy.ndarray): weights corresponding to data point to be
                                      classified
        PHI (numpy.ndarray): design matrix

        Returns:
        sigmoid (float): the sigmoid of the dot product of the weights with the
                         design matrix

        """

        dot = np.dot(weights.T, PHI)
        sigmoid = 1 / (1 + math.exp(-dot))

        return sigmoid


    def _negativeLogPosterior(self, weights, PHI, alpha):
        """Compute the negative log posterior

        Args:
        weights (numpy.ndarray):
        PHI (numpy.ndarray): design matrix
        alpha (numpy.ndarray):

        Returns:
        negativeLogPosterior (float): the negative log posterior
        TODO Jacobian?

        """

        y = self._classify(weights, PHI)

        TODO
        termA = 0

        A = np.diag(alpha)
        termB = 0.5 * np.dot(weights.T, np.dot(A, weights))


        negLogPost = termA + termB

        return negLogPost#, jacobian?


    def _hessian(self, weights, alpha, PHI):
        """Compute the Hessian matrix of the negative log posterior

        Args:
        weights (numpy.ndarray):
        PHI (numpy.ndarray): design matrix
        alpha (numpy.ndarray):

        Returns:
        hessian (numpy.ndarray): the Hessian matrix of the negative log posterior

        """

        y = self._classify(weights, PHI)

        A = np.diag(alpha)
        B = np.diag(y * (1-y))

        hessian = np.dot(PHI.T, np.dot(B, PHI)) + A

        return hessian


    def _findMode(self):
        mode = minimize(
                fun=self._negativeLogPosterior,
                hess=self._hessian,
                x0=self.weights,
                args=(self.alpha, self.phi, self.c), # self.c needs to be defined somewhere
                method='Newton-CG',
                jac=True,
                options={'maxiter': self.maxIter}
                )

        self.muPosterior = mode.x
        self.covPosterior = np.linalg.inv(
                self._hessian(self.muPosterior, self.alpha, self.phi, self.c)
                )


    def fit(self, X, T):
        """Fit the model"""

        return 0


    def predict(self, X):
        """Make class predictions"""

