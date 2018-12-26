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
EPS = 10 **-6

class RVM:
    """Relevance Vector Machine class implementation based on Mike Tipping's
    The Relevance Vector Machine (2000)

    """

    def __init__(
            self,
            X,
            T,
            kernelName,
            p=3,
            sigma=5,
            alpha=10**-6,
            beta=10**-6,
            convergenceThresh=10**-3,
            alphaThresh=10**9,
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

        self.X = X
        self.T = T
        self.N = np.shape(X)[0]
        self.phi = self._initPhi(X)
        # Why assigning to the relevance vectors additionally 1?
        # self.relevanceVectors = np.append(1, X)
        self.relevanceVectors = X

        self.beta = beta
        self.convergenceThresh = convergenceThresh
        self.alphaThresh = alphaThresh

        self.alpha = alpha * np.ones(self.N + 1)

        self.covPosterior = np.linalg.inv(
            np.diag(self.alpha) + self.beta * np.dot(self.phi.T, self.phi))
        self.muPosterior = self.beta * \
                           np.dot(np.dot(self.covPosterior, self.phi.T), self.T)

        # Why re assigning alpha?
        self.alpha = -1 * np.ones(self.N + 1)

    def _get_kernel_function(self):
        if self.kernelName == 'linearKernel':
            return linearKernel, None

        elif self.kernelName == 'RBFKernel':
            return RBFKernel, self.sigma

        elif self.kernelName == 'polynomialKernel':
            return polynomialKernel, self.p

    def rvm_output(self, unseen_x):
        kernel, args = self._get_kernel_function()
        kernel_output = [kernel(unseen_x, x_i, args) for x_i in self.relevanceVectors]
        kernel_output = np.insert(kernel_output, 0, 1)

        return np.asarray(kernel_output).dot(self.muPosterior)

    def _initPhi(self, X):
        """Initialize the design matrix based on the specified kernel function.

        Args:
        X (numpy.ndarray): the training data
        kernelName (str): the kernel function

        Returns:
        PHI (numpy.ndarray): the design matrix

        """
        N = np.shape(X)[0]
        PHI = np.ones(N * (N+1)).reshape((N, N+1))

        kernel, args = self._get_kernel_function()

        for num, x_n in enumerate(X):
            phi_x_n = [kernel(x_n, x_i, args) for x_i in X]
            phi_x_n = np.insert(phi_x_n, 0, 1)
            PHI[num] = phi_x_n

        return PHI

    def _posterior(self):
        pass

    def _prune(self):
        """Prunes alpha such that only relevant weights are kept"""
        useful = self.alpha < self.alphaThresh
        self.alpha = self.alpha[useful]

        # Shouldn't it be for both dimensions?
        self.phi = self.phi[:, useful]
        self.covPosterior = self.covPosterior[np.ix_(useful, useful)]
        self.muPosterior = self.muPosterior[useful]
        self.relevanceVectors = self.relevanceVectors[useful]

    def fit(self):
        """Fit the training data

        Args:
        T (numpy.ndarray): the targets
        kernelName (str): the kernel function

        Returns:
        self (object): the RVM model with all its properties

        """
        alphaOld = 0 * np.ones(self.N+1)

        while abs(sum(self.alpha) - sum(alphaOld)) >= self.convergenceThresh:
            alphaOld = np.array(self.alpha)

            gamma = 1 - self.alpha * np.diag(self.covPosterior)
            self.alpha = gamma / self.muPosterior**2

            self.beta = (self.N - np.sum(gamma)) / \
                    np.linalg.norm(self.T - np.dot(self.phi, self.muPosterior))**2

            self._prune()

            self._posterior()


class RVR(RVM):
    """Relevance Vector Machine regression"""
    def _posterior(self):
        """Compute the posterior distribution over the weights"""
        self.covPosterior = np.linalg.inv(
                np.diag(self.alpha) + self.beta * np.dot(self.phi.T, self.phi))
        self.muPosterior = self.beta * \
                np.dot(np.dot(self.covPosterior, self.phi.T), self.T)

    def predict(self, unseen_x):
        """Predict the value of a new data point

        Args:
        unseen x (float): unseen data point

        Returns:
        T (numpy.ndarray): predicted target values

        """
        return self.rvm_output(unseen_x)


class RVC(RVM):
    """Relevance Vector Machine classification"""

    def irls(self):
        a = np.diag(self.alpha)
        weight_old = self.muPosterior.copy()
        weights_new = np.full(self.muPosterior.shape, np.inf)

        counter =0
        while np.absolute(weights_new - weight_old).all() >= EPS:
            weight_old = weights_new.copy()
            recent_likelihood, sigmoid = self._likelihood()
            recent_likelihood_matrix = np.diag(recent_likelihood)
            second_derivative = -(np.dot(self.phi.transpose().dot(recent_likelihood_matrix), self.phi) + a)
            first_derivative = self.phi.transpose().dot(self.T - sigmoid) - a.dot(self.muPosterior)

            weights_new = self.muPosterior - np.linalg.inv(second_derivative).dot(first_derivative)
            self.covPosterior = np.linalg.inv(-second_derivative)
            self.muPosterior = weights_new.copy()

            counter+=1
            print counter

    def _likelihood(self):
        """
        Classify a data point

        Args:
        X (numpy.ndarray): datapoints

        Returns:
        sigmoid (numpy.ndarray): the sigmoid of the dot product of the weights with the
                         design matrix

        """

        sigmoid = np.asarray([ 1 / (1 + math.exp(-self.rvm_output(x))) for x in self.X])
        beta = np.multiply(sigmoid, np.ones(sigmoid.shape) - sigmoid)
        return beta, sigmoid

    # def _negativeLogPosterior(self, weights, PHI, alpha):
    #     """Compute the negative log posterior
    #
    #     Args:
    #     weights (numpy.ndarray):
    #     PHI (numpy.ndarray): design matrix
    #     alpha (numpy.ndarray):
    #
    #     Returns:
    #     negativeLogPosterior (float): the negative log posterior
    #     TODO Jacobian?
    #
    #     """
    #
    #     y = self._classify(weights, PHI)
    #
    #     termA = 0
    #
    #     A = np.diag(alpha)
    #     termB = 0.5 * np.dot(weights.T, np.dot(A, weights))
    #
    #
    #     negLogPost = termA + termB
    #
    #     return negLogPost#, jacobian?
    #
    # def _hessian(self, weights, alpha, PHI):
    #     """Compute the Hessian matrix of the negative log posterior
    #
    #     Args:
    #     weights (numpy.ndarray):
    #     PHI (numpy.ndarray): design matrix
    #     alpha (numpy.ndarray):
    #
    #     Returns:
    #     hessian (numpy.ndarray): the Hessian matrix of the negative log posterior
    #
    #     """
    #
    #     y = self._classify(weights, PHI)
    #
    #     A = np.diag(alpha)
    #     B = np.diag(y * (1-y))
    #
    #     hessian = np.dot(PHI.T, np.dot(B, PHI)) + A
    #
    #     return hessian
    #
    # def _findMode(self):
    #     mode = minimize(
    #             fun=self._negativeLogPosterior,
    #             hess=self._hessian,
    #             x0=self.weights,
    #             args=(self.alpha, self.phi, self.c), # self.c needs to be defined somewhere
    #             method='Newton-CG',
    #             jac=True,
    #             options={'maxiter': self.maxIter}
    #             )
    #
    #     self.muPosterior = mode.x
    #     self.covPosterior = np.linalg.inv(
    #             self._hessian(self.muPosterior, self.alpha, self.phi, self.c)
    #             )

    def fit(self):
        alphaOld = 0 * np.ones(self.N + 1)

        while abs(sum(self.alpha) - sum(alphaOld)) >= self.convergenceThresh:
            alphaOld = np.array(self.alpha)

            self.irls()

            gamma = 1 - self.alpha * np.diag(self.covPosterior)
            self.alpha = gamma / self.muPosterior ** 2

            self.beta = (self.N - np.sum(gamma)) / \
                        np.linalg.norm(self.T - np.dot(self.phi, self.muPosterior)) ** 2

            # self._prune()

            # self._posterior()

    def predict(self, X):
        """Make class predictions"""
        pass

