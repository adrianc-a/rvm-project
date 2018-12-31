#!/usr/bin/env python3

"""rvm.py: Relevance Vector Machine (RVM) for regression and classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"


from kernels import linearKernel, polynomialKernel, RBFKernel
from scipy.optimize import minimize
from scipy.special import expit

import math
import numpy as np


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
            # Big alpha since we want to cover a lot of weight values
            # See evidence part of last assignment
            alpha=1000,
            beta=3,
            convergenceThresh=10**-9,
            alphaThresh=10**9,
            learningRate=0.2
            ):
        """
        RVM parameters initialization

        Args:
        kernelName (string): the kernel function
        p (int): polynomial kernel function parameter
        sigma (int): RBF kernel function parameter
        alpha (float): initial alpha value
        beta (float): initial beta value
        convergenceThresh (float): threshold for convergence in fit() function
        alphaThresh (float): threshold for pruning alpha values
        maxIter (int): maximum number of iterations for the posterior mode finder
                       in RVM classification
        learningRate (float): The learning rate for the Newton/IRLS update step

        """

        self.kernelName = kernelName
        self.p = p
        self.sigma = sigma

        self.X = X
        self.T = T
        self.N = np.shape(X)[0]
        self.phi = self._initPhi(X)
        self.relevanceVectors = X
        self.relevanceVectorsT = T

        self.beta = beta
        self.convergenceThresh = convergenceThresh
        self.alphaThresh = alphaThresh
        self.learningRate = learningRate

        self.alpha = alpha * np.ones(self.N + 1)

        self._setCovAndMu()

    def _get_kernel_function(self):
        """
        Getter function for the kernel method set in the constructor together with the params
        :return: kernel function (function), additional parameters (args)
        """
        if self.kernelName == 'linearKernel':
            return linearKernel, None

        elif self.kernelName == 'RBFKernel':
            return RBFKernel, self.sigma

        elif self.kernelName == 'polynomialKernel':
            return polynomialKernel, self.p

    def rvm_output(self, unseen_x):
        """
        Calculate the output of the rvm for an unseen data point
        :param unseen_x: unseen data value (float)
        :return: the sum over the weighted kernel functions (float)
        """
        kernel, args = self._get_kernel_function()
        kernel_output = [kernel(unseen_x, x_i, args) for x_i in self.relevanceVectors]
        # if bias was not pruned
        if self.muPosterior.shape[0] == np.asarray(kernel_output)[0] + 1:
            kernel_output = np.insert(kernel_output, 0, 1)

        return np.asarray(kernel_output).dot(self.muPosterior)

    def _setCovAndMu(self):
        """
        Set the covariance and the mean according to the recent alpha and beta values
        """
        self.covPosterior = np.linalg.inv(
            np.diag(self.alpha) + self.beta * np.dot(self.phi.T, self.phi))
        subresult = np.dot(self.covPosterior, self.phi.T)
        subresult = subresult.dot(self.T)
        self.muPosterior = self.beta * subresult


    def _initPhi(self, X):
        """
        Initialize the design matrix based on the specified kernel function.

        Args:
        X (numpy.ndarray): the training data
        kernelName (str): the kernel function

        Returns:
        PHI (numpy.ndarray): the design matrix

        """
        PHI = np.ones(self.N * (self.N+1)).reshape((self.N, self.N+1))

        kernel, args = self._get_kernel_function()

        for num, x_n in enumerate(X):
            phi_x_n = [kernel(x_n, x_i, args) for x_i in X]
            phi_x_n = np.insert(phi_x_n, 0, 1)
            PHI[num] = phi_x_n

        return PHI

    def _prune(self, itCount):
        """
        Prunes alpha such that only relevant weights are kept
        """
        useful = self.alpha < self.alphaThresh
        if useful.all():
            return
        if (self.alpha < np.zeros(len(self.alpha))).any():
            print("oh no alpha contains something negative")
            print("iteration: ", itCount)
            print(self.alpha)
        # print(self.alpha)
        if self.alpha.shape[0] == self.relevanceVectors.shape[0] + 1:
            self.relevanceVectors = self.relevanceVectors[useful[1:]]
            self.relevanceVectorsT = self.relevanceVectorsT[useful[1:]]
            self.T = self.T[useful[1:]]
            self.phi = self.phi[:, useful]
            self.phi = self.phi[useful[1:], :]

        elif self.alpha.shape[0] == self.relevanceVectors.shape[0]:
            self.relevanceVectors = self.relevanceVectors[useful]
            self.relevanceVectorsT = self.relevanceVectorsT[useful]
            self.T = self.T[useful]
            self.phi = self.phi[:, useful]
            self.phi = self.phi[useful, :]

        else:
            raise RuntimeError

        self.alpha = self.alpha[useful]
        self.covPosterior = self.covPosterior[np.ix_(useful, useful)]
        self.muPosterior = self.muPosterior[useful]

    def _reestimatingAlphaBeta(self):
        """
        Re estimates alpha and beta values according to the paper
        """
        gamma = 1 - self.alpha * np.diag(self.covPosterior)
        self.alpha = gamma / (self.muPosterior ** 2)

        self.beta = (self.N - np.sum(gamma)) / \
                    (np.linalg.norm(self.T - np.dot(self.phi, self.muPosterior)) ** 2)

    def fit(self):
        """
        Dummy for the learning method
        """
        pass

    def predict(self, unseen_x):
        """
        Dummy for the predicting method
        """
        pass


class RVR(RVM):
    """
    Relevance Vector Machine regression
    """

    def _posterior(self):
        """
        Compute the posterior distribution over the weights
        """
        self._setCovAndMu()

    def fit(self):
        """
        Fit the training data
        """
        itCount = 0
        alphaOld = np.zeros(self.N+1)
        while abs(sum(self.alpha) - sum(alphaOld)) >= self.convergenceThresh:
            alphaOld = np.array(self.alpha)

            self._reestimatingAlphaBeta()
            self._prune(itCount)
            self._posterior()
            itCount = itCount + 1

    def predict(self, unseen_x):
        """
        Predict the value of a new data point

        Args:
        unseen x (float): unseen data point

        Returns:
        T (numpy.ndarray): predicted target values

        """
        return np.asarray([self.rvm_output(x) for x in unseen_x])



class RVC(RVM):
    """
    Relevance Vector Machine classification
    """

    def irls(self):
        pass

    def _likelihood(self):
        pass

    def _posterior(self, weights_new):
        pass

    def _posteriorGradient(self, weights_new):
        pass

    def fit(self):
        pass

    def predict(self, unseen_x):
        pass
