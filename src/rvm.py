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
            alpha=10**-3,
            beta=10**-3,
            convergenceThresh=10**-5,
            alphaThresh=10**9,
            learningRate=0.2
            ):
        """
        RVM parameters initialization

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
        kernel_output = np.insert(kernel_output, 0, 1)

        return np.asarray(kernel_output).dot(self.muPosterior)

    def _setCovAndMu(self):
        self.covPosterior = np.linalg.inv(
            np.diag(self.alpha) + self.beta * np.dot(self.phi.T, self.phi))
        self.muPosterior = self.beta * \
                           np.dot(np.dot(self.covPosterior, self.phi.T), self.T)

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

    def _prune(self):
        """
        Prunes alpha such that only relevant weights are kept
        """
        useful = self.alpha < self.alphaThresh
        self.alpha = self.alpha[useful]

        # Shouldn't it be for both dimensions?
        self.phi = self.phi[:, useful]
        self.covPosterior = self.covPosterior[np.ix_(useful, useful)]
        self.muPosterior = self.muPosterior[useful]

        if np.size(useful) != np.size(self.relevanceVectors):
            self.relevanceVectors = self.relevanceVectors[useful[1:]]
        else:
            self.relevanceVectors = self.relevanceVectors[useful]

    def _reestimatingAlphaBeta(self):
        """
        Re estimates alpha and beta values according to the paper
        """
        gamma = 1 - self.alpha * np.diag(self.covPosterior)
        self.alpha = gamma / self.muPosterior ** 2

        self.beta = (self.N - np.sum(gamma)) / \
                    np.linalg.norm(self.T - np.dot(self.phi, self.muPosterior)) ** 2

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
        alphaOld = 0 * np.ones(self.N+1)

        while abs(sum(self.alpha) - sum(alphaOld)) >= self.convergenceThresh:
            alphaOld = np.array(self.alpha)

            self._reestimatingAlphaBeta()
            self._prune()
            self._posterior()

    def predict(self, unseen_x):
        """
        Predict the value of a new data point

        Args:
        unseen x (float): unseen data point

        Returns:
        T (numpy.ndarray): predicted target values

        """
        return self.rvm_output(unseen_x)



class RVC(RVM):
    """
    Relevance Vector Machine classification
    """

    def irls(self):
        a = np.diag(self.alpha)
        weights_old = np.full(self.muPosterior.shape, np.inf)

        self.muPosterior = np.random.randn(self.muPosterior.shape[0])
        second_derivative = None
        iters = 0
        while iters < 100 and np.all(np.absolute(self.muPosterior - weights_old) >= self.convergenceThresh):
            recent_likelihood, sigmoid = self._likelihood()
            recent_likelihood_matrix = np.diag(recent_likelihood)
            second_derivative = -(np.dot(self.phi.transpose().dot(recent_likelihood_matrix), self.phi) + a)
            first_derivative = self.phi.transpose().dot(self.T - sigmoid) - a.dot(self.muPosterior)

            weights_old = self.muPosterior

            self.muPosterior -= self.learningRate * np.linalg.solve(second_derivative, first_derivative)
            iters += 1
        print('Iterations used: ', iters)
        self.covPosterior = np.linalg.inv(-second_derivative)


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

    def _posterior(self, weights_new):
        weights_save = self.muPosterior
        self.muPosterior = weights_new
        all_outputs, _ = self._likelihood()

        posterior = 0.0
        for i in range(len(all_outputs)):
            posterior += self.T[i] * np.log(all_outputs[i])
            posterior += (1-self.T[i]) * np.log(1 - all_outputs[i])
        posterior -= 0.5 * self.muPosterior.T @ np.diag(self.alpha) @ weights_new

        self.muPosterior = weights_save
        return posterior

    def _posteriorGradient(self, weights_new):
        weights_save = self.muPosterior
        self.muPosterior = weights_new
        all_outputs, _ = self._likelihood()
        ret = self.phi.T @ (self.T - all_outputs) - np.diag(self.alpha) @ weights_new
        self.muPosterior = weights_save

        return ret


    def fit(self):
        alphaOld = 0 * np.ones(self.N + 1)

        while abs(sum(self.alpha) - sum(alphaOld)) >= self.convergenceThresh:
            alphaOld = np.array(self.alpha)


            #self.irls()
            optRes = minimize(self._posterior, np.random.randn(self.muPosterior.shape[0]), jac=self._posteriorGradient)
            self.muPosterior = optRes.x
            recent_likelihood, sigmoid = self._likelihood()
            recent_likelihood_matrix = np.diag(recent_likelihood)
            second_derivative = -(np.dot(
                self.phi.transpose().dot(recent_likelihood_matrix),
                self.phi) + np.diag(self.alpha))

            self.covPosterior = np.linalg.inv(-second_derivative)

            self._reestimatingAlphaBeta()
            # self._prune()

    def predict(self, X):
        """Make class predictions"""
        pass

