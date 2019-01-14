#!/usr/bin/env python2

"""rvm.py: Relevance Vector Machine (RVM) for regression and classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

# from kernels import linearKernel, linearSplineKernel, polynomialKernel, \
#     RBFKernel, cosineKernel, logKernel
from kernels import get_kernel

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
            sigma=2,
            alpha=10 ** 5,
            beta=3,
            convergenceThresh=10 ** -7,
            alphaThresh=10 ** 8,
            learningRate=0.2,
            maxIter=100,
            verbosity=0,
            useFast = False
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
        self.relevanceTargets = T
        self.N = np.shape(X)[0]

        self.useFast = useFast
        self.beta = beta

        if not self.useFast:
            self.alpha = alpha * np.ones(self.N + 1)
            self.phi = self._initPhi(X)
            self.relevanceVectors = np.copy(X)
        else:
            self.alphaCompare = np.full(self.N + 1, np.inf)
            self.basisVectors = self._initPhi(X)
            if not self.beta:
                self.beta = 1. / np.var(self.T)
            self.alpha = self._computeInitAlpha(0, self.basisVectors.transpose()[0])
            self.relevanceVectors = np.asarray([])

        self.convergenceThresh = convergenceThresh
        self.alphaThresh = alphaThresh
        self.learningRate = learningRate
        self.maxIter = maxIter

        self._setCovAndMu()
        self.keptBasisFuncs = np.arange(self.N + 1)

        self.verbosity = verbosity


    def _get_kernel_function(self):
        """
        Getter function for the kernel method set in the constructor together with the params
        :return: kernel function (function), additional parameters (args)
        """
        return get_kernel(self.kernelName, sigma=self.sigma, p=self.p)

    def rvm_output(self, unseen_x):
        """
        Calculate the output of the rvm for an unseen data point
        :param unseen_x: unseen data value (float)
        :return: the sum over the weighted kernel functions (float)
        """
        kernel, args = self._get_kernel_function()
        kernel_output = [kernel(unseen_x, x_i, args) for x_i in
                         self.relevanceVectors]
        kernel_output = np.asarray(kernel_output)
        if kernel_output.shape[0] + 1 == self.muPosterior.shape[0]:
            kernel_output = np.insert(kernel_output, 0, 1)
        elif kernel_output.shape[0] != self.muPosterior.shape[0]:
            raise RuntimeError

        return kernel_output.dot(self.muPosterior)

    def _setCovAndMu(self):
        """
        Set the covariance and the mean according to the recent alpha and beta values
        """
        self.covPosterior = np.linalg.inv(
            np.diag(self.alpha) + self.beta * np.dot(self.phi.T, self.phi)
        )
        self.muPosterior = self.beta * np.dot(self.covPosterior, self.phi.T).dot(self.T)

    def _computeInitAlpha(self, index, basisVector):
        squaredNorm = np.linalg.norm(basisVector) ** 2
        denominator = (np.linalg.norm(basisVector.transpose().dot(self.T)) ** 2 / squaredNorm) - self.beta**-1

        self.alphaCompare[index] = squaredNorm / denominator
        return np.asarray([self.alphaCompare[index]])

    def _initPhi(self, X):
        """
        Initialize the design matrix based on the specified kernel function.

        Args:
        X (numpy.ndarray): the training data
        kernelName (str): the kernel function

        Returns:
        PHI (numpy.ndarray): the design matrix

        """
        # in the begining these are the same
        # but when we are calculating the predictive posterior, they will not be
        phi = np.ones((self.N ,self.N+1))

        kernel, args = self._get_kernel_function()

        for num, x_n in enumerate(X):
            phi_x_n = [kernel(x_n, x_i, args) for x_i in X]
            phi_x_n = np.insert(phi_x_n, 0, 1)
            phi[num] = phi_x_n

        if self.useFast:
            self.phi = np.ones((self.N, 1))

        return phi

    def _prune(self, alphaOld):
        """
        Prunes alpha such that only relevant weights are kept
        """
        useful = self.alpha < self.alphaThresh
        if useful.all():
            return alphaOld

        if self.alpha.shape[0] == self.relevanceVectors.shape[0] + 1:
            self.relevanceVectors = self.relevanceVectors[useful[1:]]
            self.relevanceTargets = self.relevanceTargets[useful[1:]]
            self.phi = self.phi[:, useful]

        elif self.alpha.shape[0] == self.relevanceVectors.shape[0]:
            self.relevanceVectors = self.relevanceVectors[useful]
            self.relevanceTargets = self.relevanceTargets[useful]
            self.phi = self.phi[:, useful]

        else:
            raise RuntimeError

        self.alpha = self.alpha[useful]
        self.covPosterior = self.covPosterior[np.ix_(useful, useful)]
        self.muPosterior = self.muPosterior[useful]
        self.keptBasisFuncs = self.keptBasisFuncs[useful]

        return alphaOld[useful]

    def _reestimateAlphaBeta(self):
        """
        Re-estimates alpha and beta values according to the paper
        """
        gamma = 1 - self.alpha * np.diag(self.covPosterior)
        self.alpha = gamma / (self.muPosterior ** 2)
        self.beta = (self.N - np.sum(gamma)) / (np.linalg.norm(
            self.T - np.dot(self.phi, self.muPosterior)) ** 2)

    def _reestimateAlphaBetaFastAndPrune(self, index, quality, sparsity):
        # mask = np.indices(self.alphaCompare.shape)[0][self.alphaCompare != np.inf]
        mask = self.alphaCompare != np.inf
        if self.alphaCompare[index] == np.inf:
            s = sparsity
            q = quality
        else:
            s = (self.alphaCompare[index] * sparsity) / (self.alphaCompare[index] - sparsity)
            q = (self.alphaCompare[index] * quality) / (self.alphaCompare[index] - sparsity)

        theta = (q ** 2) - s

        if theta > 0:
            maskLength = mask[mask].shape[0]
            mask[index] = True

            if maskLength != mask[mask].shape[0]:
                self.muPosterior = np.append(self.muPosterior, 1)

            newAlpha = s ** 2 / ((q ** 2) - s)
            self.alphaCompare[index] = newAlpha
            self.alpha = self.alphaCompare[mask]
            self.phi = (self.basisVectors.transpose()[mask]).transpose()
            self.keptBasisFuncs = np.indices((self.basisVectors.transpose().shape[0],)).reshape(-1)[mask]
            if self.alphaCompare[0] != np.inf:
                self.relevanceVectors = self.X[mask[1:]]
                self.relevanceTargets = self.T[mask[1:]]
            else:
                self.relevanceVectors = self.X[mask]
                self.relevanceTargets = self.T[mask]

        else:
            indices = np.indices((mask.shape[0],)).reshape(-1)[mask]
            deleteMuIndex = indices[indices == index]
            self.muPosterior = np.delete(self.muPosterior, deleteMuIndex)

            mask[index] = False

            self.alpha = self.alphaCompare[mask]
            self.phi = (self.basisVectors.transpose()[mask]).transpose()
            self.keptBasisFuncs = np.indices((self.basisVectors.transpose().shape[0],)).reshape(-1)[mask]
            if self.alphaCompare[0] != np.inf:
                self.relevanceVectors = self.X[mask[1:]]
                self.relevanceTargets = self.T[mask[1:]]
            else:
                self.relevanceVectors = self.X[mask]
                self.relevanceTargets = self.T[mask]


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

    def _computeSQ(self, basisVector):
        """
        Dummy for computing the S and the Q value for the fast implementation
        :param basisVector: the basis vector used for the calculation and
        which corresponds to the currently updated alpha
        :return: the Q and S value
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

    def _computeSQ(self, basisVector):
        a = np.linalg.inv(np.diag(self.alpha))
        cMatrix = np.linalg.inv(
            self.beta**-1 * np.eye(basisVector.shape[0]) + self.phi.dot(np.dot(a, self.phi.transpose()))
        )

        sparsity = basisVector.transpose().dot(np.dot(cMatrix, basisVector))
        quality = basisVector.transpose().dot(np.dot(cMatrix, self.T))

        return quality, sparsity

    def _noiseLevelFast(self):
        self.beta = (self.N - self.alpha.shape[0] + self.alpha.dot(np.diag(self.covPosterior))) / \
                    (np.linalg.norm(self.T - self.phi.dot(self.muPosterior))**2 + np.finfo(np.float32).eps)

    def fit(self):
        """
        Fit the training data
        """
        if not self.useFast:
            for i in range(self.maxIter):
                alphaOld = np.array(self.alpha)

                self._reestimateAlphaBeta()
                alphaOld = self._prune(alphaOld)
                self._posterior()
                if np.linalg.norm(self.alpha - alphaOld) <= self.convergenceThresh:
                    break

        else:
            # alphaOld = np.ones(self.N+1)
            counter = 0

            while counter < self.maxIter:
            # while (self.alphaCompare[self.alphaCompare != np.inf].shape[0] != alphaOld[alphaOld != np.inf].shape[0] or
            #                 np.linalg.norm(self.alphaCompare[self.alphaCompare != np.inf]
            #                 - alphaOld[alphaOld != np.inf]) > self.convergenceThresh):
            #     alphaOld = np.copy(self.alphaCompare)


                # TODO needs to bring covPosterior into shape first. Does this destroy the calculation?
                # Amazing results when re estimation of beta is commented out. epsilon?
                self._posterior()
                quality, sparsity = self._computeSQ(self.basisVectors.transpose()[counter % (self.N + 1)])
                self._noiseLevelFast()
                self._reestimateAlphaBetaFastAndPrune(counter % (self.N + 1), quality, sparsity)

                counter += 1

            # update covariance and mu after the last prune
            self._posterior()

    def predict(self, unseen_x):
        """
        Predict the value of a new data point

        Args:
        unseen x (float): unseen data point

        Returns:
        T (numpy.ndarray): predicted target values
        """

        kernel, args = self._get_kernel_function()

        if len(self.X.shape) == 1:
            self.X = self.X.reshape((self.N, 1))

        phi = np.array([[kernel(self.X[i - 1, :], xs, args) if i != 0 else 1 for i in
                          self.keptBasisFuncs] for xs in unseen_x])

        # lmk if there is a better way to fix this...
        # if self.X.shape[1] == 1:
        #     self.X = self.X.reshape(-1)

        variances = 1.0/self.beta + np.diag(phi.dot(self.covPosterior.dot(phi.T)))

        return np.asarray([self.rvm_output(x) for x in unseen_x]), variances


class RVMRS(RVR):

    def predict_rvm(self, unseen_x):
        return super().predict(unseen_x)

    def predict(self, unseen_x):
        p_mu, p_sig2 = super().predict(unseen_x)

        kernel, args = self._get_kernel_function()

        a_s = 1.0 / np.var(self.T)
        # new sample is a row
        for r in range(unseen_x.shape[0]):
            xs = unseen_x[r, :]
            phi_s = np.array([kernel(xs, xi, args) for xi in self.X]).T
            phi_xs = kernel(xs, xs, args)
            phi_x = np.array([kernel(self.X[i - 1, :], xs, args) if i != 0 else 1 for i in
                              self.keptBasisFuncs])

            q = phi_s.T.dot(self.T - self.phi.dot(self.muPosterior)) * self.beta
            e = phi_xs - self.beta * phi_x.dot(
                self.covPosterior.dot(self.phi.T.dot(phi_s)))

            a = (np.diag(1 / self.alpha)).dot(self.phi.T)
            s = phi_s.T.dot(np.linalg.inv(
                1 / self.beta * np.eye(self.N) + self.phi.dot(a))).dot(phi_s)

            p_mu[r] += (e * q) / (a_s + s)
            p_sig2[r] += (e ** 2) / (a_s + s)
        return p_mu, p_sig2


class RVC(RVM):
    """
    Relevance Vector Machine classification
    """

    def _computeSQ(self, basisVector):
        recent_likelihood, sigmoid = self._likelihood()
        a = np.diag(self.alpha)

        betaMatrix = np.diag(recent_likelihood)
        target = self.phi.dot(self.muPosterior) + np.linalg.inv(betaMatrix).dot(self.T - sigmoid)
        woodburyMatrix = np.dot(betaMatrix,
                                np.dot(self.phi,
                                       np.dot(self.covPosterior,
                                              np.dot(self.phi.transpose(),
                                                     betaMatrix
                                                     )
                                              )
                                       )
                                )
        sparsity = basisVector.transpose().dot(np.dot(betaMatrix, basisVector)) - \
                   basisVector.transpose().dot(np.dot(woodburyMatrix, basisVector))
        quality = basisVector.transpose().dot(np.dot(betaMatrix, target)) - \
                  basisVector.transpose().dot(np.dot(woodburyMatrix, target))

        return quality, sparsity

    def irls(self):
        a = np.diag(self.alpha)
        weights_old = np.full(self.muPosterior.shape, np.inf)

        second_derivative = None
        iters = 0
        while iters < self.maxIter and np.linalg.norm(
                        self.muPosterior - weights_old) >= self.convergenceThresh:
            recent_likelihood, sigmoid = self._likelihood()
            recent_likelihood_matrix = np.diag(recent_likelihood)
            second_derivative = -(np.dot(
                self.phi.transpose().dot(recent_likelihood_matrix),
                self.phi) + a)
            first_derivative = self.phi.transpose().dot(
                self.T - sigmoid) - a.dot(self.muPosterior)

            weights_old = np.copy(self.muPosterior)
            self.muPosterior -= self.learningRate * np.linalg.solve(
                second_derivative, first_derivative)
            if self.verbosity > 1:
                print(np.linalg.norm(self.muPosterior - weights_old))

            iters += 1

        if self.verbosity > 0:
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
        sigmoid = np.asarray(
            [1 / (1 + np.exp(-self.rvm_output(x))) for x in self.X])
        beta = np.multiply(sigmoid, np.ones(sigmoid.shape) - sigmoid)

        return beta, sigmoid

    def _posterior(self, weights_new):
        pass

    def _posteriorGradient(self, weights_new):
        pass

    def fit(self):
        if not self.useFast:
            alphaOld = 0 * np.ones(self.N + 1)

            iters = 0
            while np.linalg.norm(self.alpha - alphaOld) >= self.convergenceThresh \
                    and iters < 10:
                alphaOld = np.array(self.alpha)

                self.irls()
                self._reestimateAlphaBeta()
                alphaOld = self._prune(alphaOld)
                iters += 1

        else:
            alphaOld = np.ones(self.N + 1)

            counter = 0
            while (self.alphaCompare[self.alphaCompare != np.inf].shape[0] != alphaOld[alphaOld != np.inf].shape[0] or
                           np.linalg.norm(self.alphaCompare[self.alphaCompare != np.inf]
                                              - alphaOld[alphaOld != np.inf]) > self.convergenceThresh):
                alphaOld = np.copy(self.alphaCompare)
                quality, sparsity = self._computeSQ(self.basisVectors.transpose()[counter % (self.N + 1)])
                self._reestimateAlphaBetaFastAndPrune(counter % (self.N + 1), quality, sparsity)
                self.irls()
                counter += 1

    def predict(self, unseen_x):
        """
        Make predictions for unseen data
        :param unseen_x: unseen data (np.array)
        :return: prediction values and
        """
        return np.asarray(
            [1.0 / (1.0 + math.exp(-self.rvm_output(x))) for x in unseen_x]
        )
