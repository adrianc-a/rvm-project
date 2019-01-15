#!/usr/bin/env python2

"""main-sinc-noisy.py: Test RVR on the noisy sinc data set."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import sinc, initData
from rvm import RVR

import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
np.random.seed(0)


def main():
    N = 100
    noiseVariance = .03
    dataFunction = sinc

    X_train, X_test, T_train, T_test = initData(N, sinc, noiseVariance)

    clf = RVR(X_train, T_train, 'RBFKernel')
    clf.fit()

    print("The relevance vectors:")
    print(clf.relevanceVectors)

    T_pred, _ = clf.predict(X_test)

    # Plot training data
    X = np.linspace(-10, 10, 250)
    plt.plot(X, np.sin(X)/X, label='orig func')
    plt.scatter(X_train, T_train, label='Training noisy samples')

    # Plot predictions
    predictedMu, _ = clf.predict(X)
    plt.plot(X, predictedMu, label='Prediction {\mu}')

    # Plot relevance vectors
    plt.scatter(clf.relevanceVectors,
                clf.relevanceTargets,
                label="Relevance vectors",
                s=50,
                facecolors="none",
                color="k",
                zorder=1)

    plt.ylim(-0.3, 1.1)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    # plt.savefig("../plots/sincdataplot.png", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()

