#!/usr/bin/env python2

"""main-sinc.py: Test RVR for noise-less sinc function."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import sinc, initData
from rvm import RVR, RVC

import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
np.random.seed(0)


def main():
    N = 100
    noiseSpread = 0.001

    X_train, X_test, T_train, T_test = initData(N, sinc, noiseSpread)

    clf = RVR(X_train, T_train, 'linearSplineKernel')
    clf.fit()

    print("The relevance vectors:")
    print(clf.relevanceVectors)
    print("Beta:", np.sqrt(clf.beta**-1))

    T_pred, _ = clf.predict(X_test)

    # Plot training data
    plt.scatter(X_train, T_train, s=20, label='Training data')

    # Plot predictions
    plt.scatter(X_test, T_pred, s=20, color='r', label='Predictions')

    # Plot relevance vectors
    plt.scatter(clf.relevanceVectors,
                clf.relevanceTargets,
                label="Relevance vectors",
                s=50,
                facecolors="none",
                color="k",
                zorder=1)

    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    # plt.savefig("../plots/sincdataplot.png", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
