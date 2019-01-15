#!/usr/bin/env python2

"""main-airfoil.py: File for testing RVR on the airfoil data set"""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import airfoil, initData
from rvm import RVR

import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
np.random.seed(0)


def main():
    N = None
    dataFunction = airfoil

    X_train, X_test, T_train, T_test = initData(N, dataFunction)
    clf = RVR(X_train, T_train, 'RBFKernel')
    clf.fit()

    print("Number of relevance vectors:")
    print(len(clf.relevanceVectors))

    T_pred, _ = clf.predict(X_test)
    relError = np.mean([abs(true - pred) / true for true, pred in zip(T_test, T_pred)])
    absError = np.mean([abs(true - pred) for true, pred in zip(T_test, T_pred)])
    stdRelError = np.std([abs(true - pred) / true for true, pred in zip(T_test, T_pred)])
    stdAbsError = np.std([abs(true - pred) for true, pred in zip(T_test, T_pred)])
    print("Mean absolute error: ", absError, " / std: ", stdAbsError)
    print("Mean relative error: ", relError, " / std: ", stdRelError)

    # number of features - plot every feature/data dimension
    for i in range(5):
        # Plot training data
        plt.scatter(X_train[:,i], T_train, s=20, label='Training data')
        # Plot predictions
        plt.scatter(X_test[:,i], T_pred, s=20, color='r', label='Predictions')
        # Plot relevance vectors
        plt.scatter(clf.relevanceVectors[:,i],
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

