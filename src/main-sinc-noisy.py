#!/usr/bin/env python2

"""main.py: Relevance Vector Machine (RVM) for regression and classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import sinc, createSimpleClassData
from rvm import RVR, RVC

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
np.random.seed(0)


def initData(N, dataset, *args):
    """Initialize the data set traning and testing examples"""
    X, T = dataset(N, *args)

    X_train, X_test, T_train, T_test = train_test_split(
            X, T, test_size=0.2, random_state=42)

    return X_train, X_test, T_train, T_test


def main():
    N = 100 # number of data points
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
    plt.plot(X, np.sinc(X), label='orig func')
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

