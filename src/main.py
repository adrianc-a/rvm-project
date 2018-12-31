#!/usr/bin/env python3

"""main.py: Relevance Vector Machine (RVM) for regression and classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import sincNoiseFree, createSimpleClassData
from rvm import RVR, RVC
import numpy as np
import matplotlib.pyplot as plt


# Set a random seed
np.random.seed(0)


def initData(N, dataset):
    """Initialize the data set training and testing examples"""
    X, T = dataset(N)

    return X, T


def main():
    N = 200
    X, T = initData(N, sincNoiseFree)

    clf = RVR(X, T, 'RBFKernel')
    clf.fit()

    print("The relevance vectors:")
    print(clf.relevanceVectors)
    print("The relevance vectors T")
    print(clf.relevanceVectorsT)

    # This is using training data -- should be changed of course
    TPred = clf.predict(X)
    plt.scatter(X, T, label='Original Data', s=20)
    plt.scatter(X, TPred, color='r', label='Predictions', s=20)
    plt.scatter(clf.relevanceVectors, clf.relevanceVectorsT, facecolors='none', edgecolors='k', label='Relevance Vecs', s=48)


    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

