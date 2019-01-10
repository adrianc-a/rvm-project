#!/usr/bin/env python2

"""main.py: Relevance Vector Machine (RVM) for regression and classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import sincNoiseFree, createSimpleClassData
from rvm import RVR, RVC

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
np.random.seed(0)

def initData(N, dataset):
    """Initialize the data set traning and testing examples"""
    X, T = dataset(N)

    X_train, X_test, T_train, T_test = train_test_split(
        X, T, test_size=0.2, random_state=42)

    return X_train, X_test, T_train, T_test


def main():
    N = 200
    X_train, X_test, T_train, T_test = initData(N, sincNoiseFree)

    clf = RVR(X_train, T_train, 'RBFKernel')
    clf.fit()

    print("The relevance vectors:")
    print(clf.relevanceVectors)

    # This is using training data -- should be changed of course
    T_pred = clf.predict(X_train)
    plt.scatter(X_train, T_train, label='Training Data')
    plt.scatter(X_train, T_pred, color='r', label='Predictions')
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

