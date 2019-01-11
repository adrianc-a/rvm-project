#!/usr/bin/env python2

"""main.py: Relevance Vector Machine (RVM) for regression and classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import sinc, createSimpleClassData, airfoil
from rvm import RVR, RVC

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
np.random.seed(0)

# Proposal:
# Accuracy measure: mean+std difference from true value
# Plot: per dimension and one pca one
#
#
#
#

def initData(N, dataset): #*args
    """Initialize the data set traning and testing examples"""
    # X, T = dataset(N, *args)
    X, T = dataset(N)

    X_train, X_test, T_train, T_test = train_test_split(
            X, T, test_size=0.2, random_state=42)

    return X_train, X_test, T_train, T_test


def main():
    N = 1000 # number of data points
    noiseVariance = 0.01**2
    dataFunction = sinc

    # X_train, X_test, T_train, T_test = initData(N, sinc, noiseVariance)
    X_train, X_test, T_train, T_test = initData(N, airfoil)
    print("train data ", X_train)
    clf = RVR(X_train, T_train, 'RBFKernel')
    clf.fit()

    print("Number of relevance vectors:")
    print(len(clf.relevanceVectors))

    # This is using training data -- should be changed of course
    T_pred = clf.predict(X_train)

    for i in range(5):
        # Plot training data
        plt.scatter(X_train[:,i], T_train, s=20, label='Training data')

        # Plot predictions
        plt.scatter(X_train[:,i], T_pred, s=20, color='r', label='Predictions')

        # Plot relevance vectors
        plt.scatter(clf.relevanceVectors[:,i],
                clf.T,
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

