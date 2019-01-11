
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
    noiseSpread = 0.2

    X_train, X_test, T_train, T_test = initData(N, sinc, noiseSpread)

    clf = RVR(X_train, T_train, 'linearSplineKernel')
    clf.fit()

    print("The relevance vectors:")
    print(clf.relevanceVectors)
    print("Beta:", np.sqrt(clf.beta**-1))

    # This is using training data -- should be changed of course
    T_pred = clf.predict(X_test)

    # Plot training data
    plt.scatter(X_train, T_train, s=20, label='Training data')

    # Plot predictions
    plt.scatter(X_test, T_pred, s=20, color='r', label='Predictions')

    # Plot relevance vectors
    plt.scatter(clf.relevanceVectors,
                clf.predict(clf.relevanceVectors),
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
