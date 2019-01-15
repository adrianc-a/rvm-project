#!/usr/bin/env python2

"""main-sinc.py: Test RVR on the sinc data set with a fixed nuisance parameter."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import sinc, initData
from rvm import RVR

import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
np.random.seed(0)


def main():
    N = 124 # sincd we only use 80% for training (~100)
    noiseVariance = .01**2
    dataFunction = sinc

    X_train, X_test, T_train, T_test = initData(N, sinc, noiseVariance)

    clf = RVR(X_train,
              T_train,
              'linearSplineKernel',
              beta=1/noiseVariance,
              useFast=False,
              betaFixed=True)
    clf.fit()

    print("The relevance vectors (%d):" % len(clf.relevanceVectors))
    print(clf.relevanceVectors)

    T_pred = clf.predict(X_test)

    # Plot training data
    X = np.linspace(-10, 10, 250)
    plt.plot(X, np.sin(X)/X, label='Orig. func')
    plt.scatter(X_train, T_train, s=20, label='Training samples', zorder=2)

    # Plot predictions
    predictedMu = clf.predict(X)
    plt.plot(X, predictedMu, label='Pred. func (mean)', dashes=[2,2])

    # Plot relevance vectors
    plt.scatter(clf.relevanceVectors,
                clf.relevanceTargets,
                label="Relevance vectors",
                s=50,
                facecolors="none",
                color="k",
                zorder=10)

    plt.ylim(-0.3, 1.1)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    #plt.savefig("../plots/sincdataplot.png", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()

