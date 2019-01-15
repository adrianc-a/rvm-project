#!/usr/bin/env python2

"""data-test.py: File for testing data sets"""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import friedman_2, friedman_3, boston_housing, airfoil, slump, \
    banana, titanic, waveform, german, image, breast_cancer, splice, thyroid
import numpy as np

# Set a random seed
np.random.seed(0)


def main():
    """Check for dimensionality miss-matches for the data sets in data.py"""
    N = 1000
    noise = 0.08

    X, T = friedman_2(N, noise)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in Friedman 2 dataset do not match")

    X, T = friedman_3(N, noise)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in Friedman 3 dataset do not match")

    X, T = boston_housing(N)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in Boston Housing dataset do not match")

    X, T = airfoil(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in Airfoil dataset do not match")

    X, T = slump(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in Slump dataset do not match")

    X, T = banana(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in Banana dataset do not match")

    X, T = titanic(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in Titanic dataset do not match")

    X, T = waveform(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in Waveform dataset do not match")

    X, T = german(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in German dataset do not match")

    X, T = image(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in Image dataset do not match")

    X, T = breast_cancer(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in breast cancer dataset do not match")

    X, T = splice(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in splice dataset do not match")

    X, T = thyroid(None)
    if len(X) != len(T):
        raise Exception("Lengths of X and T in thyroid dataset do not match")


if __name__ == '__main__':
    main()

