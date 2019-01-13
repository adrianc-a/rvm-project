#!/usr/bin/env python3

"""main.py: Relevance Vector Machine (RVM) for regression and classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import friedman_2, friedman_3, boston_housing, airfoil, slump, banana
import numpy as np
# Set a random seed
np.random.seed(0)


def main():
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


if __name__ == '__main__':
    main()
