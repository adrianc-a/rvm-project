#!/usr/bin/env python2

"""pred_var_driver.py: """

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

import data as datagen
from rvm import RVR, RVC, RVMRS

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
np.random.seed(0)



def main():
    N = 90 # number of data points
    noiseStdDev = 0.05

    X_train,  T_train, = datagen.cos(N, noiseStdDev)

    T_train = T_train.reshape((N,))

    rvm_model = RVR(X_train, T_train, 'logKernel', convergenceThresh=10e-1)
    rvm_model.fit()

    rvm_star_model = RVMRS(X_train, T_train, 'logKernel', convergenceThresh=10e-1)
    rvm_star_model.fit()

    print("The relevance vectors:")
    print(rvm_model.relevanceVectors)

    # Plot predictions
    X = np.linspace(-10, 55, 250).reshape((250, 1))



    pred, vars = rvm_model.predict(X)
    pred_star, vars_star = rvm_star_model.predict(X)

    std_devs = np.sqrt(vars)
    std_devs_star = np.sqrt(vars_star)
    plt.plot(X, pred, label='Prediction $\mu$')
    plt.fill_between(X[:, 0], pred_star - 2 * std_devs_star , pred + 2 * std_devs_star, color='cyan', label='predictive variance')
    plt.fill_between(X[:, 0], pred - 2 * std_devs , pred + 2 * std_devs, color='gray', label='predictive variance')

    # Plot training data
    plt.plot(X, np.cos(X), label='orig func')
    plt.scatter(X_train, T_train, label='Training noisy samples')
    # plt.scatter(X_test, T_test, label='Testing noisy samples')

    # Plot relevance vectors
    # plt.scatter(clf.relevanceVectors,
    #             clf.T,
    #             label="Relevance vectors",
    #             s=50,
    #             facecolors="none",
    #             color="k",
    #             zorder=1)

    #plt.ylim(-0.3, 1.1)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    # plt.savefig("../plots/sincdataplot.png", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()

