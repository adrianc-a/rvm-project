#!/usr/bin/env python2

"""pred_var_driver.py: """

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

import data as datagen
from rvm import RVR, RVC, RVMRS
import plot as plt



def main():
    N = 30
    noiseStdDev = 0.1

    X_train,  T_train, = datagen.linear(N, noiseStdDev)

    fx = lambda x: 1.2 * x**2 + x + 2

    rvm_model = RVR(X_train, T_train, 'RBFKernel', convergenceThresh=10e-3)
    rvm_model.fit()

    plt.plot_regressor(
        rvm_model, 0, 40, train_data=(X_train, T_train), true_func=fx
    )



    rvm_star_model = RVMRS(X_train, T_train, 'RBFKernel', convergenceThresh=10e-3)
    rvm_star_model.fit()

    plt.plot_regressor(
        rvm_star_model, 0, 40, train_data=(X_train, T_train), true_func=fx
    )

    print('rel. vecs rvm:', rvm_model.keptBasisFuncs.shape[0])
    print('rel. vecs rvm*:', rvm_star_model.keptBasisFuncs.shape)

    # Plot predictions
    # plt.plot(X, pred, label='Prediction $\mu$')
    # plt.fill_between(X[:, 0], pred_star - 2 * std_devs_star , pred + 2 * std_devs_star, color='cyan', label='predictive variance')
    # plt.fill_between(X[:, 0], pred - 2 * std_devs , pred + 2 * std_devs, color='gray', label='predictive variance')
    #
    # # Plot training data
    # plt.plot(X, np.cos(X), label='orig func')
    # plt.scatter(X_train, T_train, label='Training noisy samples')
    # # plt.scatter(X_test, T_test, label='Testing noisy samples')




if __name__ == '__main__':
    main()

