#!/usr/bin/env python2

"""pred_var_driver.py: driver file for predicting the variance of RVMRS"""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

import data as datagen
from rvm import RVMRS
import plot as plt
import numpy as np


def main():
    N = 40
    noiseStdDev = 0.15

    X_train, T_train, = datagen.cos(N, noiseStdDev)

    fx = np.cos

    rvm_star_model = RVMRS(
        X_train, T_train, 'RBFKernel', alphaThresh=1e19,
        convergenceThresh=10e-8,
    )

    rvm_star_model.fit()

    plt.plot_regressor(
        rvm_star_model, 0, 20, train_data=(X_train, T_train), true_func=fx,
        save_name='rvm_star_cos.png'
    )

    print('rel. vecs rvm*:', rvm_star_model.keptBasisFuncs.shape)

if __name__ == '__main__':
    main()

