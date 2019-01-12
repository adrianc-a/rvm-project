"""
Some plotting utils specific to rvm.py
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_regressor(regressor, lo, hi, train_data=None, true_func=None,
                   save_name=None):
    n = (hi - lo) * 10
    x = np.linspace(lo, hi, n).reshape((n, 1))
    means, vars = regressor.predict(x)

    std_devs2 = 2 * np.sqrt(vars)

    fig = plt.figure()

    # std devs
    plt.fill_between(x[:, 0], means - std_devs2, means + std_devs2, color='gray',
                     label='Predictive variance')

    if true_func:
        plt.plot(x[:, 0], true_func(x), label='True function', color='green')

    plt.plot(x, means, label='Predictive mean', color='orange')

    if train_data:
        plt.scatter(train_data[0], train_data[1], marker='x',
                    label='Training points')

    if train_data:
        idx = regressor.keptBasisFuncs[regressor.keptBasisFuncs > 0] - 1
        plt.scatter(
            train_data[0][idx], train_data[1][idx],
            label='Relevance Vectors',
            color='black',
            marker='o'
        )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    if save_name:
        plt.savefig('plots/' + save_name)
    else:
        plt.show()

    return fig
