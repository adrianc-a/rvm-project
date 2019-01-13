import numpy as np
import os


def createSimpleClassData(n, w, scale=10):
    X = np.random.rand(n, 2) * scale - scale / 2.
    T = X.dot(w) > 0
    return X, T.astype(int)


def linearClassification(train_n, test_n, w, scale=10):
    return (createSimpleClassData(train_n, w, scale),
            createSimpleClassData(test_n, w, scale))


def sinc_np(n, sigma):
    """Generate noisy or noise-free data from the sinc function f(x) = sin(x pi)/x pi
    Keyword arguments:
    n -- the number of of data points to be generated
    sigma -- noise variance
    """
    X = np.linspace(-10, 10, n)
    T = np.sinc(X) + np.random.normal(0, sigma, n)

    return X, T


def sinc(n, sigma):
    """Generate noisy or noise-free data from the sinc function f(x) = sin(x)/x
    Keyword arguments:
    n -- the number of of data points to be generated
    sigma -- noise variance
    """
    X = np.linspace(-10, 10, n)
    T = np.nan_to_num(np.sin(X) / X) + np.random.normal(0, sigma, n)

    return X, T


def cos(n, sigma):
    X = np.random.uniform(0, 10, size=(n, 1))
    T = np.cos(X) + np.random.normal(0, sigma, size=(n, 1))

    return X, T.reshape((n,))


def cos_test_train(train_n, test_n, sigma):
    return cos(train_n, sigma), cos(test_n, 0.0)


def linear(n, sigma):
    x = np.random.uniform(0, 10, size=(n, 1))
    t = 1.2 * x ** 2 + x + 2 + np.random.normal(0, sigma, size=(n, 1))

    return x, t.reshape((n,))


def airfoil(n=None, path=None):
    if not path:
        path = os.getcwd()
    text_file = open(path + "/airFoil.dat", "r")

    if not n:
        lines = text_file.readlines()[:]
    else:
        lines = text_file.readlines()[:n]
    data = [line.rstrip('\n').split('\t') for line in lines]
    N = len(data)
    X = np.zeros([N, 5])
    T = np.zeros(N)
    for i in range(N):
        line = data[i]
        float_list = [float(value) for value in line]
        X[i] = float_list[:-1]
        T[i] = float_list[-1]
    text_file.close()
    return X, T
