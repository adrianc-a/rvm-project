import numpy as np


def createSimpleClassData(n, w, scale=10):
    X = np.random.rand(n, 2) * scale - scale/2.
    T = X.dot(w) > 0
    return X, T.astype(int)


def sinc_np(n, sigma):
    """Generate noisy or noise-free data from the sinc function f(x) = sin(x)/x
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


def airfoil(n1):
    text_file = open("airfoil.dat", "r")
    lines = text_file.readlines()[:n1]
    data = [line.rstrip('\n').split('\t') for line in lines]
    N = len(data)
    X = np.zeros([N, 5])
    T = np.zeros(N)
    for i in range(N):
        line = data[i]
        float_list = [float(i) for i in line]
        X[i] = float_list[:-1]
        T[i] = float_list[-1]
    text_file.close()
    return X, T

