import numpy as np
import os
import csv
from sklearn.datasets import make_friedman2, make_friedman3, load_boston, load_breast_cancer


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


def sinc_uniform(n, lower=-1, upper=1):
    """Generate noisy or noise-free data from the sinc function f(x) = sin(x)/x
    Keyword arguments:
    n -- the number of of data points to be generated
    sigma -- noise variance
    """
    X = np.linspace(-10, 10, n)
    T = np.nan_to_num(np.sin(X) / X) + np.random.uniform(low=lower, high=upper, size=n)

    return X, T


def cos(n, sigma):
    X = np.random.uniform(0, 10, size=(n, 1))
    # X = np.linspace(0, 10, n).reshape((n, 1))

    T = np.cos(X) + np.random.normal(0, sigma, size=(n, 1))

    return X, T.reshape((n,))


def cos_test_train(train_n, test_n, sigma):
    return cos(train_n, sigma), cos(test_n, 0.0)


def linear(n, sigma):
    x = np.random.uniform(0, 10, size=(n, 1))
    t = 1.2 * x ** 2 + x + 2 + np.random.normal(0, sigma, size=(n, 1))

    return x, t.reshape((n,))


def friedman_2(n, noise):
    return make_friedman2(n_samples=n, noise=noise)


def friedman_3(n, noise):
    return make_friedman3(n_samples=n, noise=noise)


def boston_housing(n=None):
    (X, T) = load_boston(return_X_y=True)
    if not n:
        return X, T
    else:
        return X[:n], T[:n]


def breast_cancer(n=None):
    (X, T) = load_breast_cancer(return_X_y=True)
    if not n:
        return X, T
    else:
        return X[:n], T[:n]


def airfoil(n=None, path=None):
    if not path:
        path = os.getcwd()
    text_file = open(path + "/data/airFoil.dat", "r")

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


def slump(n=None, path=None):
    if not path:
        path = os.getcwd()

    with open(path + "/data/slumpTest.dat") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        X = []
        T = []
        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break
            if line_count == 0:
                line_count += 1
                continue

            X.append(np.asarray(row[1: 7], dtype=np.float64))
            T.append(np.asarray(row[7:], dtype=np.float64))
            line_count += 1

        return X, T


def banana(n=None, fileNumber=1, path=None):
    if not path:
        path = os.getcwd()

    X = []
    T = []
    with open(path + "/data/banana/banana_train_data_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            X.append(np.asarray(row[:], dtype=np.float64))
            line_count += 1

    with open(path + "/data/banana/banana_train_labels_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            T.append(np.asarray(row[:], dtype=np.float64))
            if T[-1][0] < 0:
                T[-1][0] = 0
            line_count += 1

    return X, np.asarray(T).reshape(-1)


def titanic(n=None, fileNumber=1, path=None):
    if not path:
        path = os.getcwd()

    X = []
    T = []
    with open(path + "/data/titanic/titanic_train_data_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            X.append(np.asarray(row[:], dtype=np.float64))
            line_count += 1

    with open(path + "/data/titanic/titanic_train_labels_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            T.append(np.asarray(row[:], dtype=np.float64))
            if T[-1][0] < 0:
                T[-1][0] = 0
            line_count += 1

    return X, np.asarray(T).reshape(-1)


def waveform(n=None, fileNumber=1, path=None):
    if not path:
        path = os.getcwd()

    X = []
    T = []
    with open(path + "/data/waveform/waveform_train_data_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            X.append(np.asarray(row[:], dtype=np.float64))
            line_count += 1

    with open(path + "/data/waveform/waveform_train_labels_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            T.append(np.asarray(row[:], dtype=np.float64))
            if T[-1][0] < 0:
                T[-1][0] = 0
            line_count += 1

    return X, np.asarray(T).reshape(-1)


def german(n=None, fileNumber=1, path=None):
    if not path:
        path = os.getcwd()

    X = []
    T = []
    with open(path + "/data/german/german_train_data_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            X.append(np.asarray(row[:], dtype=np.float64))
            line_count += 1

    with open(path + "/data/german/german_train_labels_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            T.append(np.asarray(row[:], dtype=np.float64))
            if T[-1][0] < 0:
                T[-1][0] = 0
            line_count += 1

    return X, np.asarray(T).reshape(-1)


def image(n=None, fileNumber=1, path=None):
    if not path:
        path = os.getcwd()

    X = []
    T = []
    with open(path + "/data/image/image_train_data_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            X.append(np.asarray(row[:], dtype=np.float64))
            line_count += 1

    with open(path + "/data/image/image_train_labels_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            T.append(np.asarray(row[:], dtype=np.float64))
            if T[-1][0] < 0:
                T[-1][0] = 0
            line_count += 1

    return X, np.asarray(T).reshape(-1)