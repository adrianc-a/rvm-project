#!/usr/bin/env python2

"""data.py: Several functions for creating data sets."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

import numpy as np
import os
import csv

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman2, make_friedman3, load_boston, \
    load_breast_cancer

# Set a random seed
np.random.seed(0)


def initData(N, dataset, *args):
    """Initialize a data set with training and testing examples

    Args:
    N (int): numer of data points
    dataset (str): name of the data set
    *args (none)

    Returns:
    Data set split in training and testing examples

    """
    X, T = dataset(N, *args)

    X_train, X_test, T_train, T_test = train_test_split(
            X, T, test_size=0.2, random_state=42)

    return X_train, X_test, T_train, T_test


def createSimpleClassData(n, w, scale=10):
    """Create a toy data set for classification

    Args:
    n (int): number of data points
    w (numpy.ndarray): weight vector
    scale (float): scale factor

    Returns:
    X (numpy.ndarray): input data points
    T (numpy.ndarray): targets

    """
    X = np.random.rand(n, 2) * scale - scale / 2.
    T = X.dot(w) > 0
    return X, T.astype(int)


def linearClassification(train_n, test_n, w, scale=10):
    """Create a data set for linear classification

    Args:
    train_n (int): number of train data points
    test_n (int): number of test data points
    w (numpy.ndarray): weight vector
    scale (float): scale vector

    Returns:
    Linearly seperable classification data set
    """
    return (createSimpleClassData(train_n, w, scale),
            createSimpleClassData(test_n, w, scale))


def sinc_np(n, sigma):
    """Generate noisy or noise-free data in [-10, 10] from the sinc function
    f(x) = sin(x pi)/x pi using numpy.

    Args:
    n (int): the number of of data points to be generated
    sigma (float): noise variance

    Returns:
    X (numpy.ndarray): input data points
    T (numpy.ndarray): targets

    """
    X = np.linspace(-10, 10, n)
    T = np.sinc(X) + np.random.normal(0, sigma, n)

    return X, T


def sinc(n, sigma):
    """Generate noisy or noise-free data in [-10, 10] from the sinc function
    f(x) = sin(x)/x.

    Args:
    n (int): the number of of data points to be generated
    sigma (float): noise variance

    Returns:
    X (numpy.ndarray): input data points
    T (numpy.ndarray): targets

    """
    X = np.linspace(-10, 10, n)
    T = np.nan_to_num(np.sin(X) / X) + np.random.normal(0, sigma, n)

    return X, T


def sinc_uniform(n, lower=-1, upper=1):
    """Generate noisy data from the sinc function f(x) = sin(x)/x using uniform
    noise

    Args:
    n (int): the number of of data points to be generated
    lower (float): lower bound for uniform noise
    upper (float): upper bound for uniform noise

    Returns:
    X (numpy.ndarray): input data points
    T (numpy.ndarray): targets

    """
    X = np.linspace(-10, 10, n)
    T = np.nan_to_num(np.sin(X) / X) + np.random.uniform(low=lower, high=upper, size=n)

    return X, T


def sin(n, sigma):
    """Generate noisy or noise-free data in [-10, 10] from the sin function

    Args:
    n (int): the number of of data points to be generated
    sigma (float): noise variance

    Returns:
    X (numpy.ndarray): input data points
    T (numpy.ndarray): targets

    """
    X = np.linspace(-10, 10, n)
    T = np.sin(X) + np.random.normal(0, sigma, n)

    return X, T


def cos(n, sigma):
    """Generate noisy or noise-free data in [-10, 10] from the cos function

    Args:
    n (int): the number of of data points to be generated
    sigma (float): noise variance

    Returns:
    X (numpy.ndarray): input data points
    T (numpy.ndarray): targets

    """
    X = np.random.uniform(0, 10, size=(n, 1))
    T = np.cos(X) + np.random.normal(0, sigma, size=(n, 1))

    return X, T.reshape((n,))


def cos_test_train(train_n, test_n, sigma):
    """Create a data set based on the cos function

    Args:
    train_n (int): number of train data points
    test_n (int): number of test data points
    sigma (float): noise variance

    Returns:
    Cos data set

    """
    return cos(train_n, sigma), cos(test_n, 0.0)


def linear(n, sigma):
    """Generate noisy or noise-free linearly seperable data set

    Args:
    n (int): the number of of data points to be generated
    sigma (float): noise variance

    Returns:
    Linearly seperable data set

    """
    X = np.random.uniform(0, 10, size=(n, 1))
    T = 1.2 * X ** 2 + X + 2 + np.random.normal(0, sigma, size=(n, 1))

    return X, T.reshape((n,))


def friedman_2(n, noise):
    """Generate the friedman_2 data set

    Args:
    n (int): number of data samples
    noise (float): added noise

    Returns:
    Friedman 2 data set

    """
    return make_friedman2(n_samples=n, noise=noise)


def friedman_3(n, noise):
    """Generate the friedman_3 data set

    Args:
    n (int): number of data samples
    noise (float): added noise

    Returns:
    Friedman 3 data set

    """
    return make_friedman3(n_samples=n, noise=noise)


def boston_housing(n=None):
    """Generate the Boston housing data set

    Args:
    n (int): number of data samples

    Returns:
    Boston housing data set

    """
    (X, T) = load_boston(return_X_y=True)

    if not n:
        return X, T
    else:
        return X[:n], T[:n]


def breast_cancer(n=None):
    """Generate the sklearn breast cancer data set

    Args:
    n (int): number of data samples

    Returns:
    Sklearn breast cancer data set

    """
    (X, T) = load_breast_cancer(return_X_y=True)
    if not n:
        return X, T
    else:
        return X[:n], T[:n]


def airfoil(n=None, path=None):
    """Generate the airfoil data set

    Args:
    n (int): number of data samples
    path (str): path to .dat file

    Returns:
    Sklearn airfoil data set

    """
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
    """Generate the slump data set

    Args:
    n (int): number of data samples
    path (str): path to .dat file

    Returns:
    Sklearn slump data set

    """
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

        return np.asarray(X), np.asarray(T)


def banana(n=None, fileNumber=1, path=None):
    """Generate the banana data set

    Args:
    n (int): number of data samples
    fileNumber (int): the file number
    path (str): path to .dat file

    Returns:
    Sklearn banana data set

    """
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

    return np.asarray(X), np.asarray(T).reshape(-1)


def titanic(n=None, fileNumber=1, path=None):
    """Generate the titanic data set

    Args:
    n (int): number of data samples
    fileNumber (int): the file number
    path (str): path to .dat file

    Returns:
    Sklearn titanic data set

    """
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

    return np.asarray(X), np.asarray(T).reshape(-1)


def waveform(n=None, fileNumber=1, path=None):
    """Generate the waveform data set

    Args:
    n (int): number of data samples
    fileNumber (int): the file number
    path (str): path to .dat file

    Returns:
    Sklearn waveform data set

    """
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

    return np.asarray(X), np.asarray(T).reshape(-1)


def german(n=None, fileNumber=1, path=None):
    """Generate the german data set

    Args:
    n (int): number of data samples
    fileNumber (int): the file number
    path (str): path to .dat file

    Returns:
    Sklearn german data set

    """
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

    return np.asarray(X), np.asarray(T).reshape(-1)


def image(n=None, fileNumber=1, path=None):
    """Generate the image data set

    Args:
    n (int): number of data samples
    fileNumber (int): the file number
    path (str): path to .dat file

    Returns:
    Sklearn image data set

    """
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

    return np.asarray(X), np.asarray(T).reshape(-1)


def thyroid(n=None, fileNumber=1, path=None):
    """Generate the thyroid data set

    Args:
    n (int): number of data samples
    fileNumber (int): the file number
    path (str): path to .dat file

    Returns:
    Sklearn thyroid data set

    """
    if not path:
        path = os.getcwd()

    X = []
    T = []
    with open(path + "/data/thyroid/thyroid_train_data_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            X.append(np.asarray(row[:], dtype=np.float64))
            line_count += 1

    with open(path + "/data/thyroid/thyroid_train_labels_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            T.append(np.asarray(row[:], dtype=np.float64))
            if T[-1][0] < 0:
                T[-1][0] = 0
            line_count += 1

    return np.asarray(X), np.asarray(T).reshape(-1)


def splice(n=None, fileNumber=1, path=None):
    """Generate the splice data set

    Args:
    n (int): number of data samples
    fileNumber (int): the file number
    path (str): path to .dat file

    Returns:
    Sklearn splice data set

    """
    if not path:
        path = os.getcwd()

    X = []
    T = []
    with open(path + "/data/splice/splice_train_data_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            X.append(np.asarray(row[:], dtype=np.float64))
            line_count += 1

    with open(path + "/data/splice/splice_train_labels_" + str(fileNumber) + ".asc") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

        line_count = 0
        for row in csv_reader:
            if line_count == n:
                break

            T.append(np.asarray(row[:], dtype=np.float64))
            if T[-1][0] < 0:
                T[-1][0] = 0
            line_count += 1

    return np.asarray(X), np.asarray(T).reshape(-1)

