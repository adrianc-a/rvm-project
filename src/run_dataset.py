from sys import argv
from argparse import ArgumentParser
from sklearn import svm
from numpy import linalg as la

import time
import numpy as np

import data
import rvm

REGRESSION_DATASETS = {
    'cos': lambda: data.cos_test_train(30, 10, 0.1),
    'airfoil': lambda: data.airfoil_test_train(1050, 453)
}

CLASSIFICATION_DATASETS = {
    'linear': lambda: data.linearClassification(100, 20, np.array([1, 1]))
}


def parse_args():
    parser = ArgumentParser(
        description='Run and collect statistics for RVM/RVM*/SVM on specified data'
    )

    parser.add_argument(
        '-d', '--dataset', nargs='+', type=str,
        help='Names of dataset(s), from ' + str(REGRESSION_DATASETS) +
             str((CLASSIFICATION_DATASETS)),
        required=True
    )

    parser.add_argument('-a', '--alpha-thresh', type=float, default=10e8)
    parser.add_argument('-c', '--conv-thresh', type=float, default=10e-2)
    parser.add_argument('-k', '--kernel', type=str, default='RBFKernel')

    return parser.parse_args(argv[1:])


def time_fit(func):
    begin = time.time()
    func()
    end = time.time()
    return end - begin


def run_regression_dataset(ds, args):
    train, test = REGRESSION_DATASETS[ds]()

    # creating the models
    rvm_model = rvm.RVR(
        train[0], train[1], args.kernel, alphaThresh=args.alpha_thresh,
        convergenceThresh=args.conv_thresh
    )
    rvm_star_model = rvm.RVMRS(
        train[0], train[1], args.kernel, alphaThresh=args.alpha_thresh,
        convergenceThresh=args.conv_thresh
    )
    svm_model = svm.SVR(kernel='rbf')

    # timing the fitting...
    times = [
        time_fit(rvm_model.fit),
        time_fit(rvm_star_model.fit),
        time_fit(lambda: svm_model.fit(train[0], train[1]))
    ]

    accuracies = [
        la.norm(rvm_model.predict(test[0]) - test[1]),
        la.norm(rvm_star_model.predict(test[0]) - test[1]),
        la.norm(svm_model.predict(test[0]) - test[1]),
    ]

    num_vectors = [
        rvm_model.relevanceVectors.shape[0],
        rvm_star_model.relevanceVectors.shape[0],
        svm_model.support_vectors_.shape[0],
    ]

    return times, accuracies, num_vectors


def classification_accuracy(predictions, true_vals):
    return (predictions == true_vals).sum() / predictions.shape[0]


def run_classification_dataset(ds, args):
    train, test = CLASSIFICATION_DATASETS[ds]()

    rvm_model = rvm.RVC(
        train[0], train[1], args.kernel, alphaThresh=args.alpha_thresh,
        convergenceThresh=args.conv_thresh
    )
    svm_model = svm.SVC(kernel='rbf')
    
    train_times = [
        time_fit(rvm_model.fit),
        time_fit(lambda: svm_model.fit(train[0], train[1]))
    ]

    accuracies = [
        classification_accuracy(
            np.where(rvm_model.predict(test[0]) >= .5, 1, 0)
            , test[1]
        ),
        classification_accuracy(svm_model.predict(test[0]), test[1])
    ]
    
    num_vectors = [
        rvm_model.relevanceVectors.shape[0],
        svm_model.support_vectors_.shape[0]
    ]
    
    return train_times, accuracies, num_vectors

def main(args):
    for ds in args.dataset:
        if ds not in REGRESSION_DATASETS and ds not in CLASSIFICATION_DATASETS:
            raise RuntimeError(
                'Dataset ' + ds + 'not in list of known datasets')
        elif ds in REGRESSION_DATASETS:
            print(run_regression_dataset(ds, args))
        elif ds in CLASSIFICATION_DATASETS:
            print(run_classification_dataset(ds, args))


if __name__ == '__main__':
    main(parse_args())
