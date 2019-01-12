from sys import argv
from argparse import ArgumentParser
from sklearn import svm
from numpy import linalg as la

import time

import data
import rvm

REGRESSION_DATASETS = {
    'cos': lambda: data.cos_test_train(30, 10, 0.1),
    'airfoil': lambda: data.airfoil_test_train(1050, 453)
}

CLASSIFICATION_DATASETS = {}


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

    return parser.parse_args(argv[1:])


def time_fit(func):
    begin = time.time()
    func()
    end = time.time()
    return end - begin


def run_regression_dataset(ds):
    train, test = REGRESSION_DATASETS[ds]()

    # creating the models
    rvm_model = rvm.RVR(train[0], train[1], 'RBFKernel')
    rvm_star_model = rvm.RVMRS(train[0], train[1], 'RBFKernel')
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


def run_classification_dataset(ds):
    train, test = CLASSIFICATION_DATASETS[ds]()

    rvm_model = rvm.RVC(train[0], train[1], 'RBFKernel')
    svm_model = svm.SVC(kernel='rbf')
    
    times = [
        time_fit(rvm_model.fit),
        time_fit(lambda: svm_model.fit(train[0], train[1]))
    ]
    
    accuracies = [
        classification_accuracy(rvm_model.predict(test[0]), test[1]),
        classification_accuracy(svm_model.predict(test[0]), test[1])
    ]
    
    num_vectors = [
        rvm_model.relevanceVectors.shape[0],
        svm_model.support_vectors_.shape[0]
    ]
    
    return times, accuracies, num_vectors

def main(args):
    for ds in args.dataset:
        if ds not in REGRESSION_DATASETS and ds not in CLASSIFICATION_DATASETS:
            raise RuntimeError(
                'Dataset ' + ds + 'not in list of known datasets')
        elif ds in REGRESSION_DATASETS:
            print(run_regression_dataset(ds))
        elif ds in CLASSIFICATION_DATASETS:
            print(run_classification_dataset(ds))


if __name__ == '__main__':
    main(parse_args())
