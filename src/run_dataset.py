#!/usr/bin/env python3

"""run_dataset.py: runner file to test RVM, fast RVM, SVM and RVM*"""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from sys import argv
from argparse import ArgumentParser
from sklearn import svm
from sklearn.model_selection import cross_validate as cval
from numpy import linalg as la
from csv import DictWriter

import numpy as np

from kernels import get_kernel

import data
import rvm

REGRESSION_DATASETS = {
    'cos': lambda: data.cos(30, 0.1),
    'airfoil': data.airfoil,
    'friedman2': lambda: data.friedman_2(300, 0.1),
    'friedman3': lambda: data.friedman_2(300, 0.1),
    'boston': lambda: data.boston_housing(506),
    'slump': data.slump,
    'sinc': lambda: data.sinc(100, 0.1)
}

CLASSIFICATION_DATASETS = {
    'linear': lambda: data.createSimpleClassData(100, np.array([1, 1])),
    'breast-cancer': lambda: data.breast_cancer(569),
    'banana': data.banana,
    'titanic': data.titanic,
    'waveform': data.waveform,
    'german': data.german,
    'image': data.image
}


def parse_args():
    """Parse command line arguments"""
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
    parser.add_argument('-f', '--folds', type=int, default=5)
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-p', '--pretty-print', action='store_true')
    parser.add_argument('-l', '--latex-print', action='store_true')
    parser.add_argument('-e', '--csv-export', type=str, default='')
    parser.add_argument('-n', '--no-print', action='store_true')
    parser.add_argument('-i', '--max-iters', type=int, default=100)

    return parser.parse_args(argv[1:])


class RVRFitWrapper:
    """Wrapper class for RVR"""
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.model = None

    def fit(self, x, y):
        self.model = self.model_factory(x, y)
        self.model.fit()

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return (la.norm(self.model.predict(x) - y) ** 2) / x.shape[0]

    def get_params(self, deep=False):
        return {'model_factory': self.model_factory}


class RVCFitWrapper(RVRFitWrapper):
    """Wrapper class for RVC"""
    def predict(self, x):
        return np.where(self.model.predict(x) >= .5, 1, 0)

    def score(self, x, y):
        predictions = self.predict(x)
        return (predictions == y).sum() / predictions.shape[0]


def create_kernel_callable(kname):
    """Creates a callable for kernel

    Args:
    kname (str): name of the kernel

    Returns:
    _kernel (func): corresponding kernel function callable

    """
    kernel, argv = get_kernel(kname)

    def _kernel(x, y):
        return kernel(x, y, argv)

    return _kernel


def run_regression_dataset(ds, args):
    """Run tests of regression data set

    Args:
    ds (str): name of the data set
    args (str): command line arguments

    Returns:
    rvm_results: results of original RVM regressor
    rvm_star_results: results of RVM* regressor
    rvm_fast_results: results of fast RVM regressor
    svm_results: results of SVM regressor

    """
    x, y = REGRESSION_DATASETS[ds]()

    print('Read dataset')
    # creating the models
    rvm_model = lambda a, b: rvm.RVR(
        a, b, args.kernel, alphaThresh=args.alpha_thresh,
        convergenceThresh=args.conv_thresh,
        verbosity=args.verbosity,
        maxIter=args.max_iters
    )

    rvm_star_model = lambda a, b: rvm.RVMRS(
        a, b, args.kernel, alphaThresh=args.alpha_thresh,
        convergenceThresh=args.conv_thresh,
        verbosity=args.verbosity,
        maxIter=args.max_iters
    )

    rvm_fast_model = lambda a, b: rvm.RVR(
        a, b, args.kernel, verbosity=args.verbosity,
        maxIter=args.max_iters,
        useFast=True
    )

    svm_model = svm.SVR(kernel='rbf', gamma=2)

    rvm_wrapper = RVRFitWrapper(rvm_model)
    rvm_star_wrapper = RVRFitWrapper(rvm_star_model)
    rvm_fast_wrapper = RVRFitWrapper(rvm_fast_model)

    # results
    svm_results = cval(
        svm_model, x, y, cv=args.folds, return_train_score=False,
        scoring='neg_mean_squared_error', return_estimator=True
    )

    np.random.seed(0)
    rvm_results = cval(
        rvm_wrapper, x, y, cv=args.folds,
        return_train_score=False, return_estimator=True
    )

    np.random.seed(0)
    rvm_star_results = cval(
        rvm_star_wrapper, x, y, cv=args.folds,
        return_train_score=False, return_estimator=True
    )

    rvm_fast_results = cval(
        rvm_fast_wrapper, x, y, cv=args.folds,
        return_train_score=False, return_estimator=True
    )

    # append the number of support/relevance vectors
    svm_results['vec'] = [
        mdl.support_vectors_.shape[0] for mdl in
        svm_results['estimator']
    ]

    svm_results['test_score'] = -svm_results['test_score']

    rvm_results['vec'] = [
        mdl.model.keptBasisFuncs.shape[0] for mdl in rvm_results['estimator']
    ]
    rvm_star_results['vec'] = [
        mdl.model.keptBasisFuncs.shape[0] for mdl in
        rvm_star_results['estimator']
    ]

    rvm_fast_results['vec'] = [
        mdl.model.keptBasisFuncs.shape[0] for mdl in
        rvm_fast_results['estimator']
    ]

    return rvm_results, rvm_star_results, rvm_fast_results, svm_results


def run_classification_dataset(ds, args):
    """Run tests on classification data set

    Args:
    ds (str): name of the data set
    args (str): command line arguments

    Returns:
    rvm_results: results of original RVM classifier
    rvm_fast_results: results of fast RVM classifier
    svm_results: results of SVM classifier

    """
    x, y = CLASSIFICATION_DATASETS[ds]()

    rvm_model = lambda a, b: rvm.RVC(
        a, b, args.kernel, alphaThresh=args.alpha_thresh,
        convergenceThresh=args.conv_thresh,
        verbosity=args.verbosity,
        maxIter=args.max_iters
    )

    rvm_fast_model = lambda a, b: rvm.RVC(
        a, b, args.kernel,
        verbosity=args.verbosity,
        maxIter=args.max_iters,
        useFast=True
    )

    svm_model = svm.SVC(kernel='rbf', gamma=2)

    rvm_wrapper = RVCFitWrapper(rvm_model)
    rvm__fast_wrapper = RVCFitWrapper(rvm_fast_model)

    # results
    rvm_results = cval(
        rvm_wrapper, x, y, cv=args.folds,
        return_train_score=False, return_estimator=True
    )

    rvm_fast_results = cval(
        rvm__fast_wrapper, x, y, cv=args.folds,
        return_train_score=False, return_estimator=True
    )

    svm_results = cval(
        svm_model, x, y, cv=args.folds, return_train_score=False,
        return_estimator=True, scoring='accuracy'
    )

    svm_results['vec'] = [
        sum(mdl.n_support_) for mdl in svm_results['estimator']
    ]

    rvm_results['vec'] = [
        mdl.model.relevanceVectors.shape[0] for mdl in rvm_results['estimator']
    ]

    rvm_fast_results['vec'] = [
        mdl.model.relevanceVectors.shape[0] for mdl in rvm_fast_results['estimator']
    ]
    return rvm_results, rvm_fast_results, svm_results


def pprint(name, res):
    """Pretty print the resuylts of a test

    Args:
    name (str): name of test
    res (dict): results of test

    """
    print(name)
    print('\t   Avg score:', np.mean(res['test_score']))
    print('\t Avg no. vec:', np.mean(res['vec']))
    print('\tAvg fit time:', np.mean(res['fit_time']))


def lprint(name, res):
    """Make a latex table"""
    pass


def write_csv(name, res, is_regression=True):
    """Write results to a CSV file

    Args:
    name (str): name of test
    res (dict): results of test
    is_regression (bool): specification of whether data was regression problem

    """
    models = ['rvm', 'rvm*', 'rvmf', 'svm'] if is_regression else [' rvm', 'rvmf' ' svm']

    ext = 'regression' if is_regression else 'classification'
    with open(name + '_' + ext + '.csv', 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Model', 'Mean Score', 'Mean No. Vectors', 'Mean Fit Time']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for line in res:
            for i in range(len(models)):
                writer.writerow(
                    {
                        fieldnames[0]: line[0],
                        fieldnames[1]: models[i + 1],
                        fieldnames[2]: np.mean(line[i]['test_score']),
                        fieldnames[3]: np.mean(line[i]['vec']),
                        fieldnames[4]: np.mean(line[i]['fit_time']),
                    }
                )


def null_func(*args):
    """Null function"""
    pass


def main(args):
    if args.pretty_print:
        pfunc = pprint
    elif args.latex_print:
        pfunc = lprint
    elif args.no_print:
        pfunc = null_func
    else:
        pfunc = print

    regression_results = []
    classification_results = []

    for ds in args.dataset:
        print('Dataset', ds)
        if ds not in REGRESSION_DATASETS and ds not in CLASSIFICATION_DATASETS:
            raise RuntimeError(
                'Dataset ' + ds + 'not in list of known datasets')
        elif ds in REGRESSION_DATASETS:
            rvm_res, rvm_s_res, rvm_f_res, svm_res = run_regression_dataset(ds, args)

            pfunc(' rvm', rvm_res)
            pfunc('rvm*', rvm_s_res)
            pfunc('rvmf', rvm_f_res)
            pfunc(' svm', svm_res)
            if args.csv_export != '':
                regression_results.append((ds, rvm_res, rvm_s_res, rvm_f_res, svm_res))

        elif ds in CLASSIFICATION_DATASETS:
            rvm_res, rvm_f_res, svm_res = run_classification_dataset(ds, args)
            pfunc('rvm', rvm_res)
            pfunc('rvmf', rvm_f_res)
            pfunc('svm', svm_res)
            if args.csv_export != '':
                classification_results.append((ds, rvm_res, rvm_f_res, svm_res))
    if args.csv_export != '':
        write_csv(args.csv_export, regression_results, is_regression=True)
        write_csv(args.csv_export, classification_results, is_regression=False)

if __name__ == '__main__':
    main(parse_args())
