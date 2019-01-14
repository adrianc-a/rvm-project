from sys import argv
from argparse import ArgumentParser
from sklearn import svm
from sklearn.model_selection import cross_validate as cval
from numpy import linalg as la

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
    'banana': data.banana
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
    parser.add_argument('-f', '--folds', type=int, default=5)
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-p', '--pretty-print', action='store_true')
    parser.add_argument('-l', '--latex-print', action='store_true')

    return parser.parse_args(argv[1:])


class RVRFitWrapper:
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
    def predict(self, x):
        return np.where(self.model.predict(x) >= .5, 1, 0)

    def score(self, x, y):
        predictions = self.predict(x)
        return (predictions == y).sum() / predictions.shape[0]


def create_kernel_callable(kname):
    kernel, argv = get_kernel(kname)

    def _kernel(x, y):
        return kernel(x, y, argv)

    return _kernel


def run_regression_dataset(ds, args):
    x, y = REGRESSION_DATASETS[ds]()

    print('Read dataset')
    # creating the models
    rvm_model = lambda a, b: rvm.RVR(
        a, b, args.kernel, alphaThresh=args.alpha_thresh,
        convergenceThresh=args.conv_thresh,
        verbosity=args.verbosity
    )

    rvm_star_model = lambda a, b: rvm.RVMRS(
        a, b, args.kernel, alphaThresh=args.alpha_thresh,
        convergenceThresh=args.conv_thresh,
        verbosity=args.verbosity
    )

    svm_model = svm.SVR(kernel='rbf', gamma=2)

    rvm_wrapper = RVRFitWrapper(rvm_model)
    rvm_star_wrapper = RVRFitWrapper(rvm_star_model)


    # results
    svm_results = cval(
        svm_model, x, y, cv=args.folds, return_train_score=False,
        scoring='neg_mean_squared_error', return_estimator=True
    )

    rvm_results = cval(
        rvm_wrapper, x, y, cv=args.folds,
        return_train_score=False, return_estimator=True
    )

    rvm_star_results = cval(
        rvm_star_wrapper, x, y, cv=args.folds,
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
        mdl.model.keptBasisFuncs.shape[0] for mdl in rvm_star_results['estimator']
    ]

    return rvm_results, rvm_star_results, svm_results


def run_classification_dataset(ds, args):
    x, y = CLASSIFICATION_DATASETS[ds]()

    rvm_model = lambda a, b: rvm.RVC(
        a, b, args.kernel, alphaThresh=args.alpha_thresh,
        convergenceThresh=args.conv_thresh,
        verbosity=args.verbosity
    )

    svm_model = svm.SVC(kernel='rbf', gamma=2)

    rvm_wrapper = RVCFitWrapper(rvm_model)

    # results
    rvm_results = cval(
        rvm_wrapper, x, y, cv=args.folds,
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

    return rvm_results, svm_results


def pprint(name, res):
    print(name)
    print('\t   Avg score:', np.mean(res['test_score']))
    print('\t Avg no. vec:', np.mean(res['vec']))
    print('\tAvg fit time:', np.mean(res['fit_time']))

def lprint(name, res):
    # make a latex table...
    pass

def main(args):
    if args.pretty_print:
        pfunc = pprint
    elif args.latex_print:
        pfunc = lprint
    else:
        pfunc = print

    for ds in args.dataset:
        print('Dataset', ds)
        if ds not in REGRESSION_DATASETS and ds not in CLASSIFICATION_DATASETS:
            raise RuntimeError(
                'Dataset ' + ds + 'not in list of known datasets')
        elif ds in REGRESSION_DATASETS:
            rvm_res, rvm_s_res, svm_res = run_regression_dataset(ds, args)

            pfunc(' rvm', rvm_res)
            pfunc('rvm*', rvm_s_res)
            pfunc(' svm', svm_res)

        elif ds in CLASSIFICATION_DATASETS:
            rvm_res, svm_res = run_classification_dataset(ds, args)
            pfunc('rvm', rvm_res)
            pfunc('svm', svm_res)


if __name__ == '__main__':
    main(parse_args())
