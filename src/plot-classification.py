#!/usr/bin/env python3

"""plot-classification.py: make plots for the RVM classification."""

__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
              and Leo Zeitler"

from data import createSimpleClassData, banana
from rvm import RVR, RVC
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from matplotlib.patches import Circle
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set()

# Set a random seed
np.random.seed(0)

def main():
    data = 'simple'
    kernel = 'polynomialKernel'
    N = 300
    if data == 'banana':
        mainBanana(kernel, N)
    else:
        mainSimple(kernel, N)

def mainSimple(kernel, N):
    w = np.array([4, 2])
    Xtrain, Ttrain = createSimpleClassData(N, w)
    Xtest, Ttest = createSimpleClassData(int(N / 3), w)
    clf = RVC(Xtrain, Ttrain, kernel, alphaThresh=10e8, convergenceThresh=10e-2)
    clf.fit()
    TPred = clf.predict(Xtest)

    ############ CREATE GRID ###############################
    x_grid = np.linspace(-7, 7, 20)
    y_grid = np.linspace(-7, 7, 20)
    grid = [[float(clf.predict(np.array([np.array([x, y])]))) for x in x_grid] for y in y_grid]
    ############ PLOT TRAIN DATA WITH REL VECTORS ##################################################
    plt.subplot(131)
    correct_classifications = Xtrain.dot(w) > 0
    pos_data = Xtrain[correct_classifications == True]
    neg_data = Xtrain[correct_classifications == False]
    #data points
    plt.scatter(pos_data[:, 0], pos_data[:, 1], color='red')
    plt.scatter(neg_data[:, 0], neg_data[:, 1], color='blue')
    #relevance vectors
    plt.scatter(
        clf.relevanceVectors[:, 0], clf.relevanceVectors[:, 1],
        label='Relevance Vectors',
        s=100,
        facecolors="none",
        color="k",
    )
    # decision boundary
    plt.contour(x_grid, y_grid, grid, (0.0, 0.5, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    plt.xlabel("X[0]")
    plt.ylabel("X[1]")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend(loc='lower center')
    plt.title("Train data with rel vectors")

    ################## PLOT CONTOUR ##########################################
    plt.subplot(132)
    h = 0.3  # step size in the mesh
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='coolwarm', zorder=1)
    plt.contour(x_grid, y_grid, grid, (0.0, 0.5, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel("X[0]")
    plt.ylabel("X[1]")
    plt.title("Posterior class prob")

    ############ PLOT PREDICTIONS ############################################################
    plt.subplot(133)
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=TPred, cmap='coolwarm', label='Predictions', vmin=0, vmax=1)

    true_test = Xtest.dot(w) > 0
    pos_data = Xtest[true_test == True]
    neg_data = Xtest[true_test == False]

    plt.scatter(pos_data[:, 0], pos_data[:, 1], s=100, facecolors='none', color='red', label='True class Pos')
    plt.scatter(neg_data[:, 0], neg_data[:, 1], s=100, facecolors='none', color='blue', label='True class neg')

    plt.contour(x_grid, y_grid, grid, (0.0, 0.5, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1),
                label='decision boundary')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel("X[0]")
    plt.ylabel("X[1]")
    legend_elements = [Circle((0, 0), radius=5, color='k', label='Posterior prob'),
                       Circle((0, 0), radius=5, color='k', label='True class',
                              facecolor='none')]
    plt.legend(handles=legend_elements, loc='lower center')
    plt.title("Test data posterior prob")
    plt.show()

    # compute and print prediction
    prediction_classification = TPred > 0.5
    print('accuracy', (true_test == prediction_classification).sum() / true_test.shape[0])

def mainBanana(kernel, N):
    X, T = banana(N)
    Xtrain, Xtest, Ttrain, Ttest = train_test_split(
        X, T, test_size=0.2, random_state=42)
    clf = RVC(Xtrain, Ttrain, kernel, alphaThresh=10e8, convergenceThresh=10e-2)
    clf.fit()
    TPred = clf.predict(Xtest)

    ######## CREATE GRID ##################3
    x_grid = np.linspace(-3, 3, 60)
    y_grid = np.linspace(-3, 3, 60)
    grid = [[float(clf.predict(np.array([np.array([x, y])]))) for x in x_grid] for y in y_grid]

    ############ PLOT TRAIN DATA WITH REL VECTORS ##################################################
    plt.figure()
    posIndices = np.where(Ttrain > 0.5)[0].tolist()
    negIndices = np.where(Ttrain <= 0.5)[0].tolist()
    pos_data = np.array([Xtrain[posIndex] for posIndex in posIndices])
    neg_data = np.array([Xtrain[negIndex] for negIndex in negIndices])
    #data points
    plt.scatter(pos_data[:, 0], pos_data[:, 1], color='red')
    plt.scatter(neg_data[:, 0], neg_data[:, 1], color='blue')
    #relevance vectors
    plt.scatter(
        clf.relevanceVectors[:, 0], clf.relevanceVectors[:, 1],
        label='Relevance Vectors',
        s=100,
        facecolors="none",
        color="k",
    )
    #decision boundary
    plt.contour(x_grid, y_grid, grid, (0.0, 0.5, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.xlabel("X[0]")
    plt.ylabel("X[1]")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend(loc='lower center')
    plt.title("Train data with relevance vectors")
    plt.show()

    # ################## PLOT CONTOUR ##########################################
    plt.figure()
    ax = sns.heatmap(grid,vmin=0, vmax=1, cmap='coolwarm')
    ax.set_xlim([0, 60])
    ax.set_ylim([0, 60])
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.set_title("RVM posterior probability")
    # scale everything because heatmap is (0,60) instead of (-3,3)
    scaledX = [(i+3)*10 for i in x_grid]
    scaledY = [(i + 3) * 10 for i in y_grid]
    ax.contour(scaledX, scaledY, grid, (0.0, 0.5, 1), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.show()
    #
    # ############ PLOT PREDICTIONS ############################################################
    plt.figure()
    # plot predictions
    plt.scatter([elem[0] for elem in Xtest], [elem[1] for elem in Xtest], c=TPred, cmap='coolwarm', label='Predictions', vmin=0, vmax=1)

    posIndices = np.where(Ttest > 0.5)[0].tolist()
    negIndices = np.where(Ttest <= 0.5)[0].tolist()
    pos_data = np.array([Xtest[posIndex] for posIndex in posIndices])
    neg_data = np.array([Xtest[negIndex] for negIndex in negIndices])
    #plot true data points
    plt.scatter(pos_data[:, 0], pos_data[:, 1], s=100, facecolors='none', color='red', label='True class Pos')
    plt.scatter(neg_data[:, 0], neg_data[:, 1], s=100, facecolors='none', color='blue', label='True class neg')
    #plot decision boundary
    plt.contour(x_grid, y_grid, grid, (0.0, 0.5, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.xlabel("X[0]")
    plt.ylabel("X[1]")
    h1, = ax.plot([], ls='none', marker='o',color='k')
    h2, = ax.plot([], ls='none', marker='o', fillstyle='none', color='k')
    handles = [h1,h2]
    labels = ['Prediction', 'True class']
    plt.legend(handles, labels, loc='lower center')
    plt.title("Classification results")
    plt.show()

    #compute and print accuracy
    prediction_classification = TPred > 0.5
    print('accuracy', (Ttest == prediction_classification).sum() / Ttest.shape[0])

if __name__ == '__main__':
    main()
