# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score


def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    accuracy = correct / float(len(y_true)) * 100.0
    return accuracy


def Recall(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    # print("sklearns",recall_score(y_true, y_pred, average='macro'))
    # print(np.mean(recall))
    return np.mean(recall)


def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    return np.mean(precision)


def WCSS(Clusters, centroids):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    #     wcss=[]
    #     total=0
    val = 0.0
    for clus in range(len(Clusters)):
        #         dis_square=0.0
        distances = Clusters[clus] - centroids[clus]
        dis_square = distances * distances
        val += np.sum(dis_square)

    #         for i in range(len(clusters[clus])):
    #             distances=(np.linalg.norm(clusters[clus][i] - centroids[clus], axis=1))
    #             dis_square=distances*distances
    #             val=val+dis_square
    #         wcss.append(val)
    return val


def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    no_of_classes = len(np.unique(Y))
    array = y_true * no_of_classes + y_pred
    x = np.histogram(array, bins=range(min(array), (pow(no_of_classes, 2) + (min(array) + 1))))
    cm = np.asarray(x[0]).reshape(no_of_classes, no_of_classes)
    return cm


def KNN(X_train,X_test,Y_train):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    :rtype: numpy.ndarray
    """
    dists = np.zeros((X_test.shape[0], X_train.shape[0]))
    dists = np.sqrt(
        np.sum(X_test.values ** 2, axis=1)[:, np.newaxis] + np.sum(X_train.values ** 2, axis=1) - 2 * np.dot(X_test.values, X_train.values.T))
    # dists = np.argsort(dists, axis = 1)
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = []
        y_indicies = np.argsort(dists[i, :], axis=0)
        closest_y = Y_train[y_indicies[:K]]
        y_pred[i] = np.argmax(np.bincount(closest_y))
    return y_pred.astype(int)


def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    # training
    numberSamples = X_train.shape[0]
    featuresCount = X_train.shape[1]
    featuresInDecisionTree = int(np.ceil(np.sqrt(featuresCount)))
    samplesInDecisionTree = int(np.ceil(0.8 * numberSamples))
    randomForestSize = 20
    randomForest = []
    maxDepth = 10
    for i in range(randomForestSize):
        print('Training Decision Tree ==> ', str(i + 1), ' / ', randomForestSize)
        selectedFeatures = np.random.randint(featuresCount, size=featuresInDecisionTree)
        selectedSamples = np.random.randint(numberSamples, size=samplesInDecisionTree)
        Xselected = X_train[selectedSamples][:, selectedFeatures]
        Yselected = Y_train[selectedSamples]
        tree = DecisionTreeBuilder(Xselected, Yselected, maxDepth)
        randomForest.append(tree)

    # predicting
    predictions = []
    print('Predicting Decision Tree ==> ', str(i + 1), ' / ', randomForestSize)
    for tree in randomForest:
        prediction = predictFromDecisionTree(tree, X_test)
        predictions.append(prediction)

    predictions = np.asarray(predictions)
    return np.max(predictions, axis=0)

    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    mean = np.mean(X_train.T, axis=1)
    center = X_train - mean
    cov = np.cov(center.T)
    eig_val, eig_vect = np.linalg.eig(cov)
    PC = N
    P = eig_vect.T[0:PC]
    new_data = np.dot(X_train, P.T)
    # print(new_data.shape)
    return new_data


def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    K = N
    m = X_train.shape[0]
    n = X_train.shape[1]
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    centers = np.random.randn(K, n) * std + mean
    centers_old = np.zeros(centers.shape)
    centers_new = centers.copy()
    clusters = np.zeros(m)
    distances = np.zeros((m, K))
    error = np.linalg.norm(centers_new - centers_old)
    while error != 0:
        cluster_mem = []
        for i in range(K):
            distances[:, i] = np.linalg.norm(X_train - centers[i], axis=1)
        clusters = np.argmin(distances, axis=1)
        centers_old = centers_new.copy()
        for i in range(K):
            centers_new[i] = np.mean(X_train[clusters == i], axis=0)
            cluster_mem.append(X_train[clusters == i])
        error = np.linalg.norm(centers_new - centers_old)
    # print(type(clusters))
    # print(len(cluster_mem))
    # print(centers_new)
    return cluster_mem, clusters, centers_new


def SklearnSupervisedLearning(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    Y_pred = []
    # SVM
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, Y_train)
    Y_pred.append(svclassifier.predict(X_test))

    # logistic regression
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, Y_train)
    Y_pred.append(logisticRegr.predict(X_test))

    # Decision Tree
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    Y_pred.append(clf.predict(X_test))

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)
    Y_pred.append(knn.predict(X_test))

    return Y_pred

def SklearnVotingClassifier(X_train,Y_train,X_test,Y_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    model1 = SVC(kernel='linear')
    model2 = LogisticRegression()
    model3 = DecisionTreeClassifier()
    model4 = KNeighborsClassifier(n_neighbors=5)
    model = VotingClassifier(estimators=[('svc', model1), ('lr', model2), ('dt', model3), ('knn', model4)], voting='hard')
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    accuracy = model.score(X_test,Y_test)
    print("Voting Classifier Accuracy",accuracy * 100)
    return Y_pred

"""
Utility Functions
"""
def normalization(X_train):
    X_train= (X_train - X_train.mean()) / (X_train.max() - X_train.min())
    return X_train


def DecisionTreeBuilder(X, y, maxDepth):
    return DecisionTreeBuilderUtil(X, y, [], maxDepth, 0)


def predictFromDecisionTree(tree, X):
    y = []
    for Xi in X:
        root = tree
        while not root.isLeaf:
            if Xi[root.featureIndex] < root.featureThreshold:
                root = root.left
            else:
                root = root.right
        y.append(root.prediction)
    return np.asarray(y)


def DecisionTreeBuilderUtil(X, y, featuresConsidered, maxDepth, depth=None):
    classCount = countEachClassInSamples(y)
    totalSamples = X.shape[0]
    featuresCount = X.shape[1]
    root = Node()
    root.giniImpurity = giniImpurity(classCount, totalSamples)
    root.prediction = max(classCount, key=classCount.get)
    root.totalSamples = totalSamples
    root.classCount = classCount
    root.featuresCount = featuresCount
    if depth == maxDepth or len(classCount) <= 1:
        root.isLeaf = True
#         print('gini impurity of leaf node', root.giniImpurity)
#         print('prediction of leaf node', root.prediction)
#         print('class counts of leaf node', root.classCount)
#         print('\n')
        return root
    else:
        featureIndex, featureThreshold = calculateChildSplit(X, y, root, featuresConsidered)
        featuresConsidered.append(featureIndex)
        leftSamples = X[:, featureIndex] < featureThreshold
        X_left = X[leftSamples]
        y_left = y[leftSamples]
        X_right = X[~leftSamples]
        y_right = y[~leftSamples]
        root.featureIndex = featureIndex
        root.featureThreshold = featureThreshold
        root.left = DecisionTreeBuilderUtil(X_left, y_left, featuresConsidered, maxDepth, depth+1)
        root.right = DecisionTreeBuilderUtil(X_right, y_right, featuresConsidered, maxDepth, depth+1)
        return root


# def calculateChildSplit(X, y, root):
#     totalSamples = root.totalSamples
#     featuresCount = root.featuresCount
#     minGiniImpurity = root.giniImpurity
#     featureIndex = None
#     featureThreshold = None
#     for i in range(featuresCount):
#         print(i, featureIndex, featureThreshold)
#         for j in range(1, totalSamples):
#             split = X[j-1][i] + (X[j-1][i] + X[j][i])/2
#             ind = X[:, i] < split
#             yLeft = y[ind]
#             leftSamplesCount = yLeft.shape[0]
#             yRight = y[~ind]
#             rightSamplesCount = yRight.shape[0]
#             giniLeft = giniImpurity(countEachClassInSamples(yLeft), leftSamplesCount)
#             giniRight = giniImpurity(countEachClassInSamples(yRight), rightSamplesCount)
#             giniImpurityLocal = (leftSamplesCount*giniLeft + rightSamplesCount*giniRight) / totalSamples
#             if giniImpurityLocal < minGiniImpurity:
#                 minGiniImpurity = giniImpurityLocal
#                 featureIndex = i
#                 featureThreshold = split
#     return featureIndex, featureThreshold


def calculateChildSplit(X, y, root, featuresConsidered):
    totalSamples = root.totalSamples
    classCount = root.classCount
    featuresCount = root.featuresCount
    minGiniImpurity = root.giniImpurity
    featureIndex = None
    featureThreshold = None
    for i in range(featuresCount):
#         print(i, featureIndex, featureThreshold)
#         if i in featuresConsidered:
#             continue
        sortedIndex = X[:, i].argsort()
        featureValues = X[sortedIndex, i]
        classValues = y[sortedIndex]
        left = dict(zip(classCount.keys(), [0]*len(classCount)))
        right = countEachClassInSamples(y)
        for k in range(1, totalSamples):
            left[classValues[k-1]] += 1
            right[classValues[k-1]] -= 1
            if featureValues[k] == featureValues[k - 1]:
                continue
            giniImpurityLocal = (k*giniImpurity(left, k) + (totalSamples-k)*giniImpurity(right, totalSamples-k)) / totalSamples
            if giniImpurityLocal < minGiniImpurity:
                minGiniImpurity = giniImpurityLocal
                featureIndex = i
                featureThreshold = (featureValues[k] + featureValues[k-1])/2
    return featureIndex, featureThreshold


def giniImpurity(classCount, numSamples):
    return 1.0 - sum(pow((count/numSamples), 2) for count in classCount.values())


def countEachClassInSamples(y):
    classKey, countValue = np.unique(y, return_counts=True)
    return dict(zip(classKey, countValue))


class Node:
    def __init__(self):
        self.giniImpurity = None
        self.left = None
        self.right = None
        self.prediction = None
        self.isLeaf = False
        self.classCount = None
        self.featuresCount = None
        self.totalSamples = None
        self.featureIndex = None
        self.featureThreshold = None

    def __repr__(self):
        return 'giniImpurity = ' + str(self.giniImpurity) \
               + '\nleft = ' + str(self.left) \
               + '\nright = ' + str(self.right) \
               + '\nprediction = ' + str(self.prediction) \
               + '\nfeaturesCount = ' + str(self.featuresCount) \
               + '\nclassCount = ' + str(self.classCount) \
               + '\ntotalSamples = ' + str(self.totalSamples) \
               + '\nfeatureIndex = ' + str(self.featureIndex) \
               + '\nfeatureThreshold = ' + str(self.featureThreshold)


"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""

def ConfusionMatrixvisualizations(Y_test, Y_pred):
    cm1 = ConfusionMatrix(Y_test, Y_pred[0])
    cm2 = ConfusionMatrix(Y_test, Y_pred[1])
    cm3 = ConfusionMatrix(Y_test, Y_pred[2])
    cm4 = ConfusionMatrix(Y_test, Y_pred[3])
    cm5 = ConfusionMatrix(Y_test, Y_pred[4])
    fig, ((ax1, ax2),(ax3, ax4),(ax5,ax6)) = plt.subplots(3,2, figsize = (14,14))
    ax1.matshow(cm1)
    ax2.matshow(cm2)
    ax3.matshow(cm3)
    ax4.matshow(cm4)
    ax5.matshow(cm5)
    ax6.remove()
    ax1.title.set_text('SVM')
    ax1.set(xlabel='Predicted', ylabel='True')
    ax2.title.set_text('LR')
    ax1.set(xlabel='Predicted', ylabel='True')
    ax3.title.set_text('DT')
    ax1.set(xlabel='Predicted', ylabel='True')
    ax4.title.set_text('KNN')
    ax1.set(xlabel='Predicted', ylabel='True')
    ax5.title.set_text('Ensemble Model')
    ax1.set(xlabel='Predicted', ylabel='True')
    fig.savefig('confusion_matrix.png')

def GridSearchPlots(X_train, y_train):
    # KNN
    paramGrid = {'n_neighbors': [3, 5, 7, 9, 11], 'metric': ['euclidean', 'manhattan']}
    gridKNN = GridSearchCV(KNeighborsClassifier(), paramGrid, refit=True, verbose=1, cv=3, n_jobs=-1)
    gridKNN.fit(X_train, y_train)
    print(gridKNN.cv_results_)
    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = plt.axes()

    params = []
    for param in gridKNN.cv_results_['params']:
        p = ''
        for k, v in param.items():
            p += k + ' : ' + str(v) + '\n'
        params.append(p)
    bestParam = params[9] + 'mean_test_score : ' + str(gridKNN.cv_results_['mean_test_score'][9])

    ax.plot(params, gridKNN.cv_results_['mean_test_score'], '-o')
    props = dict(facecolor='lightblue', alpha=0.5)
    ax.text(0.05, 0.95, 'best estimator params\n' + bestParam, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.xticks(rotation='vertical')
    plt.title('K-Nearest Neighbors Grid Search', fontsize=20)
    plt.xlabel('Grid Search Parameters', fontsize=16)
    plt.ylabel('Mean Test Score', fontsize=16)
    plt.close(fig)

    # DT
    paramGridDT = [{'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']}]
    gridDT = GridSearchCV(DecisionTreeClassifier(), paramGridDT, refit=True, verbose=1, cv=3, n_jobs=-1)
    gridDT.fit(X_train, y_train)
    print(gridDT.cv_results_)
    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = plt.axes()

    params = []
    for param in gridDT.cv_results_['params']:
        p = ''
        for k, v in param.items():
            p += k + ' : ' + str(v) + '\n'
        params.append(p)

    bestParam = params[2] + 'mean_test_score : ' + str(gridDT.cv_results_['mean_test_score'][2])

    ax.plot(params, gridDT.cv_results_['mean_test_score'], '-o')
    props = dict(facecolor='lightblue', alpha=0.5)
    ax.text(0.1, 0.95, 'best estimator params\n' + bestParam, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.title('Decision Tree Grid Search', fontsize=20)
    plt.xlabel('Grid Search Parameters', fontsize=16)
    plt.ylabel('Mean Test Score', fontsize=16)
    plt.close(fig)

    # SVC
    paramGrid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    gridSVC = GridSearchCV(SVC(), paramGrid, refit=True, verbose=1, cv=3, n_jobs=-1)
    gridSVC.fit(X_train, y_train)
    print(gridSVC.cv_results_)
    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = plt.axes()

    params = []
    for param in gridSVC.cv_results_['params']:
        p = ''
        for k, v in param.items():
            p += k + ' : ' + str(v) + '\n'
        params.append(p)

    bestParam = params[7] + 'mean_test_score : ' + str(gridSVC.cv_results_['mean_test_score'][7])

    ax.plot(params, gridSVC.cv_results_['mean_test_score'], '-o')
    props = dict(facecolor='lightblue', alpha=0.5)
    ax.text(0.1, 0.95, 'best estimator params\n' + bestParam, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.title('Support Vector Classifier Grid Search', fontsize=20)
    plt.xlabel('Grid Search Parameters', fontsize=16)
    plt.ylabel('Mean Test Score', fontsize=16)
    plt.close(fig)


if __name__ == '__main__':
    print('DIC Assignment 1')

    # pre-processing-kundan
    # df = pd.read_csv('data.csv')
    # print(df.head())
    # v = df.values
    # np.random.seed(100)
    # np.random.shuffle(v)
    # X = v[:, :-1]
    # y = v[:, -1].astype(int)
    # n = X.shape[0]
    # n_train = round(0.8 * n)
    # n_test = n - n_train
    # X_train, y_train = X[:n_train], y[:n_train]
    # X_test, y_test = X[n_train:], y[n_train:]

    # pre-processing-varsha
    # df = pd.read_csv('data.csv')
    # X = df.drop(['48'], axis=1)
    # X = (X - np.min(X)) / (np.max(X) - np.min(X)).values
    # Y = df['48'].values
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    # Y_pred = SklearnSupervisedLearning(X_train, Y_train, X_test, Y_test)
    # Y_pred.append(SklearnVotingClassifier(X_train, Y_train, X_test))
    # ConfusionMatrixvisualizations(Y_test, Y_pred)
    #
    # Y_predict = KNN(X_train, X_test, Y_train, K=5)
    # accuracy = Accuracy(Y_test, Y_predict)
    # print(accuracy)
    # Recall = Recall(Y_test, Y_predict)
    # print(Recall)
    # Precision = Precision(Y_test, Y_predict)
    # print(Precision)

    # pre-processing-viswanathan
    # X_train = pd.read_csv('data.csv')
    # X_train = X_train.drop('48', axis=1)
    # X_train = X_train.values[:, 0:48]