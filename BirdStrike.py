#!/usr/bin/python

from sklearn.preprocessing import StandardScaler
# Import K-Nearest Neighbor Classifier and accuracy_score

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def scaling(numeric, train_X, test_X):
    # Intialize a scaler
    scaler = StandardScaler()
    
    # Fit on training data
    scaler.fit(train_X[numeric])

    # Transform training and test data
    train_numeric_transform = scaler.transform(train_X[numeric])
    test_numeric_transform = scaler.transform(test_X[numeric])
    return train_numeric_transform, test_numeric_transform


def KNN(train_X,train_Y,test_X, test_Y):


    # Instantiate K Nearest Neighbors with 6 neighbors
    knn = KNeighborsClassifier(n_neighbors = 6)

    # Fit on training data
    knn.fit(train_X, train_Y)

    # Create Predictions
    pred_test_Y = knn.predict(test_X)
    pred_train_Y = knn.predict(train_X)

    # Calculate accuracy score on testing data
    test_accuracy = accuracy_score(test_Y, pred_test_Y)
    train_accuracy = accuracy_score(train_Y, pred_train_Y)
    return test_accuracy, train_accuracy


def Logistic(train_X,train_Y,test_X, test_Y):


    # Instantiate K Nearest Neighbors with 6 neighbors
    lr = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)

    # Fit on training data
    lr.fit(train_X, train_Y)
    

    # Create Predictions
    pred_test_Y = lr.predict(test_X)
    pred_train_Y = lr.predict(train_X)

    # Calculate accuracy score on testing data
    test_accuracy = accuracy_score(test_Y, pred_test_Y)
    train_accuracy = accuracy_score(train_Y, pred_train_Y)
    return test_accuracy, train_accuracy


def RF(train_X,train_Y,test_X, test_Y):

    rand_forest = RandomForestClassifier(random_state = 123, n_estimators = 200, max_depth = 4)

    # Fit decision tree and random forest on data
    rand_forest.fit(train_X, train_Y)



    # Create Predictions on test and train data using random forest
    pred_test_Y_forest = rand_forest.predict(test_X)
    pred_train_Y_forest = rand_forest.predict(train_X)



    # Calculate test and train accuracy score on random forest
    test_accuracy_forest = accuracy_score(test_Y, pred_test_Y_forest)
    train_accuracy_forest = accuracy_score(train_Y, pred_train_Y_forest)
    return test_accuracy_forest, train_accuracy_forest


def DT(train_X,train_Y,test_X, test_Y):


    # Instantiate decision tree and random forest classifiers
    dec_tree = DecisionTreeClassifier(random_state = 123)

    dec_tree.fit(train_X, train_Y)

    # Create Predictions on test and train data using decision tree
    pred_test_Y_tree = dec_tree.predict(test_X)
    pred_train_Y_tree = dec_tree.predict(train_X)

    # Calculate test and train accuracy score on decision tree
    test_accuracy_tree = accuracy_score(test_Y, pred_test_Y_tree)
    train_accuracy_tree = accuracy_score(train_Y, pred_train_Y_tree)
    return test_accuracy_tree, train_accuracy_tree