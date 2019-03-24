

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


def decision_tree_algorithm():
    # Importing the dataset
    dataset = pd.read_csv("framingham.csv")
    dataset.drop(['education'], axis=1, inplace=True)
    dataset.head()

    # Calculating the total observations with NULL values
    dataset.isnull().sum()
    count = 0
    for i in dataset.isnull().sum(axis=1):
        if i > 0:
            count = count + 1
    print('Total number of rows with missing values is ', count)

    # Creating matrix of independent features
    X = dataset.iloc[:, :-1].values

    # Creating dependent variable vector
    Y = dataset.iloc[:, 14].values

    # Dealing with missing values
    imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imputer = imputer.fit(X[:, 0:14])
    X[:, 0:14] = imputer.transform(X[:, 0:14])

    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Feature Scaling

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Fitting Decision Tree to the Training set

    classifier = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=10, random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Converting Input from user to Dataset and getting the output
    sample = {'col0': [1], 'col1': [40], 'col2': [1], 'col3': [4], 'col4': [0], 'col5': [0], 'col6': [1], 'col7': [1],
              'col8': [395], 'col9': [100], 'col10': [70], 'col11': [23], 'col12': [80], 'col13': [70]}
    sample_re = pd.DataFrame(data=sample)
    sample_re = sc_X.transform(sample_re)
    y_re = classifier.predict(sample_re)

    # Making the Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)

    # Applying k-Fold Cross Validation

    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    print(accuracies.mean() * 100)
    accuracies.std()

    # Applying Grid Search to find the best model and the best parameters

    parameters = [{'criterion': ['gini'], 'splitter': ['best', 'random'], 'max_depth': [10, 100, 500]},
                  {'criterion': ['entropy'], 'splitter': ['best', 'random'], 'max_depth': [10, 100, 500]}]
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=10,
                               n_jobs=-1)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    # Accuracy
    return (sklearn.metrics.accuracy_score(y_test, y_pred) * 100)



