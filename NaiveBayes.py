# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:17:54 2019

@author: Anvi Puri
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn

#Importing the dataset
dataset=pd.read_csv("framingham.csv")
dataset.drop(['education'],axis=1,inplace=True)
dataset.head()

#Calculating the total observations with NULL values 
dataset.rename(columns={'male':'Sex_male'},inplace=True)
dataset.isnull().sum()
count=0
for i in dataset.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)

#Creating matrix of independent features
X=dataset.iloc[:,:-1].values

#Creating dependent variable vector
Y=dataset.iloc[:,14].values

# Dealing with missing values 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0) 
imputer = imputer.fit(X[:, 0:14])
X[:, 0:14] = imputer.transform(X[:, 0:14])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('Mean accuracy on applying k fold cross validation:',accuracies.mean())
accuracies.std()

#Accuracy
print('Accuracy:',sklearn.metrics.accuracy_score(y_test,y_pred))





