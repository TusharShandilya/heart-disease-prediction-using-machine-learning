# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 00:17:29 2019

@author: Anvi Puri
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Importing the dataset
dataset=pd.read_csv("framingham.csv")
dataset.drop(['education'],axis=1,inplace=True)
dataset.head()

#Calculating the total observations with NULL values 
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

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=14, units=8, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=8, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy
print('Accuracy:',sklearn.metrics.accuracy_score(y_test,y_pred))
