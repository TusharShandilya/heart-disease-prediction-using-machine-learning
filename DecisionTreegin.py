# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:53:26 2019

@author: Anvi Puri
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm #For the estimation of many statistical models as well as to conduct statistical tests 
import scipy.stats as ss#contains probability distributions as well as statistical functions
import seaborn as sb #A python data visualization library based on matplotlib
import sklearn
import matplotlib.mlab as mlab#Numerical python functions written for compatibility with MATLAB commands with the same name
   
#Importing the dataset
dataset=pd.read_csv("framingham.csv")
dataset.drop(['education'],axis=1,inplace=True)
dataset.head()
dataset.isnull().sum()

#Counting number of columns with missing values
count=0
for i in dataset.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
#print('since it is only',round((count/len(dataset.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')
#dataset.dropna(axis=0,inplace=True)
#dataset.isnull().sum()

#Creating matrix of independent features
X=dataset.iloc[:,:-1].values

#Creating dependent variable vector
Y=dataset.iloc[:,14].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0) #Observation:Out of the 3 strategies, most_frequent worked the best
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

#Analysing the Dataset
def draw_histograms(dataset, features, row, col):
    fig=plt.figure(figsize=(300,500))
    for i, feature in enumerate(features): # It allows us to loop over something and have an automatic counter
        lay=fig.add_subplot(row,col,i+1)
        dataset[feature].hist(bins=20,ax=lay,facecolor='yellow')
        lay.set_title(feature+" Distribution",color='blue')
      
    fig.tight_layout()  
    plt.show()
draw_histograms(dataset,dataset.columns,4240,15)

# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini',random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy
print(sklearn.metrics.accuracy_score(y_test,y_pred))


