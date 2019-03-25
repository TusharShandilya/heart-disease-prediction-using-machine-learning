# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:32:31 2019

@author: Anvi Puri
"""
#common part starts here-->
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn

#Importing the dataset
dataset=pd.read_csv("framingham.csv")
dataset.drop(['education'],axis=1,inplace=True)

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
#common part ends here

#specific to algorithms-->
# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', splitter='random',max_depth=10,random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Converting Input from user to Dataset and getting the output
sample={'col0':[1],'col1':[ 40],'col2': [1],'col3':[4],'col4':[0],'col5':[0],'col6':[1],'col7':[1],'col8':[395],'col9':[100],'col10':[70],'col11':[23],'col12':[80],'col13':[70]}
sample_re= pd.DataFrame(data=sample)
sample_re=sc_X.transform(sample_re)
y_re=classifier.predict(sample_re)

#Accuracy
print('Accuracy after applying k fold cross validation and Grid Search:',sklearn.metrics.accuracy_score(y_test,y_pred))

