#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:52:00 2018

@author: j
@title: training on imputation and pipeline
"""

# ta hjälp av kod jag har i  min anteckning MLnotes.py

# 1... docs

import pandas as pd
import numpy as np
# scale data
from sklearn.preprocessing import StandardScaler
# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
# metrics
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# save models
from sklearn.externals import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
1+1
estimator = Pipeline([("imputer",
                        Imputer(missing_values='NaN',
                                  strategy="mean",
                                  axis=0)),
                      ("reg",
                      LogisticRegression(C=[0.1, 10])
                      #LogisticRegression(C=[0.1, 10], 'penalty'=['l1', 'l2'])
                      )
                      ])

# estimator = Pipeline([("imputer", Imputer(missing_values='NaN',
#                                           strategy="mean",
#                                           axis=0)),
#                       ("forest", RandomForestRegressor(random_state=0,
#                                                        n_estimators=100))])
score = cross_val_score(estimator, Xm, ym, cv=5).mean()
print("Score after imputation of the missing values = %.2f" % score)

# 2... docs w gridsearch

### sklearn docs: model selection

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

## pipe basics

estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)

pipe.set_params(clf__C=10)

## gridsearch

param_grid = dict(reduce_dim__n_components=[2, 5, 10],
                  clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)

from sklearn.linear_model import LogisticRegression
param_grid = dict(reduce_dim=[None, PCA(5), PCA(10)],
                  clf=[SVC(), LogisticRegression()],
                  clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)

## jacob (ej kört så vet ej om den funkar )
estimators = [('imputer', Imputer()),
              ('scaling', StandardScaler()),
              ('reduce_dim', PCA()),
              ('clf', SVC())
              ]
pipe = Pipeline(estimators)
param_grid = dict(Imputer(strategy=['mean', 'most_frequent']),
                  reduce_dim=[None, PCA(5), PCA(10)],
                  clf=[SVC(), LogisticRegression(C=[.1, 10, 100])],)
grid_search = GridSearchCV(pipe, param_grid=param_grid)
# qq jag fattar inte riktigt hur man kan mixa syntaxen här. så jag kanske bör göra ngt basic?
# använd detta i klr. tex



# 3... datacamp

### impoutation in a pipeline

# source https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines?ex=8

# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [
        ('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
         #('scaler', StandardScaler()),
        ('SVM', SVC())
        ]
# straget can aslo be: meaan (default), median,-

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


### pipeline, scaling och grid search

# source https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines?ex=12

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(estimator = pipeline, param_grid = parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
