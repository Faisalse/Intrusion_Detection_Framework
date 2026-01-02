# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023
@author: shefai
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DTree(BaseEstimator, ClassifierMixin):
    def __init__(self, criterion = "gini", max_depth = 10, splitter = "random"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        
    def fit(self, X, y):
        model = DecisionTreeClassifier(criterion = self.criterion, max_depth = self.max_depth)
        model.fit(X, y)
        self.model = model
        
    def predict(self, X):

        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):

        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob
    
    def use_hyperparameter_value(self):
        return f"Max_depth: {int(self.max_depth)} criterion: {self.criterion} splitter: {self.splitter}"
    
    def clear(self):
        self.max_depth = 0
        self.criterion = ""
        self.splitter = 0
    
    
    
    