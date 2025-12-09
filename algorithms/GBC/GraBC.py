# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023
@author: shefai
"""

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class GBC(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators = 30, learning_rate = 0.1, max_depth = None, 
                 min_samples_leaf = 1, subsample = 0.8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample

    def use_hyperparameter_value(self):
        return f"n_estimators: {str(self.n_estimators)} learning_rate: {str(self.learning_rate)} max_depth: {str(self.max_depth)},  min_samples_leaf: {str(self.min_samples_leaf)} subsample: {str(self.subsample)}"
        
    def fit(self, X, y):
        self.model = GradientBoostingClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate, 
                                                max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf, 
                                                subsample = self.subsample, random_state=42)
        self.model.fit(X, y)
        
    def predict(self, X):

        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):

        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob
    
    def clear(self):
        self.n_estimators = 0
        self.learning_rate = 0
        self.max_depth = 0
        self.min_samples_split = 0
        self.min_samples_leaf = 0
    
    
    
    