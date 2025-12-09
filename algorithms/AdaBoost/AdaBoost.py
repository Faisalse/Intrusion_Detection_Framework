# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class AdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators = 50, learning_rate = 1.0):
        
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        
        
    def fit(self, X, y):
        self.model = AdaBoostClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate)
        self.model.fit(X, y)

    def use_hyperparameter_value(self):
        return f"n_estimators: {str(self.n_estimators)} learning_rate: {str(self.learning_rate)}"
        
    def predict(self, X):
        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):
        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob
    
    def clear(self):
        self.n_estimators = 0
        self.learning_rate = 0
    
    
    
    