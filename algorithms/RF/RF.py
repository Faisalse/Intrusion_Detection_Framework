# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RF:
    def __init__(self, n_estimators = 30, max_depth = None, min_samples_split = 10, min_samples_leaf = 1):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def fit(self, X, y):
        model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth,
                                     min_samples_split = self.min_samples_split,
                                     min_samples_leaf = self.min_samples_leaf)
        
        model.fit(X, y)
        self.model = model
        
    def predict(self, X):
        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):

        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob
    
    def clear(self):
        self.n_estimators = 0
        self.max_depth = 0
        self.min_samples_split = 0
        self.min_samples_leaf = 0
    
    
    
    