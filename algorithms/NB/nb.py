# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 15:34:57 2023

@author: shefai
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin

class NB(BaseEstimator, ClassifierMixin):
    def __init__(self, var_smoothing = 0.000001):
        self.var_smoothing = var_smoothing

    def use_hyperparameter_value(self):
        return f"var_smooting: {str(self.var_smoothing)}"
        
    def fit(self, X, y):
        model = GaussianNB(var_smoothing=self.var_smoothing)
        model.fit(X, y)
        self.model = model
        
    def predict(self, X):

        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):

        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob
    
    def clear(self):
        self.var_smoothing = 0