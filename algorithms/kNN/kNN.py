# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 15:34:57 2023

@author: shefai
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors = 10, weights  = "uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
          
    def fit(self, X, y):
        model = KNeighborsClassifier(
                n_neighbors = self.n_neighbors,
                weights = self.weights
            )
        model.fit(X, y)
        self.model = model
        
    def predict(self, X):

        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):

        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob
    
    def clear(self):
        self.solver = 0
        self.penalty = 0
        self.C = 0
        