# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 15:34:57 2023

@author: shefai
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# l2 solver support solvers: sag, saga, lbfgs
# [0.001, 0.01, 0.1, 1, 10, 100]
class LR(BaseEstimator, ClassifierMixin):
    def __init__(self, solver = "lbfgs", penalty = "l2", C = 100):
        self.solver = solver
        self.penalty = penalty
        self.C = C

    def use_hyperparameter_value(self):
        return f"solver: {str(self.solver)} penalty: {str(self.penalty)} C: {str(self.C)}"
             
        
    def fit(self, X, y):

        if len(np.unique(y)) > 2: 

            model = LogisticRegression(multi_class='multinomial', solver = self.solver, penalty= self.penalty, C = self.C, random_state=0)
            model.fit(X, y)
            self.model = model
        else:
            model = LogisticRegression(solver = self.solver, penalty= self.penalty, C = self.C, random_state=0)
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
        