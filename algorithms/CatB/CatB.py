# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CatB(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations = 100, learning_rate = 0.03, depth = 6, l2_leaf_reg = 3):
        
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
       
    def use_hyperparameter_value(self):
        return f"iterations: {str(self.iterations)} learning_rate: {str(self.learning_rate)} depth: {str(self.depth)},  l2_leaf_reg: {str(self.l2_leaf_reg)}"
        
    def fit(self, X, y):

        if len(np.unique(y)) > 2:
            model = CatBoostClassifier(iterations = self.iterations, learning_rate = self.learning_rate, 
                                 depth = self.depth, l2_leaf_reg = self.l2_leaf_reg, loss_function='MultiClass', eval_metric='MultiClass', verbose=100, random_seed=42)
        else:
            model = CatBoostClassifier(iterations = self.iterations, learning_rate = self.learning_rate, 
                                 depth = self.depth, l2_leaf_reg = self.l2_leaf_reg)
            

        
        model.fit(X, y)
        self.model = model
        
    def predict(self, X):

        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):

        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob
    
    def clear(self):
        self.iterations = 0
        self.learning_rate = 0
        self.depth = 0
        self.l2_leaf_reg = 0
    
    
    
    