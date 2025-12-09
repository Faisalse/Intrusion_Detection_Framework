# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class XGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators = 500, max_depth = 6, learning_rate = 0.3, subsample = 0.6, colsample_bytree = 0.6, gamma = 0, reg_alpha = 0, reg_lambda = 0):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

    def use_hyperparameter_value(self):
        return f"n_estimators: {str(self.n_estimators)} max_depth: {str(self.max_depth)} subsample: {str(self.subsample)} colsample_bytree: {str(self.colsample_bytree)}  gamma: {str(self.gamma)} reg_alpha: {str(self.reg_alpha)}  reg_lambda: {str(self.reg_lambda)}"
    
       
    def fit(self, X, y):

        if len(np.unique(y)) > 2:
            model = XGBClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate,  
                                max_depth = self.max_depth, subsample = self.subsample, colsample_bytree = self.colsample_bytree, 
                                gamma = self.gamma, reg_alpha = self.reg_alpha, reg_lambda = self.reg_lambda,
                                objective='multi:softmax', num_class=len(np.unique(y)) )
        else:
            model = XGBClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate,  
                                max_depth = self.max_depth)
                                     
        model.fit(X, y)
        self.model = model
        
    def predict(self, X):

        y_predict = self.model.predict(X)
        return y_predict
    
    def predict_proba(self, X):

        y_predict_prob = self.model.predict_proba(X)
        return y_predict_prob
    def return_model(self):

        return self.model
    
    def clear(self):
        self.n_estimators = 0
        self.learning_rate = 0
        self.max_depth = 0
        self.min_samples_split = 0
        self.min_samples_leaf = 0
    
    
    
    