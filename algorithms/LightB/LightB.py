# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

class LightB(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators = 100, learning_rate = 0.1, max_depth = None, 
                 num_leaves = 31, min_child_samples = 20):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
    
    def use_hyperparameter_value(self):
        return f"n_estimators: {str(self.n_estimators)} learning_rate: {str(self.learning_rate)} max_depth: {str(self.max_depth)},  num_leaves: {str(self.num_leaves)} min_child_samples: {str(self.min_child_samples)}"
        
    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            model = lgb.LGBMClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate,  
                                    max_depth = self.max_depth,
                                        num_leaves = self.num_leaves,
                                        min_child_samples = self.min_child_samples,
                                        objective='multiclass', num_class = len(np.unique(y)), random_state=42)
        else:

            model = lgb.LGBMClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate,  
                                    max_depth = self.max_depth,
                                        num_leaves = self.num_leaves,
                                        min_child_samples = self.min_child_samples
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
        self.n_estimators = 0
        self.learning_rate = 0
        self.max_depth = 0
        self.min_samples_split = 0
        self.min_samples_leaf = 0
    
    
    
    