import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


class Accuracy:
    def __init__(self):
        a = 5
        pass
    
    def compute(self, y_actual, y_predict):
        measure = accuracy_score(y_actual, y_predict)
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("Accuracy:  ",  self.measure)
    

class Precision:
    def __init__(self):
        pass
    
    def compute(self, y_actual, y_predict):
        measure = precision_score(y_actual, y_predict)
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("Precision:  ",  self.measure)

class Recall:
    def __init__(self):
        pass
    
    def compute(self, y_actual, y_predict):
        measure = recall_score(y_actual, y_predict)
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("Recall:  ",  self.measure)

class F1_score:
    def __init__(self):
        pass
    
    def compute(self, y_actual, y_predict):
        measure = f1_score(y_actual, y_predict)
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("F1-score:  ",  self.measure) 
    
class Roc_Auc_score:
    def __init__(self):
        pass
    
    def compute(self, y_actual, y_predict):
        measure = roc_auc_score(y_actual, y_predict)
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("Roc_Auc_score:  ",  self.measure)

class Precision_Recall_score:
    def __init__(self):
        pass
    
    def compute(self, y_actual, y_predict):
        lr_precision, lr_recall, _ = precision_recall_curve(y_actual, y_predict)
        measure = auc(lr_recall, lr_precision)
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("Precision_Recall_score:  ",  self.measure)



