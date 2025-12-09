import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


class Acc:
    def __init__(self):
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
        measure = precision_score(y_actual, y_predict, average='weighted')
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("Precision:  ",  self.measure)

class Recall:
    def __init__(self):
        pass
    
    def compute(self, y_actual, y_predict):
        measure = recall_score(y_actual, y_predict, average='weighted')
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("Recall:  ",  self.measure)

class F1_score:
    def __init__(self):
        pass
    
    def compute(self, y_actual, y_predict):
        measure = f1_score(y_actual, y_predict, average='weighted')
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("F1-score:  ",  self.measure) 
    
class Roc_Auc_score:
    def __init__(self):
        pass
    
    def compute(self, y_actual, y_predict):
        measure = roc_auc_score(y_actual, y_predict, average='weighted', multi_class='ovr')
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("Roc_Auc_score:  ",  self.measure)

class Precision_Recall_score:
    def __init__(self):
        pass
    
    def compute(self, y_actual, y_predict):
        measure = average_precision_score(y_actual, y_predict, average='weighted')
        self.measure = np.round(measure, decimals = 4)
        
    
    def result(self):
        return ("Precision_Recall_score:  ",  self.measure)



