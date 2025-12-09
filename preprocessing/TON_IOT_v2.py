# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 21:29:36 2023

@author: shefai
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

DATA_PATH = r'./data/raw/'
DATA_PATH_PROCESSED = r'./data/TON_IOT/fulltrain/'
DATA_FILE = "NF-ToN-IoT-v2"


def data_precessing(data_path = DATA_PATH, path_processed = DATA_PATH_PROCESSED, data_name = DATA_FILE):
    x, y = data_load(data_path, data_name)
    data_spilit(x, y, 0.20, 0.10, path_processed)
    

def data_load(path, name):
    path_name = path+name+".csv"
    data = pd.read_csv(path_name)
    
    print("Shape of data  ", data.shape)
    # Delete unnecessasy_columns
    #columns_to_drop = ["src_ip", "src_port", "dst_ip", "dst_port", "src_pkts", "dst_pkts", "type"]
    # Drop the columns
    #data = data.drop(columns=columns_to_drop)
    # we remove duplicated rows. drop_duplicates function only keeps the first occurence and remove the remaining rows................
    #data = data.drop_duplicates()

    """
    
    Encoding scheme for categorical data.....
    Label encoding
    Simple but assume the order of categorical values, for example, low = 0, middle = 1, high = 2, but if there is no order in the data, then this
    label encoding highly impact the resultss

    One-hot encoding
    Do not consider order in the data, however, it increases the data dimensionality. Therefore, it is not suitable for a column/feature, which has multiple categories

    Ordinal encoding
    """
    
    ########################### working categorical data #############################
    object_column_names = data.select_dtypes(include='object').columns.tolist()
    count_unique_values_forEach_category = dict()
    for i in object_column_names:
        count_unique_values_forEach_category[i] = len(data[i].unique())
        print("Column_name "+str(i) + " number of unique values: "+str(len(data[i].unique())))

    print("Remove categorical columns")
    object_column_names = data.select_dtypes(include='object').columns.tolist()
    data = data.drop(columns=object_column_names)

    y = data['Label']
    del data['Label']
    X = data

    print("Number of missing values   ", X.isnull().sum().sum())
    thefy = len(  [i for i in y if i == 1 ]  )
    normal = len(  [i for i in y if i == 0 ]  )
    print("Thefty samples: "+str(thefy))
    print("Normal samples: "+str(normal))
    return X, y

def split_data_train_test(X, y, ratio, validation_ratio):
    X_train, X_test, y_train, y_test = train_test_split( X, y, stratify = y, test_size=ratio, random_state=42, shuffle = True)
    return X_train, X_test, y_train, y_test

def data_spilit(X, y, ratio, validation_ratio, path_processed):
    X_train, X_test, y_train, y_test = train_test_split( X, y, stratify = y, test_size=ratio, random_state=42, shuffle = True)
    
    train_full = X_train.copy()
    train_full["label"] = y_train
    test = X_test.copy()
    test["label"] = y_test
    
    dataName = "TON_IOT"
    
    train_full.to_csv(path_processed+dataName+"_train_full.txt", index = False)
    test.to_csv(path_processed+dataName+"_test.txt", index = False)
    
    X_train_tr, X_test_tr, y_train_tr, y_test_tr = train_test_split( X_train, y_train, stratify = y_train, 
                                                                    test_size=validation_ratio, random_state=42, shuffle = True)
    train_tr = X_train_tr.copy()
    train_tr["label"] = y_train_tr
    test_tr = X_test_tr.copy()
    test_tr["label"] = y_test_tr 

    train_tr.to_csv(path_processed+dataName+"_train_tr.txt", index = False)
    test_tr.to_csv(path_processed+dataName+"_train_valid.txt", index = False)
    
    
if __name__ =="__main__":
    data_precessing()

    
    
    


