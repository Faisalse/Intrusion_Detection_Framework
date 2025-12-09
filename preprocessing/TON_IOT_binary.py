# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 21:29:36 2023

@author: shefai
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_PATH = r'./data/raw/'
DATA_PATH_PROCESSED = r'./data/TON_IOT/multiclassification/fulltrain/'
DATA_FILE = "ToN_IoT_train_test_network"


def data_precessing(data_path = DATA_PATH, path_processed = DATA_PATH_PROCESSED, data_name = DATA_FILE):
    x, y = data_load(data_path, data_name)
    data_spilit(x, y, 0.20, 0.10, path_processed)
    

def data_load(path, name):
    path_name = path+name+".csv"
    data = pd.read_csv(path_name)
    
    print("Shape of data  ", data.shape)
    # Delete unnecessasy_columns, which have direct correlation with label column
    columns_to_drop = ["src_ip", "src_port", "dst_ip", "dst_port", "src_pkts", "dst_pkts", "label"]
    # Drop the columns
    data = data.drop(columns=columns_to_drop)
    # we remove duplicated rows. drop_duplicates function only keeps the first occurence and remove the remaining rows................
    data = data.drop_duplicates()

    data.replace("-", np.nan, inplace=True) # Replace "-" values with NAN values.....
    
    """
    
    Encoding scheme for categorical data.....
    Label encoding
    Simple but assume the order of categorical values, for example, low = 0, middle = 1, high = 2, but if there is no order in the data, then this
    label encoding highly impact the resultss

    One-hot encoding
    Do not consider order in the data, however, it increases the data dimensionality. Therefore, it is not suitable for a column/feature, which has multiple categories

    
    """

    columnswithHighestMissingValues = list()
    for i in data.columns:
        print("Column name: "+str(i)+" Number of null values: "+str(data[i].isnull().sum()))
        if data[i].isnull().sum() > len(data) / 2:
            columnswithHighestMissingValues.append(i)

    data = data.drop(columns=columnswithHighestMissingValues)
    
    multi_class_label = data["type"]
    del data["type"]
    ########################### working categorical data #############################
    object_column_names = data.select_dtypes(include='object').columns.tolist()
    count_unique_values_forEach_category = dict()
    for i in object_column_names:
        count_unique_values_forEach_category[i] = len(data[i].unique())
    
    
    data_cate = data[count_unique_values_forEach_category]
    data_num = data.drop(columns=count_unique_values_forEach_category)
    
    proto_encoded = pd.get_dummies(data_cate['proto'], dummy_na=False)
    service_encoded = pd.get_dummies(data_cate['service'], dummy_na=False)
    conn_state_encoded = pd.get_dummies(data_cate['conn_state'], dummy_na=False)
        
    category_df = pd.concat([proto_encoded, service_encoded, conn_state_encoded], axis=1)
    dict_cal_calsses = dict()
    for count_ in multi_class_label:
        if count_ in dict_cal_calsses:
           dict_cal_calsses[count_] +=1
        else:
             dict_cal_calsses[count_] =1
    print("Number of instances/records for each class")
    for key, values in dict_cal_calsses.items():
        print(key +": "+ str(values) + "("+ str( round((values/ len(multi_class_label)  * 100), 2)  ) + "%)")
    le = LabelEncoder()
    multi_class_label = le.fit_transform(multi_class_label)
    scaler = MinMaxScaler()
    data_num = pd.DataFrame(scaler.fit_transform(data_num), columns=data_num.columns)
    combine_num_cat = pd.concat([data_num.reset_index(drop=True), category_df.reset_index(drop=True)], axis=1)
    combine_num_cat["label"] = multi_class_label
    print(combine_num_cat.shape)
    combine_num_cat = combine_num_cat.dropna()
    print(combine_num_cat.shape)
    
    y = combine_num_cat['label']
    del combine_num_cat['label']
    X = combine_num_cat
    print("Number of missing values   ", X.isnull().sum().sum())
    y = [0 if y[i] == 0 else 1 for i in range(len(y))]
    y = pd.Series(y)
    return X, y


def split_data_train_test(X, y, ratio = 0.20, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split( X, y, stratify = y, test_size=ratio, random_state=random_state, shuffle = True)
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

    
    
    


