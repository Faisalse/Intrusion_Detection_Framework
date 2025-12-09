
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from pathlib import Path
import random
import pandas as pd
import numpy as np
import pandas as pd
import time
import pandas as pd
import numpy as np
import torch
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X, leaf_idx, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.leaf_idx = torch.tensor(leaf_idx, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.leaf_idx[i], self.y[i]


class DeepGBMNet(nn.Module):
        def __init__(self, n_num, y, leaf_train, leaf_val, leaf_test, n_trees, emb_dim, hidden_dim, dropout):
            super().__init__()
            # Compute leaf range across train/val to avoid out-of-range errors
            leaf_all = np.vstack([leaf_train, leaf_val, leaf_test])
            max_leaf_ids = leaf_all.max(axis=0)

            # Embedding layers for each tree
            self.leaf_embs = nn.ModuleList([
                nn.Embedding(int(max_leaf_ids[t]) + 1, emb_dim)
                for t in range(n_trees)
            ])

            # MLP backbone
            input_dim = n_num + n_trees * emb_dim
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            layers += [nn.Linear(hidden_dim, len(np.unique(y)))]
            self.net = nn.Sequential(*layers)

        def forward(self, x_num, leaf_idx):
            leaf_vecs = []
            for t, emb in enumerate(self.leaf_embs):
                idx = leaf_idx[:, t].clamp(0, emb.num_embeddings - 1)
                leaf_vecs.append(emb(idx))
            leaf_concat = torch.cat(leaf_vecs, dim=1)
            out = torch.cat([x_num, leaf_concat], dim=1)
            return self.net(out)


def load_data_with_meta_and_original_features_without_feature_extraction():
    path = Path("data/save_meta_features_without_feature_extractions")


    meta_features = pd.read_csv(path/"meta_features_k_fold_cross_validation_X_Y.csv", sep = ";")

    original_features = pd.read_csv(path/"original_features_k_fold_cross_validation_X_Y.csv", sep = ";")
    
    return meta_features, original_features


def apply_random_scaling_df_global(df, ratio=0.1, scale=0.1, seed=43):
    """
    Randomly selects a ratio of values in the DataFrame and scales them by a given factor.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        ratio (float): Fraction of total elements to scale (default is 0.1 for 10%).
        scale (float): Multiplicative scale factor (default is 0.1).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Modified DataFrame with scaled values.
    """
    if seed is not None:
        np.random.seed(seed)

    df_copy = df.copy()
    total_elements = df_copy.size
    num_to_scale = int(total_elements * ratio)

    # Generate random (row, column) index pairs
    row_indices = np.random.randint(0, df_copy.shape[0], num_to_scale)
    col_indices = np.random.randint(0, df_copy.shape[1], num_to_scale)

    for row, col in zip(row_indices, col_indices):
        col_name = df_copy.columns[col]
        df_copy.iat[row, col] *= scale

    return df_copy


def apply_random_scaling_df_local(df, ratio=0.1, seed=43):
    
    if seed is not None:
        np.random.seed(seed)
    cols = df.columns
    df_copy = df.copy()
    continuous_cols = df_copy.select_dtypes(include=['float']).columns.tolist()

    selected_col = continuous_cols[random.choice([i for i in range(len(continuous_cols))])]
    df_copy[selected_col] = df_copy[selected_col] * ratio
    return df_copy


def k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                accuracy_objects_dict, n_splits = 5, random_state = 42):
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    

    meta_features = dict()
    for key in models_object_dict.keys():
        meta_features[key] = list()
    meta_y = list()

    # We are using out of fold strategy to avoid data leakage issue............
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


    original_features = pd.DataFrame()
    for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        X = X_train[test_index]
        y = y_train[test_index]
        df = pd.DataFrame(X)
        df["label"] = y
        original_features = pd.concat([original_features, df], ignore_index=True)

    result_dataframe = dict()
    for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        for key in models_object_dict.keys():
            print(f"*********************** {key} ***********************")
            models_object_dict[key].fit(X_train[train_index], y_train[train_index])

            y_predict = models_object_dict[key].predict(X_train[test_index])
            y_predict_prob = models_object_dict[key].predict_proba(X_train[test_index])

            column_names = [i for i in range(y_predict_prob.shape[1])]
            temp_df = pd.DataFrame(y_predict_prob, columns= column_names)
            meta_features[key].append(temp_df)
            
            # print accuracy values on out of fold
            temp = dict()
            for acc_key, acc_object in accuracy_objects_dict.items():
                acc_object.compute(y_predict, y_train[test_index])
                temp[acc_key] = round(acc_object.result()[1], 4)
                print(acc_object.result())
                
            result_dataframe[str(key) +"_fold_"+str(fold)] = temp

        meta_y.append(y_train[test_index])   
    
    

    meta_features_df = pd.DataFrame()
    for key, value in meta_features.items():
        temp = pd.concat(value, axis=0, ignore_index=True)
        meta_features_df = pd.concat([meta_features_df, temp], axis= 1, ignore_index=True)

    # make meta features frame.........................................................
    new_column_names = [i for i in   range(  meta_features_df.shape[1] )]  
    meta_features_df.columns = new_column_names
    meta_features_y = [item   for sublist in meta_y for item in list(sublist)]

    return meta_features_df, meta_features_y, original_features, result_dataframe



def return_metafeatures_for_single_splits(X_train, y_train, X_test, y_test, models_object_dict, accuracy_objects_dict):
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    result_dataframe = dict()

    meta_features = dict()
    for key in models_object_dict.keys():
        meta_features[key] = list()
    # results with full data
    for key in models_object_dict.keys():
        print(f"*********************** Results with full data: {key} ***********************")
        start = time.time()
        models_object_dict[key].fit(X_train, y_train)

        
        y_predict = models_object_dict[key].predict(X_test)
        y_predict_prob = models_object_dict[key].predict_proba(X_test)

        column_names = [i for i in range(y_predict_prob.shape[1])]
        temp_df = pd.DataFrame(y_predict_prob, columns= column_names)
        meta_features[key].append(temp_df)
            
            # print accuracy values on out of fold
        temp = dict()
        for acc_key, acc_object in accuracy_objects_dict.items():
            acc_object.compute(y_predict, y_test)
            temp[acc_key] = round(acc_object.result()[1], 4)
            print(acc_object.result())
            
        end = time.time()
        temp["time"] = end - start
        result_dataframe[str(key)] = temp
    
    meta_features_df = pd.DataFrame()
    for key, value in meta_features.items():
        meta_features_df = pd.concat([meta_features_df, value[0]], axis= 1, ignore_index=True)

    # make meta features frame.........................................................
    new_column_names = [i for i in   range(  meta_features_df.shape[1] )]  
    meta_features_df.columns = new_column_names
    return meta_features_df, y_test, result_dataframe

def stacked_model_object_dictAND_accuracy_dict(X_train, y_train, X_test, y_test, models_object_dict, accuracy_objects_dict):

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    
    result_dataframe = dict()
    for key in models_object_dict.keys():
        print(f"*********************** Results with full data: {key} ***********************")
        start = time.time()
        models_object_dict[key].fit(X_train, y_train)
        

        y_predict = models_object_dict[key].predict(X_test)
        y_predict_prob = models_object_dict[key].predict_proba(X_test)

          
        # print accuracy values on out of fold
        temp = dict()
        for acc_key, acc_object in accuracy_objects_dict.items():
            acc_object.compute(y_predict, y_test)
            print(acc_object.result())
            temp[acc_key] = round(acc_object.result()[1], 4)
        
        end = time.time()
        temp["time"] = end - start
        result_dataframe[key] = temp
        

    return result_dataframe