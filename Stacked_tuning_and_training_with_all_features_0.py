from tqdm import tqdm
import matplotlib.pyplot as plt
from helper_functions import *
import seaborn as sns
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
from Optimization_files.opt_CatBoost import *
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
from bayes_opt import BayesianOptimization
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import argparse
from sklearn.model_selection import train_test_split
from functools import partial
from bayes_opt import BayesianOptimization
from Optimization_files.opt_XGBoost import *
from algorithms.XGBoost.XGBoost import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--type', type = str, default='meta', help="meta/meta_original/meta_original_statistical")
    args = parser.parse_args()
    meta_features, original_features = load_data_with_meta_and_original_features_without_feature_extraction()
    if args.type == "meta":


        y = meta_features["label"]
        del meta_features["label"]
        X = meta_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_validation = np.array(X_test)
        y_validation = np.array(y_test)
    """
    elif args.type == "meta_original":

        meta_features_trainX = pd.concat([meta_features_trainX.reset_index(drop=True), train_X.reset_index(drop=True)], axis=1)
        meta_features_testX = pd.concat([meta_features_testX.reset_index(drop=True), test_X.reset_index(drop=True)], axis=1)


        X_train = np.array(meta_features_trainX)
        y_train = np.array(meta_features_trainY)

        #X_validation, X_test, y_validation, y_test = train_test_split(meta_features_testX, meta_features_testY, test_size=0.5, shuffle=True, random_state=42)
        
        X_test = np.array(meta_features_testX)
        y_test = np.array(meta_features_testY)

        X_validation = np.array(meta_features_testX)
        y_validation = np.array(meta_features_testY)

    """    


    #print(f"train: {len( np.unique(y_train)  )} validation:  {len( y_validation.unique()  )}  test: {len( y_test.unique()  )}")
    meta_df = pd.DataFrame()

    start = time.time()    

    INTIAL_POINTS = 5
    N_ITERATIONS = 30
    #X_validation, X_test, y_validation, y_test = np.array(X_validation), np.array(X_test), np.array(y_validation), np.array(y_test)
    
    
    opt_func = partial(optimize_xgb_single, X_train = X_train, y_train = y_train, X_valid = X_validation, y_valid= y_validation )
    optimizer = BayesianOptimization(
        f=opt_func,
        pbounds=xgbounds,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
    xgb_optimal_hyperparameter_values = optimizer.max
    
    meta_df["boostOptimalFeatures"] = [xgb_optimal_hyperparameter_values]

    cls = XGBoost(n_estimators=int(xgb_optimal_hyperparameter_values["params"]["n_estimators"]), 
                                        max_depth = int(xgb_optimal_hyperparameter_values["params"]["max_depth"]), 
                                        subsample = float(xgb_optimal_hyperparameter_values["params"]["subsample"]),
                                        colsample_bytree = float(xgb_optimal_hyperparameter_values["params"]["colsample_bytree"]),
                                        gamma = float(xgb_optimal_hyperparameter_values["params"]["gamma"]), 
                                        reg_alpha = float(xgb_optimal_hyperparameter_values["params"]["reg_alpha"]),  
                                        reg_lambda = float(xgb_optimal_hyperparameter_values["params"]["reg_lambda"])
                                        )
    
    cls.fit(X_train, y_train)
    model = cls.return_model()
    print(f"validation f1-score: { round(f1_score(y_validation, model.predict(X_validation), average='weighted'), 4) }")
    print(f"test f1-score: { round(f1_score(y_test, model.predict(X_test), average='weighted'), 4) }")


    leaf_train = model.apply(X_train)
    leaf_val = model.apply(X_validation)
    leaf_test = model.apply(X_test)

    leaf_train = leaf_train.astype(np.int64)
    leaf_val = leaf_val.astype(np.int64)
    leaf_test = leaf_test.astype(np.int64)


    train_ds = TabDataset(X_train, leaf_train, y_train)
    val_ds = TabDataset(X_validation, leaf_val, y_validation)
    test_ds = TabDataset(X_test, leaf_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

    # FUNCTION FOR TUNING.....
    def train_deepgbm(lr, weight_decay, emb_dim, hidden_dim, dropout, epoch):
    # Convert types safely
        emb_dim = int(round(emb_dim))
        hidden_dim = int(round(hidden_dim))
        dropout = float(dropout)
        lr = float(lr)
        weight_decay = float(weight_decay)
        epoch = int(epoch)

        # Initialize model and optimizer
        model = DeepGBMNet(
            n_num=X_train.shape[1],
            y = y_train,
            leaf_train = leaf_train, 
            leaf_val = leaf_val, 
            leaf_test = leaf_test,
            n_trees=leaf_train.shape[1],
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        print(f"Number of epochs: {epoch}")
        for epoch1 in tqdm(range(epoch)):
            
            model.train()
            for xb, leafb, yb in train_loader:
                opt.zero_grad()
                out = model(xb, leafb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()

        # Evaluate on validation set → Macro-F1
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, leafb, yb in val_loader:
                logits = model(xb, leafb)
                preds = logits.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.cpu().numpy())

        val_f1 = f1_score(y_true, y_pred, average='weighted')
        return val_f1
    
    pbounds = {
    "lr": (1e-4, 5e-3),
    "weight_decay": (1e-6, 5e-3),
    "emb_dim": (4, 24),
    "hidden_dim": (16, 64),
    "dropout": (0.0, 0.5),
    "epoch": (5, 30),
    }

    optimizer = BayesianOptimization(
        f=train_deepgbm,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    optimizer.maximize(
        init_points = 5,   # random exploration
        n_iter = 30        # optimization iterations
    )
    
    best_params = optimizer.max
    meta_df["deep_optimal_HPs"] = [best_params]
    print("Best hyperparameters")
    print(best_params)
    meta_df["stacked_tuning_time_"+args.type] = [time.time() - start]
    meta_df["stacked_optimal_hps_boost"] = [ xgb_optimal_hyperparameter_values ]
    meta_df["stacked_optimal_hps_deep"] = [ best_params]
    start = time.time()
    model = DeepGBMNet(
                n_num=X_train.shape[1],
                y = y_train,
                leaf_train = leaf_train, 
                leaf_val = leaf_val, 
                leaf_test = leaf_test,
                n_trees=leaf_train.shape[1],
                emb_dim = int(round(best_params["params"]["emb_dim"])),
                hidden_dim = int(round(best_params["params"]["hidden_dim"])),
                hidden_dim2 = int(round(best_params["params"]["hidden_dim2"])),
                n_layers = int(round(best_params["params"]["n_layers"])),
                dropout = float(best_params["params"]["dropout"])
            )    

    opt = torch.optim.AdamW(model.parameters(), lr = float(best_params["params"]["lr"]), weight_decay = float(best_params["params"]["weight_decay"]))
    criterion = nn.CrossEntropyLoss()

    train_f1 = list()

    valid_f1_score = list()
    valid_accuracy = list()
    valid_recall = list()
    valid_precision = list()
    

    test_f1_score = list()
    test_accuracy = list()
    test_recall = list()
    test_precision = list()
    
    #for epoch1 in range(int(round(best_params["params"]["epoch"]))):
    for epoch1 in range(10):
        print(f"Epoch number: "+str(epoch1))
        model.train()
        for xb, leafb, yb in train_loader:
            opt.zero_grad()
            out = model(xb, leafb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

        # Evaluate on validation set → Macro-F1
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, leafb, yb in train_loader:
                logits = model(xb, leafb)
                preds = logits.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.cpu().numpy())

        val_f1 = f1_score(y_true, y_pred, average='weighted')
        train_f1.append(val_f1)
        print(f"F1-score training data: {round(val_f1, 4)}")

        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, leafb, yb in val_loader:
                logits = model(xb, leafb)
                preds = logits.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.cpu().numpy())

        val_f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"F1-score validation data: {round(val_f1, 4)}")

        
        valid_accuracy.append(accuracy_score(y_true, y_pred))
        valid_recall.append(recall_score(y_true, y_pred, average='weighted'))
        valid_precision.append(precision_score(y_true, y_pred, average='weighted'))
        valid_f1_score.append(val_f1)


        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, leafb, yb in test_loader:
                logits = model(xb, leafb)
                preds = logits.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.cpu().numpy())

        val_f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"F1-score test data: {round(val_f1, 4)}")

        test_accuracy.append(accuracy_score(y_true, y_pred))
        test_recall.append(recall_score(y_true, y_pred, average='weighted'))
        test_precision.append(precision_score(y_true, y_pred, average='weighted'))
        test_f1_score.append(val_f1)


    print("Results on validation data.................")
    print(f"Accuracy: {round(max(valid_accuracy), 4)}, Precision: {round(max(valid_precision), 4)}, Recall: {round(max(valid_recall), 4)}, F1-score: {round(max(valid_f1_score), 4)}  ")

    print("Results on test data.................")
    print(f"Accuracy: {round(max(test_accuracy), 4)}, Precision: {round(max(test_precision), 4)}, Recall: {round(max(test_recall), 4)}, F1-score: {round(max(test_f1_score), 4)}  ")


    meta_df["stacked_training_time_"+args.type] = [time.time() - start]

    # training and testing curves
    train_acc = train_f1
    val_acc   = valid_f1_score
    test_acc   = test_f1_score
    epochs = range(1, len(train_acc) + 1)

    # --- Style configuration (publication ready) ---
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3
    })

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    
    plt.plot(epochs, train_acc, 
            marker='o', linewidth=2.5, markersize=6,
            color='#1f77b4', label='Training F1-score')
    plt.plot(epochs, val_acc, 
            marker='s', linewidth=2.5, markersize=6,
            color='#ff7f0e', label='Validation F1-score')
    plt.plot(epochs, test_acc, 
            marker='s', linewidth=2.5, markersize=6,
            color="#3d44ac", label='Test F1-score')

    # --- Labels and title ---
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')

    # --- Grid, limits, legend ---
    plt.grid(True, linestyle='--', linewidth=0.7)
    plt.ylim(0.9, 1.0)
    plt.xlim(1, len(epochs))
    plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, borderpad=0.8)

    # --- Annotate final values ---
    plt.text(epochs[-1]+0.2, train_acc[-1], f"{train_acc[-1]:.4f}", color='#1f77b4', fontsize=10)
    plt.text(epochs[-1]+0.2, val_acc[-1], f"{val_acc[-1]:.4f}", color='#ff7f0e', fontsize=10)
    plt.text(epochs[-1]+0.2, test_acc[-1], f"{val_acc[-1]:.4f}", color='#3d44ac', fontsize=10)

    # --- Tidy layout ---
    sns.despine()
    plt.tight_layout()
    path = Path("results/multi/")
    
    

    path.mkdir(parents=True, exist_ok=True)
    name = args.type +"features.pdf"
    plt.savefig(path / name, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    meta_df.to_csv(path / "meta_info.txt", sep = "\t", index = False)



