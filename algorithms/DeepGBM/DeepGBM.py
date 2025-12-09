# DeepGBM-style Multi-Class Example with Learning Curves
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
import torch.nn as nn
import pandas as pd
import numpy as np
import torch

# original features
train_X = pd.read_csv("data/original_features_k_fold_training.csv", sep = ";")
train_Y = train_X["label"]
del train_X["label"]

test_X = pd.read_csv("data/original_features_test.csv", sep = ";")
yest_Y = test_X["label"]
del test_X["label"]

meta_features_trainX = pd.read_csv("data/training.csv", sep = ";")
meta_features_trainY = meta_features_trainX["label"]
del meta_features_trainX["label"]

meta_features_testX = pd.read_csv("data/testing.csv", sep = ";")
meta_features_testY = meta_features_testX["label"]

del meta_features_testX["label"]


meta_features_trainX = pd.concat([meta_features_trainX.reset_index(drop=True), train_X.reset_index(drop=True)], axis=1)
meta_features_testX = pd.concat([meta_features_testX.reset_index(drop=True), test_X.reset_index(drop=True)], axis=1)


X_train = np.array(meta_features_trainX)
X_val = np.array(meta_features_testX)

y_train = np.array(meta_features_trainY)
y_val = np.array(meta_features_testY)

X = X_train
y = y_train

cat_model = CatBoostClassifier(
    iterations=50,
    depth=3,
    learning_rate=0.1,
    verbose=0
)

cat_model.fit(X_train, y_train)
# Get leaf indices for each sample and tree
leaf_train = cat_model.calc_leaf_indexes(X_train)
leaf_val = cat_model.calc_leaf_indexes(X_val)

leaf_train = leaf_train.astype(np.int64)
leaf_val = leaf_val.astype(np.int64)

print("Leaf index matrix shape:", leaf_train.shape)
n_trees = leaf_train.shape[1]
n_classes = len(np.unique(y))


class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X, leaf_idx, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.leaf_idx = torch.tensor(leaf_idx, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.leaf_idx[i], self.y[i]

train_ds = TabDataset(X_train, leaf_train, y_train)
val_ds = TabDataset(X_val, leaf_val, y_val)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)

# Neural model.....
class DeepGBMNet(nn.Module):
        def __init__(self, n_num, n_trees, n_classes, emb_dim, hidden_dim, dropout):
            super().__init__()
            leaf_all = np.vstack([leaf_train, leaf_val])
            max_leaf_ids = leaf_all.max(axis=0)  # max index per tree

            self.leaf_embs = nn.ModuleList([
                nn.Embedding(int(max_leaf_ids[t]) + 1, emb_dim)
                for t in range(n_trees)
            ])

            input_dim = n_num + n_trees * emb_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)
            )

        def forward(self, x_num, leaf_idx):
            leaf_vecs = [emb(leaf_idx[:, t]) for t, emb in enumerate(self.leaf_embs)]
            leaf_concat = torch.cat(leaf_vecs, dim=1)
            out = torch.cat([x_num, leaf_concat], dim=1)
            return self.net(out)


# Function to optimize
def tune_deepgbm(lr, emb_dim, hidden_dim, dropout, epoch):
    emb_dim = int(emb_dim)
    hidden_dim = int(hidden_dim)
    dropout = float(dropout)
    epoch_ = int(epoch)
    
    model = DeepGBMNet(X.shape[1], leaf_train.shape[1], len(np.unique(y)), emb_dim, hidden_dim, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train for few epochs (for BO speed)
    for epoch in range(epoch_):
        model.train()
        for xb, leafb, yb in train_loader:
            opt.zero_grad()
            out = model(xb, leafb)
            loss = criterion(out, yb)
            loss.backward(); opt.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, leafb, yb in val_loader:
            logits = model(xb, leafb)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.cpu().numpy())

    val_f1 = f1_score(y_true, y_pred, average='weighted')
    return np.round(val_f1, decimals = 4)

# Function to train.............S
def train_deepgbm(lr, emb_dim, hidden_dim, dropout, epoch):
    emb_dim = int(emb_dim)
    hidden_dim = int(hidden_dim)
    dropout = float(dropout)
    epoch_ = int(epoch)
    
    model = DeepGBMNet(X.shape[1], leaf_train.shape[1], len(np.unique(y)), emb_dim, hidden_dim, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    

    train_accu_ = []
    vali_accu_ = []

    train_loss_ = []
    vali_loss_ = []

    for epoch in range(epoch_):
        print(f"\nEpoch {epoch+1}/{epoch_}")
        model.train()
        running_loss = 0.0
        total_batches = 0

        # ---------------- TRAINING LOOP ----------------
        for xb, leafb, yb in train_loader:
            opt.zero_grad()
            out = model(xb, leafb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            total_batches += 1

        # Mean training loss for this epoch
        epoch_train_loss = running_loss / total_batches
        train_loss_.append(epoch_train_loss)

        # Evaluate training F1
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, leafb, yb in train_loader:
                logits = model(xb, leafb)
                preds = logits.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.cpu().numpy())
        train_f1 = f1_score(y_true, y_pred, average='weighted')
        train_accu_.append(train_f1)

        # ---------------- VALIDATION LOOP ----------------
        val_running_loss = 0.0
        val_batches = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for xb, leafb, yb in val_loader:
                logits = model(xb, leafb)
                val_loss = criterion(logits, yb)
                val_running_loss += val_loss.item()
                val_batches += 1

                preds = logits.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.cpu().numpy())

        # Mean validation loss
        epoch_val_loss = val_running_loss / val_batches
        vali_loss_.append(epoch_val_loss)

        # Validation F1
        val_f1 = f1_score(y_true, y_pred, average='weighted')
        vali_accu_.append(val_f1)

    acc =  np.round(accuracy_score(y_true, y_pred), decimals = 4)
    pre =  np.round(precision_score(y_true, y_pred, average='weighted'), decimals = 4)
    rec =  np.round(recall_score(y_true, y_pred, average='weighted'), decimals = 4)
    f1_ =  np.round(f1_score(y_true, y_pred, average='weighted'), decimals = 4)
    return train_accu_, vali_accu_, train_loss_, vali_loss_,  acc, pre, rec, f1_

        

# hyperparameter ranges......................
pbounds = {
    'lr': (1e-4, 1e-2),          # learning rate
    'emb_dim': (16, 32),          # leaf embedding dim
    'hidden_dim': (64, 128),     # hidden layer size
    'dropout': (0.0, 0.5), # dropout rate
    'epoch': (10, 30),               
}

# Baysian optimization
optimizer = BayesianOptimization(
    f=tune_deepgbm,   # function to maximize
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

