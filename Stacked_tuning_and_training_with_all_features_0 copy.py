

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from helper_functions import *
from sklearn.model_selection import train_test_split
import numpy as np


class NumericDataset(Dataset):
    def __init__(self, X_num, y):
        self.X_num = X_num      # [N, n_num] float32
        self.y = y              # [N] int64 (class index)

    def __len__(self):
        return self.X_num.size(0)

    def __getitem__(self, idx):
        return self.X_num[idx], self.y[idx]


meta_features, original_features = load_data_with_meta_and_original_features_without_feature_extraction()
    

y = meta_features["label"]
del meta_features["label"]
X = meta_features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

        

# ---- fake example data (replace with your own) ----
N = X_train.shape[0]          # samples
n_num = X_train.shape[1]        # numeric features
n_classes = len(np.unique(y_train))     # multi-class

X_num =  torch.from_numpy(X_train)
y =  torch.from_numpy(y_train)

dataset = NumericDataset(X_num, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)



class FTTransformerNumeric(nn.Module):
    def __init__(
        self,
        n_num,
        n_classes,
        d_token=32,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.n_num = n_num
        self.d_token = d_token

        # One learnable embedding vector per numeric feature
        # shape: [1, n_num, d_token]
        self.feature_embedding = nn.Parameter(
            torch.randn(1, n_num, d_token) * 0.02
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_token)
        )

        # Positional / feature-wise embeddings for (CLS + n_num features)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1 + n_num, d_token)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=False,   # expects [S, B, E]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Classification head on top of CLS token
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, n_classes)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x_num):
        """
        x_num: [B, n_num] float
        """
        B = x_num.size(0)

        # Expand feature embeddings to batch: [B, n_num, d_token]
        feat_emb = self.feature_embedding.expand(B, -1, -1)

        # Each numeric value scales its feature embedding vector:
        # x_num: [B, n_num] -> [B, n_num, 1]
        tokens = feat_emb * x_num.unsqueeze(-1)  # [B, n_num, d_token]

        # CLS token: [B, 1, d_token]
        cls_token = self.cls_token.expand(B, -1, -1)

        # Concatenate CLS + feature tokens: [B, 1 + n_num, d_token]
        tokens = torch.cat([cls_token, tokens], dim=1)

        # Add positional/feature-wise embeddings
        tokens = tokens + self.pos_embedding   # [B, 1+n_num, d_token]

        # Transformer expects [S, B, E]
        tokens = tokens.transpose(0, 1)        # [1+n_num, B, d_token]

        out = self.transformer(tokens)         # [1+n_num, B, d_token]

        # CLS output (first token): [B, d_token]
        cls_out = out[0]

        logits = self.head(cls_out)            # [B, n_classes]
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FTTransformerNumeric(
    n_num=n_num,
    n_classes=n_classes,
    d_token=32,     # try 32 or 64
    n_heads=4,
    n_layers=3,
    dropout=0.1,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

n_epochs = 20

for epoch in range(1, n_epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_num_batch, y_batch in loader:
        x_num_batch = x_num_batch.to(device).float()
        y_batch = y_batch.to(device).long()

        optimizer.zero_grad()
        logits = model(x_num_batch)          # [B, n_classes]
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y_batch.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch:02d} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f}")
