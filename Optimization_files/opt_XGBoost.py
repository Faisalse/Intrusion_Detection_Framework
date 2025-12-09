from algorithms.XGBoost.XGBoost import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

xgbounds = {
    "n_estimators": (50, 500),           # Integer
    "max_depth": (3, 15),                  # Integer
    "learning_rate": (0.01, 0.3),          # Float
    "subsample": (0.5, 1.0),               # Float
    "colsample_bytree": (0.5, 1.0),        # Float
    "gamma": (0, 10),                      # Float
    "reg_alpha": (0, 10),                  # Float
    "reg_lambda": (0, 10)                  # Float
}


def optimize_xgb(n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
                   gamma, reg_alpha, reg_lambda,
                   X, y, cv):
    
    model = XGBoost(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )
    
    f1_macro = make_scorer(f1_score, average='weighted')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores.mean()

def optimize_xgb_single(n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
                   gamma, reg_alpha, reg_lambda,
                   X_train, y_train, X_valid, y_valid):
    
    model = XGBoost(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )
    
    model.fit(X_train, y_train)
    # Predictions
    y_pred = model.predict(X_valid)
    # Weighted F1 score
    weighted_f1 = f1_score(y_valid, y_pred, average='weighted')
    return weighted_f1



