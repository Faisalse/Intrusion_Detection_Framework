# Optimization_files/opt_RandomForest.py

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

# adjust the import path to where you saved the wrapper
from algorithms.RF.RF import *  # e.g., algorithms/RandomForest/RF.py


# Bounds for Bayesian Optimization
rfbounds = {
    "n_estimators": (10, 500),
    "max_depth": (1, 100),            # wrapper expects int or None (we'll pass int)
    "min_samples_split": (2, 50),
    "min_samples_leaf": (1, 30),
}


def optimize_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf, X, y, cv):
    """
    Returns mean CV F1-macro for the RF wrapper.
    """

    model = RF(
        n_estimators=int(round(n_estimators)),
        max_depth=int(round(max_depth)),
        min_samples_split=int(round(min_samples_split)),
        min_samples_leaf=int(round(min_samples_leaf)),
    )

    f1_macro = make_scorer(f1_score, average="macro")
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores.mean()
