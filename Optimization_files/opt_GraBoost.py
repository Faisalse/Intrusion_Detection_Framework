from algorithms.GBC.GraBC import * 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

gbcbounds = {
    "n_estimators": (50, 500),           # Integer
    "max_depth": (3, 15),                  # Integer
    "learning_rate": (0.01, 0.3),          # Float
    "subsample": (0.5, 1.0),
    "min_samples_leaf": (5, 50)
                                  # Float
}


def optimize_gbc(n_estimators, max_depth, learning_rate, subsample, min_samples_leaf, X,y, cv):
    
    model = GBC(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate = float(learning_rate),
        subsample = float(subsample),
        min_samples_leaf = int(min_samples_leaf)
        )
    
    f1_macro = make_scorer(f1_score, average='weighted')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores.mean()




