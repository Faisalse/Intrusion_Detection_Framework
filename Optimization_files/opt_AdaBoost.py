from algorithms.AdaBoost.AdaBoost import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

adaboost_search_space = {
    "n_estimators": (50, 800),        # int
    "learning_rate": (1e-3, 2.0)
}


def optimize_adaboost(n_estimators, learning_rate,
                      X, y, cv):
    model = AdaBoost(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate)
    )
    f1_macro = make_scorer(f1_score, average='weighted')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores.mean()




