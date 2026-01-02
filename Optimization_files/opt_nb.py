from sklearn.model_selection import cross_val_score
from algorithms.NB.nb import *
from sklearn.metrics import make_scorer, f1_score



nb_pbounds = {
    "var_smoothing": (1e-12, 1e-2)
}

def optimize_nb(var_smoothing, X, y, cv):
    model = NB(var_smoothing=var_smoothing)

    f1_macro = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores.mean()







