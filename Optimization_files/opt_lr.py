from sklearn.model_selection import cross_val_score
from algorithms.LR.lr import *
from sklearn.metrics import make_scorer, f1_score

solver_map = {0: 'lbfgs', 1: 'saga'}
penalty_map = {0: 'l2', 1: 'none'}



lrpbounds = {
    "solver": (0, 1),          # 0: 'lbfgs', 1: 'saga'
    "penalty": (0, 1),         # 0: 'l2', 1: 'none'
    "C": (0.001, 1000)         # Inverse regularization strength
}





from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

def optimize_lr(solver, penalty, C, X, y, cv):
    solver_str = solver_map[int(round(solver))]
    penalty_str = penalty_map[int(round(penalty))]
    
    # Compatibility check
    try:
        model = LR(
            solver=solver_str,
            penalty=penalty_str,
            C=C
        )

        f1_macro = make_scorer(f1_score, average='macro')
        scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)

        return scores.mean()
    
    except ValueError:
        return 0  # Return worst score if configuration fails






