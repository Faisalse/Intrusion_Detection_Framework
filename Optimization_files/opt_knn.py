from sklearn.model_selection import cross_val_score
from algorithms.kNN.kNN import *
from sklearn.metrics import make_scorer, f1_score

weights_map = {0: "uniform", 1: "distance"}

knnpbounds = {
    "n_neighbors": (1, 100),     # Integer range
    "weights": (0, 1)           # 0: uniform, 1: distance
}


def optimize_knn(n_neighbors, weights, X, y, cv):
    weight_str = weights_map[int(round(weights))]
    model = kNN(
        n_neighbors=int(n_neighbors),
        weights=weight_str
    )

    f1_macro = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)

    return scores.mean()







