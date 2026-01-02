from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from accuracy.multi_accuracy import *
import pandas as pd
import time
from helfer_functions.import_models import *
from functools import partial
from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization
# upload optimization files...
from Optimization_files.opt_AdaBoost import *
from Optimization_files.opt_CatBoost import *
from Optimization_files.opt_GraBoost import *
from Optimization_files.opt_XGBoost import *
from Optimization_files.opt_LightGBM import *

from Optimization_files.opt_DT import *
from Optimization_files.opt_lr import *
from Optimization_files.opt_nb import *
from Optimization_files.opt_knn import *
from Optimization_files.opt_rf import *


################# IMPORT ACCURACY MEASURES ##################
accuracy_objects_dict = dict()
accuracy_objects_dict["Accuracy"] = Acc()
accuracy_objects_dict["Precision"] = Precision()
accuracy_objects_dict["Recall"] = Recall()
accuracy_objects_dict["F1_score"] = F1_score()
models_object_dict = dict()



DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/multi/all_features/")
path.mkdir(parents=True, exist_ok=True)

X, y = data_load(DATA_PATH, data_name)
#X_train, X_test, y_train, y_test = split_data_train_test(X, y)

INTIAL_POINTS = 1
N_ITERATIONS = 1
cv_strategy = StratifiedKFold(n_splits=5)

meta_df = pd.DataFrame()


############################# RF 
start = time.time()
opt_func = partial(optimize_lr, X = X, y = y, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=lrpbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
LR_optimal_hyperparameter_values = optimizer.max
meta_df["LR_tuning_time"] = [time.time() - start]
meta_df["LR_HP"] = [LR_optimal_hyperparameter_values]

############################# LR 
start = time.time()
opt_func = partial(optimize_lr, X = X, y = y, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=lrpbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
LR_optimal_hyperparameter_values = optimizer.max
meta_df["LR_tuning_time"] = [time.time() - start]
meta_df["LR_HP"] = [LR_optimal_hyperparameter_values]

############################# NB 
start = time.time()
opt_func = partial(optimize_nb, X = X, y = y, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=nb_pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
nb_optimal_hyperparameter_values = optimizer.max
meta_df["NB_tuning_time"] = [time.time() - start]
meta_df["NB_HP"] = [nb_optimal_hyperparameter_values]


############################# DT 
start = time.time()
opt_func = partial(optimize_dtree, X = X, y = y, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=dtbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
dt_optimal_hyperparameter_values = optimizer.max
meta_df["DT_tuning_time"] = [time.time() - start]
meta_df["DT_HP"] = [dt_optimal_hyperparameter_values]

############################# KNN
start = time.time()
opt_func = partial(optimize_knn, X = X, y = y, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=knnpbounds ,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
knn_optimal_hyperparameter_values = optimizer.max
meta_df["KNN_tuning_time"] = [time.time() - start]
meta_df["kNN_HP"] = [knn_optimal_hyperparameter_values]





models_object_dict["LR"] = LR(solver = solver_map[round(LR_optimal_hyperparameter_values["params"]["solver"])], 
                              penalty = penalty_map[round(LR_optimal_hyperparameter_values["params"]["penalty"])], 
                              C = float(LR_optimal_hyperparameter_values["params"]["C"]))

models_object_dict["NB"] = NB(var_smoothing = float(nb_optimal_hyperparameter_values["params"]["var_smoothing"]))

models_object_dict["DTree"] = DTree(criterion = criterion_map[round(dt_optimal_hyperparameter_values["params"]["criterion"])],
                                    splitter = splitter_map[round(dt_optimal_hyperparameter_values["params"]["splitter"])],
                                    max_depth = round(dt_optimal_hyperparameter_values["params"]["max_depth"]))

models_object_dict["KNN"] = kNN(n_neighbors = round(knn_optimal_hyperparameter_values["params"]["n_neighbors"]), 
                                weights  = weights_map[round(knn_optimal_hyperparameter_values["params"]["weights"])])

"""
#######################        optimization for adaBoost
start = time.time()
opt_func = partial(optimize_adaboost, X = X, y = y, cv = cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=adaboost_search_space,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
ada_optimal_hyperparameter_values = optimizer.max

meta_df["adaboost_tuning_time"] = [time.time() - start]
meta_df["adaboost_HP"] = [ada_optimal_hyperparameter_values]

#######################        optimization for CatB ########################################

opt_func = partial(optimize_catb, X = X, y = y, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=catbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
cat_optimal_hyperparameter_values = optimizer.max
meta_df["catboost_tuning_time"] = [time.time() - start]
meta_df["catboost_HP"] = [cat_optimal_hyperparameter_values]

#################################################################################################
#######################        optimization for GraB ########################################

start = time.time()
opt_func = partial(optimize_gbc, X = X, y = y, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=gbcbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
gbc_optimal_hyperparameter_values = optimizer.max
meta_df["graboost_tuning_time"] = [time.time() - start]
meta_df["graboost_HP"] = [gbc_optimal_hyperparameter_values]

#######################        optimization for LightBoost ########################################
start = time.time()
opt_func = partial(optimize_lightb, X = X, y = y, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=lightbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
light_optimal_hyperparameter_values = optimizer.max
meta_df["lightboost_tuning_time"] = [time.time() - start]
meta_df["lightboost_HP"] = [light_optimal_hyperparameter_values]
#######################        optimization for XGBoost ########################################

start = time.time()
opt_func = partial(optimize_xgb, X = X, y = y, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=xgbounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
xgb_optimal_hyperparameter_values = optimizer.max
meta_df["xgboost_tuning_time"] = [time.time() - start]
meta_df["xgboost_HP"] = [xgb_optimal_hyperparameter_values]

################################### END optimization ###########################################





# import models --- > baselines

models_object_dict = dict()

models_object_dict["AdaBoost"] = AdaBoost(n_estimators= int(ada_optimal_hyperparameter_values["params"]["n_estimators"]), 
                                          learning_rate = ada_optimal_hyperparameter_values["params"]["learning_rate"])

models_object_dict["XGBoost"] = XGBoost(n_estimators=int(xgb_optimal_hyperparameter_values["params"]["n_estimators"]), 
                                        max_depth = int(xgb_optimal_hyperparameter_values["params"]["max_depth"]), 
                                        subsample = float(xgb_optimal_hyperparameter_values["params"]["subsample"]),
                                        colsample_bytree = float(xgb_optimal_hyperparameter_values["params"]["colsample_bytree"]),
                                        gamma = float(xgb_optimal_hyperparameter_values["params"]["gamma"]), 
                                        reg_alpha = float(xgb_optimal_hyperparameter_values["params"]["reg_alpha"]),  
                                        reg_lambda = float(xgb_optimal_hyperparameter_values["params"]["reg_lambda"]))

models_object_dict["CatBoost"] = CatB(iterations= round(cat_optimal_hyperparameter_values["params"]["iterations"]), 
                                      learning_rate = cat_optimal_hyperparameter_values["params"]["learning_rate"], 
                                      depth = round(cat_optimal_hyperparameter_values["params"]["depth"]), 
                                      l2_leaf_reg = cat_optimal_hyperparameter_values["params"]["l2_leaf_reg"])



models_object_dict["LightBoost"] = LightB(n_estimators = int(light_optimal_hyperparameter_values["params"]["n_estimators"]), 
                                          learning_rate = float(light_optimal_hyperparameter_values["params"]["learning_rate"]), 
                                          max_depth = int(light_optimal_hyperparameter_values["params"]["max_depth"]), 
                                          num_leaves = int(light_optimal_hyperparameter_values["params"]["num_leaves"]), 
                                          min_child_samples = int(light_optimal_hyperparameter_values["params"]["min_child_samples"]))



models_object_dict["GraBoost"] = GBC(n_estimators= int(gbc_optimal_hyperparameter_values["params"]["n_estimators"]), learning_rate = float(gbc_optimal_hyperparameter_values["params"]["learning_rate"]),
                                     max_depth = int(gbc_optimal_hyperparameter_values["params"]["max_depth"]), min_samples_leaf = int(gbc_optimal_hyperparameter_values["params"]["min_samples_leaf"]), subsample = float(gbc_optimal_hyperparameter_values["params"]["subsample"]))


"""
meta_features_X, meta_features_Y, original_features_k_fold, result_dataframe = k_fold_return_meta_features(X, y, 
                                                                                                           models_object_dict, accuracy_objects_dict)

# Assign column names.............
label = original_features_k_fold["label"]
del original_features_k_fold["label"]
original_features_k_fold.columns = X.columns
original_features_k_fold["label"] = label
meta_features_X["label"] = meta_features_Y


temp = dict()
for key in result_dataframe.keys():
    split_key = key.split("_")
    temp_nested_dict = result_dataframe[key]
    temp_nested_dict["Fold"] = str(split_key[2]) 
    temp[str(split_key[0])+"_"+str(split_key[2])  ] = temp_nested_dict


results_ = pd.DataFrame(temp).transpose()
results_.to_csv(path / "results_baselines_cross_validation.txt", sep = "\t")
meta_df.to_csv(path / "optimal_hyperparameter_values.txt", sep = "\t")

path = Path("data/save_meta_features_without_feature_extractions")
original_features_k_fold.to_csv(path / "meta_features_k_fold_cross_validation_X_Y.csv", sep = ";", index = False)
original_features_k_fold.to_csv(path / "original_features_k_fold_cross_validation_X_Y.csv", sep = ";", index = False)


