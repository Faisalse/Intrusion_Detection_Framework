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


DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/multi/all_features/")
path.mkdir(parents=True, exist_ok=True)

X, y = data_load(DATA_PATH, data_name)
#X_train, X_test, y_train, y_test = split_data_train_test(X, y)

INTIAL_POINTS = 5
N_ITERATIONS = 50
cv_strategy = StratifiedKFold(n_splits=5)

meta_df = pd.DataFrame()
start = time.time()
#######################        optimization for adaBoost
opt_func = partial(optimize_adaboost, X= X_train, y=y_train, cv=cv_strategy)
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
#################################################################################################
#######################        optimization for CatB ########################################
start = time.time()
opt_func = partial(optimize_catb, X= X_train, y=y_train, cv=cv_strategy)
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
opt_func = partial(optimize_gbc, X= X_train, y=y_train, cv=cv_strategy)
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
opt_func = partial(optimize_lightb, X= X_train, y=y_train, cv=cv_strategy)
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
opt_func = partial(optimize_xgb, X= X_train, y=y_train, cv=cv_strategy)
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



accuracy_objects_dict = dict()
accuracy_objects_dict["Accuracy"] = Acc()
accuracy_objects_dict["Precision"] = Precision()
accuracy_objects_dict["Recall"] = Recall()
accuracy_objects_dict["F1_score"] = F1_score()

# import models --- > baselines
models_object_dict = dict()
models_object_dict["AdaBoost"] = AdaBoost(n_estimators= int(ada_optimal_hyperparameter_values["params"]["n_estimators"]), 
                                          learning_rate = ada_optimal_hyperparameter_values["params"]["learning_rate"])

models_object_dict["CatBoost"] = CatB(iterations= round(cat_optimal_hyperparameter_values["params"]["iterations"]), 
                                      learning_rate = cat_optimal_hyperparameter_values["params"]["learning_rate"], 
                                      depth = round(cat_optimal_hyperparameter_values["params"]["depth"]), 
                                      l2_leaf_reg = cat_optimal_hyperparameter_values["params"]["l2_leaf_reg"])

models_object_dict["XGBoost"] = XGBoost(n_estimators=int(xgb_optimal_hyperparameter_values["params"]["n_estimators"]), 
                                        max_depth = int(xgb_optimal_hyperparameter_values["params"]["max_depth"]), 
                                        subsample = float(xgb_optimal_hyperparameter_values["params"]["subsample"]),
                                        colsample_bytree = float(xgb_optimal_hyperparameter_values["params"]["colsample_bytree"]),
                                        gamma = float(xgb_optimal_hyperparameter_values["params"]["gamma"]), 
                                        reg_alpha = float(xgb_optimal_hyperparameter_values["params"]["reg_alpha"]),  
                                        reg_lambda = float(xgb_optimal_hyperparameter_values["params"]["reg_lambda"]))

models_object_dict["LightBoost"] = LightB(n_estimators = int(light_optimal_hyperparameter_values["params"]["n_estimators"]), 
                                          learning_rate = float(light_optimal_hyperparameter_values["params"]["learning_rate"]), 
                                          max_depth = int(light_optimal_hyperparameter_values["params"]["max_depth"]), 
                                          num_leaves = int(light_optimal_hyperparameter_values["params"]["num_leaves"]), 
                                          min_child_samples = int(light_optimal_hyperparameter_values["params"]["min_child_samples"]))



models_object_dict["GraBoost"] = GBC(n_estimators= int(gbc_optimal_hyperparameter_values["params"]["n_estimators"]), learning_rate = float(gbc_optimal_hyperparameter_values["params"]["learning_rate"]),
                                     max_depth = int(gbc_optimal_hyperparameter_values["params"]["max_depth"]), min_samples_leaf = int(gbc_optimal_hyperparameter_values["params"]["min_samples_leaf"]), subsample = float(gbc_optimal_hyperparameter_values["params"]["subsample"]))



meta_features_trainX, meta_features_trainY, original_features_k_fold = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                         accuracy_objects_dict, path)

# Assign column names.............
label = original_features_k_fold["label"]
del original_features_k_fold["label"]
original_features_k_fold.columns = X.columns
original_features_k_fold["label"] = label


meta_features_testX, meta_features_testY, result_dataframe = return_metafeatures_for_single_splits(X_train, y_train, X_test, 
                                                                                                       y_test, models_object_dict, 


                                                                                                       accuracy_objects_dict)
result_dataframe = pd.DataFrame(result_dataframe)
result_dataframe = result_dataframe.transpose()

result_dataframe = result_dataframe.sort_values(by ="F1_score")
result_dataframe.to_csv(path / "baseline_results.csv", sep = "\t")
meta_df.to_csv(path / "baseline_tuning_time.txt", sep = "\t")

# save meta_features..................
path = Path("data/save_meta_features_without_feature_extractions/")
path.mkdir(parents=True, exist_ok=True)

meta_features_trainX["label"] = meta_features_trainY
meta_features_testX["label"] = meta_features_testY

meta_features_trainX.to_csv(path / "training_meta.csv", sep = ";", index = False)
meta_features_testX.to_csv(path / "testing_meta.csv", sep = ";", index = False)


original_features_k_fold.to_csv(path / "original_features_k_fold_training.csv", sep = ";", index = False)
X_test["label"] = y_test
X_test.to_csv(path / "original_features_test.csv", sep = ";", index = False)
