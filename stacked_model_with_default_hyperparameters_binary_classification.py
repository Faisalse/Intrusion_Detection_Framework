from preprocessing.TON_IOT_binary import *
from helper_functions import *
from accuracy.bin_accuracy import *
import pandas as pd

# stacked models
from algorithms.CatB.CatB import *
from algorithms.LightB.LightB import *
from algorithms.XGBoost.XGBoost import *
from algorithms.AdaBoost.AdaBoost import *
from algorithms.GBC.GraBC import *

accuracy_objects_dict = dict()
accuracy_objects_dict["Accuracy"] = Accuracy()
accuracy_objects_dict["Precision"] = Precision()
accuracy_objects_dict["Recall"] = Recall()
accuracy_objects_dict["F1_score"] = F1_score()

# import models --- > baselines
models_object_dict = dict()
models_object_dict["XGBoost"] = XGBoost()
models_object_dict["CatBoost"] = CatB()
models_object_dict["LightBoost"] = LightB()
models_object_dict["AdaBoost"] = AdaBoost()
models_object_dict["GraBoost"] = GBC()


DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/binary/")
path.mkdir(parents=True, exist_ok=True)

X, y = data_load(DATA_PATH, data_name)


X_train, X_test, y_train, y_test = split_data_train_test(X, y)
meta_features_trainX, meta_features_trainY = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                         accuracy_objects_dict, path)

meta_features_testX, meta_features_testY, result_dataframe = return_metafeatures_for_single_splits(X_train, y_train, X_test, 
                                                                                                       y_test, models_object_dict, 
                                                                                                       accuracy_objects_dict)
# use XGBoost as stacked model.............................................
stacked_model_dict = dict()
stacked_model_dict["XGBoost"] = XGBoost()



df = pd.DataFrame.from_dict(result_dataframe, orient="index")
df.to_csv(path / "defaultHyperparameters.txt", index = True, sep = "\t")


