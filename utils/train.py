import pandas as pd

from models.ensembling_models import HardVotingModel, SoftVotingModel, StackingModel
from data.feature_engineering import get_aac_features, get_dipeptide_features, get_tripeptide_frequency, get_svd_embedding_features
from models.random_forest import RandomForestModel as rf
from models.lightgbm import LightGBMModel as lgb
from models.svm import SVMModel as svm


train_data = pd.read_csv("SwissProt-EC-train_updated.csv")

base_estimators = [
    ("rf", rf),
    ("lgb", lgb),
    ("svm", svm)
]


needed_df = train_data[["seq", "main_ec"]]
X_train = needed_df["seq"]
y_train = needed_df["main_ec"]


