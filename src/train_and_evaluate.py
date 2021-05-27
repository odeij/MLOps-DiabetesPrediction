# load the train and test dataset
# Training algorithm
# save the metrics, params

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred)
    return accuracy, precision, recall, roc_auc

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    penalty = config["estimators"]["LogisticRegression"]["params"]["penalty"]
    l1_ratio = config["estimators"]["LogisticRegression"]["params"]["l1_ratio"]
    solver = config["estimators"]["LogisticRegression"]["params"]["solver"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    lr = LogisticRegression(
        solver=solver,
        penalty=penalty,
        l1_ratio=l1_ratio,
        random_state=random_state)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    
    (accuracy, precision, recall, roc_auc) = eval_metrics(test_y, predicted_qualities)

    print("LogisticRegression model (penalty = %s)" % solver)
    print("LogisticRegression model (penalty = %s)" % penalty)
    print("LogisticRegression model (l1_ratio = %f)" % l1_ratio)
    print("  accuracy : %f" % accuracy)
    print("  precision: %f" % precision)
    print("  recall: %f" % recall)
    print("  area under roc: %f" % roc_auc)

    scores_file = config["reports"]["scores"]   #updating scores from scores.json
    params_file = config["reports"]["params"]   #updating parameters from params.json

    with open(scores_file, "w") as f:
        scores = {
            "accuracy_score": accuracy,
            "precision_score": precision,
            "recall_score": recall,
            "roc_auc_score": roc_auc
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "solver": solver,
            "penalty": penalty,
            "l1_ratio": l1_ratio
        }
        json.dump(params, f, indent=4)


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path) #saving the model



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
