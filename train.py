# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

from funcy import reraise
from dvc.repo import Repo
from dvc.exceptions import OutputNotFoundError, PathMissingError


def get_data_url(path, repo, rev, remote):
    with Repo.open(repo, rev=rev, subrepos=True, uninitialized=True) as _repo:
        fs_path = _repo.dvcfs.from_os_path(path)

        with reraise(FileNotFoundError, PathMissingError(path, repo)):
            info = _repo.dvcfs.info(fs_path)
        dvc_info = info.get("dvc_info")

        dvc_repo = info["repo"]
        md5 = dvc_info["md5"]

        data_url = remote + "/" + md5[0:2] + "/" + md5[2:]
        return data_url

mlflow.set_experiment("azure_dvc_mlflow_demo")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, 
                        help="Location and file name of the target, relative to the root of the project")
    parser.add_argument("--repo", required=True, 
                        help="Specifies the location of the DVC project")
    parser.add_argument("--rev", required=True, 
                        help="Git commit")
    parser.add_argument("--remote", required=True, 
                        help="Name of the DVC remote to use to form the returned URL string.")
    parser.add_argument("--alpha", default=0.7)
    parser.add_argument("--l1_ratio", default=1.9)
    args = parser.parse_args()

    data_url = get_data_url(args.path, args.repo, args.rev, args.remote)

    data = pd.read_csv(data_url, sep=",")
    mlflow.log_param("data_url", data_url)
    mlflow.log_param("data_version", args.rev)
    mlflow.log_param("input_rows", data.shape[0])
    mlflow.log_param("input_cols", data.shape[1])

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(args.alpha) if args.alpha > 1 else 0.5
    l1_ratio = float(args.l1_ratio) if args.l1_ratio > 2 else 0.5

#    with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(lr, "model")
