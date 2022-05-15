"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""

from typing import Any, Dict, List, Union

import pandas as pd
from surprise import AlgoBase, Dataset, Reader, accuracy
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import cross_validate

from recommender_system_starter.utils import import_class


def build_dataset(
    data: pd.DataFrame,
    rating_scale: Dict[str, int],
    user_column: str,
    item_column: str,
    rating_column: str,
) -> Dataset:
    """This node build a dataset."""
    reader = Reader(rating_scale=(rating_scale["min"], rating_scale["max"]))
    dataset = Dataset.load_from_df(
        data[[user_column, item_column, rating_column]], reader
    )
    return dataset


def search_best_model(
    train_dataset: Dataset,
    model_dict: Dict[str, Any],
    fold: Dict[str, Any],
    metric: str,
    verbose: int,
    n_jobs: int,
) -> pd.DataFrame:
    """This node fit a model"""
    model_class = import_class(model_dict["model_class"])
    cv = import_class(fold["class"])(**fold["kwargs"])

    search = _build_search(
        model_dict["params_search"], model_class, cv, metric, verbose, n_jobs
    )
    search.fit(train_dataset)
    return {"model": search.best_estimator[metric], "metric": search.best_score[metric]}


def _build_search(
    search_dict: Dict[str, Any],
    model_class: AlgoBase,
    cv: Any,
    metric: str,
    verbose: int,
    n_jobs: int,
):
    search_class = import_class(search_dict["class"])
    params = {}
    for p_name, p_value in search_dict["params"].items():
        params[p_name] = p_value
    search = search_class(
        model_class,
        params,
        cv=cv,
        measures=[metric],
        n_jobs=n_jobs,
        joblib_verbose=verbose,
        **search_dict["kwargs"],
    )
    return search


def model_selection(
    train_dataset: Dataset,
    *models_results,
):
    """This node select the best model"""
    models_results = list(models_results)
    models_results.sort(key=lambda result: result["metric"])
    model = models_results[0]["model"]
    if isinstance(train_dataset, DatasetAutoFolds):
        train_dataset = train_dataset.build_full_trainset()
    model.fit(train_dataset)
    return model


def evaluate_model(
    model: AlgoBase,
    test_dataset: Dataset,
    metric: str,
) -> Dict[str, Union[float, List[float]]]:
    if isinstance(test_dataset, DatasetAutoFolds):
        test_dataset = test_dataset.build_full_trainset().build_testset()
    predictions = model.test(test_dataset)
    if metric == "rmse":
        metric_value = accuracy.rmse(predictions)
    elif metric == "mse":
        metric_value = accuracy.mse(predictions)
    elif metric == "mae":
        metric_value = accuracy.mae(predictions)
    else:
        ValueError("Invalid metric name")
    return {
        metric : {"value": metric_value, "step": 1},
    }

def fit_model(
    dataset: Dataset,
    model: AlgoBase
):
    if isinstance(dataset, DatasetAutoFolds):
        dataset = dataset.build_full_trainset()
    model.fit(dataset)
    return model