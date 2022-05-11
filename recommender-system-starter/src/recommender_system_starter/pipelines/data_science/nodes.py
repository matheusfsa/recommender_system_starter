"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""
from typing import Any, Dict

import pandas as pd
from surprise import AlgoBase

from recommender_system_starter.utils import import_class
from surprise import Reader
from surprise import Dataset

def fit_model(
    train_data: pd.DataFrame,
    model_dict: Dict[str, Any],
    fold: Dict[str, Any],
    metric: str,
    verbose: int,
    n_jobs: int,
    rating_scale: Dict[str, int],
    user_column: str,
    item_column: str,
    rating_column: str
) -> pd.DataFrame:
    """This node fit a model"""


    reader = Reader(rating_scale=(rating_scale["min"], rating_scale["max"]))
    train_data = Dataset.load_from_df(train_data[[user_column, item_column, rating_column]], reader)

    model_class = import_class(model_dict["model_class"])
    cv = import_class(fold["class"])(**fold["kwargs"])

    search = _build_search(model_dict["params_search"], model_class, cv, metric, verbose, n_jobs)
    search.fit(train_data)
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
        model_class, params, cv=cv, measures=[metric],
        n_jobs=n_jobs, joblib_verbose=verbose,
        **search_dict["kwargs"]
    )
    return search
