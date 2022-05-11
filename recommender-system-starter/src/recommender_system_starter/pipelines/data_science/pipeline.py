"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""

from functools import reduce
from operator import add
from itertools import product

from kedro.pipeline import Pipeline, node
from kedro.framework.session.session import _active_session
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import fit_model

def _training_template(name):
    return Pipeline(
        [
            node(
                func=fit_model,
                inputs={
                    "train_data": "train_data",
                    "model_dict": "params:model_dict",
                    "fold": "params:fold",
                    "metric": "params:metric",
                    "verbose": "params:verbose",
                    "n_jobs": "params:n_jobs",
                    "user_column": "params:user_column",
                    "item_column": "params:item_column",
                    "rating_column": "params:rating_column",
                    "rating_scale": "params:rating_scale"
                },
                outputs="model_result",
                name=f"fit_{name}"
            )
        ]
    )

def create_pipeline(**kwargs) -> Pipeline:
    session = _active_session.load_context()
    models = session.catalog.load("params:models")

    training_pipelines = [
        pipeline(
            pipe=_training_template(model),
            parameters={"params:model_dict": f"params:models.{model}"},
            outputs={
                "model_result": f"{model}_result"
            },
        )
        for model in models
    ]
    training_pipeline = reduce(add, training_pipelines)
    return training_pipeline
