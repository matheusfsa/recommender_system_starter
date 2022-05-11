"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""

from functools import reduce
from operator import add

from kedro.framework.session.session import _active_session
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import build_dataset, fit_model, model_selection


def _training_template(name):
    return Pipeline(
        [
            node(
                func=fit_model,
                inputs={
                    "train_dataset": "train_dataset",
                    "model_dict": "params:model_dict",
                    "fold": "params:fold",
                    "metric": "params:metric",
                    "verbose": "params:verbose",
                    "n_jobs": "params:n_jobs",
                },
                outputs="model_result",
                name=f"fit_{name}",
            )
        ]
    )


def _dataset_template(name):
    return Pipeline(
        [
            node(
                func=build_dataset,
                inputs={
                    "data": "data",
                    "user_column": "params:user_column",
                    "item_column": "params:item_column",
                    "rating_column": "params:rating_column",
                    "rating_scale": "params:rating_scale",
                },
                outputs="dataset",
                name=f"dataset_{name}",
            )
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    session = _active_session.load_context()
    models = session.catalog.load("params:models")
    refs = ["train", "test"]

    dataset_pipelines = [
        pipeline(
            pipe=_dataset_template(ref),
            inputs={"data": f"{ref}_data"},
            parameters={},
            outputs={"dataset": f"{ref}_dataset"},
        )
        for ref in refs
    ]
    dataset_pipeline = reduce(add, dataset_pipelines)

    training_pipelines = [
        pipeline(
            pipe=_training_template(model),
            parameters={"params:model_dict": f"params:models.{model}"},
            outputs={"model_result": f"{model}_result"},
        )
        for model in models
    ]
    training_pipeline = reduce(add, training_pipelines)

    model_selection_pipeline = Pipeline(
        [
            node(
                func=model_selection,
                inputs=["train_dataset"] + [f"{model}_result" for model in models],
                outputs="model",
                name="model_selection",
            )
        ]
    )
    return dataset_pipeline + training_pipeline + model_selection_pipeline
