"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess, split_train_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess,
                inputs={
                    "data": "master_data",
                    "user_column": "params:user_column",
                    "item_column": "params:item_column",
                    "rating_column": "params:rating_column",
                },
                outputs="preprocessed_data",
            ),
            node(
                func=split_train_test,
                inputs={
                    "data": "preprocessed_data",
                    "user_column": "params:user_column",
                    "test_size": "params:test_size",
                },
                outputs=["train_data", "test_data"],
            ),
        ]
    )
