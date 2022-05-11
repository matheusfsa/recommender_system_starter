"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from recommender_system_starter.pipelines import preprocessing as pp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    pp_pipeline = pp.create_pipeline()
    return {
        "pp": pp_pipeline,
        "__default__": pp_pipeline,
    }
