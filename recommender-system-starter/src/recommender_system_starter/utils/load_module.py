from importlib import import_module
from typing import Any, Dict, Optional


def import_class(class_path: str) -> Any:
    """This function import a class from the class path.

    Args:
        class_path (str): Class path

    Returns:
        Any: Class
    """
    lib, _, class_name = class_path.rpartition(".")
    lib = import_module(lib)
    return getattr(lib, class_name)


def import_model(model_name: str, default_args: Optional[Dict[str, Any]] = None) -> Any:
    """
    This function load model from string
    Args:
        model_name: Path to model.
        default_args: Default model args.
    Return:
        Model instance.
    """
    model_class = getattr(
        import_module((".").join(model_name.split(".")[:-1])),
        model_name.rsplit(".")[-1],
    )
    if default_args is None:
        return model_class()
    else:
        return model_class(**default_args)
