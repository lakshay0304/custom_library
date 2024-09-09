from typing import Union
from .test_run import global_test_run_manager

def process_hyperparameters(hyperparameters) -> Union[dict, None]:
    if hyperparameters is None:
        return None

    if not isinstance(hyperparameters, dict):
        raise TypeError("Hyperparameters must be a dictionary or None")

    processed_hyperparameters = {}

    for key, value in hyperparameters.items():
        if not isinstance(key, str):
            raise TypeError(f"Hyperparameter key '{key}' must be a string")

        if value is None:
            continue

        if not isinstance(value, (str, int, float)):
            raise TypeError(
                f"Hyperparameter value for key '{key}' must be a string, integer, or float"
            )

        processed_hyperparameters[key] = str(value)

    return processed_hyperparameters