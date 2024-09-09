from contextvars import ContextVar
from enum import Enum
import copy
import os
import json
import time
from typing import Any, Dict, Optional, List, Union, Tuple
from collections.abc import Iterable
import tqdm
import re
import string
import numpy as np
from dataclasses import asdict, is_dataclass
import re
import asyncio
import nest_asyncio
import uuid
from pydantic import BaseModel
from base_metric import BaseMetric, BaseConversationalMetric
from test_case import LLMTestCase, ConversationalTestCase

from base_model import DeepEvalBaseLLM

from test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
    Message,
)


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print(
                "Event loop is already running. Applying nest_asyncio patch to allow async execution..."
            )
            nest_asyncio.apply()

        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def prettify_list(lst: List[Any]):
    if len(lst) == 0:
        return "[]"

    formatted_elements = []
    for item in lst:
        if isinstance(item, str):
            formatted_elements.append(f'"{item}"')
        elif isinstance(item, BaseModel):
            try:
                jsonObj = item.model_dump()
            except AttributeError:
                # Pydantic version below 2.0
                jsonObj = item.dict()

            formatted_elements.append(
                json.dumps(jsonObj, indent=4).replace("\n", "\n    ")
            )
        else:
            formatted_elements.append(repr(item))  # Fallback for other types

    formatted_list = ",\n    ".join(formatted_elements)
    return f"[\n    {formatted_list}\n]"



def construct_verbose_logs(metric: BaseMetric, steps: List[str]) -> str:
    verbose_logs = ""
    for i in range(len(steps) - 1):
        verbose_logs += steps[i]

        # don't add new line for penultimate step
        if i < len(steps) - 2:
            verbose_logs += " \n \n"

    if metric.verbose_mode:
        # only print reason and score for deepeval
        print_verbose_logs(metric.__name__, verbose_logs + f"\n \n{steps[-1]}")

    return verbose_logs

def trimAndLoadJson(
    input_string: str, metric: Optional[BaseMetric] = None
) -> Any:
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        if metric is not None:
            metric.error = error_str
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")
        

def check_llm_test_case_params(
    test_case: LLMTestCase,
    test_case_params: List[LLMTestCaseParams],
    metric: BaseMetric,
):
    if isinstance(test_case, LLMTestCase) is False:
        error_str = f"Unable to evaluate test cases that are not of type 'LLMTestCase' using the non-conversational '{metric.__name__}' metric."
        metric.error = error_str
        raise ValueError(error_str)

    missing_params = []
    for param in test_case_params:
        if getattr(test_case, param.value) is None:
            missing_params.append(f"'{param.value}'")

    if missing_params:
        if len(missing_params) == 1:
            missing_params_str = missing_params[0]
        elif len(missing_params) == 2:
            missing_params_str = " and ".join(missing_params)
        else:
            missing_params_str = (
                ", ".join(missing_params[:-1]) + ", and " + missing_params[-1]
            )

        error_str = f"{missing_params_str} cannot be None for the '{metric.__name__}' metric"
        metric.error = error_str
        raise ValueError(error_str)

def initialize_model(
    model: Optional[Union[str, DeepEvalBaseLLM, GPTModel]] = None,
) -> Tuple[DeepEvalBaseLLM, bool]:
    """
    Returns a tuple of (initialized DeepEvalBaseLLM, using_native_model boolean)
    """
    # If model is a GPTModel, it should be deemed as using native model
    if isinstance(model, GPTModel):
        return model, True
    # If model is a DeepEvalBaseLLM but not a GPTModel, we can not assume it is a native model
    if isinstance(model, DeepEvalBaseLLM):
        return model, False
    # Otherwise (the model is a string or None), we initialize a GPTModel and use as a native model
    return GPTModel(model=model), True

def copy_metrics(
    metrics: Union[List[BaseMetric], List[BaseConversationalMetric]]
) -> Union[List[BaseMetric], List[BaseConversationalMetric]]:
    copied_metrics = []
    for metric in metrics:
        metric_class = type(metric)
        args = vars(metric)

        signature = inspect.signature(metric_class.__init__)
        valid_params = signature.parameters.keys()
        valid_args = {key: args[key] for key in valid_params if key in args}

        copied_metrics.append(metric_class(**valid_args))
    return copied_metrics