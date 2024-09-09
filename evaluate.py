import asyncio
from copy import deepcopy
import os
from typing import List, Optional, Union, Dict
import time
from dataclasses import dataclass
from rich.console import Console
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from base_metric import BaseMetric, BaseConversationalMetric
from test_case import LLMTestCase, ConversationalTestCase

def evaluate(
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]],
    metrics: List[BaseMetric],
    hyperparameters: Optional[Dict[str, Union[str, int, float]]] = None,
    run_async: bool = True,
    show_indicator: bool = True,
    print_results: bool = True,
    write_cache: bool = True,
    use_cache: bool = False,
    ignore_errors: bool = False,
    verbose_mode: Optional[bool] = None,
    throttle_value: int = 0,
):
    if hyperparameters is not None:
        if (
            hyperparameters.get("model") is None
            or hyperparameters.get("prompt template") is None
        ):
            raise ValueError(
                "A `model` and `prompt template` key must be provided when logging `hyperparameters`."
            )
        hyperparameters = process_hyperparameters(hyperparameters)

    global_test_run_manager.reset()
    start_time = time.perf_counter()

    if show_indicator:
        console = Console()
        for metric in metrics:
            console.print(
                format_metric_description(metric, async_mode=run_async)
            )

    with capture_evaluation_run("evaluate()"):
        if run_async:
            loop = get_or_create_event_loop()
            test_results = loop.run_until_complete(
                a_execute_test_cases(
                    test_cases,
                    metrics,
                    ignore_errors=ignore_errors,
                    use_cache=use_cache,
                    verbose_mode=verbose_mode,
                    save_to_disk=write_cache,
                    show_indicator=show_indicator,
                    throttle_value=throttle_value,
                )
            )
        else:
            test_results = execute_test_cases(
                test_cases,
                metrics,
                ignore_errors=ignore_errors,
                use_cache=use_cache,
                verbose_mode=verbose_mode,
                save_to_disk=write_cache,
                show_indicator=show_indicator,
            )

    end_time = time.perf_counter()
    run_duration = end_time - start_time
    if print_results:
        for test_result in test_results:
            print_test_result(test_result)

        aggregate_metric_pass_rates(test_results)

    test_run = global_test_run_manager.get_test_run()
    test_run.hyperparameters = hyperparameters
    global_test_run_manager.save_test_run()
    global_test_run_manager.wrap_up_test_run(run_duration, display_table=False)
    return test_results