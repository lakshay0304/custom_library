from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from contextlib import contextmanager
import sys
from typing import List, Optional, Union
import time
import asyncio

from deepeval.metrics import BaseMetric, BaseConversationalMetric




@contextmanager
def metric_progress_indicator(
    metric: BaseMetric,
    async_mode: Optional[bool] = None,
    _show_indicator: bool = True,
    total: int = 9999,
    transient: bool = True,
):
    with capture_metric_type(metric.__name__):
        console = Console(file=sys.stderr)  # Direct output to standard error
        if _show_indicator:
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                TextColumn("[progress.description]{task.description}"),
                console=console,  # Use the custom console
                transient=transient,
            ) as progress:
                progress.add_task(
                    description=format_metric_description(metric, async_mode),
                    total=total,
                )
                yield
        else:
            yield