from dataclasses import dataclass
from typing import Any, Optional
from .metrics_logger import MetricsLogger

@dataclass
class TrainingState:
    epoch: int
    step: int
    model: Any
    optimizer: Any
    scheduler: Optional[Any]
    device: str
    params: dict
    metrics: MetricsLogger