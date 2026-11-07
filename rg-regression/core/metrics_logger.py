from collections import defaultdict
from typing import Dict, List, Any, Optional

class MetricsLogger:
    def __init__(self, initial_data: Optional[Dict[str, List[Any]]] = None):
        if initial_data is not None:
            self._data: Dict[str, List[Any]] = defaultdict(list, {
                k: list(v) for k, v in initial_data.items()
            })
        else:
            self._data: Dict[str, List[Any]] = defaultdict(list)

    def log(self, **metrics):
        for key, value in metrics.items():
            if value is not None:
                self._data[key].append(value)

    def get(self, key: str):
        return self._data.get(key, [])

    def as_dict(self) -> Dict[str, List[Any]]:
        return dict(self._data)

    def reset(self):
        self._data.clear()
