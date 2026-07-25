"""Training history containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class History:
    """Track metrics and losses across epochs."""

    records: Dict[str, List[float]] = field(default_factory=dict)

    def append(self, logs: Dict[str, Any]) -> None:
        for key, value in logs.items():
            self.records.setdefault(key, []).append(float(value))

    def as_dict(self) -> Dict[str, List[float]]:
        return self.records
