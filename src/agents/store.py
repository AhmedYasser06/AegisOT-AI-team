"""In-memory incident store (bounded). Replace with a DB table if you need persistence;
the backend already stores every assessment when PUSH_RESULT_TO_BACKEND=true."""
from __future__ import annotations

import threading
from collections import OrderedDict

from .schemas import IncidentAssessment


class IncidentStore:
    def __init__(self, max_items: int = 1000):
        self._d: OrderedDict[str, IncidentAssessment] = OrderedDict()
        self._lock, self._max = threading.Lock(), max_items

    def put(self, a: IncidentAssessment) -> None:
        with self._lock:
            self._d[a.incident_id] = a
            self._d.move_to_end(a.incident_id)
            while len(self._d) > self._max:
                self._d.popitem(last=False)

    def get(self, incident_id: str) -> IncidentAssessment | None:
        with self._lock:
            return self._d.get(incident_id)

    def latest(self, n: int = 20) -> list[IncidentAssessment]:
        with self._lock:
            return list(reversed(list(self._d.values())))[:n]
