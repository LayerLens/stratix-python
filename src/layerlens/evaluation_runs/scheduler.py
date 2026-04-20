"""Lightweight thread-backed scheduler for recurring evaluation runs."""

from __future__ import annotations

import uuid
import logging
import threading
from typing import Dict, List, Callable, Optional
from dataclasses import field, dataclass

from .models import EvaluationRun

log = logging.getLogger(__name__)

RunFactory = Callable[[], EvaluationRun]


@dataclass
class ScheduledRun:
    id: str
    interval_seconds: float
    factory: RunFactory
    name: Optional[str] = None
    last_run: Optional[EvaluationRun] = None
    history: List[EvaluationRun] = field(default_factory=list)
    history_limit: int = 20
    _timer: Optional[threading.Timer] = field(default=None, repr=False)
    _stopped: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _record(self, run: EvaluationRun) -> None:
        """Atomically append a run and trim to ``history_limit``."""
        with self._lock:
            self.last_run = run
            self.history.append(run)
            if len(self.history) > self.history_limit:
                self.history = self.history[-self.history_limit :]

    def _tick(self) -> None:
        with self._lock:
            if self._stopped:
                return
        try:
            run = self.factory()
        except Exception as exc:
            log.warning("scheduled run %s failed: %s", self.id, exc)
            run = None
        if run is not None:
            self._record(run)
        with self._lock:
            self._arm_locked()

    def _arm(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._arm_locked()

    def _arm_locked(self) -> None:
        t = threading.Timer(self.interval_seconds, self._tick)
        t.daemon = True
        self._timer = t
        t.start()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            timer = self._timer
        if timer is not None:
            timer.cancel()


class RunScheduler:
    """In-process scheduler. Swap for a cron/queue backend in production."""

    def __init__(self) -> None:
        self._scheduled: Dict[str, ScheduledRun] = {}
        self._lock = threading.Lock()

    def schedule(
        self,
        factory: RunFactory,
        *,
        interval_seconds: float,
        name: Optional[str] = None,
        run_immediately: bool = False,
    ) -> ScheduledRun:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        schedule_id = f"sched_{uuid.uuid4().hex[:12]}"
        sched = ScheduledRun(
            id=schedule_id,
            interval_seconds=interval_seconds,
            factory=factory,
            name=name,
        )
        with self._lock:
            self._scheduled[schedule_id] = sched
        if run_immediately:
            sched._tick()  # arms next tick at the end
        else:
            sched._arm()
        return sched

    def list(self) -> List[ScheduledRun]:
        with self._lock:
            return list(self._scheduled.values())

    def get(self, schedule_id: str) -> Optional[ScheduledRun]:
        with self._lock:
            return self._scheduled.get(schedule_id)

    def cancel(self, schedule_id: str) -> bool:
        with self._lock:
            sched = self._scheduled.pop(schedule_id, None)
        if sched is None:
            return False
        sched.stop()
        return True

    def cancel_all(self) -> None:
        with self._lock:
            scheduled = list(self._scheduled.values())
            self._scheduled.clear()
        for s in scheduled:
            s.stop()

    def trigger_now(self, schedule_id: str) -> Optional[EvaluationRun]:
        sched = self.get(schedule_id)
        if sched is None:
            return None
        try:
            run = sched.factory()
        except Exception as exc:
            log.warning("trigger_now failed for %s: %s", schedule_id, exc)
            return None
        sched._record(run)
        return run
