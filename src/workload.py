"""
Workload data structures.

A `Task` is a unit of compute work that arrives at a known time, has a
required amount of work (in Compute Units), a deadline, a priority and
flags describing whether it may be delayed / interrupted.

A `WorkloadSet` is the collection of tasks for one experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

import config


@dataclass
class Task:
    task_id: str
    category: str                 # "urgent" | "flexible" | "training"
    arrival_slot: int             # slot index when the task becomes available
    workload_cu: float            # total Compute Units required
    deadline_slot: int            # slot by which the task must be completed
    priority: int                # higher = more important
    delayable: bool
    interruptible: bool
    penalty_per_cu: float = 1.0   # $/CU applied if deadline missed

    # ---- derived helpers -------------------------------------------------
    @property
    def window_slots(self) -> List[int]:
        """Slots in which this task may legally be processed."""
        start = self.arrival_slot
        end = self.deadline_slot          # inclusive deadline slot
        return list(range(start, end + 1))

    @property
    def is_urgent(self) -> bool:
        return self.category == "urgent"

    def to_row(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "arrival_slot": self.arrival_slot,
            "arrival_time": config.slot_to_time_str(self.arrival_slot),
            "workload_cu": self.workload_cu,
            "deadline_slot": self.deadline_slot,
            "deadline_time": config.slot_to_time_str(self.deadline_slot),
            "priority": self.priority,
            "delayable": self.delayable,
            "interruptible": self.interruptible,
            "penalty_per_cu": self.penalty_per_cu,
        }


@dataclass
class WorkloadSet:
    tasks: List[Task] = field(default_factory=list)

    # ---- access ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def add(self, task: Task) -> None:
        self.tasks.append(task)

    @property
    def total_workload_cu(self) -> float:
        return sum(t.workload_cu for t in self.tasks)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_row() for t in self.tasks])

    # ---- summary by category --------------------------------------------
    def summary(self) -> pd.DataFrame:
        rows = []
        for cat in sorted({t.category for t in self.tasks}):
            ts = [t for t in self.tasks if t.category == cat]
            rows.append({
                "category": cat,
                "n_tasks": len(ts),
                "total_cu": round(sum(t.workload_cu for t in ts), 2),
                "avg_deadline_slots": (
                    round(sum(t.deadline_slot - t.arrival_slot for t in ts) / len(ts), 2)
                    if ts else 0.0
                ),
            })
        return pd.DataFrame(rows)
