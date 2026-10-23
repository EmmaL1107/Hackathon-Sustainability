"""
Metrics computation for the co-scheduling experiments.

Computes economic, environmental, service and computational metrics from
actual Schedule objects. All definitions follow the competition spec:

    cost reduction (%)  = (baseline_cost - opt_cost) / baseline_cost * 100
    emissions red. (%)  = (baseline_emis - opt_emis) / baseline_emis * 100

Zero denominators are handled safely (returns 0.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict

import numpy as np
import pandas as pd

import config
from src.optimizer import Schedule


@dataclass
class MetricSet:
    name: str
    feasible: bool
    # economic
    total_cost_usd: float
    total_carbon_kg: float
    completed_cu: float
    total_cu: float
    cost_per_cu: float
    # environmental
    renewable_used_kwh: float
    total_energy_kwh: float
    renewable_utilization_pct: float       # used / available
    renewable_share_pct: float             # renewable / total electricity
    # service
    missed_deadlines: int
    deadline_miss_rate_pct: float
    avg_delay_slots: float
    urgent_service_rate_pct: float
    urgent_completed_cu: float
    urgent_total_cu: float
    # per-category unmet work (Feedback 2: distinguish Unmet CU vs Missed tasks
    # vs Urgent service rate, broken down by category)
    unmet_urgent_cu: float
    unmet_flexible_cu: float
    unmet_training_cu: float
    missed_urgent: int
    missed_flexible: int
    missed_training: int
    # computational
    runtime_s: float
    solver_status: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(schedule: Schedule, data) -> MetricSet:
    tasks = schedule.tasks.tasks
    n_tasks = len(tasks)
    comp = schedule.per_slot_compute()
    e_cu = config.KW_PER_CU * config.SLOT_HOURS
    e_idle = config.IDLE_POWER_KW * config.SLOT_HOURS

    completed_per_task = schedule.x.sum(axis=1)
    total_per_task = np.array([t.workload_cu for t in tasks], dtype=float)
    shortfall = schedule.shortfall

    # ---- economic ----
    total_cost = float((data.price * schedule.grid_energy).sum())
    total_carbon_g = float((data.carbon * schedule.grid_energy).sum())
    completed_cu = float(completed_per_task.sum())
    total_cu = float(total_per_task.sum())
    cost_per_cu = float(total_cost / completed_cu) if completed_cu > 0 else 0.0

    # ---- environmental ----
    renewable_used = float(schedule.renewable_used.sum())
    renewable_available = float(data.renewable.sum())
    total_energy = float(schedule.grid_energy.sum() + schedule.renewable_used.sum())
    ren_util = (renewable_used / renewable_available * 100.0
                if renewable_available > 0 else 0.0)
    ren_share = (renewable_used / total_energy * 100.0
                 if total_energy > 0 else 0.0)

    # ---- service ----
    # a task counts as missed only if a meaningful amount of work is unmet
    # (>0.01 CU) -- below that it is floating-point residue, treated complete
    missed = int(np.sum(shortfall > 1e-2))
    miss_rate = (missed / n_tasks * 100.0) if n_tasks > 0 else 0.0

    # average completion delay (slots between arrival and last executed slot),
    # only over fully-completed tasks
    delays = []
    for ti, t in enumerate(tasks):
        if shortfall[ti] > 1e-6:
            continue
        executed_slots = np.where(schedule.x[ti] > 1e-9)[0]
        if len(executed_slots) == 0:
            continue
        last_slot = executed_slots[-1]
        delays.append(max(0, last_slot - t.arrival_slot))
    avg_delay = float(np.mean(delays)) if delays else 0.0

    urgent_idx = [i for i, t in enumerate(tasks) if t.is_urgent]
    urgent_comp = float(completed_per_task[urgent_idx].sum()) if urgent_idx else 0.0
    urgent_tot = float(total_per_task[urgent_idx].sum()) if urgent_idx else 0.0
    urgent_rate = (urgent_comp / urgent_tot * 100.0
                   if urgent_tot > 0 else 100.0)

    # ---- per-category unmet work & missed counts (Feedback 2) ----
    unmet_by_cat = {}
    missed_by_cat = {}
    for cat in ("urgent", "flexible", "training"):
        cat_idx = [i for i, t in enumerate(tasks) if t.category == cat]
        unmet_by_cat[cat] = float(shortfall[cat_idx].sum()) if cat_idx else 0.0
        missed_by_cat[cat] = int(np.sum(shortfall[cat_idx] > 1e-2)) if cat_idx else 0

    return MetricSet(
        name=schedule.name,
        feasible=schedule.feasible,
        total_cost_usd=round(total_cost, 4),
        total_carbon_kg=round(total_carbon_g / 1000.0, 4),
        completed_cu=round(completed_cu, 2),
        total_cu=round(total_cu, 2),
        cost_per_cu=round(cost_per_cu, 6),
        renewable_used_kwh=round(renewable_used, 2),
        total_energy_kwh=round(total_energy, 2),
        renewable_utilization_pct=round(ren_util, 2),
        renewable_share_pct=round(ren_share, 2),
        missed_deadlines=missed,
        deadline_miss_rate_pct=round(miss_rate, 2),
        avg_delay_slots=round(avg_delay, 2),
        urgent_service_rate_pct=round(urgent_rate, 2),
        urgent_completed_cu=round(urgent_comp, 2),
        urgent_total_cu=round(urgent_tot, 2),
        unmet_urgent_cu=round(unmet_by_cat.get("urgent", 0.0), 2),
        unmet_flexible_cu=round(unmet_by_cat.get("flexible", 0.0), 2),
        unmet_training_cu=round(unmet_by_cat.get("training", 0.0), 2),
        missed_urgent=missed_by_cat.get("urgent", 0),
        missed_flexible=missed_by_cat.get("flexible", 0),
        missed_training=missed_by_cat.get("training", 0),
        runtime_s=round(schedule.runtime_s, 4),
        solver_status=str(schedule.solver_status)[:80],
    )


# ---------------------------------------------------------------------------
# Comparison between baseline and optimized
# ---------------------------------------------------------------------------
def compare(baseline_m: MetricSet, opt_m: MetricSet) -> pd.DataFrame:
    # direction: True  => lower is better  (improvement = (base-new)/base)
    #            False => higher is better  (improvement = (new-base)/base)
    spec = [
        ("Total electricity cost [$]", baseline_m.total_cost_usd, opt_m.total_cost_usd, True),
        ("Total carbon emissions [kgCO2]", baseline_m.total_carbon_kg, opt_m.total_carbon_kg, True),
        ("Cost per completed CU [$/CU]", baseline_m.cost_per_cu, opt_m.cost_per_cu, True),
        ("Renewable utilization [%]", baseline_m.renewable_utilization_pct, opt_m.renewable_utilization_pct, False),
        ("Renewable share of electricity [%]", baseline_m.renewable_share_pct, opt_m.renewable_share_pct, False),
        ("Unmet workload (total CU)", baseline_m.unmet_urgent_cu + baseline_m.unmet_flexible_cu + baseline_m.unmet_training_cu,
         opt_m.unmet_urgent_cu + opt_m.unmet_flexible_cu + opt_m.unmet_training_cu, True),
        ("Unmet urgent workload [CU]", baseline_m.unmet_urgent_cu, opt_m.unmet_urgent_cu, True),
        ("Unmet flexible workload [CU]", baseline_m.unmet_flexible_cu, opt_m.unmet_flexible_cu, True),
        ("Unmet training workload [CU]", baseline_m.unmet_training_cu, opt_m.unmet_training_cu, True),
        ("Missed deadlines [count]", float(baseline_m.missed_deadlines), float(opt_m.missed_deadlines), True),
        ("Missed urgent tasks [count]", float(baseline_m.missed_urgent), float(opt_m.missed_urgent), True),
        ("Deadline miss rate [%]", baseline_m.deadline_miss_rate_pct, opt_m.deadline_miss_rate_pct, True),
        ("Avg completion delay [slots]", baseline_m.avg_delay_slots, opt_m.avg_delay_slots, True),
        ("Urgent service rate [%]", baseline_m.urgent_service_rate_pct, opt_m.urgent_service_rate_pct, False),
        ("Completed workload [CU]", baseline_m.completed_cu, opt_m.completed_cu, False),
        ("Optimization runtime [s]", baseline_m.runtime_s, opt_m.runtime_s, True),
    ]
    rows = []
    for label, base, new, lower_better in spec:
        if abs(base) < 1e-12:
            imp = 0.0
        elif lower_better:
            imp = (base - new) / base * 100.0
        else:
            imp = (new - base) / base * 100.0
        rows.append((label, base, new, round(imp, 2)))
    return pd.DataFrame(rows, columns=["metric", "baseline", "optimized",
                                       "improvement_vs_baseline_pct"])


def metrics_to_dataframe(m: MetricSet) -> pd.DataFrame:
    d = m.to_dict()
    return pd.DataFrame({"metric": list(d.keys()),
                          "value": list(d.values())})


# ---------------------------------------------------------------------------
# Miss-cause diagnosis (Feedback 2)
# ---------------------------------------------------------------------------
def diagnose_misses(schedule: Schedule, data) -> pd.DataFrame:
    """For each task that missed its deadline, diagnose the ROOT CAUSE.

    Three distinct causes are distinguished:

    1. **window_too_short** — the task's execution window cannot physically
       hold its workload even with *exclusive* access to full per-slot capacity.
       (sum of capacity across window slots < workload_cu.)

    2. **horizon_boundary** — a sub-case where the window is truncated because
       the task's deadline falls at or near the last slot of the 24h horizon.
       The work cannot be extended beyond the planning horizon.

    3. **capacity_bound** — the window IS long enough (total window capacity >=
       workload) but the shortfall arises because competing tasks consumed the
       shared capacity during overlapping windows.

    These distinctions prevent the false assumption that all end-of-day
    shortfalls are "unavoidable" — some may be scheduling failures, not
    fundamental infeasibility.
    """
    tasks = schedule.tasks.tasks
    rows = []
    for ti, t in enumerate(tasks):
        shortfall_cu = schedule.shortfall[ti]
        if shortfall_cu <= 1e-2:
            continue   # task met its deadline (or only floating-point residue)

        window = t.window_slots
        window_total_capacity = float(sum(data.capacity[s] for s in window))
        can_finish_exclusive = window_total_capacity >= t.workload_cu - 1e-6
        at_horizon_end = t.deadline_slot >= data.n_slots - 1

        if not can_finish_exclusive:
            cause = "horizon_boundary" if at_horizon_end else "window_too_short"
        else:
            cause = "capacity_bound"

        rows.append({
            "task_id": t.task_id,
            "category": t.category,
            "workload_cu": round(t.workload_cu, 2),
            "shortfall_cu": round(float(shortfall_cu), 2),
            "arrival_time": config.slot_to_time_str(t.arrival_slot),
            "deadline_time": config.slot_to_time_str(t.deadline_slot),
            "window_slots": len(window),
            "window_capacity_cu": round(window_total_capacity, 1),
            "cause": cause,
        })
    return pd.DataFrame(rows)
