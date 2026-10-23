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
        ("Missed deadlines [count]", float(baseline_m.missed_deadlines), float(opt_m.missed_deadlines), True),
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
