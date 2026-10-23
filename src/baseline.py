"""
Baseline scheduler: a fair, non-energy-aware policy.

Policy ("earliest-deadline-first / as-early-as-possible"):
    - Sort tasks by (priority desc, arrival, deadline).
    - For each task, greedily place its required CU into the earliest slots
      within [arrival, deadline] that still have spare capacity.
    - Electricity is drawn physically: renewable first (it is available and
      effectively free), then the grid for the remainder. The baseline does
      NOT look at price or carbon to *decide when* to run -- it just runs
      as early as capacity allows. This is a realistic "dumb" data center.

This guarantees the baseline and optimizer process the SAME workload on the
SAME electricity/capacity data, so improvements are attributable to smarter
scheduling, not an unfair workload.
"""

from __future__ import annotations

import time

import numpy as np

import config
from src.data_generator import ScenarioData
from src.optimizer import Schedule


def schedule(data: ScenarioData) -> Schedule:
    t0 = time.perf_counter()
    tasks = data.workload.tasks
    n_tasks = len(tasks)
    n_slots = data.n_slots
    e_cu = config.KW_PER_CU * config.SLOT_HOURS
    e_idle = config.IDLE_POWER_KW * config.SLOT_HOURS

    x = np.zeros((n_tasks, n_slots))
    remaining_cap = data.capacity.astype(float).copy()
    shortfall = np.zeros(n_tasks)

    # order: urgent first, then by arrival, then deadline
    order = sorted(range(n_tasks),
                   key=lambda i: (-tasks[i].priority,
                                   tasks[i].arrival_slot,
                                   tasks[i].deadline_slot))
    for ti in order:
        t = tasks[ti]
        todo = t.workload_cu
        for s in range(t.arrival_slot, t.deadline_slot + 1):
            if todo <= 1e-9:
                break
            avail = remaining_cap[s]
            if avail <= 1e-9:
                continue
            placed = min(avail, todo)
            x[ti, s] = placed
            remaining_cap[s] -= placed
            todo -= placed
        shortfall[ti] = todo   # whatever could not be placed (capacity bound)

    # per-slot compute energy
    compute_energy = x.sum(axis=0) * e_cu
    total_demand_energy = compute_energy + e_idle          # per slot

    # renewable first, grid for the rest
    renewable_used = np.minimum(data.renewable, total_demand_energy)
    grid_energy = np.maximum(total_demand_energy - renewable_used, 0.0)

    # cost terms (for fair comparison reporting)
    grid_cost = float((data.price * grid_energy).sum())
    grid_carbon_g = float((data.carbon * grid_energy).sum())
    deadline_pen = float(sum(t.penalty_per_cu * shortfall[i]
                              for i, t in enumerate(tasks)))
    urgent_short = float(sum(shortfall[i] for i, t in enumerate(tasks)
                              if t.is_urgent))

    obj_terms = {
        "electricity_cost_usd": grid_cost,
        "carbon_emissions_kg": grid_carbon_g / 1000.0,
        "deadline_penalty_usd": deadline_pen,
        "urgent_shortfall_cu": urgent_short,
        "weighted_cost_term": config.W_COST * grid_cost,
        "weighted_carbon_term": config.W_CARBON * grid_carbon_g / 1000.0,
        "weighted_deadline_term": config.W_DEADLINE * deadline_pen,
        "weighted_urgent_term": config.W_URGENT * urgent_short,
    }

    return Schedule(
        name="baseline",
        feasible=True,
        solver_status="greedy-EDF (no solver)",
        n_tasks=n_tasks,
        n_slots=n_slots,
        x=x,
        grid_energy=grid_energy,
        renewable_used=renewable_used,
        shortfall=shortfall,
        objective_value=float(config.W_COST * grid_cost
                              + config.W_CARBON * grid_carbon_g / 1000.0
                              + config.W_DEADLINE * deadline_pen
                              + config.W_URGENT * urgent_short),
        cost_terms=obj_terms,
        runtime_s=time.perf_counter() - t0,
        tasks=data.workload,
    )
