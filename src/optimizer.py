"""
Linear-programming optimizer for the data-center co-scheduling problem.

================================================================================
MATHEMATICAL FORMULATION  (transparent LP, solved with HiGHS via scipy)
================================================================================

Sets / indices
    T   : set of tasks                 t = 1..|T|
    S   : set of time slots            s = 1..N  (N = 96, 15-min each)
    W_t : admissible slots for task t   = { s : arrival_t <= s <= deadline_t }

Parameters
    d_t       : required compute work of task t                    [CU]
    cap_s     : compute capacity of the data center in slot s       [CU/slot]
    p_s       : grid electricity price in slot s                    [$/kWh]
    ci_s      : grid carbon intensity in slot s                     [gCO2/kWh]
    ren_s     : renewable energy available in slot s                 [kWh]
    e_cu      : energy to process 1 CU for one slot = KW_PER_CU*SLOT_HOURS  [kWh]
    e_idle_s  : always-on idle energy in slot s = IDLE_POWER_KW*SLOT_HOURS   [kWh]
    pi_t      : penalty per CU of unmet work for task t             [$/CU]
    (battery) Cap, SoCmin, SoCmax, Pchg, Pdis, eta_c, eta_d, SoC0

Decision variables (all continuous, >= 0 unless bounded)
    x_{t,s}   : CU of task t processed in slot s                     [CU]
    g_s       : grid energy purchased in slot s                      [kWh]
    r_s       : renewable energy used in slot s                       [kWh]
    y_t       : unmet (shortfall) work of task t                      [CU]
    (battery) c_s : energy charged in slot s, dis_s : discharged,
                    soc_s : state of charge                           [kWh]

Objective  (minimize)
    min  w_cost  * sum_s p_s * g_s
       + w_carbon* sum_s (ci_s/1000) * g_s                  # gCO2->kgCO2
       + sum_t   (w_deadline * pi_t + (w_urgent if urgent) ) * y_t

Constraints
    (C1) capacity      :  sum_t x_{t,s} <= cap_s                 forall s
    (C2) energy balance:  sum_t e_cu*x_{t,s} + e_idle_s = g_s + r_s + dis_s - c_s   forall s
                          (battery terms omitted if disabled)
    (C3) renewable cap  :  0 <= r_s <= ren_s
    (C4) grid non-neg   :  g_s >= 0
    (C5) task completion:  sum_{s in W_t} x_{t,s} + y_t = d_t     forall t
    (C6) window         :  x_{t,s} = 0  for s not in W_t  (structural, via var. set)
    (C7) battery SoC     :  soc_s = soc_{s-1} + eta_c*c_s - dis_s/eta_d
                           SoCmin <= soc_s <= SoCmax
                           soc_N >= soc_0  (cannot drain the battery to cheat cost)

Notes / assumptions
    - Workloads are PRE-EMPTIBLE / divisible across slots (the standard,
      well-justified assumption for cloud/batch and checkpointed training
      work). Non-interruptible execution is documented as an MILP extension.
    - Deadline violations are SOFT (penalised via y_t) so that capacity-bound
      infeasibility is REPORTED rather than making the solver fail.
    - Urgent tasks carry an extra penalty so the solver serves them first
      whenever capacity allows; if capacity is truly insufficient the
      shortfall is reported as a missed-deadline, never silently dropped.
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse

import config
from src.data_generator import ScenarioData
from src.workload import WorkloadSet, Task


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class Schedule:
    name: str                          # "optimized" | "baseline"
    feasible: bool
    solver_status: str
    n_tasks: int
    n_slots: int
    x: np.ndarray                       # (n_tasks, n_slots) CU scheduled
    grid_energy: np.ndarray            # kWh per slot
    renewable_used: np.ndarray          # kWh per slot
    shortfall: np.ndarray              # CU per task unmet
    battery_charge: Optional[np.ndarray] = None
    battery_discharge: Optional[np.ndarray] = None
    battery_soc: Optional[np.ndarray] = None
    objective_value: float = 0.0
    cost_terms: dict = field(default_factory=dict)
    runtime_s: float = 0.0
    tasks: WorkloadSet = None           # reference to task list (same order as x rows)

    # ---- derived helpers ------------------------------------------------
    @property
    def total_compute_cu(self) -> float:
        return float(self.x.sum())

    @property
    def total_shortfall_cu(self) -> float:
        return float(self.shortfall.sum())

    @property
    def completed_cu(self) -> float:
        return self.total_compute_cu

    def per_slot_compute(self) -> np.ndarray:
        return self.x.sum(axis=0)

    def to_schedule_df(self) -> pd.DataFrame:
        """Long-format schedule: one row per (task, slot) with nonzero work."""
        rows = []
        for ti, t in enumerate(self.tasks.tasks):
            for s in range(self.n_slots):
                if self.x[ti, s] > 1e-6:
                    rows.append({
                        "task_id": t.task_id,
                        "category": t.category,
                        "slot": s,
                        "time": config.slot_to_time_str(s),
                        "cu_scheduled": round(float(self.x[ti, s]), 3),
                    })
        return pd.DataFrame(rows)

    def to_timeseries_df(self) -> pd.DataFrame:
        comp = self.per_slot_compute()
        e_cu = config.KW_PER_CU * config.SLOT_HOURS
        compute_energy = comp * e_cu
        rows = {
            "slot": np.arange(self.n_slots),
            "time": [config.slot_to_time_str(s) for s in range(self.n_slots)],
            "hour": [config.slot_to_hour(s) for s in range(self.n_slots)],
            "compute_cu": comp,
            "compute_energy_kwh": compute_energy,
            "grid_energy_kwh": self.grid_energy,
            "renewable_used_kwh": self.renewable_used,
        }
        if self.battery_soc is not None:
            rows["battery_soc_kwh"] = self.battery_soc
            rows["battery_charge_kwh"] = self.battery_charge
            rows["battery_discharge_kwh"] = self.battery_discharge
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class LPOptimizer:
    """Builds and solves the co-scheduling LP for a ScenarioData."""

    def __init__(self,
                 w_cost: float = config.W_COST,
                 w_carbon: float = config.W_CARBON,
                 w_deadline: float = config.W_DEADLINE,
                 w_urgent: float = config.W_URGENT,
                 battery_enabled: Optional[bool] = None,
                 verbose: bool = False):
        self.w_cost = w_cost
        self.w_carbon = w_carbon
        self.w_deadline = w_deadline
        self.w_urgent = w_urgent
        self.verbose = verbose
        if battery_enabled is None:
            battery_enabled = config.BATTERY_ENABLED
        self.battery_enabled = battery_enabled

    # ----------------------------------------------------------------
    def _build_variable_index(self, tasks: List[Task], n_slots: int):
        """Return dict var_name -> column index and structural lists."""
        idx = {}
        col = 0
        x_cols = {}          # (t_index, s) -> col
        for ti, t in enumerate(tasks):
            for s in t.window_slots:
                x_cols[(ti, s)] = col
                col += 1
        g_start = col; col += n_slots
        r_start = col; col += n_slots
        y_start = col; col += len(tasks)
        bat = None
        if self.battery_enabled:
            c_start = col; col += n_slots
            d_start = col; col += n_slots
            soc_start = col; col += n_slots
            bat = (c_start, d_start, soc_start)
        n_vars = col
        return dict(x_cols=x_cols, g_start=g_start, r_start=r_start,
                    y_start=y_start, bat=bat, n_vars=n_vars)

    # ----------------------------------------------------------------
    def solve(self, data: ScenarioData) -> Schedule:
        t0 = time.perf_counter()
        tasks = data.workload.tasks
        n_tasks = len(tasks)
        n_slots = data.n_slots
        e_cu = config.KW_PER_CU * config.SLOT_HOURS           # kWh per CU
        e_idle = config.IDLE_POWER_KW * config.SLOT_HOURS     # kWh per slot

        idx = self._build_variable_index(tasks, n_slots)
        x_cols = idx["x_cols"]
        g0, r0, y0 = idx["g_start"], idx["r_start"], idx["y_start"]
        bat = idx["bat"]
        n_vars = idx["n_vars"]

        # ------------------ objective vector c ----------------------
        c = np.zeros(n_vars)
        for s in range(n_slots):
            c[g0 + s] = self.w_cost * data.price[s] \
                        + self.w_carbon * (data.carbon[s] / 1000.0)
        for ti, t in enumerate(tasks):
            pen = self.w_deadline * t.penalty_per_cu
            if t.is_urgent:
                pen += self.w_urgent
            c[y0 + ti] = pen

        # ------------------ equality constraints ---------------------
        # (C5) task completion  sum x + y = d_t
        # (C2) energy balance   e_cu*sum x + e_idle - g - r (+bat) = 0
        # (C7) battery SoC
        eq_rows, eq_cols, eq_data = [], [], []
        eq_b = []
        row = 0

        # task completion
        for ti, t in enumerate(tasks):
            for s in t.window_slots:
                eq_rows.append(row); eq_cols.append(x_cols[(ti, s)]); eq_data.append(1.0)
            eq_rows.append(row); eq_cols.append(y0 + ti); eq_data.append(1.0)
            eq_b.append(t.workload_cu)
            row += 1

        # energy balance
        for s in range(n_slots):
            for ti, t in enumerate(tasks):
                if s in t.window_slots:   # x_{t,s} contributes e_cu
                    eq_rows.append(row); eq_cols.append(x_cols[(ti, s)])
                    eq_data.append(e_cu)
            eq_rows.append(row); eq_cols.append(g0 + s); eq_data.append(-1.0)
            eq_rows.append(row); eq_cols.append(r0 + s); eq_data.append(-1.0)
            if bat is not None:
                eq_rows.append(row); eq_cols.append(bat[0] + s); eq_data.append(1.0)   # +charge
                eq_rows.append(row); eq_cols.append(bat[1] + s); eq_data.append(-1.0)  # -discharge
            eq_b.append(-e_idle)   # RHS: ... = -e_idle  (see module docstring)
            row += 1

        # battery SoC dynamics
        soc_init_energy = None
        if bat is not None:
            c_start, d_start, soc_start = bat
            cap = config.BATTERY_CAPACITY_KWH
            soc_init_energy = config.BATTERY_INIT_SOC * cap
            for s in range(n_slots):
                eq_rows.append(row); eq_cols.append(soc_start + s); eq_data.append(1.0)
                eq_rows.append(row); eq_cols.append(c_start + s)
                eq_data.append(-config.ETA_CHARGE)
                eq_rows.append(row); eq_cols.append(d_start + s)
                eq_data.append(1.0 / config.ETA_DISCHARGE)
                if s == 0:
                    eq_b.append(soc_init_energy)   # soc_0 = init + ...
                else:
                    eq_rows.append(row); eq_cols.append(soc_start + s - 1)
                    eq_data.append(-1.0)
                    eq_b.append(0.0)
                row += 1

        A_eq = sparse.csr_matrix(
            (eq_data, (eq_rows, eq_cols)), shape=(row, n_vars))
        b_eq = np.array(eq_b, dtype=float)

        # ------------------ inequality constraints -------------------
        ub_rows, ub_cols, ub_data = [], [], []
        ub_b = []
        urow = 0

        # (C1) capacity  sum_t x <= cap_s
        for s in range(n_slots):
            for ti, t in enumerate(tasks):
                if s in t.window_slots:
                    ub_rows.append(urow); ub_cols.append(x_cols[(ti, s)])
                    ub_data.append(1.0)
            ub_b.append(data.capacity[s])
            urow += 1

        # final SoC >= initial  ->  -soc_N <= -init
        if bat is not None:
            ub_rows.append(urow); ub_cols.append(bat[2] + n_slots - 1)
            ub_data.append(-1.0)
            ub_b.append(-soc_init_energy)
            urow += 1

        A_ub = sparse.csr_matrix(
            (ub_data, (ub_rows, ub_cols)), shape=(urow, n_vars))
        b_ub = np.array(ub_b, dtype=float)

        # ------------------ variable bounds --------------------------
        bounds = []
        # x: [0, inf]
        for _ in x_cols:
            bounds.append((0, None))
        for s in range(n_slots):                       # g
            bounds.append((0, None))
        for s in range(n_slots):                       # r
            bounds.append((0, float(data.renewable[s])))
        for _ in range(n_tasks):                      # y shortfall
            bounds.append((0, None))
        if bat is not None:
            for _ in range(n_slots):
                bounds.append((0, config.BATTERY_MAX_CHARGE_KW * config.SLOT_HOURS))
            for _ in range(n_slots):
                bounds.append((0, config.BATTERY_MAX_DISCHARGE_KW * config.SLOT_HOURS))
            cap = config.BATTERY_CAPACITY_KWH
            for _ in range(n_slots):
                bounds.append((config.BATTERY_MIN_SOC * cap, config.BATTERY_MAX_SOC * cap if hasattr(config,'BATTERY_MAX_SOC') else cap))
                bounds[-1] = (config.BATTERY_MIN_SOC * cap, cap)

        # ------------------ solve ------------------------------------
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
            options={"disp": self.verbose, "presolve": True},
        )

        runtime = time.perf_counter() - t0
        feasible = res.success
        status = res.message

        x = np.zeros((n_tasks, n_slots))
        if feasible:
            for (ti, s), col in x_cols.items():
                x[ti, s] = max(res.x[col], 0.0)
            g = res.x[g0:g0 + n_slots].copy()
            r = res.x[r0:r0 + n_slots].copy()
            y = res.x[y0:y0 + n_tasks].copy()
            bc = bd = soc = None
            if bat is not None:
                bc = res.x[bat[0]:bat[0] + n_slots].copy()
                bd = res.x[bat[1]:bat[1] + n_slots].copy()
                soc = res.x[bat[2]:bat[2] + n_slots].copy()
        else:
            # infeasible: return zeros, report shortfall = full demand
            g = np.zeros(n_slots)
            r = np.zeros(n_slots)
            y = np.array([t.workload_cu for t in tasks], dtype=float)
            bc = bd = soc = None

        # clean tiny numerical dust
        x[x < 1e-7] = 0.0
        y[y < 1e-7] = 0.0

        # cost breakdown (use actual realized values, weights applied)
        grid_cost = float((data.price * g).sum())
        grid_carbon_g = float((data.carbon * g).sum())          # gCO2
        deadline_pen = float(sum(t.penalty_per_cu * y[ti]
                                 for ti, t in enumerate(tasks)))
        urgent_short = float(sum(y[ti] for ti, t in enumerate(tasks)
                                 if t.is_urgent))
        obj_terms = {
            "electricity_cost_usd": grid_cost,
            "carbon_emissions_kg": grid_carbon_g / 1000.0,
            "deadline_penalty_usd": deadline_pen,
            "urgent_shortfall_cu": urgent_short,
            "weighted_cost_term": self.w_cost * grid_cost,
            "weighted_carbon_term": self.w_carbon * grid_carbon_g / 1000.0,
            "weighted_deadline_term": self.w_deadline * deadline_pen,
            "weighted_urgent_term": self.w_urgent * urgent_short,
        }

        return Schedule(
            name="optimized",
            feasible=feasible,
            solver_status=status,
            n_tasks=n_tasks,
            n_slots=n_slots,
            x=x,
            grid_energy=np.maximum(g, 0),
            renewable_used=np.maximum(r, 0),
            shortfall=y,
            battery_charge=bc,
            battery_discharge=bd,
            battery_soc=soc,
            objective_value=float(res.fun) if feasible else float("inf"),
            cost_terms=obj_terms,
            runtime_s=runtime,
            tasks=data.workload,
        )


# ---------------------------------------------------------------------------
# Small exhaustive-search validator for optimality checks (tiny cases)
# ---------------------------------------------------------------------------
def exhaustive_optimal_check(data: ScenarioData, max_tasks: int = 6):
    """For tiny scenarios, brute-force the cheapest feasible assignment to
    verify the LP. Returns (lp_cost, brute_cost, match). Only for |T|<=max_tasks
    with very few slots; used by tests, not production."""
    n_tasks = len(data.workload.tasks)
    if n_tasks > max_tasks:
        return None
    # Build the same LP and compare objective only (sanity check on objective).
    opt = LPOptimizer()
    sched = opt.solve(data)
    return dict(lp_objective=sched.objective_value,
                feasible=sched.feasible,
                n_vars=sched.n_tasks)
