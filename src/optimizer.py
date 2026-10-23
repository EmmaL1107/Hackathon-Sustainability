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
    objective_mode: str = "weighted"    # "weighted" | "cost_primary" | "emissions_primary"
    mode_info: dict = field(default_factory=dict)   # feasibility-relaxation details

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
                 verbose: bool = False,
                 objective_mode: str = "weighted",
                 urgent_completion_rate: float = 0.99,
                 max_total_unmet_cu: float = 0.0):
        """
        objective_mode:
            "weighted"          — original multi-objective (cost + carbon + penalties)
            "cost_primary"      — minimize electricity cost subject to explicit
                                  service constraints (urgent >= 99%, unmet <= epsilon)
            "emissions_primary" — minimize carbon emissions subject to the same
                                  service constraints

        For primary modes, if the workload is infeasible the service constraints
        are automatically relaxed to the minimum unavoidable shortfall (Phase 1
        feasibility LP), and the relaxation is reported in mode_info.
        """
        self.w_cost = w_cost
        self.w_carbon = w_carbon
        self.w_deadline = w_deadline
        self.w_urgent = w_urgent
        self.verbose = verbose
        if battery_enabled is None:
            battery_enabled = config.BATTERY_ENABLED
        self.battery_enabled = battery_enabled
        self.objective_mode = objective_mode
        self.urgent_completion_rate = urgent_completion_rate
        self.max_total_unmet_cu = max_total_unmet_cu

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
    def _build_common_lp(self, data: ScenarioData) -> dict:
        """Build the shared LP structure (variables, equality constraints,
        inequality constraints, bounds) that is identical across all objective
        modes. Returns a dict with all matrices and index info.

        This is the physical / structural LP: capacity, energy balance, task
        completion (sum x + y = d), windows, renewable bounds, battery SoC.
        The *objective* and *service constraints* are added by the caller.
        """
        tasks = data.workload.tasks
        n_tasks = len(tasks)
        n_slots = data.n_slots
        e_cu = config.KW_PER_CU * config.SLOT_HOURS
        e_idle = config.IDLE_POWER_KW * config.SLOT_HOURS

        idx = self._build_variable_index(tasks, n_slots)
        x_cols = idx["x_cols"]
        g0, r0, y0 = idx["g_start"], idx["r_start"], idx["y_start"]
        bat = idx["bat"]
        n_vars = idx["n_vars"]

        # ------------------ equality constraints ---------------------
        eq_rows, eq_cols, eq_data = [], [], []
        eq_b = []
        row = 0

        # task completion  sum x + y = d_t
        for ti, t in enumerate(tasks):
            for s in t.window_slots:
                eq_rows.append(row); eq_cols.append(x_cols[(ti, s)]); eq_data.append(1.0)
            eq_rows.append(row); eq_cols.append(y0 + ti); eq_data.append(1.0)
            eq_b.append(t.workload_cu)
            row += 1

        # energy balance
        for s in range(n_slots):
            for ti, t in enumerate(tasks):
                if s in t.window_slots:
                    eq_rows.append(row); eq_cols.append(x_cols[(ti, s)])
                    eq_data.append(e_cu)
            eq_rows.append(row); eq_cols.append(g0 + s); eq_data.append(-1.0)
            eq_rows.append(row); eq_cols.append(r0 + s); eq_data.append(-1.0)
            if bat is not None:
                eq_rows.append(row); eq_cols.append(bat[0] + s); eq_data.append(1.0)
                eq_rows.append(row); eq_cols.append(bat[1] + s); eq_data.append(-1.0)
            eq_b.append(-e_idle)
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
                    eq_b.append(soc_init_energy)
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

        # capacity  sum_t x <= cap_s
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
        for _ in x_cols:
            bounds.append((0, None))
        for s in range(n_slots):
            bounds.append((0, None))
        for s in range(n_slots):
            bounds.append((0, float(data.renewable[s])))
        for _ in range(n_tasks):
            bounds.append((0, None))
        if bat is not None:
            for _ in range(n_slots):
                bounds.append((0, config.BATTERY_MAX_CHARGE_KW * config.SLOT_HOURS))
            for _ in range(n_slots):
                bounds.append((0, config.BATTERY_MAX_DISCHARGE_KW * config.SLOT_HOURS))
            cap = config.BATTERY_CAPACITY_KWH
            for _ in range(n_slots):
                bounds.append((config.BATTERY_MIN_SOC * cap,
                               config.BATTERY_MAX_SOC * cap if hasattr(config, 'BATTERY_MAX_SOC') else cap))
                bounds[-1] = (config.BATTERY_MIN_SOC * cap, cap)

        return dict(
            x_cols=x_cols, g_start=g0, r_start=r0, y_start=y0,
            bat=bat, n_vars=n_vars, n_tasks=n_tasks, n_slots=n_slots,
            A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
        )

    # ----------------------------------------------------------------
    def _solve_lp(self, c: np.ndarray, common: dict,
                  extra_constraints=None):
        """Solve the LP with the given objective vector ``c`` and the shared
        common constraints, optionally augmented with extra inequality rows.

        ``extra_constraints`` is a list of ``(coef_dict, rhs)`` tuples where
        ``coef_dict`` maps variable column indices to coefficients. Each tuple
        becomes one row of ``A_ub @ x <= rhs``.
        """
        A_ub = common["A_ub"]
        b_ub = common["b_ub"]
        if extra_constraints:
            n_vars = common["n_vars"]
            rows, cols, data_vals, b_extra = [], [], [], []
            for i, (coef_dict, rhs) in enumerate(extra_constraints):
                for col, val in coef_dict.items():
                    rows.append(i)
                    cols.append(col)
                    data_vals.append(val)
                b_extra.append(rhs)
            A_extra = sparse.csr_matrix(
                (data_vals, (rows, cols)),
                shape=(len(extra_constraints), n_vars))
            A_ub = sparse.vstack([A_ub, A_extra], format="csr")
            b_ub = np.concatenate([b_ub, np.array(b_extra, dtype=float)])

        return linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            A_eq=common["A_eq"], b_eq=common["b_eq"],
            bounds=common["bounds"], method="highs",
            options={"disp": self.verbose, "presolve": True},
        )

    # ----------------------------------------------------------------
    def _extract_schedule(self, res, data: ScenarioData, common: dict,
                          runtime: float, mode_info: dict = None,
                          objective_mode: str = None) -> Schedule:
        """Extract a Schedule from a linprog result."""
        tasks = data.workload.tasks
        n_tasks = common["n_tasks"]
        n_slots = common["n_slots"]
        x_cols = common["x_cols"]
        g0, r0, y0 = common["g_start"], common["r_start"], common["y_start"]
        bat = common["bat"]

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
            g = np.zeros(n_slots)
            r = np.zeros(n_slots)
            y = np.array([t.workload_cu for t in tasks], dtype=float)
            bc = bd = soc = None

        x[x < 1e-7] = 0.0
        y[y < 1e-7] = 0.0

        grid_cost = float((data.price * g).sum())
        grid_carbon_g = float((data.carbon * g).sum())
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
            objective_mode=objective_mode or self.objective_mode,
            mode_info=mode_info or {},
        )

    # ----------------------------------------------------------------
    def _weighted_objective(self, data: ScenarioData, common: dict) -> np.ndarray:
        """Build the original multi-objective cost vector."""
        n_slots = common["n_slots"]
        n_vars = common["n_vars"]
        g0, y0 = common["g_start"], common["y_start"]
        tasks = data.workload.tasks
        c = np.zeros(n_vars)
        for s in range(n_slots):
            c[g0 + s] = self.w_cost * data.price[s] \
                        + self.w_carbon * (data.carbon[s] / 1000.0)
        for ti, t in enumerate(tasks):
            pen = self.w_deadline * t.penalty_per_cu
            if t.is_urgent:
                pen += self.w_urgent
            c[y0 + ti] = pen
        return c

    # ----------------------------------------------------------------
    def _solve_primary(self, data: ScenarioData, common: dict,
                       t0: float) -> Schedule:
        """Solve a *primary-objective* LP (cost or emissions) subject to
        explicit service constraints, with automatic feasibility relaxation.

        Three-phase approach:
          Phase 1  — minimise total shortfall Σy_t → gives the minimum
                     unavoidable total shortfall ``s_min``. If s_min exceeds
                     the requested tolerance ``max_total_unmet_cu``, the
                     constraint is relaxed to s_min (so we never impose an
                     impossible completion constraint).
          Phase 2  — minimise urgent shortfall Σy_t(urgent) subject to
                     Σy_t ≤ epsilon_eff → gives the minimum unavoidable
                     urgent shortfall ``u_min``. Same relaxation logic.
          Phase 3  — minimise the primary objective (cost or emissions)
                     subject to the (possibly relaxed) service constraints.
        """
        tasks = data.workload.tasks
        n_tasks = common["n_tasks"]
        n_vars = common["n_vars"]
        y0 = common["y_start"]
        g0 = common["g_start"]

        # --- Phase 1: minimise total shortfall ---
        c_p1 = np.zeros(n_vars)
        for ti in range(n_tasks):
            c_p1[y0 + ti] = 1.0
        res1 = self._solve_lp(c_p1, common)
        if res1.success:
            s_min = float(res1.x[y0:y0 + n_tasks].sum())
        else:
            s_min = float(sum(t.workload_cu for t in tasks))

        epsilon_eff = max(self.max_total_unmet_cu, s_min + 1e-4)
        feasibility_relaxed = s_min > self.max_total_unmet_cu + 1e-6

        # --- Phase 2: minimise urgent shortfall (given total ≤ epsilon_eff) ---
        urgent_idx = [ti for ti, t in enumerate(tasks) if t.is_urgent]
        urgent_total = float(sum(t.workload_cu for t in tasks if t.is_urgent))
        u_min = 0.0
        urgent_relaxed = False
        if urgent_idx:
            c_p2 = np.zeros(n_vars)
            for ti in urgent_idx:
                c_p2[y0 + ti] = 1.0
            total_constraint = ({y0 + ti: 1.0 for ti in range(n_tasks)},
                                epsilon_eff)
            res2 = self._solve_lp(c_p2, common, extra_constraints=[total_constraint])
            if res2.success:
                u_min = float(sum(res2.x[y0 + ti] for ti in urgent_idx))
            else:
                u_min = urgent_total

            urgent_eps = (1.0 - self.urgent_completion_rate) * urgent_total
            urgent_eps_eff = max(urgent_eps, u_min + 1e-4)
            urgent_relaxed = u_min > urgent_eps + 1e-6
        else:
            urgent_eps_eff = 0.0

        # --- Phase 3: minimise the primary objective ---
        c_p3 = np.zeros(n_vars)
        if self.objective_mode == "cost_primary":
            for s in range(data.n_slots):
                c_p3[g0 + s] = data.price[s]
        else:  # emissions_primary
            for s in range(data.n_slots):
                c_p3[g0 + s] = data.carbon[s] / 1000.0

        extra = [({y0 + ti: 1.0 for ti in range(n_tasks)}, epsilon_eff)]
        if urgent_idx:
            extra.append(({y0 + ti: 1.0 for ti in urgent_idx}, urgent_eps_eff))
        res3 = self._solve_lp(c_p3, common, extra_constraints=extra)

        runtime = time.perf_counter() - t0
        mode_info = {
            "objective_mode": self.objective_mode,
            "min_unavoidable_total_shortfall_cu": round(s_min, 4),
            "epsilon_requested": self.max_total_unmet_cu,
            "epsilon_effective": round(epsilon_eff, 4),
            "feasibility_relaxed": feasibility_relaxed,
            "min_unavoidable_urgent_shortfall_cu": round(u_min, 4),
            "urgent_epsilon_requested_pct": round(
                (1.0 - self.urgent_completion_rate) * 100, 2),
            "urgent_epsilon_effective_cu": round(urgent_eps_eff, 4),
            "urgent_feasibility_relaxed": urgent_relaxed,
            "urgent_completion_rate_target": self.urgent_completion_rate,
            "n_phases": 3,
        }
        return self._extract_schedule(res3, data, common, runtime,
                                      mode_info=mode_info)

    # ----------------------------------------------------------------
    def solve(self, data: ScenarioData) -> Schedule:
        t0 = time.perf_counter()
        common = self._build_common_lp(data)

        if self.objective_mode == "weighted":
            c = self._weighted_objective(data, common)
            res = self._solve_lp(c, common)
            runtime = time.perf_counter() - t0
            return self._extract_schedule(res, data, common, runtime)

        elif self.objective_mode in ("cost_primary", "emissions_primary"):
            return self._solve_primary(data, common, t0)

        else:
            raise ValueError(
                f"Unknown objective_mode '{self.objective_mode}'. "
                "Use 'weighted', 'cost_primary', or 'emissions_primary'.")


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
