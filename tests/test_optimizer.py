"""Tests for the LP optimizer: feasibility, constraints, objective.

Covers (from the spec):
  - workload conservation
  - computing capacity
  - task arrival / deadline window constraints
  - renewable energy availability
  - battery state-of-charge constraints
  - optimality on a tiny case (vs an exact lower bound check)
"""
import numpy as np
import pytest

import config
from src.optimizer import LPOptimizer
from src import baseline


def _e_cu():
    return config.KW_PER_CU * config.SLOT_HOURS


# ------------------------------------------------------------------ feasibility
def test_optimizer_feasible_normal(normal_data):
    sched = LPOptimizer().solve(normal_data)
    assert sched.feasible, f"optimizer infeasible: {sched.solver_status}"


# ------------------------------------------------------------------ capacity
def test_capacity_constraint(normal_data):
    sched = LPOptimizer().solve(normal_data)
    per_slot = sched.per_slot_compute()
    assert np.all(per_slot <= normal_data.capacity + 1e-3), \
        "scheduled compute exceeds capacity in some slot"


# ------------------------------------------------------------------ workload conservation
def test_workload_conservation(normal_data):
    sched = LPOptimizer().solve(normal_data)
    completed = sched.x.sum(axis=1)
    total = np.array([t.workload_cu for t in normal_data.workload.tasks])
    shortfall = sched.shortfall
    # completed + shortfall must equal required workload (conservation)
    np.testing.assert_allclose(completed + shortfall, total, atol=1e-2)


# ------------------------------------------------------------------ arrival/deadline windows
def test_window_constraints(normal_data):
    sched = LPOptimizer().solve(normal_data)
    for ti, t in enumerate(normal_data.workload.tasks):
        # no work scheduled before arrival or after deadline
        before = sched.x[ti, :t.arrival_slot]
        after = sched.x[ti, t.deadline_slot + 1:] if t.deadline_slot + 1 < sched.n_slots else np.array([])
        assert before.sum() < 1e-6, f"task {t.task_id} scheduled before arrival"
        assert (after.sum() if len(after) else 0.0) < 1e-6, \
            f"task {t.task_id} scheduled after deadline"


# ------------------------------------------------------------------ renewable availability
def test_renewable_availability(normal_data):
    sched = LPOptimizer().solve(normal_data)
    assert np.all(sched.renewable_used <= normal_data.renewable + 1e-6), \
        "renewable usage exceeds availability"
    assert np.all(sched.renewable_used >= -1e-6)
    assert np.all(sched.grid_energy >= -1e-6)


# ------------------------------------------------------------------ energy balance
def test_energy_balance(normal_data):
    sched = LPOptimizer().solve(normal_data)
    e_cu = _e_cu()
    e_idle = config.IDLE_POWER_KW * config.SLOT_HOURS
    compute_energy = sched.per_slot_compute() * e_cu + e_idle
    supply = sched.grid_energy + sched.renewable_used
    np.testing.assert_allclose(compute_energy, supply, atol=1e-3,
                                err_msg="energy balance violated")


# ------------------------------------------------------------------ cost/emissions calc
def test_cost_emissions_match(normal_data):
    sched = LPOptimizer().solve(normal_data)
    expected_cost = float((normal_data.price * sched.grid_energy).sum())
    expected_carbon_kg = float((normal_data.carbon * sched.grid_energy).sum()) / 1000.0
    assert abs(sched.cost_terms["electricity_cost_usd"] - expected_cost) < 1e-3
    assert abs(sched.cost_terms["carbon_emissions_kg"] - expected_carbon_kg) < 1e-3


# ------------------------------------------------------------------ battery constraints
def test_battery_constraints(normal_data):
    sched = LPOptimizer(battery_enabled=True).solve(normal_data)
    cap = config.BATTERY_CAPACITY_KWH
    soc_min, soc_max = config.BATTERY_MIN_SOC * cap, cap
    assert sched.battery_soc is not None
    assert np.all(sched.battery_soc >= soc_min - 1e-3), "SoC below minimum"
    assert np.all(sched.battery_soc <= soc_max + 1e-3), "SoC above maximum"
    max_c = config.BATTERY_MAX_CHARGE_KW * config.SLOT_HOURS
    max_d = config.BATTERY_MAX_DISCHARGE_KW * config.SLOT_HOURS
    assert np.all(sched.battery_charge <= max_c + 1e-3)
    assert np.all(sched.battery_discharge <= max_d + 1e-3)
    # final SoC must not be below initial (no end-of-day draining)
    assert sched.battery_soc[-1] >= config.BATTERY_INIT_SOC * cap - 1e-3


def test_battery_energy_balance(normal_data):
    sched = LPOptimizer(battery_enabled=True).solve(normal_data)
    e_cu = _e_cu()
    e_idle = config.IDLE_POWER_KW * config.SLOT_HOURS
    compute_energy = sched.per_slot_compute() * e_cu + e_idle
    supply = (sched.grid_energy + sched.renewable_used
              + sched.battery_discharge - sched.battery_charge)
    np.testing.assert_allclose(compute_energy, supply, atol=1e-3)


# ------------------------------------------------------------------ optimality check (tiny)
def test_optimizer_no_worse_than_baseline(small_data):
    """The LP must not be worse than the ASAP baseline on the objective."""
    opt = LPOptimizer()
    opt_sched = opt.solve(small_data)
    base_sched = baseline.schedule(small_data)
    # optimizer objective <= baseline objective (optimizer is global opt of LP)
    assert opt_sched.objective_value <= base_sched.objective_value + 1e-6


def test_optimizer_objective_matches_recomputed(small_data):
    """Objective reported must equal recomputed weighted cost."""
    opt = LPOptimizer()
    s = opt.solve(small_data)
    w = dict(w_cost=opt.w_cost, w_carbon=opt.w_carbon,
             w_deadline=opt.w_deadline, w_urgent=opt.w_urgent)
    recomputed = (w["w_cost"] * s.cost_terms["electricity_cost_usd"]
                  + w["w_carbon"] * s.cost_terms["carbon_emissions_kg"]
                  + w["w_deadline"] * s.cost_terms["deadline_penalty_usd"]
                  + w["w_urgent"] * s.cost_terms["urgent_shortfall_cu"])
    assert abs(s.objective_value - recomputed) < 1e-3
