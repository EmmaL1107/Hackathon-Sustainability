"""
Experiment driver: runs baseline vs optimizer on a scenario and provides
rolling-horizon online scheduling.

Provides:
  * run_experiment  - one scenario, full horizon, baseline + optimized
  * run_all_scenarios - sweep every stress-test scenario (optionally multi-seed)
  * rolling_horizon - online re-optimization (perfect-forecast or forecast-error)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

import config
from src.data_generator import generate_scenario, ScenarioData, save_scenario
from src.workload import WorkloadSet, Task
from src.optimizer import LPOptimizer, Schedule
from src import baseline
from src.metrics import compute_metrics, compare, MetricSet
from src import scenarios as scn


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------
def run_experiment(scenario_name: str = "normal", seed: int = 42,
                   battery_enabled: bool = False,
                   weights: Optional[dict] = None) -> dict:
    """Run baseline + optimized on one scenario. Returns schedules + metrics."""
    data = scn.get_scenario(scenario_name, seed=seed)
    data.battery_enabled = battery_enabled

    # weights: scenario-builtin overrides -> explicit arg
    eff_w = dict(w_cost=config.W_COST, w_carbon=config.W_CARBON,
                 w_deadline=config.W_DEADLINE, w_urgent=config.W_URGENT)
    if weights:
        eff_w.update(weights)
    elif "weights" in data.extra:
        eff_w.update(data.extra["weights"])

    opt = LPOptimizer(battery_enabled=battery_enabled, **eff_w)
    opt_sched = opt.solve(data)
    base_sched = baseline.schedule(data)

    bm = compute_metrics(base_sched, data)
    om = compute_metrics(opt_sched, data)
    return dict(
        scenario=scenario_name, seed=seed, data=data,
        baseline_schedule=base_sched, optimized_schedule=opt_sched,
        baseline_metrics=bm, optimized_metrics=om,
        comparison=compare(bm, om),
        weights=eff_w,
    )


# ---------------------------------------------------------------------------
# Scenario sweep
# ---------------------------------------------------------------------------
def run_all_scenarios(seeds: Optional[List[int]] = None,
                      battery_enabled: bool = False) -> pd.DataFrame:
    """Run every scenario (optionally across seeds) and return a summary."""
    if seeds is None:
        seeds = [42]
    records = []
    for sname in scn.SCENARIOS:
        for seed in seeds:
            res = run_experiment(sname, seed=seed,
                                 battery_enabled=battery_enabled)
            bm, om = res["baseline_metrics"], res["optimized_metrics"]
            records.append({
                "scenario": sname,
                "label": scn.SCENARIOS[sname].label,
                "seed": seed,
                "baseline_cost_usd": bm.total_cost_usd,
                "optimized_cost_usd": om.total_cost_usd,
                "cost_reduction_pct": (
                    (bm.total_cost_usd - om.total_cost_usd)
                    / bm.total_cost_usd * 100 if bm.total_cost_usd else 0),
                "baseline_carbon_kg": bm.total_carbon_kg,
                "optimized_carbon_kg": om.total_carbon_kg,
                "carbon_reduction_pct": (
                    (bm.total_carbon_kg - om.total_carbon_kg)
                    / bm.total_carbon_kg * 100 if bm.total_carbon_kg else 0),
                "baseline_missed": bm.missed_deadlines,
                "optimized_missed": om.missed_deadlines,
                "baseline_urgent_rate": bm.urgent_service_rate_pct,
                "optimized_urgent_rate": om.urgent_service_rate_pct,
                "optimized_runtime_s": om.runtime_s,
                "optimized_feasible": om.feasible,
            })
    df = pd.DataFrame(records)
    if len(seeds) > 1:
        agg = df.groupby("scenario").agg(
            cost_reduction_pct=("cost_reduction_pct", "mean"),
            cost_reduction_std=("cost_reduction_pct", "std"),
            carbon_reduction_pct=("carbon_reduction_pct", "mean"),
            carbon_reduction_std=("carbon_reduction_pct", "std"),
            optimized_missed_mean=("optimized_missed", "mean"),
        ).reset_index()
        return df, agg
    return df


# ---------------------------------------------------------------------------
# Rolling-horizon online scheduling
# ---------------------------------------------------------------------------
def _build_residual_scenario(actual: ScenarioData, forecast: ScenarioData,
                              t: int, remaining_cu: np.ndarray) -> ScenarioData:
    """Build a planning sub-scenario for slots [t, N) using forecast data and
    each task's remaining workload (CU not yet completed)."""
    n = actual.n_slots
    rem_slots = list(range(t, n))
    ns = len(rem_slots)
    if ns == 0:
        return None
    tasks = []
    for ti, task in enumerate(actual.workload.tasks):
        rem = remaining_cu[ti]
        if rem <= 1e-6:
            continue
        # convert absolute slots to relative indices within [0, ns)
        arr = max(task.arrival_slot, t) - t
        dl = task.deadline_slot - t
        if dl < 0:
            # already past deadline: represent as 1-slot task at 0 with full
            # remaining (will create shortfall -> reported as missed)
            arr = 0
            dl = 0
        dl = min(dl, ns - 1)
        tasks.append(Task(
            task_id=task.task_id + "_r",
            category=task.category,
            arrival_slot=arr,
            workload_cu=round(float(rem), 4),
            deadline_slot=dl,
            priority=task.priority,
            delayable=task.delayable,
            interruptible=task.interruptible,
            penalty_per_cu=task.penalty_per_cu,
        ))
    ws = WorkloadSet(tasks=tasks)
    sub = ScenarioData(
        name=actual.name + f"_rh{t}",
        seed=actual.seed,
        price=forecast.price[t:n].copy(),
        carbon=forecast.carbon[t:n].copy(),
        renewable=forecast.renewable[t:n].copy(),
        capacity=actual.capacity[t:n].copy(),
        baseline_demand=actual.baseline_demand[t:n].copy(),
        workload=ws,
        battery_enabled=actual.battery_enabled,
    )
    return sub


def rolling_horizon(actual: ScenarioData,
                    use_forecast: bool = False,
                    forecast: Optional[ScenarioData] = None,
                    weights: Optional[dict] = None,
                    battery_enabled: bool = False) -> Schedule:
    """Online rolling-horizon scheduler.

    At each slot t we re-optimize the remaining horizon using either the
    *forecast* (imperfect) or the *actual* (perfect-forecast) data, execute
    only slot t's compute decision, then advance. Executed-slot energy is
    recomputed against ACTUAL renewable (realized truth).
    """
    n = actual.n_slots
    n_tasks = len(actual.workload.tasks)
    e_cu = config.KW_PER_CU * config.SLOT_HOURS
    e_idle = config.IDLE_POWER_KW * config.SLOT_HOURS

    if use_forecast:
        if forecast is None:
            forecast = scn.make_forecast(actual)
    else:
        forecast = actual   # perfect forecast

    eff_w = dict(w_cost=config.W_COST, w_carbon=config.W_CARBON,
                 w_deadline=config.W_DEADLINE, w_urgent=config.W_URGENT)
    if weights:
        eff_w.update(weights)

    x = np.zeros((n_tasks, n))
    remaining = np.array([t.workload_cu for t in actual.workload.tasks],
                         dtype=float)
    grid = np.zeros(n)
    ren_used = np.zeros(n)
    total_runtime = 0.0
    status = "rolling-horizon"

    for t in range(n):
        sub = _build_residual_scenario(actual, forecast, t, remaining)
        if sub is None or len(sub.workload) == 0:
            # no remaining work; idle energy still drawn
            ren_used[t] = min(actual.renewable[t], e_idle)
            grid[t] = max(e_idle - ren_used[t], 0)
            continue
        opt = LPOptimizer(battery_enabled=battery_enabled, **eff_w)
        sched = opt.solve(sub)
        total_runtime += sched.runtime_s
        if not sched.feasible:
            status = "rolling-horizon (infeasible sub-problem at t=%d)" % t
        # map slot t of sub-problem (= slot 0) back to actual slot t
        col0 = sched.x[:, 0] if sched.x.shape[1] > 0 else np.zeros(len(sub.workload))
        # apply executed compute for slot t to global task list
        placed = 0.0
        for si, stask in enumerate(sub.workload.tasks):
            # find original task index (task_id without "_r")
            orig_id = stask.task_id[:-2]
            ti = next(i for i, tk in enumerate(actual.workload.tasks)
                      if tk.task_id == orig_id)
            amt = min(float(col0[si]), remaining[ti])
            x[ti, t] = amt
            remaining[ti] -= amt
            placed += amt
        # executed energy using ACTUAL renewable for realized slot t
        compute_energy = placed * e_cu + e_idle
        ru = min(actual.renewable[t], compute_energy)
        ren_used[t] = ru
        grid[t] = max(compute_energy - ru, 0)

    # shortfall = whatever remains unmet at end. Clean floating-point residue:
    # any remainder below 0.01 CU (vs task sizes of hundreds-to-thousands) is
    # numerical noise from 96-step accumulation, treated as fully completed.
    shortfall = remaining.copy()
    shortfall[shortfall < 1e-2] = 0.0

    grid_cost = float((actual.price * grid).sum())
    grid_carbon_g = float((actual.carbon * grid).sum())
    deadline_pen = float(sum(t.penalty_per_cu * shortfall[i]
                              for i, t in enumerate(actual.workload.tasks)))
    urgent_short = float(sum(shortfall[i] for i, t in enumerate(actual.workload.tasks)
                              if t.is_urgent))
    obj_terms = {
        "electricity_cost_usd": grid_cost,
        "carbon_emissions_kg": grid_carbon_g / 1000.0,
        "deadline_penalty_usd": deadline_pen,
        "urgent_shortfall_cu": urgent_short,
    }
    return Schedule(
        name="rolling_horizon" + ("_forecast" if use_forecast else "_perfect"),
        feasible=True,
        solver_status=status,
        n_tasks=n_tasks,
        n_slots=n,
        x=x,
        grid_energy=grid,
        renewable_used=ren_used,
        shortfall=shortfall,
        objective_value=float(config.W_COST * grid_cost
                              + config.W_CARBON * grid_carbon_g / 1000.0
                              + config.W_DEADLINE * deadline_pen
                              + config.W_URGENT * urgent_short),
        cost_terms=obj_terms,
        runtime_s=total_runtime,
        tasks=actual.workload,
    )


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------
def export_experiment(res: dict, out_prefix: str) -> None:
    save_scenario(res["data"], out_prefix + "_data")
    res["baseline_schedule"].to_timeseries_df().to_csv(
        out_prefix + "_baseline_schedule.csv", index=False)
    res["optimized_schedule"].to_timeseries_df().to_csv(
        out_prefix + "_optimized_schedule.csv", index=False)
    res["optimized_schedule"].to_schedule_df().to_csv(
        out_prefix + "_task_gantt.csv", index=False)
    res["comparison"].to_csv(out_prefix + "_comparison.csv", index=False)


# ---------------------------------------------------------------------------
# Objective-mode comparison (Feedback 1: weighted vs cost_primary vs emissions_primary)
# ---------------------------------------------------------------------------
def run_objective_comparison(scenario_name: str = "normal", seed: int = 42,
                             battery_enabled: bool = False) -> pd.DataFrame:
    """Run all three objective modes on the same scenario and return a
    side-by-side comparison table.

    Modes:
      - weighted:          original multi-objective (cost + carbon + penalties)
      - cost_primary:      minimize electricity cost subject to service constraints
      - emissions_primary: minimize carbon emissions subject to service constraints
    """
    data = scn.get_scenario(scenario_name, seed=seed)
    data.battery_enabled = battery_enabled
    base_sched = baseline.schedule(data)
    bm = compute_metrics(base_sched, data)
    records = [{
        "method": "Baseline",
        "objective_mode": "—",
        "cost_usd": bm.total_cost_usd,
        "carbon_kg": bm.total_carbon_kg,
        "missed": bm.missed_deadlines,
        "unmet_cu": round(bm.unmet_urgent_cu + bm.unmet_flexible_cu + bm.unmet_training_cu, 1),
        "unmet_urgent_cu": bm.unmet_urgent_cu,
        "unmet_flexible_cu": bm.unmet_flexible_cu,
        "unmet_training_cu": bm.unmet_training_cu,
        "urgent_rate_pct": bm.urgent_service_rate_pct,
        "feasibility_relaxed": False,
        "min_unavoidable_shortfall_cu": 0.0,
    }]
    for mode in ["weighted", "cost_primary", "emissions_primary"]:
        opt = LPOptimizer(battery_enabled=battery_enabled, objective_mode=mode)
        sched = opt.solve(data)
        m = compute_metrics(sched, data)
        mi = sched.mode_info
        records.append({
            "method": f"Optimized ({mode})",
            "objective_mode": mode,
            "cost_usd": m.total_cost_usd,
            "carbon_kg": m.total_carbon_kg,
            "missed": m.missed_deadlines,
            "unmet_cu": round(m.unmet_urgent_cu + m.unmet_flexible_cu + m.unmet_training_cu, 1),
            "unmet_urgent_cu": m.unmet_urgent_cu,
            "unmet_flexible_cu": m.unmet_flexible_cu,
            "unmet_training_cu": m.unmet_training_cu,
            "urgent_rate_pct": m.urgent_service_rate_pct,
            "feasibility_relaxed": mi.get("feasibility_relaxed", False),
            "min_unavoidable_shortfall_cu": mi.get("min_unavoidable_total_shortfall_cu", 0.0),
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=== single experiment (normal) ===")
    res = run_experiment("normal", seed=42)
    print(res["comparison"].to_string(index=False))
    print("\n=== scenario sweep ===")
    df = run_all_scenarios()
    print(df.to_string(index=False))
    print("\n=== objective-mode comparison (normal) ===")
    print(run_objective_comparison("normal").to_string(index=False))
