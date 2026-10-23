"""
Sensitivity / robustness experiments (Feedback 3).

Three complementary sweeps that test how the optimizer's conclusions hold up
when the underlying assumptions change. Each returns a tidy DataFrame so the
dashboard can plot it directly.

1. ``sensitivity_sweep``         - one-parameter-at-a-time sensitivity analysis
   that varies the compute-to-power coefficient (KW_PER_CU), renewable
   availability and electricity price, in turn, on the *normal* scenario.

2. ``forecast_error_sweep``     - how rolling-horizon performance degrades as
   the price/renewable forecast becomes more inaccurate. Compares against the
   perfect-forecast rolling horizon and the full-horizon LP optimum.

3. ``workload_flexibility_comparison`` - three workload profiles (flexible,
   strict, balanced) with different category mixes and deadlines, showing
   how much the optimizer saves in each regime.

All numbers come from real solver executions (HiGHS LP via scipy.optimize.linprog
and the greedy EDF baseline), no placeholders.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.data_generator import generate_scenario, ScenarioData
from src import baseline
from src.optimizer import LPOptimizer
from src.metrics import compute_metrics
from src.simulator import rolling_horizon


# ---------------------------------------------------------------------------
# Tiny helper: temporarily override config.* attributes
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _temp_config(**overrides):
    """Context manager that temporarily swaps config attributes and restores
    them on exit. Used because optimizer/baseline/metrics read config at
    call-time, so the override actually takes effect."""
    old = {k: getattr(config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def _run_pair(data: ScenarioData) -> dict:
    """Run baseline + weighted optimizer on one ScenarioData and return the
    compact metric record used by all sweeps."""
    base_sched = baseline.schedule(data)
    opt = LPOptimizer(battery_enabled=data.battery_enabled)
    opt_sched = opt.solve(data)
    bm = compute_metrics(base_sched, data)
    om = compute_metrics(opt_sched, data)
    return dict(
        baseline_cost_usd=bm.total_cost_usd,
        optimized_cost_usd=om.total_cost_usd,
        cost_reduction_pct=(
            (bm.total_cost_usd - om.total_cost_usd)
            / bm.total_cost_usd * 100 if bm.total_cost_usd > 0 else 0.0),
        baseline_carbon_kg=bm.total_carbon_kg,
        optimized_carbon_kg=om.total_carbon_kg,
        carbon_reduction_pct=(
            (bm.total_carbon_kg - om.total_carbon_kg)
            / bm.total_carbon_kg * 100 if bm.total_carbon_kg > 0 else 0.0),
        baseline_missed=bm.missed_deadlines,
        optimized_missed=om.missed_deadlines,
        baseline_urgent_rate_pct=bm.urgent_service_rate_pct,
        optimized_urgent_rate_pct=om.urgent_service_rate_pct,
        optimized_unmet_cu=round(
            om.unmet_urgent_cu + om.unmet_flexible_cu + om.unmet_training_cu, 1),
        optimized_runtime_s=om.runtime_s,
        optimized_feasible=om.feasible,
    )


# ===========================================================================
# 1. Sensitivity sweep  (one-parameter-at-a-time)
# ===========================================================================
def sensitivity_sweep(seed: int = 42) -> pd.DataFrame:
    """One-parameter-at-a-time sensitivity analysis on the *normal* scenario.

    Three sub-sweeps are concatenated in the returned DataFrame:
      - ``kw_per_cu``        : compute-to-power coefficient [0.30, 0.40, 0.50,
                                0.60, 0.80] kW per active CU. Changes how much
                                electrical energy each CU draws.
      - ``renewable_scale`` : renewable availability multiplier [0.25, 0.50,
                                1.00, 1.50, 2.00] on solar+wind peaks.
      - ``price_scale``     : grid electricity price multiplier [0.50, 1.00,
                                1.50, 2.00, 3.00].

    Returns a DataFrame with one row per (parameter, value) pair and the
    baseline/optimized metrics for that perturbation.
    """
    records = []

    # --- 1a. KW_PER_CU sweep (override config; data unchanged) ---
    kw_values = [0.30, 0.40, 0.50, 0.60, 0.80]
    for kw in kw_values:
        with _temp_config(KW_PER_CU=kw):
            data = generate_scenario("normal", seed=seed)
            rec = _run_pair(data)
        rec.update(parameter="kw_per_cu", value=kw, baseline_value=0.50)
        records.append(rec)

    # --- 1b. renewable_scale sweep (generate scenario with multiplier) ---
    ren_values = [0.25, 0.50, 1.00, 1.50, 2.00]
    for r in ren_values:
        data = generate_scenario("normal", seed=seed, renewable_scale=r)
        rec = _run_pair(data)
        rec.update(parameter="renewable_scale", value=r, baseline_value=1.00)
        records.append(rec)

    # --- 1c. price_scale sweep ---
    price_values = [0.50, 1.00, 1.50, 2.00, 3.00]
    for p in price_values:
        data = generate_scenario("normal", seed=seed, price_scale=p)
        rec = _run_pair(data)
        rec.update(parameter="price_scale", value=p, baseline_value=1.00)
        records.append(rec)

    return pd.DataFrame(records)


# ===========================================================================
# 2. Forecast-error sweep (rolling horizon degradation)
# ===========================================================================
def _make_perturbed_forecast(actual: ScenarioData,
                             perturbation: float,
                             seed: int = 42) -> ScenarioData:
    """Build an imperfect forecast by perturbing price and renewable by
    ``perturbation * N(0,1)``. perturbation=0 -> perfect forecast."""
    rng = np.random.default_rng(seed + 999)
    f = generate_scenario(actual.name + "_forecast", seed=seed)
    if perturbation <= 0:
        # perfect forecast: copy actual exactly
        f.price = actual.price.copy()
        f.renewable = actual.renewable.copy()
        f.carbon = actual.carbon.copy()
    else:
        f.price = np.clip(
            actual.price * (1 + perturbation * rng.standard_normal(actual.n_slots)),
            0.01, None)
        f.renewable = np.clip(
            actual.renewable * (1 + perturbation * rng.standard_normal(actual.n_slots)),
            0, None)
        f.carbon = np.clip(
            actual.carbon * (1 + 0.5 * perturbation * rng.standard_normal(actual.n_slots)),
            80, 700)
    f.capacity = actual.capacity
    f.workload = actual.workload
    f.baseline_demand = actual.baseline_demand
    return f


def forecast_error_sweep(seed: int = 42,
                         scenario_name: str = "normal"
                         ) -> pd.DataFrame:
    """Run the rolling-horizon scheduler with forecasts of varying accuracy
    and measure how the realized cost degrades vs the perfect-forecast
    rolling horizon and the full-horizon LP optimum.

    Perturbation levels: 0%, 5%, 10%, 20%, 30%, 50% multiplicative noise on
    price and renewable. ``perturbation=0`` is the perfect-forecast case.

    Returns a DataFrame with one row per perturbation level and the metrics:
      - rolling-horizon cost (realized against actual data)
      - cost degradation vs perfect-forecast rolling horizon [%]
      - cost degradation vs full-horizon LP optimum [%]
      - missed deadlines, urgent service rate
    """
    from src import scenarios as scn
    actual = scn.get_scenario(scenario_name, seed=seed)

    # references: perfect-forecast rolling horizon + full-horizon LP optimum
    rh_perfect = rolling_horizon(actual, use_forecast=False)
    rh_perfect_metrics = compute_metrics(rh_perfect, actual)
    opt_full = LPOptimizer().solve(actual)
    opt_full_metrics = compute_metrics(opt_full, actual)

    perturbations = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]
    records = []
    for p in perturbations:
        forecast = _make_perturbed_forecast(actual, p, seed=seed)
        rh = rolling_horizon(actual, use_forecast=True, forecast=forecast)
        m = compute_metrics(rh, actual)
        # degradations: positive means we got *worse* (higher cost)
        deg_vs_perfect = (
            (m.total_cost_usd - rh_perfect_metrics.total_cost_usd)
            / rh_perfect_metrics.total_cost_usd * 100
            if rh_perfect_metrics.total_cost_usd > 0 else 0.0)
        deg_vs_optimal = (
            (m.total_cost_usd - opt_full_metrics.total_cost_usd)
            / opt_full_metrics.total_cost_usd * 100
            if opt_full_metrics.total_cost_usd > 0 else 0.0)
        records.append({
            "perturbation_pct": round(p * 100, 1),
            "rh_cost_usd": m.total_cost_usd,
            "rh_carbon_kg": m.total_carbon_kg,
            "rh_missed": m.missed_deadlines,
            "rh_urgent_rate_pct": m.urgent_service_rate_pct,
            "rh_runtime_s": m.runtime_s,
            "perfect_forecast_cost_usd": rh_perfect_metrics.total_cost_usd,
            "full_horizon_optimum_cost_usd": opt_full_metrics.total_cost_usd,
            "degradation_vs_perfect_pct": round(deg_vs_perfect, 2),
            "degradation_vs_optimal_pct": round(deg_vs_optimal, 2),
        })
    return pd.DataFrame(records)


# ===========================================================================
# 3. Workload flexibility comparison
# ===========================================================================
# (category_shares, deadline_slots) for each profile.
# deadline_slots = None means "use config defaults".
_WORKLOAD_PROFILES = {
    "flexible": dict(
        shares={"urgent": 0.10, "flexible": 0.40, "training": 0.50},
        deadline_slots={"urgent": 8, "flexible": 32, "training": 80},
        label="Highly flexible (long deadlines, training-heavy)",
    ),
    "balanced": dict(
        shares={"urgent": 0.20, "flexible": 0.45, "training": 0.35},
        deadline_slots=None,
        label="Balanced (default config)",
    ),
    "strict": dict(
        shares={"urgent": 0.40, "flexible": 0.35, "training": 0.25},
        deadline_slots={"urgent": 2, "flexible": 8, "training": 24},
        label="Strict (short deadlines, urgent-heavy)",
    ),
}


def _override_categories(deadline_slots: Optional[dict]) -> dict:
    """Build a deep-copied CATEGORIES dict with overridden deadline_slots."""
    import copy
    cats = copy.deepcopy(config.CATEGORIES)
    if deadline_slots is not None:
        for cat, dl in deadline_slots.items():
            if cat in cats:
                cats[cat]["deadline_slots"] = dl
    return cats


def workload_flexibility_comparison(seed: int = 42) -> pd.DataFrame:
    """Compare the optimizer's gains across three workload regimes:
    highly flexible, balanced (default), and strict (urgent-heavy, short
    deadlines). For each profile we generate a fresh scenario with the
    appropriate category mix AND temporarily override ``config.CATEGORIES``
    so the generator's deadline_slots match.

    Returns a DataFrame with one row per profile and the baseline/optimized
    cost, carbon, service metrics.
    """
    records = []
    for profile_name, spec in _WORKLOAD_PROFILES.items():
        cats_override = _override_categories(spec["deadline_slots"])
        with _temp_config(CATEGORIES=cats_override):
            data = generate_scenario(
                f"flex_{profile_name}", seed=seed,
                category_shares=spec["shares"])
            rec = _run_pair(data)
        rec.update(
            workload_profile=profile_name,
            profile_label=spec["label"],
            urgent_share=spec["shares"]["urgent"],
            flexible_share=spec["shares"]["flexible"],
            training_share=spec["shares"]["training"],
            total_daily_cu=120000.0,
        )
        records.append(rec)
    return pd.DataFrame(records)


# ===========================================================================
# CLI smoke test
# ===========================================================================
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("=== 1. Sensitivity sweep ===")
    df1 = sensitivity_sweep()
    print(df1.to_string(index=False))

    print("\n=== 2. Forecast-error sweep ===")
    df2 = forecast_error_sweep()
    print(df2.to_string(index=False))

    print("\n=== 3. Workload flexibility comparison ===")
    df3 = workload_flexibility_comparison()
    print(df3.to_string(index=False))
