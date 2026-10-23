"""
Scenario definitions for robustness / stress testing.

Each scenario calls generate_scenario(...) with multipliers that perturb the
*same underlying generator*, so baseline and optimizer always see identical
data. Weights override objective weights when relevant.

The 8 required scenarios:
 1. normal            - moderate workload, ordinary renewables
 2. renewable_short   - renewable availability falls substantially
 3. price_spike       - grid electricity temporarily very expensive
 4. high_demand       - compute demand approaches / exceeds capacity
 5. workload_surge    - many tasks arrive unexpectedly (late seed diff)
 6. forecast_error    - predicted price/renewable differ from actual
 7. carbon_priority   - high carbon weight
 8. cost_priority     - high cost weight
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

import config
from src.data_generator import generate_scenario, ScenarioData


@dataclass
class ScenarioConfig:
    name: str
    label: str
    description: str
    build: Callable[..., ScenarioData]
    weights: dict = None           # override objective weights
    battery_enabled: bool = False


def _normal(seed: int = 42, **kw) -> ScenarioData:
    return generate_scenario("normal", seed=seed, **kw)


def _renewable_short(seed: int = 42, **kw) -> ScenarioData:
    return generate_scenario("renewable_short", seed=seed,
                             renewable_scale=0.25, **kw)


def _price_spike(seed: int = 42, **kw) -> ScenarioData:
    d = generate_scenario("price_spike", seed=seed, **kw)
    # superimpose a sharp price spike in the evening (17:00-21:00)
    spike_slots = [s for s in range(d.n_slots)
                   if 17 <= config.slot_to_hour(s) < 21]
    d.price = d.price.copy()
    d.price[spike_slots] += 0.60        # +$0.60/kWh
    return d


def _high_demand(seed: int = 42, **kw) -> ScenarioData:
    return generate_scenario("high_demand", seed=seed,
                             total_daily_cu=320000.0, **kw)


def _workload_surge(seed: int = 42, **kw) -> ScenarioData:
    # different seed -> different arrival pattern, plus more urgent tasks
    d = generate_scenario("workload_surge", seed=seed + 100,
                           total_daily_cu=180000.0,
                           category_shares={"urgent": 0.40,
                                            "flexible": 0.35,
                                            "training": 0.25}, **kw)
    return d


def _forecast_error(seed: int = 42, **kw) -> ScenarioData:
    """Returns the *actual* scenario; the simulator separately builds a
    *forecast* scenario with perturbed price/renewable used only for the
    rolling-horizon decisions. See simulator.rolling_horizon."""
    return generate_scenario("forecast_error_actual", seed=seed, **kw)


def make_forecast(actual: ScenarioData, seed: int = 42) -> ScenarioData:
    """Perturb the actual scenario to produce an imperfect forecast used
    ONLY inside rolling-horizon re-optimization (never the ground truth)."""
    rng = np.random.default_rng(seed + 999)
    f = generate_scenario(actual.name + "_forecast", seed=seed)
    f.price = np.clip(actual.price * (1 + 0.20 * rng.standard_normal(actual.n_slots)), 0.01, None)
    f.renewable = np.clip(actual.renewable * (1 + 0.25 * rng.standard_normal(actual.n_slots)), 0, None)
    # carbon roughly tracks actual (slight noise)
    f.carbon = actual.carbon * (1 + 0.10 * rng.standard_normal(actual.n_slots))
    f.carbon = np.clip(f.carbon, 80, 700)
    # capacity & workload identical (observed truth)
    f.capacity = actual.capacity
    f.workload = actual.workload
    f.baseline_demand = actual.baseline_demand
    return f


def _carbon_priority(seed: int = 42, **kw) -> ScenarioData:
    d = generate_scenario("carbon_priority", seed=seed, **kw)
    d.extra["weights"] = dict(w_cost=0.05, w_carbon=2.0,
                              w_deadline=config.W_DEADLINE,
                              w_urgent=config.W_URGENT)
    return d


def _cost_priority(seed: int = 42, **kw) -> ScenarioData:
    d = generate_scenario("cost_priority", seed=seed, **kw)
    d.extra["weights"] = dict(w_cost=3.0, w_carbon=0.001,
                              w_deadline=config.W_DEADLINE,
                              w_urgent=config.W_URGENT)
    return d


SCENARIOS: Dict[str, ScenarioConfig] = {
    "normal": ScenarioConfig("normal", "Normal conditions",
        "Moderate workload, ordinary renewable generation.", _normal),
    "renewable_short": ScenarioConfig("renewable_short", "Renewable shortage",
        "Renewable availability falls to ~25% of normal.", _renewable_short),
    "price_spike": ScenarioConfig("price_spike", "Electricity price spike",
        "Sharp $0.60/kWh price spike 17:00-21:00.", _price_spike),
    "high_demand": ScenarioConfig("high_demand", "High demand",
        "Compute demand near data-center capacity.", _high_demand),
    "workload_surge": ScenarioConfig("workload_surge", "Sudden workload surge",
        "More urgent tasks, larger total workload, different arrival pattern.",
        _workload_surge),
    "forecast_error": ScenarioConfig("forecast_error", "Forecast error",
        "Imperfect price/renewable forecasts (used by rolling horizon).",
        _forecast_error),
    "carbon_priority": ScenarioConfig("carbon_priority", "Carbon priority",
        "Optimizer weights emissions heavily.", _carbon_priority,
        weights=dict(w_cost=0.05, w_carbon=2.0)),
    "cost_priority": ScenarioConfig("cost_priority", "Cost priority",
        "Optimizer weights expenditure heavily.", _cost_priority,
        weights=dict(w_cost=3.0, w_carbon=0.001)),
}


def get_scenario(name: str, seed: int = 42) -> ScenarioData:
    return SCENARIOS[name].build(seed=seed)


def list_scenarios() -> Dict[str, str]:
    return {k: v.label for k, v in SCENARIOS.items()}
