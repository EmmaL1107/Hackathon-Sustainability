"""
Synthetic data generator for the AI data-center scheduling problem.

Generates, for a 24h horizon split into 15-min slots:
  * grid electricity price        [$/kWh]
  * grid carbon intensity         [gCO2/kWh]
  * available renewable energy    [kWh per slot]   (solar + wind)
  * max computing capacity        [CU per slot]
  * baseline workload demand      [CU per slot]     (total arrivals)
plus a concrete list of `Task` objects.

Everything is driven by a random seed and a set of multipliers so that the
same scenario can be reproduced and so that `scenarios.py` can perturb it.

ASSUMPTIONS (documented):
  - Solar follows a clipped sin bell, peaking near solar noon (13:00 local).
  - Wind is a slow random walk plus a diurnal term (often stronger at night).
  - Grid price has a low overnight base, a morning shoulder bump and a sharp
    evening peak (typical of real time-of-use tariffs).
  - Carbon intensity is inversely related to renewable share: it falls during
    the sunny/windy hours and rises when the grid leans on fossil peakers
    (evening). We model this explicitly rather than as pure noise.
  - Renewable generation and demand do NOT peak together: demand is biased
    toward business hours while solar peaks midday and wind often at night.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.workload import Task, WorkloadSet


# ---------------------------------------------------------------------------
# Container for everything one experiment needs
# ---------------------------------------------------------------------------
@dataclass
class ScenarioData:
    name: str
    seed: int
    price: np.ndarray            # $/kWh, len N_SLOTS
    carbon: np.ndarray           # gCO2/kWh
    renewable: np.ndarray        # kWh available per slot
    capacity: np.ndarray         # CU per slot
    baseline_demand: np.ndarray  # CU arriving per slot (informational)
    workload: WorkloadSet = None
    battery_enabled: bool = False
    curtailed_renewable: np.ndarray = None   # kWh curtailed (set by optimizer)
    extra: dict = field(default_factory=dict)

    # convenience
    @property
    def n_slots(self) -> int:
        return len(self.price)

    def to_timeseries_df(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "slot": np.arange(self.n_slots),
            "time": [config.slot_to_time_str(s) for s in range(self.n_slots)],
            "hour": [config.slot_to_hour(s) for s in range(self.n_slots)],
            "price_usd_per_kwh": self.price,
            "carbon_gco2_per_kwh": self.carbon,
            "renewable_kwh": self.renewable,
            "capacity_cu": self.capacity,
            "baseline_demand_cu": self.baseline_demand,
        })
        return df


# ---------------------------------------------------------------------------
# Electricity + capacity generation
# ---------------------------------------------------------------------------
def _solar_profile(n_slots: int, peak_kwh: float, seed: int) -> np.ndarray:
    """Bell-shaped solar generation peaking around 13:00."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n_slots) * config.SLOT_HOURS
    # sunrise ~6:00, sunset ~18:30, peak ~13:00
    center = 13.0
    width = 3.2
    base = np.exp(-0.5 * ((hours - center) / width) ** 2)
    base = np.where((hours >= 5.5) & (hours <= 18.5), base, 0.0)
    noise = 1.0 + 0.12 * rng.standard_normal(n_slots)
    return np.clip(base * peak_kwh * noise, 0, None)


def _wind_profile(n_slots: int, peak_kwh: float, seed: int) -> np.ndarray:
    """Wind: slow random walk + slightly stronger at night."""
    rng = np.random.default_rng(seed + 7)
    hours = np.arange(n_slots) * config.SLOT_HOURS
    diurnal = 0.6 + 0.4 * np.cos((hours - 2.0) / 24.0 * 2 * np.pi)  # peak ~02:00
    walk = np.cumsum(rng.standard_normal(n_slots)) * 0.05
    walk -= walk.mean()
    profile = diurnal + 0.4 * (walk - walk.min()) / (walk.max() - walk.min() + 1e-9)
    return np.clip(profile * peak_kwh, 0, None)


def _price_profile(n_slots: int, seed: int,
                   base: float = 0.05, peak: float = 0.22) -> np.ndarray:
    """Time-of-use price: cheap night, morning bump, sharp evening peak."""
    rng = np.random.default_rng(seed + 3)
    hours = np.arange(n_slots) * config.SLOT_HOURS
    # overnight valley 0-6, morning rise 6-10, midday moderate 10-16,
    # evening peak 16-21, late decline 21-24
    price = np.where(
        hours < 6, base,
        np.where(hours < 10, base + 0.06 * (hours - 6) / 4,
        np.where(hours < 16, base + 0.05,
        np.where(hours < 21, peak - 0.05 * (hours - 16) / 5,
                 base + 0.03)))
    )
    price = price + 0.01 * rng.standard_normal(n_slots)
    return np.clip(price, 0.01, None)


def _carbon_profile(price: np.ndarray, renewable: np.ndarray,
                    seed: int) -> np.ndarray:
    """Carbon intensity drops when renewables are abundant; rises at peak."""
    rng = np.random.default_rng(seed + 11)
    n = len(price)
    hours = np.arange(n) * config.SLOT_HOURS
    # fossil-heavy baseline (coal/gas) gCO2/kWh
    base = 450.0
    # renewables displace fossil: more renewable => lower carbon
    max_ren = renewable.max() + 1e-9
    renewable_share = np.clip(renewable / (max_ren * 1.5), 0, 1)
    carbon = base * (1 - 0.6 * renewable_share)
    # evening peakers (gas) push carbon up a bit
    evening = 0.15 * np.exp(-0.5 * ((hours - 19) / 1.8) ** 2)
    carbon = carbon * (1 + evening)
    carbon = carbon + 15 * rng.standard_normal(n)
    return np.clip(carbon, 80.0, 700.0)


# ---------------------------------------------------------------------------
# Workload (tasks) generation
# ---------------------------------------------------------------------------
def _generate_tasks(seed: int, total_daily_cu: float,
                    category_shares: dict, n_slots: int) -> WorkloadSet:
    rng = np.random.default_rng(seed + 21)
    ws = WorkloadSet()
    task_counter = 0
    for cat, share in category_shares.items():
        spec = config.CATEGORIES[cat]
        cat_cu = total_daily_cu * share
        # number of tasks in this category (bigger tasks => fewer LP vars)
        if cat == "urgent":
            n_tasks = max(1, int(cat_cu / 400))      # small frequent tasks
            per = cat_cu / n_tasks
        elif cat == "flexible":
            n_tasks = max(1, int(cat_cu / 2000))
            per = cat_cu / n_tasks
        else:  # training
            n_tasks = max(1, int(cat_cu / 10000))
            per = cat_cu / n_tasks
        # arrival pattern per category
        hours = np.arange(n_slots) * config.SLOT_HOURS
        if cat == "urgent":
            arr_w = 0.7 + 0.3 * np.exp(-0.5 * ((hours - 14) / 5) ** 2)  # day-biased
        elif cat == "flexible":
            arr_w = 0.6 + 0.4 * np.exp(-0.5 * ((hours - 12) / 6) ** 2)
        else:  # training arrives mostly off-peak evening
            arr_w = 0.4 + 0.6 * np.exp(-0.5 * ((hours - 22) / 4) ** 2)
        arr_w = arr_w / arr_w.sum()
        arrivals = rng.choice(n_slots, size=n_tasks, p=arr_w / arr_w.sum(),
                              replace=True)
        for i in range(n_tasks):
            arrival = int(arrivals[i])
            deadline = min(n_slots - 1,
                           arrival + spec["deadline_slots"])
            # small variability in per-task size
            work = max(1.0, per * (0.7 + 0.6 * rng.random()))
            task_counter += 1
            ws.add(Task(
                task_id=f"{cat[:3].upper()}-{task_counter:04d}",
                category=cat,
                arrival_slot=arrival,
                workload_cu=round(work, 2),
                deadline_slot=deadline,
                priority=spec["priority"],
                delayable=spec["delayable"],
                interruptible=spec["interruptible"],
                penalty_per_cu=(20.0 if cat == "urgent"
                                else 5.0 if cat == "flexible" else 1.0),
            ))
    return ws


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------
def generate_scenario(
    name: str = "normal",
    seed: int = 42,
    total_daily_cu: float = 120000.0,
    renewable_scale: float = 1.0,
    price_scale: float = 1.0,
    carbon_scale: float = 1.0,
    capacity_scale: float = 1.0,
    demand_scale: float = 1.0,
    battery_enabled: bool = False,
    category_shares: Optional[dict] = None,
    n_slots: int = config.N_SLOTS,
) -> ScenarioData:
    """Generate a fully specified ScenarioData for one experiment.

    Multipliers let scenarios.py scale renewable/price/demand/etc. without
    touching the underlying generator logic, so baseline and optimizer
    always see identical data.
    """
    if category_shares is None:
        category_shares = {c: config.CATEGORIES[c]["share"]
                           for c in config.CATEGORIES}

    solar_peak = 350.0 * renewable_scale          # kWh per slot
    wind_peak = 150.0 * renewable_scale
    solar = _solar_profile(n_slots, solar_peak, seed)
    wind = _wind_profile(n_slots, wind_peak, seed)
    renewable = solar + wind

    price = _price_profile(n_slots, seed) * price_scale
    carbon = _carbon_profile(price, renewable, seed) * carbon_scale

    capacity = np.full(n_slots, config.MAX_CAPACITY_CU * capacity_scale)

    # baseline demand (informational per-slot arrival envelope)
    hours = np.arange(n_slots) * config.SLOT_HOURS
    demand_shape = (0.5 + 0.5 * np.exp(-0.5 * ((hours - 14) / 5) ** 2)
                    + 0.25 * np.exp(-0.5 * ((hours - 22) / 3) ** 2))
    baseline_demand = demand_shape / demand_shape.mean() * (
        total_daily_cu * demand_scale / n_slots)

    workload = _generate_tasks(seed, total_daily_cu * demand_scale,
                               category_shares, n_slots)

    return ScenarioData(
        name=name,
        seed=seed,
        price=price,
        carbon=carbon,
        renewable=renewable,
        curtailed_renewable=np.zeros(n_slots),
        capacity=capacity,
        baseline_demand=baseline_demand,
        workload=workload,
        battery_enabled=battery_enabled,
        extra={
            "solar": solar,
            "wind": wind,
            "total_daily_cu": total_daily_cu * demand_scale,
        },
    )


def save_scenario(data: ScenarioData, path: str) -> None:
    """Persist the per-slot timeseries and task list as CSV."""
    data.to_timeseries_df().to_csv(f"{path}_timeseries.csv", index=False)
    data.workload.to_dataframe().to_csv(f"{path}_tasks.csv", index=False)


if __name__ == "__main__":
    # quick smoke test
    d = generate_scenario("normal", seed=42)
    print(d.to_timeseries_df().describe().round(3))
    print(d.workload.summary())
