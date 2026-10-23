"""Energy-balance and renewable-availability tests across methods."""
import numpy as np

import config
from src.optimizer import LPOptimizer
from src import baseline
from src.simulator import rolling_horizon


def _check_balance(sched, data, battery=False):
    e_cu = config.KW_PER_CU * config.SLOT_HOURS
    e_idle = config.IDLE_POWER_KW * config.SLOT_HOURS
    compute_energy = sched.per_slot_compute() * e_cu + e_idle
    if battery:
        supply = (sched.grid_energy + sched.renewable_used
                  + sched.battery_discharge - sched.battery_charge)
    else:
        supply = sched.grid_energy + sched.renewable_used
    np.testing.assert_allclose(compute_energy, supply, atol=1e-3)


def test_baseline_energy_balance(normal_data):
    _check_balance(baseline.schedule(normal_data), normal_data)


def test_optimizer_energy_balance(normal_data):
    _check_balance(LPOptimizer().solve(normal_data), normal_data)


def test_battery_energy_balance(normal_data):
    _check_balance(LPOptimizer(battery_enabled=True).solve(normal_data),
                   normal_data, battery=True)


def test_rolling_horizon_energy_balance(normal_data):
    rh = rolling_horizon(normal_data, use_forecast=False)
    _check_balance(rh, normal_data)


def test_renewable_never_exceeds_available(normal_data):
    for s in (baseline.schedule(normal_data), LPOptimizer().solve(normal_data)):
        assert np.all(s.renewable_used <= normal_data.renewable + 1e-6)
