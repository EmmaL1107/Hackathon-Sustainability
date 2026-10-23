"""Tests for the workload data model and synthetic data realism."""
import numpy as np
import pytest

import config
from src.data_generator import generate_scenario
from src.workload import Task, WorkloadSet


def test_window_slots_inclusive():
    t = Task("T", "flexible", arrival_slot=10, workload_cu=100,
            deadline_slot=20, priority=2, delayable=True, interruptible=True)
    assert t.window_slots == list(range(10, 21))


def test_urgent_has_short_deadline(normal_data):
    for t in normal_data.workload.tasks:
        if t.category == "urgent":
            assert t.deadline_slot - t.arrival_slot <= config.CATEGORIES["urgent"]["deadline_slots"]


def test_three_categories_present(normal_data):
    cats = {t.category for t in normal_data.workload.tasks}
    assert {"urgent", "flexible", "training"}.issubset(cats)


def test_solar_higher_in_daytime(normal_data):
    solar = normal_data.extra["solar"]
    day = solar[24:72]      # 06:00 - 18:00
    night = np.concatenate([solar[:24], solar[72:]])
    assert day.mean() > night.mean() * 3   # much more solar in daytime


def test_demand_and_renewable_not_always_peaked_together(normal_data):
    # Genuine temporal misalignment: substantial demand persists in the
    # evening (training surge) while solar is ~0 -- this is exactly what makes
    # energy-aware scheduling valuable (work can be shifted to the solar peak).
    solar = normal_data.extra["solar"]
    demand = normal_data.baseline_demand
    evening = np.array([s for s in range(96) if 20 <= config.slot_to_hour(s) < 24])
    day_solar = np.array([s for s in range(96) if 10 <= config.slot_to_hour(s) < 14])
    assert demand[evening].mean() > 0.3 * demand[day_solar].mean()  # real evening load
    assert solar[evening].mean() < 0.05 * solar[day_solar].mean()    # ~no solar at night


def test_horizon_24h_15min_slots(normal_data):
    assert normal_data.n_slots == config.N_SLOTS == 96
    assert config.SLOT_HOURS == 0.25


def test_total_workload_positive(normal_data):
    assert normal_data.workload.total_workload_cu > 0
    assert len(normal_data.workload.tasks) > 0


def test_workloadset_dataframe_and_summary(normal_data):
    df = normal_data.workload.to_dataframe()
    assert {"task_id", "category", "arrival_slot", "deadline_slot",
            "workload_cu", "priority"}.issubset(df.columns)
    summ = normal_data.workload.summary()
    assert len(summ) == 3
