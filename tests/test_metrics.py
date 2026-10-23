"""Tests for metrics, cost/emissions reductions, and reproducibility."""
import numpy as np
import pytest

import config
from src.data_generator import generate_scenario
from src.optimizer import LPOptimizer
from src import baseline
from src.metrics import compute_metrics, compare
from src.simulator import run_experiment


# ------------------------------------------------------------------ metric defs
def test_cost_reduction_definition(normal_data):
    base = compute_metrics(baseline.schedule(normal_data), normal_data)
    opt = compute_metrics(LPOptimizer().solve(normal_data), normal_data)
    cmp = compare(base, opt)
    cost_row = cmp[cmp["metric"] == "Total electricity cost [$]"].iloc[0]
    expected = ((base.total_cost_usd - opt.total_cost_usd)
               / base.total_cost_usd * 100) if base.total_cost_usd else 0
    # compare() rounds the reported pct to 2 decimals -> allow that tolerance
    assert abs(cost_row["improvement_vs_baseline_pct"] - expected) < 1e-2


def test_zero_denominator_safe(normal_data):
    base = compute_metrics(baseline.schedule(normal_data), normal_data)
    # construct an optimized with zero cost -> reduction should be 100, not crash
    opt_sched = LPOptimizer().solve(normal_data)
    opt_sched.grid_energy = np.zeros_like(opt_sched.grid_energy)
    opt = compute_metrics(opt_sched, normal_data)
    cmp = compare(base, opt)
    row = cmp[cmp["metric"] == "Total electricity cost [$]"].iloc[0]
    assert row["improvement_vs_baseline_pct"] == 100.0


def test_renewable_share_definition(normal_data):
    sched = LPOptimizer().solve(normal_data)
    m = compute_metrics(sched, normal_data)
    total_energy = sched.grid_energy.sum() + sched.renewable_used.sum()
    expected_share = sched.renewable_used.sum() / total_energy * 100
    # metric rounds to 2 decimals
    assert abs(m.renewable_share_pct - expected_share) < 1e-2


# ------------------------------------------------------------------ reproducibility
def test_reproducibility_same_seed():
    d1 = generate_scenario("normal", seed=42)
    d2 = generate_scenario("normal", seed=42)
    np.testing.assert_allclose(d1.price, d2.price)
    np.testing.assert_allclose(d1.renewable, d2.renewable)
    assert d1.workload.total_workload_cu == d2.workload.total_workload_cu
    # optimizer deterministic
    s1 = LPOptimizer().solve(d1)
    s2 = LPOptimizer().solve(d2)
    np.testing.assert_allclose(s1.x, s2.x, atol=1e-9)


def test_reproducibility_different_seed():
    d1 = generate_scenario("normal", seed=42)
    d2 = generate_scenario("normal", seed=7)
    # different seeds -> different data
    assert not np.allclose(d1.price, d2.price)


# ------------------------------------------------------------------ comparison fairness
def test_equal_workload_baseline_optimized(normal_data):
    """Baseline and optimizer must process the same total workload."""
    bs = baseline.schedule(normal_data)
    os = LPOptimizer().solve(normal_data)
    assert abs(bs.x.sum() - os.x.sum()) < 1.0  # within 1 CU (numerical)


def test_optimized_beats_baseline_cost(normal_data):
    bm = compute_metrics(baseline.schedule(normal_data), normal_data)
    om = compute_metrics(LPOptimizer().solve(normal_data), normal_data)
    # optimizer should not spend more (it minimizes cost)
    assert om.total_cost_usd <= bm.total_cost_usd + 1e-3


# ------------------------------------------------------------------ experiment runner
def test_run_experiment_returns_all_pieces(normal_data):
    res = run_experiment("normal", seed=42)
    for key in ("baseline_schedule", "optimized_schedule",
                "baseline_metrics", "optimized_metrics", "comparison"):
        assert key in res
    assert res["comparison"].shape[0] == 11
