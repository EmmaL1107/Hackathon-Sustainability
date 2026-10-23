# AI Data Center Power–Computing Co-Scheduling

An intelligent scheduling system for an AI data center that jointly optimizes
**when computing tasks run**, **how much capacity is used each period**, and
**how grid vs. renewable electricity (and an optional battery) is allocated** —
minimizing electricity cost and carbon emissions while respecting task
deadlines and urgent service requirements.

Built for **HackSHSID 2026 — Track 2**. The optimizer is a transparent
**linear program** solved with HiGHS (via `scipy.optimize.linprog`), not a
black-box ML model.

---

## 1. Quick start

```bash
# from the project root
pip install -r requirements.txt

# run the test suite (32 tests)
python -m pytest tests/ -q

# run one experiment + the full 8-scenario sweep (prints tables)
python -m src.simulator

# launch the interactive dashboard
streamlit run app.py
```

Open the dashboard at the URL Streamlit prints (default http://localhost:8501).

---

## 2. Project structure

```
app.py                  Streamlit multi-section dashboard
config.py               unit conventions + all tunable constants
requirements.txt
src/
    data_generator.py   synthetic data + ScenarioData container
    workload.py         Task / WorkloadSet data model
    optimizer.py        LP optimizer (scipy HiGHS)
    baseline.py         non-energy-aware ASAP baseline
    simulator.py        experiment driver + rolling-horizon
    metrics.py          economic/environmental/service/computational metrics
    scenarios.py        8 stress-test scenarios + forecast generator
    visualization.py    matplotlib chart helpers
tests/
    conftest.py
    test_optimizer.py      feasibility, capacity, windows, battery, optimality
    test_metrics.py        metric definitions, reproducibility, fairness
    test_energy_balance.py energy balance across all methods
    test_workloads.py      workload model + data realism
data/generated/         saved CSV datasets
results/                saved experiment CSVs
```

---

## 3. Unit conventions (consistent across the whole system)

| Quantity | Unit | Notes |
|---|---|---|
| Time | 15-min slot | 24 h horizon → 96 slots; `slot_hours = 0.25` |
| Compute work / capacity | CU (Compute Unit) | 1 CU = a fixed normalized chunk of work; capacity is CU/slot |
| Compute → power | `KW_PER_CU` kW per active CU | single transparent coefficient |
| Energy | kWh | `energy = power × slot_hours` → `e_cu = KW_PER_CU × 0.25` kWh per CU |
| Price | $/kWh | grid time-of-use |
| Carbon intensity | gCO2/kWh | reported as kgCO2 (÷1000) |
| Battery | kWh (energy terms) | SoC in kWh, charge/discharge in kWh/slot |

---

## 4. Mathematical model

### Sets / indices
- `T`: tasks `t`; `S`: slots `s = 1..96`; `W_t`: admissible slots
  `{s : arrival_t ≤ s ≤ deadline_t}`.

### Decision variables (continuous, ≥ 0 unless bounded)
- `x_{t,s}` — CU of task `t` processed in slot `s`
- `g_s` — grid energy purchased [kWh]
- `r_s ∈ [0, ren_s]` — renewable energy used [kWh]
- `y_t` — unmet (shortfall) work of task `t` [CU]
- (battery) `c_s, dis_s, soc_s` — charge, discharge, state of charge [kWh]

### Objective (minimize)

```
min  w_cost   · Σ_s p_s · g_s
   + w_carbon · Σ_s (ci_s / 1000) · g_s
   + Σ_t ( w_deadline · π_t  +  w_urgent·𝟙[t urgent] ) · y_t
```

`π_t` is a per-category penalty (`urgent`=20, `flexible`=5, `training`=1 $/CU).

### Constraints

- **(C1) Capacity** — `Σ_t x_{t,s} ≤ cap_s`
- **(C2) Energy balance** — `Σ_t e_cu·x_{t,s} + e_idle = g_s + r_s + dis_s − c_s`
- **(C3) Renewable cap** — `0 ≤ r_s ≤ ren_s`
- **(C4) Grid non-negativity** — `g_s ≥ 0`
- **(C5) Task completion** — `Σ_{s∈W_t} x_{t,s} + y_t = d_t`
- **(C6) Window** — `x_{t,s} = 0` for `s ∉ W_t` (structural)
- **(C7) Battery SoC** — `soc_s = soc_{s−1} + η_c·c_s − dis_s/η_d`,
  `SoC_min ≤ soc_s ≤ SoC_max`, `soc_N ≥ soc_0` (no end-of-day draining)

The whole model is a **linear program**, solved to optimality by HiGHS.

---

## 5. Algorithm explanation

**Optimizer (proposed).** The co-scheduling problem is formulated as the LP
above. Because workloads are divisible across slots, all decision variables
are continuous, so HiGHS returns the **global optimum of the LP relaxation**
in ~0.02–0.05 s for ~90 tasks × 96 slots (~2 000 variables). The solver
shifts deferrable work (flexible/training) into the renewable-rich, low-carbon,
cheap midday window while serving urgent work as soon as it arrives.

**Baseline (comparison).** A realistic "dumb" data center: an
earliest-deadline-first greedy that runs each task **as early as capacity
allows** within its window, drawing renewable first then the grid. It does
**not** look at price or carbon when deciding *when* to run. This makes the
comparison fair: identical workload, identical electricity/capacity data.

**Rolling horizon (online).** At each slot `t` the system re-solves the LP
over the remaining horizon `[t, N)` using either the **perfect forecast**
(actual future values) or an **imperfect forecast** (perturbed price/renewable),
executes only slot `t`'s decision against the *realized* renewable, then
advances. This lets us distinguish perfect-forecast optimization from
realistic online scheduling.

### Why this is credible
- Transparent LP, no black box — every variable/constraint is inspectable.
- HiGHS returns a proven-optimal LP solution (solver status reported).
- Battery dynamics, energy balance and SoC bounds verified to machine precision
  (see `test_optimizer.py`, `test_energy_balance.py`).
- All metrics computed from actual solver outputs — no hardcoded numbers.

### When it can fail / struggle
- **Renewable shortage** (`renewable_short`): only ~15 % carbon reduction
  (little green energy to shift into).
- **High demand** (`high_demand`): capacity saturates, so the optimizer cannot
  move work freely — 8 end-of-horizon tasks cannot finish within 24 h
  (reported honestly as missed deadlines; total unmet CU equals baseline's).
- **Forecast error**: rolling-horizon with imperfect forecasts costs more than
  the full-horizon optimum (~$175 vs ~$70) but still beats the baseline ($454).
- Tasks are modelled as **pre-emptible**; strict non-interruptible execution
  would need binary variables (MILP) — noted as an extension.

---

## 6. Metric definitions

```
cost reduction (%)   = (baseline_cost − opt_cost) / baseline_cost × 100
emissions red. (%)   = (baseline_emis − opt_emis) / baseline_emis × 100
renewable share (%)  = renewable_used / (renewable_used + grid_energy) × 100
renewable util (%)   = renewable_used / renewable_available × 100
deadline miss rate   = missed_deadlines / n_tasks × 100
urgent service rate  = urgent_completed_CU / urgent_total_CU × 100
```

Zero denominators return 0 (handled in `config.safe_div` / `metrics.compare`).
For metrics where *higher is better* (renewable share, urgent rate), the
improvement % sign is flipped so positive always means "optimizer better".

---

## 7. Reproducing the experiments

```bash
# 1. tests
python -m pytest tests/ -q

# 2. single experiment + 8-scenario sweep (prints tables)
python -m src.simulator

# 3. per-scenario CSV exports + multi-seed variability (seeds 1-5) written to results/
python -c "from src.simulator import run_experiment, export_experiment; \
 from src import scenarios as scn; import config, os; \
 [export_experiment(run_experiment(n, seed=42), os.path.join(config.RESULTS_DIR, n)) for n in scn.SCENARIOS]"

# 4. interactive dashboard
streamlit run app.py
```

Generated CSVs land in `results/` (`<scenario>_timeseries.csv`,
`<scenario>_tasks.csv`, `<scenario>_comparison.csv`,
`<scenario>_optimized_schedule.csv`, `<scenario>_task_gantt.csv`,
`multiseed_normal.csv`, `scenario_sweep.csv`).

---

## 8. Results (seed = 42, actual runs)

| scenario | baseline cost | opt cost | cost ↓ | carbon ↓ | missed |
|---|---|---|---|---|---|
| normal | $453.8 | $69.6 | 84.7% | 83.7% | 0 |
| renewable_short | $1004.4 | $569.5 | 43.3% | 14.7% | 0 |
| price_spike | $930.1 | $69.6 | 92.5% | 83.7% | 0 |
| high_demand | $1917.7 | $1526.4 | 20.4% | 18.2% | 8 |
| workload_surge | $990.5 | $254.7 | 74.3% | 63.3% | 0 |
| forecast_error* | $453.8 | $69.6 | 84.7% | 83.7% | 0 |
| carbon_priority | $453.8 | $74.2 | 83.6% | 83.7% | 0 |
| cost_priority | $453.8 | $69.6 | 84.7% | 83.7% | 0 |

\* full-horizon optimizer; the *rolling* forecast-error variant costs $175.1
(still 61% below baseline).

**Multi-seed (normal, seeds 1–5):** mean cost reduction **87.4 % ± 14.0 %**,
mean carbon reduction **78.0 % ± 24.7 %** — variability reported honestly.

---

## 9. Assumptions & limitations

- Workloads are **pre-emptible/divisible** across slots (standard for cloud
  batch and checkpointed training). Non-interruptible execution = MILP
  extension.
- Deadline violations are **soft** (penalised via `y_t`) so capacity-bound
  infeasibility is *reported*, never silently dropped.
- Single transparent compute→power coefficient `KW_PER_CU`.
- Renewable is free and used first; curtailment allowed.
- Horizon = 24 h; late-arriving tasks with end-of-day deadlines may be
  infeasible within the window (reported as missed, not hidden).
- No global-optimality claim beyond the LP solver status; the LP is a
  relaxation of the true integer scheduling problem.
- Synthetic data is configurable and seed-driven; assumptions (solar bell,
  wind walk, time-of-use price, carbon–renewable coupling) are documented in
  `src/data_generator.py` so real data can replace it later.
