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
    simulator.py        experiment driver + rolling-horizon + objective-mode comparison
    metrics.py          economic/environmental/service/computational metrics
                         + per-category unmet CU + miss-cause diagnosis
    scenarios.py        8 stress-test scenarios + forecast generator
    sensitivity.py      sensitivity / forecast-error / workload-flexibility sweeps
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

The optimizer supports **three objective modes** so the cost–carbon–service
trade-off is explicit rather than hidden behind arbitrary weights.

**Mode A — weighted multi-objective (default):**

```
min  w_cost   · Σ_s p_s · g_s
   + w_carbon · Σ_s (ci_s / 1000) · g_s
   + Σ_t ( w_deadline · π_t  +  w_urgent·𝟙[t urgent] ) · y_t
```

`π_t` is a per-category penalty (`urgent`=20, `flexible`=5, `training`=1 $/CU).

**Mode B — cost-priority with explicit service constraints:**

```
min  Σ_s p_s · g_s
s.t. Σ_{t ∈ U} y_t ≤ (1 − ρ) · Σ_{t ∈ U} d_t          (urgent completion ≥ ρ)
     Σ_t y_t    ≤ ε                                    (total unmet ≤ ε)
```

**Mode C — emissions-priority with the same service constraints:**

```
min  Σ_s (ci_s / 1000) · g_s
s.t. (same service constraints as Mode B)
```

`ρ = 99%` is the urgent-completion target; `U` is the urgent-task set.
Adding $'s of cost + kg of CO₂ + CU of unmet work in a single objective
is only valid if the weights encode a defensible trade-off and have
consistent units. The primary modes (B, C) avoid that question entirely by
making the trade-off a hard constraint: minimize one target subject to a
reportable service requirement.

**Feasibility relaxation.** If the workload is infeasible (e.g. `high_demand`),
`ε` is not silently dropped. A three-phase procedure is used:

1. Phase 1 — minimize total shortfall `s_min = min Σ_t y_t` (finds the
   minimum unavoidable unmet work).
2. Phase 2 — minimize urgent shortfall `u_min` subject to `Σ_t y_t = s_min`.
3. Phase 3 — minimize the primary objective subject to
   `Σ_t y_t ≤ s_min + 1e-4` and `Σ_{t∈U} y_t ≤ u_min + 1e-4`.

So infeasible workloads report the minimum unavoidable shortfall and the
relaxation is *exactly* that value, never an arbitrary one.

### Constraints

- **(C1) Capacity** — `Σ_t x_{t,s} ≤ cap_s`
- **(C2) Energy balance** — `Σ_t e_cu·x_{t,s} + e_idle = g_s + r_s + dis_s − c_s`
- **(C3) Renewable cap** — `0 ≤ r_s ≤ ren_s`
- **(C4) Grid non-negativity** — `g_s ≥ 0`
- **(C5) Task completion** — `Σ_{s∈W_t} x_{t,s} + y_t = d_t`
- **(C6) Window** — `x_{t,s} = 0` for `s ∉ W_t` (structural)
- **(C7) Battery SoC** — `soc_s = soc_{s−1} + η_c·c_s − dis_s/η_d`,
  `SoC_min ≤ soc_s ≤ SoC_max`, `soc_N ≥ soc_0` (no end-of-day draining)

The whole model is a **linear program**, solved to optimality by HiGHS in
all three modes.

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
  (little green energy to shift into). The sensitivity sweep confirms this —
  dropping `renewable_scale` to 0.25 collapses the optimizer's cost
  reduction from ~85 % to ~43 %, while raising it to 2.0 yields little
  extra benefit (renewable already covers most demand).
- **High demand** (`high_demand`): capacity saturates, so the optimizer cannot
  move work freely — 8 end-of-horizon tasks cannot finish within 24 h
  (reported honestly as missed deadlines; total unmet CU equals baseline's).
  Miss-cause diagnosis classifies each of the 8 misses as
  `capacity_bound` (window long enough but competing tasks consumed capacity)
  or `horizon_boundary` (deadline at the last slot, horizon-truncated);
  zero are `window_too_short`, so the shortfall is a capacity issue, not a
  modelling artefact.
- **Forecast error**: rolling-horizon with imperfect forecasts costs more than
  the full-horizon optimum (~$175 vs ~$70) but still beats the baseline ($454).
  The forecast-error sweep shows cost degrades gracefully with perturbation
  magnitude while the urgent service rate stays at 100 % throughout — the
  optimizer never drops urgent work, only settles for a more expensive
  electricity mix.
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

**Per-category unmet work** (Feedback 2) — three separate metrics that are
NOT interchangeable:

| Metric | What it measures |
|---|---|
| Unmet CU (total) | Total computing work left unfinished, `Σ_t y_t` |
| Missed tasks | Number of tasks whose deadline was missed (count) |
| Urgent service rate | Proportion of *urgent* computing work completed |

Each is also broken down by category — urgent / flexible / training — so a
judge can see whether a shortfall hit urgent inference or only deferrable
training. A system that leaves 50 000 CU of training unfinished but completes
all urgent inference provides better service than one with the same total
unmet CU but several missed urgent deadlines; the per-category breakdown
makes that distinction visible.

**Miss-cause diagnosis** (`metrics.diagnose_misses`) — instead of assuming
all end-of-day shortfalls are unavoidable, every missed task is classified:

| Cause | Definition |
|---|---|
| `window_too_short` | The task's execution window cannot physically hold its workload even with exclusive access to full per-slot capacity. |
| `horizon_boundary` | Sub-case where the window is truncated because the deadline falls at the last slot of the 24 h horizon. |
| `capacity_bound` | The window IS long enough, but competing tasks consumed the shared capacity during overlapping windows. |

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

# 4. feedback experiments (objective modes, sensitivity, forecast-error, workload flexibility)
python -m src.sensitivity

# 5. regenerate evidence images (8 PNGs + pytest log) into evidence/
python evidence/generate_evidence.py

# 6. interactive dashboard
streamlit run app.py
```

Generated CSVs land in `results/` (`<scenario>_timeseries.csv`,
`<scenario>_tasks.csv`, `<scenario>_comparison.csv`,
`<scenario>_optimized_schedule.csv`, `<scenario>_task_gantt.csv`,
`multiseed_normal.csv`, `scenario_sweep.csv`).

Feedback-experiment CSVs land in `results/feedback/`
(`sensitivity_sweep.csv`, `forecast_error_sweep.csv`,
`workload_flexibility.csv`, `objective_modes_<scenario>.csv`).

Evidence images (8 PNGs) land in `evidence/` and are reproducible via
`evidence/generate_evidence.py`.

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

### 8.1 Objective-mode comparison (Feedback 1)

Side-by-side run of the three objective modes (seed 42). The primary modes
make the cost–carbon–service trade-off explicit instead of hiding behind
arbitrary weights.

**Normal scenario (workload feasible):**

| Mode | Cost [$] | Carbon [kg] | Missed | Urgent rate |
|---|---|---|---|---|
| Baseline | 453.76 | 2256.4 | 0 | 100 % |
| Weighted (default) | 69.61 | 367.8 | 0 | 100 % |
| Cost-primary | 69.61 | 367.8 | 0 | 100 % |
| Emissions-primary | 74.23 | 367.3 | 0 | 100 % |

The weighted and cost-primary modes coincide on a feasible workload (the
service constraint is non-binding). Emissions-primary pays $4.62 more to
save 0.47 kg CO₂ — the explicit marginal cost of decarbonization.

**High-demand scenario (workload infeasible — 52 145.08 CU unavoidable shortfall):**

| Mode | Cost [$] | Carbon [kg] | Missed | Urgent rate | Relaxed? |
|---|---|---|---|---|---|
| Baseline | 1917.66 | 7032.1 | 7 | 100 % | — |
| Weighted | 1526.35 | 5755.0 | 8 | 100 % | no |
| Cost-primary | 1526.30 | 5760.2 | 12 | 99 % | **yes** |
| Emissions-primary | 1765.69 | 5398.5 | 9 | 99 % | **yes** |

The primary modes relax the urgent constraint from 100 % to 99 % (the
configured `ρ`) and report the minimum unavoidable shortfall of
52 145.08 CU rather than hiding it. Emissions-primary pays $239 more than
cost-primary to save 362 kg CO₂ — again, an explicit, reportable trade-off.

### 8.2 Sensitivity & robustness sweeps (Feedback 3)

**One-parameter-at-a-time sensitivity (normal scenario, seed 42).** Cost
reduction % reported; full table in `results/feedback/sensitivity_sweep.csv`.

| Parameter | Value range | Optimizer cost-reduction range |
|---|---|---|
| `KW_PER_CU` (compute→power) | 0.30 → 0.80 | 84.0 % → 86.7 % |
| `renewable_scale` | 0.25 → 2.00 | 43.3 % → 89.9 % |
| `price_scale` | 0.50 → 3.00 | 84.7 % (constant) |

The optimizer's advantage is **robust** to the compute-to-power coefficient
and to electricity-price scaling (the relative saving is invariant because
both baseline and optimizer scale linearly with price). It is **most
sensitive** to renewable availability: at 25 % of default renewable, cost
reduction falls from 85 % to 43 % (no green window to shift into); beyond
1.0× renewable, additional green energy yields diminishing returns.

**Forecast-error sweep (rolling horizon, normal scenario).** Realized cost
vs forecast perturbation; urgent service rate stays at 100 % throughout.

| Perturbation | RH cost [$] | vs perfect-forecast | vs full-horizon optimum |
|---|---|---|---|
| 0 % | 69.61 | 0 % | 0 % |
| 5 % | 87.55 | +25.8 % | +25.8 % |
| 10 % | 112.31 | +61.3 % | +61.3 % |
| 20 % | 151.15 | +117.1 % | +117.1 % |
| 30 % | 199.02 | +185.9 % | +185.9 % |
| 50 % | 216.40 | +210.9 % | +210.9 % |

At 50 % forecast error the realized cost is +211 % above the full-horizon
optimum — but the rolling horizon never drops a single urgent task. The
degradation is purely in the electricity mix the optimizer commits to, not
in service.

**Workload-flexibility comparison (seed 42).**

| Profile | Urgent share | Cost reduction | Carbon reduction | Missed |
|---|---|---|---|---|
| Flexible (long deadlines, training-heavy) | 10 % | 85.4 % | 83.1 % | 1 |
| Balanced (default) | 20 % | 84.7 % | 83.7 % | 0 |
| Strict (short deadlines, urgent-heavy) | 40 % | 88.5 % | 76.6 % | 0 |

Counter-intuitive but explainable: the **strict** workload yields the
highest *cost* reduction because the baseline greedily runs urgent work
early at peak prices, while the optimizer still defers the flexible/training
portion to cheap slots. The same profile yields the lowest *carbon*
reduction because less work is shiftable into the renewable window.

---

## 9. Assumptions & limitations

- Workloads are **pre-emptible/divisible** across slots (standard for cloud
  batch and checkpointed training). Non-interruptible execution = MILP
  extension.
- Deadline violations are **soft** (penalised via `y_t`) so capacity-bound
  infeasibility is *reported*, never silently dropped. The three objective
  modes (weighted / cost-primary / emissions-primary) make the trade-off
  explicit; when the workload is infeasible the service constraint relaxes
  to the *minimum unavoidable shortfall* found by a Phase-1 feasibility LP,
  never to an arbitrary value.
- Single transparent compute→power coefficient `KW_PER_CU`. The sensitivity
  sweep (§8.2) shows the optimizer's relative advantage stays at 84–87 %
  across `KW_PER_CU ∈ [0.30, 0.80]`.
- Renewable is free and used first; curtailment allowed.
- Horizon = 24 h; late-arriving tasks with end-of-day deadlines may be
  infeasible within the window (reported as missed, not hidden, and each
  miss is classified by root cause — capacity-bound, window-too-short, or
  horizon-boundary — via `metrics.diagnose_misses`).
- No global-optimality claim beyond the LP solver status; the LP is a
  relaxation of the true integer scheduling problem.
- Synthetic data is configurable and seed-driven; assumptions (solar bell,
  wind walk, time-of-use price, carbon–renewable coupling) are documented in
  `src/data_generator.py` so real data can replace it later. The sensitivity,
  forecast-error, and workload-flexibility sweeps (§8.2) bound how much the
  conclusions change when these assumptions vary.
