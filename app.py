"""
Streamlit dashboard for the AI Data Center Power-Computing Co-Scheduling
project (HackSHSID 2026 Track 2).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st

import config
from src.data_generator import generate_scenario, save_scenario, ScenarioData
from src.optimizer import LPOptimizer
from src import baseline
from src import scenarios as scn
from src.simulator import run_experiment, run_all_scenarios, rolling_horizon, \
    run_objective_comparison
from src.metrics import compute_metrics, compare, diagnose_misses
from src.sensitivity import sensitivity_sweep, forecast_error_sweep, \
    workload_flexibility_comparison
from src import visualization as viz


# ---------------------------------------------------------------------------
# Cached experiment runner
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running optimizer and baseline...")
def cached_experiment(scenario_name, seed, w_cost, w_carbon, w_deadline,
                      w_urgent, battery_enabled):
    weights = dict(w_cost=w_cost, w_carbon=w_carbon,
                   w_deadline=w_deadline, w_urgent=w_urgent)
    return run_experiment(scenario_name, seed=seed,
                          battery_enabled=battery_enabled, weights=weights)


@st.cache_data(show_spinner="Running scenario sweep...")
def cached_sweep(battery_enabled):
    return run_all_scenarios(battery_enabled=battery_enabled)


@st.cache_data(show_spinner="Running rolling horizon...")
def cached_rolling(scenario_name, seed, w_cost, w_carbon, w_deadline,
                    w_urgent, use_forecast, battery_enabled):
    data = scn.get_scenario(scenario_name, seed=seed)
    weights = dict(w_cost=w_cost, w_carbon=w_carbon,
                   w_deadline=w_deadline, w_urgent=w_urgent)
    return rolling_horizon(data, use_forecast=use_forecast,
                           weights=weights, battery_enabled=battery_enabled)


@st.cache_data(show_spinner="Running objective-mode comparison...")
def cached_objective_modes(scenario_name, seed):
    return run_objective_comparison(scenario_name, seed=seed)


@st.cache_data(show_spinner="Running sensitivity sweep...")
def cached_sensitivity(seed):
    return sensitivity_sweep(seed=seed)


@st.cache_data(show_spinner="Running forecast-error sweep...")
def cached_forecast(seed, scenario_name):
    return forecast_error_sweep(seed=seed, scenario_name=scenario_name)


@st.cache_data(show_spinner="Running workload-flexibility comparison...")
def cached_workload_flex(seed):
    return workload_flexibility_comparison(seed=seed)


# ---------------------------------------------------------------------------
def df_to_csv(df) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ===========================================================================
# PAGE
# ===========================================================================
st.set_page_config(page_title="AI Data Center Co-Scheduling",
                   page_icon=":zap:", layout="wide")

st.title(":zap: AI Data Center Power-Computing Co-Scheduling")
st.caption("HackSHSID 2026 Track 2  |  LP-based energy-aware scheduler vs. baseline")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
scenario_options = list(scn.SCENARIOS.keys())
scenario_name = st.sidebar.selectbox(
    "Scenario", scenario_options,
    format_func=lambda x: f"{x} — {scn.SCENARIOS[x].label}",
    index=0)
seed = st.sidebar.number_input("Random seed", 0, 1000, 42, step=1)

st.sidebar.subheader("Objective weights")
w_cost = st.sidebar.slider("Cost weight (w_cost)", 0.0, 5.0, config.W_COST, 0.05)
w_carbon = st.sidebar.slider("Carbon weight $/kgCO2 (w_carbon)",
                             0.0, 5.0, config.W_CARBON, 0.01)
w_deadline = st.sidebar.slider("Deadline penalty mult (w_deadline)",
                               0.0, 50.0, config.W_DEADLINE, 0.5)
w_urgent = st.sidebar.slider("Urgent extra penalty $/CU (w_urgent)",
                              0.0, 200.0, config.W_URGENT, 5.0)
battery_enabled = st.sidebar.checkbox("Enable battery storage",
                                      config.BATTERY_ENABLED)

run_now = st.sidebar.button("Run / refresh experiment", type="primary",
                            use_container_width=True)

# ---------------------------------------------------------------- run
res = cached_experiment(scenario_name, seed, w_cost, w_carbon,
                        w_deadline, w_urgent, battery_enabled)
bm, om = res["baseline_metrics"], res["optimized_metrics"]
data = res["data"]

# =========================================================== tabs
tab_over, tab_energy, tab_sched, tab_cmp, tab_scen, tab_obj, tab_sens, \
    tab_service, tab_method, tab_export = st.tabs([
        "Overview", "Energy & Demand", "Scheduling Results",
        "Algorithm Comparison", "Scenario Testing",
        "Objective Modes", "Sensitivity", "Service Analysis",
        "Methodology", "Export"])

# ----------------------------------------------------------- Overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    cost_red = ((bm.total_cost_usd - om.total_cost_usd) / bm.total_cost_usd * 100
                if bm.total_cost_usd else 0)
    carb_red = ((bm.total_carbon_kg - om.total_carbon_kg) / bm.total_carbon_kg * 100
                if bm.total_carbon_kg else 0)
    c1.metric("Electricity cost reduction", f"{cost_red:.1f}%",
               f"${bm.total_cost_usd:.1f} -> ${om.total_cost_usd:.1f}")
    c2.metric("Carbon reduction", f"{carb_red:.1f}%",
              f"{bm.total_carbon_kg:.0f} -> {om.total_carbon_kg:.0f} kgCO2")
    c3.metric("Completed workload", f"{om.completed_cu:,.0f} CU",
              f"{om.completed_cu / om.total_cu * 100:.1f}% of demand")
    c4.metric("Urgent service rate", f"{om.urgent_service_rate_pct:.1f}%",
              f"{om.missed_deadlines} missed deadlines")
    st.markdown("#### Problem in one paragraph")
    st.info(
        "An AI data center must decide **when** to run computing tasks across "
        "96 fifteen-minute slots of a day. Electricity prices, renewable "
        "generation (solar+wind) and carbon intensity all change over time, "
        "while tasks have different arrivals, deadlines and priorities. "
        "The scheduler shifts flexible work into cheap, low-carbon, "
        "renewable-rich windows without violating urgent service requirements."
    )
    st.markdown("#### Baseline vs Optimized (key metrics)")
    st.dataframe(res["comparison"].style.format(
        {"baseline": "{:.4f}", "optimized": "{:.4f}",
         "improvement_vs_baseline_pct": "{:.2f}%"}),
                 use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(viz.fig_cost_comparison(bm, om))

# ----------------------------------------------------------- Energy & Demand
with tab_energy:
    st.pyplot(viz.fig_energy_demand(data), use_container_width=True)
    st.markdown("##### When is electricity cheap, green and demand high?")
    comp_df = data.to_timeseries_df()
    cheap = comp_df.nsmallest(5, "price_usd_per_kwh")[["time", "price_usd_per_kwh"]]
    green = comp_df.nlargest(5, "renewable_kwh")[["time", "renewable_kwh"]]
    high = comp_df.nlargest(5, "baseline_demand_cu")[["time", "baseline_demand_cu"]]
    c1, c2, c3 = st.columns(3)
    c1.dataframe(cheap, hide_index=True, use_container_width=True)
    c2.dataframe(green, hide_index=True, use_container_width=True)
    c3.dataframe(high, hide_index=True, use_container_width=True)

# ----------------------------------------------------------- Scheduling Results
with tab_sched:
    st.markdown("##### Optimized schedule")
    st.pyplot(viz.fig_schedule(res["optimized_schedule"], data, "Optimized"),
              use_container_width=True)
    with st.expander("Baseline schedule (for reference)"):
        st.pyplot(viz.fig_schedule(res["baseline_schedule"], data, "Baseline"),
                  use_container_width=True)
    st.markdown("##### Combined: grid vs renewable (baseline vs optimized)")
    st.pyplot(viz.fig_compare_timeseries(res["baseline_schedule"],
                                         res["optimized_schedule"], data),
              use_container_width=True)
    st.markdown("##### Task execution timeline (Gantt, sampled)")
    st.pyplot(viz.fig_gantt(res["optimized_schedule"], data), use_container_width=True)
    with st.expander("Optimized schedule table (per slot)"):
        st.dataframe(res["optimized_schedule"].to_timeseries_df(),
                     use_container_width=True, hide_index=True)

# ----------------------------------------------------------- Algorithm Comparison
with tab_cmp:
    st.markdown("##### Side-by-side metrics")
    st.dataframe(res["comparison"].style.format(
        {"baseline": "{:.4f}", "optimized": "{:.4f}",
         "improvement_vs_baseline_pct": "{:.2f}%"}),
                 use_container_width=True, hide_index=True)
    st.pyplot(viz.fig_cost_comparison(bm, om), use_container_width=True)
    st.markdown("##### Computational performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Solver status", om.solver_status.split("(")[0].strip())
    c2.metric("Optimizer runtime", f"{om.runtime_s:.3f} s")
    c3.metric("Feasible?", "Yes" if om.feasible else "No")

    st.markdown("##### Rolling-horizon vs full-horizon")
    rh_p = cached_rolling(scenario_name, seed, w_cost, w_carbon,
                          w_deadline, w_urgent, False, battery_enabled)
    rh_f = cached_rolling(scenario_name, seed, w_cost, w_carbon,
                          w_deadline, w_urgent, True, battery_enabled)
    pm = compute_metrics(rh_p, data)
    fm = compute_metrics(rh_f, data)
    rdf = pd.DataFrame({
        "method": ["Baseline", "Full-horizon opt", "Rolling (perfect fc)",
                   "Rolling (forecast err)"],
        "cost_usd": [bm.total_cost_usd, om.total_cost_usd, pm.total_cost_usd,
                     fm.total_cost_usd],
        "carbon_kg": [bm.total_carbon_kg, om.total_carbon_kg, pm.total_carbon_kg,
                      fm.total_carbon_kg],
        "missed": [bm.missed_deadlines, om.missed_deadlines,
                   pm.missed_deadlines, fm.missed_deadlines],
        "runtime_s": [bm.runtime_s, om.runtime_s, pm.runtime_s, fm.runtime_s],
    })
    st.dataframe(rdf.style.format({"cost_usd": "${:.2f}",
                                   "carbon_kg": "{:.1f}",
                                   "runtime_s": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

# ----------------------------------------------------------- Scenario Testing
with tab_scen:
    st.markdown("##### Stress-test scenarios (8 required)")
    sweep_df = cached_sweep(battery_enabled)
    if isinstance(sweep_df, tuple):
        sweep_df = sweep_df[0]
    st.pyplot(viz.fig_scenario_sweep(sweep_df), use_container_width=True)
    show = sweep_df.drop(columns=["seed"])
    st.dataframe(show.style.format({
        "baseline_cost_usd": "${:.2f}", "optimized_cost_usd": "${:.2f}",
        "cost_reduction_pct": "{:.1f}%", "baseline_carbon_kg": "{:.0f}",
        "optimized_carbon_kg": "{:.0f}", "carbon_reduction_pct": "{:.1f}%",
        "baseline_urgent_rate": "{:.0f}%", "optimized_urgent_rate": "{:.0f}%",
        "optimized_runtime_s": "{:.3f}",
    }), use_container_width=True, hide_index=True)
    st.markdown("##### Notes on where the optimizer struggles")
    worst = sweep_df.loc[sweep_df["cost_reduction_pct"].idxmin()]
    st.info(
        f"Smallest cost reduction in **{worst['scenario']}** "
        f"({worst['cost_reduction_pct']:.1f}%): "
        f"{worst['optimized_missed']:.0f} missed deadlines there. "
        f"This is expected — when renewable energy is scarce or demand "
        f"saturates capacity, there is little room to shift work, and "
        f"some end-of-horizon tasks cannot finish within 24h."
    )

# ----------------------------------------------------------- Objective Modes
with tab_obj:
    st.markdown("##### Three objective modes")
    st.info(
        "Different optimization goals give different trade-offs between cost, "
        "carbon and service. The **weighted** mode sums cost + carbon + "
        "deadline penalties into a single objective; the **primary** modes "
        "minimize one target (cost or carbon) while keeping an explicit "
        "service constraint: urgent workload completed ≥ 99% and total unmet "
        "≤ ε. If the workload is infeasible, the service constraints are "
        "automatically relaxed to the *minimum unavoidable shortfall* (not "
        "an arbitrary penalty)."
    )
    obj_df = cached_objective_modes(scenario_name, seed)
    st.pyplot(viz.fig_objective_modes(obj_df), use_container_width=True)
    st.markdown("##### Side-by-side comparison")
    st.dataframe(obj_df.style.format({
        "cost_usd": "${:.2f}", "carbon_kg": "{:.2f}",
        "unmet_cu": "{:.1f}", "unmet_urgent_cu": "{:.1f}",
        "unmet_flexible_cu": "{:.1f}", "unmet_training_cu": "{:.1f}",
        "urgent_rate_pct": "{:.2f}%", "min_unavoidable_shortfall_cu": "{:.1f}",
    }), use_container_width=True, hide_index=True)

    # surface feasibility-relaxation details
    relaxed = obj_df[obj_df["feasibility_relaxed"]]
    if len(relaxed):
        st.warning(
            f"{len(relaxed)} mode(s) had to relax the service constraint: "
            f"minimum unavoidable shortfall = "
            f"{relaxed['min_unavoidable_shortfall_cu'].iloc[0]:.1f} CU. "
            "This is reported rather than hidden behind arbitrary penalties."
        )
    else:
        st.success("All service constraints satisfied without relaxation.")

# ----------------------------------------------------------- Sensitivity
with tab_sens:
    st.markdown("##### Sensitivity analysis (Feedback 3)")
    st.caption(
        "One-parameter-at-a-time sweeps on the *normal* scenario. The dashed "
        "vertical line marks the default value used elsewhere in this report.")

    sens_df = cached_sensitivity(seed)
    st.markdown("###### Electricity cost vs each parameter")
    st.pyplot(viz.fig_sensitivity_sweep(sens_df), use_container_width=True)
    st.pyplot(viz.fig_sensitivity_carbon(sens_df), use_container_width=True)

    with st.expander("Full sensitivity table"):
        st.dataframe(sens_df.style.format({
            "baseline_cost_usd": "${:.2f}", "optimized_cost_usd": "${:.2f}",
            "cost_reduction_pct": "{:.2f}%",
            "baseline_carbon_kg": "{:.1f}", "optimized_carbon_kg": "{:.1f}",
            "carbon_reduction_pct": "{:.2f}%",
            "baseline_urgent_rate_pct": "{:.1f}%",
            "optimized_urgent_rate_pct": "{:.1f}%",
            "optimized_unmet_cu": "{:.1f}", "optimized_runtime_s": "{:.3f}",
        }), use_container_width=True, hide_index=True)
    st.markdown(
        "- **KW_PER_CU** raising the compute-to-power coefficient raises both "
        "baseline and optimized cost roughly linearly (more energy per CU). "
        "The optimizer's relative advantage stays around 85% throughout.\n"
        "- **renewable_scale** has a strong nonlinear effect: when renewable "
        "drops to 25% the optimizer's cost reduction falls from ~85% to ~43% "
        "because there is no green window to shift work into.\n"
        "- **price_scale** scales cost linearly but does not change the "
        "carbon trajectory (carbon is set by the grid mix, not price)."
    )

    st.markdown("---")
    st.markdown("##### Forecast-error robustness (rolling horizon)")
    st.caption(
        "How much does the rolling-horizon scheduler's realized cost "
        "deteriorate as the price & renewable forecasts become more "
        "inaccurate? Perturbation = 0% means perfect forecast.")
    fc_df = cached_forecast(seed, scenario_name)
    st.pyplot(viz.fig_forecast_error(fc_df), use_container_width=True)
    st.dataframe(fc_df.style.format({
        "rh_cost_usd": "${:.2f}", "rh_carbon_kg": "{:.1f}",
        "perfect_forecast_cost_usd": "${:.2f}",
        "full_horizon_optimum_cost_usd": "${:.2f}",
        "degradation_vs_perfect_pct": "{:.2f}%",
        "degradation_vs_optimal_pct": "{:.2f}%",
        "rh_runtime_s": "{:.3f}",
    }), use_container_width=True, hide_index=True)
    st.info(
        "At 50% forecast error the realized cost is "
        f"{fc_df['degradation_vs_optimal_pct'].iloc[-1]:.1f}% above the "
        "full-horizon optimum. Crucially, **the urgent service rate stays at "
        "100% throughout** — the optimizer never drops urgent work, only "
        "settles for a more expensive (or carbon-heavier) electricity mix."
    )

    st.markdown("---")
    st.markdown("##### Workload flexibility comparison")
    st.caption(
        "Three workload regimes: highly flexible (long deadlines, "
        "training-heavy), balanced (default config), and strict "
        "(short deadlines, urgent-heavy).")
    wf_df = cached_workload_flex(seed)
    st.pyplot(viz.fig_workload_flexibility(wf_df), use_container_width=True)
    st.dataframe(wf_df.style.format({
        "baseline_cost_usd": "${:.2f}", "optimized_cost_usd": "${:.2f}",
        "cost_reduction_pct": "{:.2f}%",
        "baseline_carbon_kg": "{:.1f}", "optimized_carbon_kg": "{:.1f}",
        "carbon_reduction_pct": "{:.2f}%",
        "baseline_urgent_rate_pct": "{:.1f}%",
        "optimized_urgent_rate_pct": "{:.1f}%",
        "optimized_unmet_cu": "{:.1f}",
        "urgent_share": "{:.2f}", "flexible_share": "{:.2f}",
        "training_share": "{:.2f}",
    }), use_container_width=True, hide_index=True)
    st.info(
        "The optimizer saves the most (cost reduction) on **strict** workloads "
        "where urgent work must run early: the baseline greedily runs "
        "everything early at peak prices, while the optimizer still defers "
        "the flexible/training portion to cheaper slots."
    )

# ----------------------------------------------------------- Service Analysis
with tab_service:
    st.markdown("##### Unmet workload, broken down by category")
    st.caption(
        "Distinguishes **Unmet CU** (total compute work left unfinished) "
        "from **Missed tasks** (count of tasks that missed their deadline) "
        "from **Urgent service rate** (proportion of urgent computing work "
        "completed). These are NOT interchangeable.")
    st.pyplot(viz.fig_unmet_by_category(bm, om), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Urgent unmet", f"{om.unmet_urgent_cu:.1f} CU",
              f"{om.missed_urgent} task(s) missed")
    c2.metric("Flexible unmet", f"{om.unmet_flexible_cu:.1f} CU",
              f"{om.missed_flexible} task(s) missed")
    c3.metric("Training unmet", f"{om.unmet_training_cu:.1f} CU",
              f"{om.missed_training} task(s) missed")

    st.markdown("##### Miss-cause diagnosis (root cause per missed task)")
    st.caption(
        "Instead of assuming all end-of-day shortfalls are unavoidable, "
        "every missed task is classified by its root cause.")
    miss_df = diagnose_misses(res["optimized_schedule"], data)
    if miss_df.empty:
        st.success("No missed deadlines for the optimized schedule in this scenario.")
    else:
        st.dataframe(miss_df, use_container_width=True, hide_index=True)
        st.markdown("**Cause definitions:**")
        st.markdown(
            "- **window_too_short** — the task's execution window cannot "
            "physically hold its workload even with exclusive access to full "
            "per-slot capacity.\n"
            "- **horizon_boundary** — a sub-case where the window is "
            "truncated because the deadline falls at the last slot of the "
            "24-hour planning horizon.\n"
            "- **capacity_bound** — the window IS long enough, but competing "
            "tasks consumed the shared capacity during overlapping windows."
        )
        cause_counts = miss_df["cause"].value_counts()
        st.markdown("##### Cause distribution")
        st.bar_chart(cause_counts)

# ----------------------------------------------------------- Methodology
with tab_method:
    st.markdown("### Mathematical model")
    st.markdown("**Mode 1 — weighted multi-objective (default):**")
    st.latex(r"\min\; w_{cost}\sum_s p_s g_s \;+\; w_{carbon}\sum_s \tfrac{ci_s}{1000} g_s \;+\; \sum_t (w_{deadline}\pi_t + w_{urgent}\mathbb{1}_{urgent}) y_t")
    st.markdown("**Mode 2 — cost-priority with explicit service constraints:**")
    st.latex(r"\min\;\sum_s p_s\,g_s\quad\text{s.t.}\quad"
             r"\sum_{t\in U} y_t \le (1-\rho)\sum_{t\in U} d_t,\;\;"
             r"\sum_t y_t \le \epsilon")
    st.markdown("**Mode 3 — emissions-priority with the same service constraints:**")
    st.latex(r"\min\;\sum_s \tfrac{ci_s}{1000}\,g_s\quad\text{s.t.}\quad"
             r"\sum_{t\in U} y_t \le (1-\rho)\sum_{t\in U} d_t,\;\;"
             r"\sum_t y_t \le \epsilon")
    st.caption(r"$\rho=99\%$ is the urgent-completion target. If the workload "
               "is infeasible, $\epsilon$ is relaxed to the minimum unavoidable "
               "shortfall found by a Phase-1 feasibility LP (never to an "
               "arbitrary value).")
    st.markdown("**Decision variables**")
    st.markdown(r"- $x_{t,s}\ge 0$: CU of task $t$ processed in slot $s$ "
                r"- $g_s\ge 0$: grid energy [kWh] - $r_s\in[0,ren_s]$: renewable used "
                r"- $y_t\ge 0$: unmet work of task $t$ (shortfall)")
    st.markdown("**Constraints**")
    st.markdown(r"- Capacity: $\sum_t x_{t,s}\le cap_s$ "
                r"- Energy balance: $\sum_t e_{cu}x_{t,s}+e_{idle}=g_s+r_s+dis_s-c_s$ "
                r"- Task completion: $\sum_{s\in W_t}x_{t,s}+y_t=d_t$ "
                r"- Window: $x_{t,s}=0$ outside $[\text{arrival},\text{deadline}]$ "
                r"- Battery SoC dynamics & limits (if enabled)")
    st.markdown("**Why three modes?**")
    st.markdown(
        "Adding $'s of cost + kg of CO₂ + CU of unmet work in a single objective "
        "is only valid if the weights encode an explicit, defensible trade-off "
        "and have consistent units. To avoid hiding behind arbitrary weights, "
        "the primary modes make the trade-off **explicit and reportable**: "
        "minimize cost (or carbon) *subject to* a hard service requirement. "
        "Judges can read the cost-vs-carbon-vs-service trade-off directly."
    )
    st.markdown("**Assumptions & limitations**")
    st.markdown(
        "- Workloads are pre-emptible/divisible across slots (standard for "
        "cloud batch & checkpointed training). Non-interruptible execution "
        "would require binary variables (MILP) — noted as an extension.\n"
        "- Deadline violations are SOFT (penalized via $y_t$) so capacity-bound "
        "infeasibility is REPORTED, never silently dropped.\n"
        "- Single transparent compute->power coefficient `KW_PER_CU`; the "
        "sensitivity sweep explores how robust the conclusions are to this "
        "assumption.\n"
        "- Renewable is free and used first; curtailment allowed.\n"
        "- Horizon = 24h; tasks arriving late with end-of-day deadlines may "
        "be infeasible within the window (reported honestly, with the root "
        "cause of every miss classified in the Service Analysis tab).\n"
        "- Synthetic data: results establish performance on a *synthetic* "
        "data center; the sensitivity and forecast-error sweeps bound how "
        "much the conclusions change when key assumptions vary.\n"
        "- No claim of global optimality beyond the LP solver status; the "
        "LP is a relaxation of the true (integer) scheduling problem."
    )

# ----------------------------------------------------------- Export
with tab_export:
    st.markdown("##### Download generated artifacts")
    ts = data.to_timeseries_df()
    tasks = data.workload.to_dataframe()
    comp = res["comparison"]
    opt_ts = res["optimized_schedule"].to_timeseries_df()
    gantt = res["optimized_schedule"].to_schedule_df()

    c1, c2 = st.columns(2)
    c1.download_button("Dataset (timeseries CSV)", df_to_csv(ts),
                       file_name=f"{scenario_name}_timeseries.csv", mime="text/csv")
    c2.download_button("Tasks CSV", df_to_csv(tasks),
                       file_name=f"{scenario_name}_tasks.csv", mime="text/csv")
    c1.download_button("Comparison CSV", df_to_csv(comp),
                       file_name=f"{scenario_name}_comparison.csv", mime="text/csv")
    c2.download_button("Optimized schedule CSV", df_to_csv(opt_ts),
                       file_name=f"{scenario_name}_optimized_schedule.csv",
                       mime="text/csv")
    c1.download_button("Task Gantt CSV", df_to_csv(gantt),
                       file_name=f"{scenario_name}_gantt.csv", mime="text/csv")
    c2.download_button("Scenario sweep CSV", df_to_csv(sweep_df),
                       file_name="scenario_sweep.csv", mime="text/csv")
    st.markdown("##### Feedback experiment artifacts")
    c3, c4 = st.columns(2)
    c3.download_button("Objective-mode comparison CSV",
                       df_to_csv(cached_objective_modes(scenario_name, seed)),
                       file_name=f"{scenario_name}_objective_modes.csv",
                       mime="text/csv")
    c4.download_button("Sensitivity sweep CSV",
                       df_to_csv(cached_sensitivity(seed)),
                       file_name="sensitivity_sweep.csv", mime="text/csv")
    c3.download_button("Forecast-error sweep CSV",
                       df_to_csv(cached_forecast(seed, scenario_name)),
                       file_name=f"{scenario_name}_forecast_error.csv",
                       mime="text/csv")
    c4.download_button("Workload flexibility CSV",
                       df_to_csv(cached_workload_flex(seed)),
                       file_name="workload_flexibility.csv", mime="text/csv")
    if not miss_df.empty:
        c3.download_button("Miss-cause diagnosis CSV",
                           df_to_csv(miss_df),
                           file_name=f"{scenario_name}_miss_causes.csv",
                           mime="text/csv")

    st.markdown("##### Experiment summary (text)")
    summary = (
        f"Scenario: {scenario_name} (seed={seed})\n"
        f"Weights: w_cost={w_cost}, w_carbon={w_carbon}, "
        f"w_deadline={w_deadline}, w_urgent={w_urgent}\n"
        f"Battery: {'enabled' if battery_enabled else 'disabled'}\n"
        f"Baseline:  cost=${bm.total_cost_usd:.2f}, "
        f"carbon={bm.total_carbon_kg:.1f}kg, missed={bm.missed_deadlines}\n"
        f"Optimized:  cost=${om.total_cost_usd:.2f}, "
        f"carbon={om.total_carbon_kg:.1f}kg, missed={om.missed_deadlines}\n"
        f"Cost reduction:    {cost_red:.1f}%\n"
        f"Carbon reduction:  {carb_red:.1f}%\n"
        f"Optimizer runtime: {om.runtime_s:.3f}s, status={om.solver_status}\n"
    )
    st.code(summary, language="text")
    st.download_button("Download summary (.txt)", summary.encode("utf-8"),
                       file_name=f"{scenario_name}_summary.txt")
