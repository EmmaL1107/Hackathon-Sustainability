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
from src.simulator import run_experiment, run_all_scenarios, rolling_horizon
from src.metrics import compute_metrics, compare
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
tab_over, tab_energy, tab_sched, tab_cmp, tab_scen, tab_method, tab_export = \
    st.tabs(["Overview", "Energy & Demand", "Scheduling Results",
             "Algorithm Comparison", "Scenario Testing",
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

# ----------------------------------------------------------- Methodology
with tab_method:
    st.markdown("### Mathematical model")
    st.latex(r"\min\; w_{cost}\sum_s p_s g_s \;+\; w_{carbon}\sum_s \tfrac{ci_s}{1000} g_s \;+\; \sum_t (w_{deadline}\pi_t + w_{urgent}\mathbb{1}_{urgent}) y_t")
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
    st.markdown("**Assumptions & limitations**")
    st.markdown(
        "- Workloads are pre-emptible/divisible across slots (standard for "
        "cloud batch & checkpointed training). Non-interruptible execution "
        "would require binary variables (MILP) — noted as an extension.\n"
        "- Deadline violations are SOFT (penalized via $y_t$) so capacity-bound "
        "infeasibility is REPORTED, never silently dropped.\n"
        "- Single transparent compute->power coefficient `KW_PER_CU`.\n"
        "- Renewable is free and used first; curtailment allowed.\n"
        "- Horizon = 24h; tasks arriving late with end-of-day deadlines may "
        "be infeasible within the window (reported honestly).\n"
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
