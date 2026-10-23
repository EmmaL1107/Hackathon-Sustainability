"""
Matplotlib chart helpers for the dashboard.

Each function returns a `matplotlib.figure.Figure` (never calls plt.show),
so Streamlit can render them with st.pyplot(fig).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


# consistent color palette
C_BASE = "#d6604d"      # baseline (red-ish)
C_OPT = "#2c7fb8"       # optimized (blue)
C_GRID = "#e41a1c"
C_REN = "#4daf4a"
C_COMP = "#377db4"
C_BAT = "#984ea3"
C_PRICE = "#ff7f00"
C_CARBON = "#666666"


def _hours(n):
    return np.arange(n) * config.SLOT_HOURS


# ---------------------------------------------------------------------------
def fig_energy_demand(data) -> plt.Figure:
    """Electricity price, renewable generation, computing demand."""
    n = data.n_slots
    h = _hours(n)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    axes[0].plot(h, data.price, color=C_PRICE, lw=2)
    axes[0].set_ylabel("Grid price [$/kWh]")
    axes[0].set_title("Electricity price")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(h, data.extra.get("solar", np.zeros(n)), 0,
                         color="#fee08b", label="solar")
    axes[1].fill_between(h, data.renewable,
                         data.extra.get("solar", np.zeros(n)),
                         color="#74add1", label="wind")
    axes[1].set_ylabel("Renewable [kWh]")
    axes[1].set_title("Renewable generation")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(h, data.baseline_demand, color=C_COMP, lw=2)
    axes[2].set_ylabel("Demand [CU/slot]")
    axes[2].set_title("Baseline computing demand")
    axes[2].set_xlabel("Hour of day")
    axes[2].grid(alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
def fig_schedule(schedule, data, title=None) -> plt.Figure:
    """Time-series of scheduled compute, grid, renewable, battery."""
    n = schedule.n_slots
    h = _hours(n)
    e_cu = config.KW_PER_CU * config.SLOT_HOURS
    comp = schedule.per_slot_compute()
    has_bat = schedule.battery_soc is not None

    rows = 4 if has_bat else 3
    fig, axes = plt.subplots(rows, 1, figsize=(9, 2.4 * rows), sharex=True)
    if title:
        fig.suptitle(title, y=0.995)

    axes[0].plot(h, comp, color=C_COMP, lw=2)
    axes[0].axhline(data.capacity[0], color="gray", ls="--", lw=1,
                    label="capacity")
    axes[0].set_ylabel("Compute [CU]")
    axes[0].set_title(f"{schedule.name}: scheduled compute")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(h, schedule.grid_energy, 0, color=C_GRID, alpha=0.6,
                         label="grid")
    axes[1].fill_between(h, schedule.renewable_used,
                         schedule.grid_energy, color=C_REN, alpha=0.6,
                         label="renewable")
    axes[1].set_ylabel("Energy [kWh]")
    axes[1].set_title("Electricity sources")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(h, data.price, color=C_PRICE, lw=1.5, label="price")
    axes[2].set_ylabel("Price [$/kWh]")
    axes[2].set_title("Grid price (reference)")
    axes[2].set_xlabel("Hour of day")
    axes[2].grid(alpha=0.3)

    if has_bat:
        ax = axes[3]
        ax.plot(h, schedule.battery_soc, color=C_BAT, lw=2)
        ax.set_ylabel("SoC [kWh]")
        ax.set_title("Battery state of charge")
        ax.set_xlabel("Hour of day")
        ax.grid(alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
def fig_compare_timeseries(base, opt, data) -> plt.Figure:
    """Side-by-side grid vs renewable for baseline and optimized."""
    n = base.n_slots
    h = _hours(n)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, sched, lab in [(axes[0], base, "Baseline"),
                           (axes[1], opt, "Optimized")]:
        ax.fill_between(h, sched.grid_energy, 0, color=C_GRID, alpha=0.6,
                        label="grid")
        ax.fill_between(h, sched.renewable_used, sched.grid_energy,
                        color=C_REN, alpha=0.6, label="renewable")
        ax.set_ylabel("Energy [kWh]")
        ax.set_title(f"{lab}: grid vs renewable")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Hour of day")
    return fig


# ---------------------------------------------------------------------------
def fig_cost_comparison(bm, om) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    labels = ["Baseline", "Optimized"]
    axes[0].bar(labels, [bm.total_cost_usd, om.total_cost_usd],
                color=[C_BASE, C_OPT])
    axes[0].set_ylabel("Cost [$]")
    axes[0].set_title("Electricity cost")
    for i, v in enumerate([bm.total_cost_usd, om.total_cost_usd]):
        axes[0].text(i, v, f"${v:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(labels, [bm.total_carbon_kg, om.total_carbon_kg],
                color=[C_BASE, C_OPT])
    axes[1].set_ylabel("Carbon [kgCO2]")
    axes[1].set_title("Carbon emissions")
    for i, v in enumerate([bm.total_carbon_kg, om.total_carbon_kg]):
        axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def fig_scenario_sweep(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(df))
    wl = 0.4
    axes[0].bar(x - wl/2, df["baseline_cost_usd"], wl, color=C_BASE, label="baseline")
    axes[0].bar(x + wl/2, df["optimized_cost_usd"], wl, color=C_OPT, label="optimized")
    axes[0].set_xticks(x); axes[0].set_xticklabels(df["scenario"], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Cost [$]"); axes[0].set_title("Cost by scenario")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x - wl/2, df["baseline_carbon_kg"], wl, color=C_BASE, label="baseline")
    axes[1].bar(x + wl/2, df["optimized_carbon_kg"], wl, color=C_OPT, label="optimized")
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["scenario"], rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Carbon [kgCO2]"); axes[1].set_title("Carbon by scenario")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def fig_gantt(schedule, data, max_tasks: int = 40) -> plt.Figure:
    """Gantt-style timeline of task execution windows (sample if many)."""
    sched_df = schedule.to_schedule_df()
    tasks_present = sched_df["task_id"].unique()
    # sample to keep chart readable
    if len(tasks_present) > max_tasks:
        rng = np.random.default_rng(data.seed)
        keep = rng.choice(tasks_present, size=max_tasks, replace=False)
        sched_df = sched_df[sched_df["task_id"].isin(keep)]
    cat_colors = {"urgent": C_GRID, "flexible": C_OPT, "training": C_BAT}
    fig, ax = plt.subplots(figsize=(9, max(3, 0.25 * len(sched_df["task_id"].unique()))))
    yticks = []
    ylabels = []
    for i, (tid, g) in enumerate(sched_df.groupby("task_id", sort=False)):
        cat = g["category"].iloc[0]
        for _, r in g.iterrows():
            ax.barh(i, 0.25, left=r["slot"] * config.SLOT_HOURS,
                    color=cat_colors.get(cat, "gray"), alpha=0.9,
                    edgecolor="none")
        yticks.append(i); ylabels.append(tid)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("Hour of day"); ax.set_title("Task execution timeline (sampled)")
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in cat_colors.items()],
              fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
