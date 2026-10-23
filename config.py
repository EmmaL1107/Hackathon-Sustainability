"""
Global configuration and unit conventions for the
AI Data Center Power-Computing Co-Scheduling system.

=============================================================================
UNIT CONVENTION (used consistently across the whole project)
=============================================================================
Time axis:
    - Planning horizon: 24 hours, divided into 96 slots of 15 minutes.
    - slot_hours = 0.25  (energy = power * slot_hours)
    - Slots indexed 0..95. Slot s covers [s*0.25h, (s+1)*0.25h).

Computing:
    - Workload of a task is measured in "Compute Units" (CU). 1 CU is an
      arbitrary but consistent normalization of compute work (e.g. one
      inference batch of fixed size).
    - Data-center capacity is measured in CU per slot (how many CU can be
      processed in one 15-min slot).
    - x[t,s] = CU of task t processed during slot s  (decision variable).

Power / Energy:
    - kW_per_CU: electrical power drawn from the grid/renewables while
      processing 1 CU of compute in one slot, expressed in kW.
      (Data-center average ~ PUE * chip power; here a single transparent
       coefficient converts compute to power.)
    - Energy drawn in slot s for c CU = c * kW_per_CU * slot_hours  [kWh].

Electricity:
    - price[s] in $/kWh  (grid electricity price).
    - carbon[s] in gCO2/kWh (grid carbon intensity).
    - renewable[s] in kWh available during slot s.

Battery (optional):
    - Capacity in kWh. State of charge (SoC) in kWh.
    - charge/discharge in kWh per slot (energy terms).
    - Round-trip efficiencies eta_c (charge) and eta_d (discharge).

Cost / Carbon objective (per slot):
    - electricity_cost[s] = price[s] * grid_energy[s]            [$]
    - carbon_emission[s]  = carbon[s] * grid_energy[s] / 1000     [kgCO2]
      (we keep gCO2 internally, report kgCO2; 1 kWh grid * carbon gCO2/kWh)
=============================================================================
"""

import numpy as np

# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------
HORIZON_HOURS = 24.0
SLOT_HOURS = 0.25                      # 15-minute intervals
N_SLOTS = int(HORIZON_HOURS / SLOT_HOURS)   # 96
SLOT_LABEL_FMT = "%H:%M"               # for display

# ---------------------------------------------------------------------------
# Compute -> power conversion (transparent single coefficient)
#   Processing 1 CU for one full slot draws `KW_PER_CU` kilowatts.
#   i.e. energy per CU per slot = KW_PER_CU * SLOT_HOURS  [kWh]
# ---------------------------------------------------------------------------
KW_PER_CU = 0.5                        # kW per active CU
PUE = 1.0                              # already folded into KW_PER_CU; keep 1.0

# ---------------------------------------------------------------------------
# Data-center hardware
# ---------------------------------------------------------------------------
MAX_CAPACITY_CU = 4000.0               # CU that can be processed per slot
IDLE_POWER_KW = 50.0                    # always-on baseline load (kW)

# ---------------------------------------------------------------------------
# Default objective weights (all configurable from the dashboard)
# ---------------------------------------------------------------------------
W_COST = 1.0                            # weight on electricity expenditure [$]
W_CARBON = 0.02                         # weight on carbon [$/kgCO2] (carbon tax-like)
W_DEADLINE = 5.0                         # weight on deadline-miss penalty [$/CU]
W_URGENT = 50.0                         # extra weight on unmet urgent demand [$/CU]
W_SWITCH = 0.0                          # weight on workload switching (optional)

# ---------------------------------------------------------------------------
# Battery (optional, disabled by default; enable via scenario/dashboard)
# ---------------------------------------------------------------------------
BATTERY_ENABLED = False
BATTERY_CAPACITY_KWH = 2000.0
BATTERY_INIT_SOC = 0.5                  # fraction of capacity
BATTERY_MIN_SOC = 0.05
BATTERY_MAX_CHARGE_KW = 500.0          # per-slot energy (kWh)
BATTERY_MAX_DISCHARGE_KW = 500.0
ETA_CHARGE = 0.95
ETA_DISCHARGE = 0.95

# ---------------------------------------------------------------------------
# Renewable curtailment (fraction of renewable we are allowed to curtail)
# ---------------------------------------------------------------------------
ALLOW_CURTAILMENT = True

# ---------------------------------------------------------------------------
# Workload categories
# ---------------------------------------------------------------------------
CATEGORIES = {
    "urgent": {
        "label": "Urgent inference",
        "share": 0.20,                 # fraction of total arriving CU
        "deadline_slots": 4,           # must finish within 1 hour
        "interruptible": True,
        "delayable": False,
        "priority": 3,
    },
    "flexible": {
        "label": "Flexible batch inference",
        "share": 0.45,
        "deadline_slots": 16,          # 4 hours
        "interruptible": True,
        "delayable": True,
        "priority": 2,
    },
    "training": {
        "label": "Model training",
        "share": 0.35,
        "deadline_slots": 48,          # 12 hours
        "interruptible": True,
        "delayable": True,
        "priority": 1,
    },
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "generated")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def slot_to_time_str(slot: int) -> str:
    """Convert a slot index (0..95) to a 'HH:MM' string."""
    minutes = int(slot * SLOT_HOURS * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def slot_to_hour(slot: int) -> float:
    """Return the (fractional) hour-of-day at the start of a slot."""
    return slot * SLOT_HOURS


def safe_div(num, den):
    """Safe division returning 0.0 (and 0% for pct) when denominator is 0."""
    den = np.asarray(den, dtype=float)
    out = np.zeros_like(np.asarray(num, dtype=float))
    mask = den != 0
    out[mask] = np.asarray(num, dtype=float)[mask] / den[mask]
    return out
