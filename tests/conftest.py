"""Pytest fixtures shared across tests."""
import os
import sys

# make project root importable so `import config` / `from src...` work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

import config
from src.data_generator import generate_scenario


@pytest.fixture
def normal_data():
    return generate_scenario("normal", seed=42)


@pytest.fixture
def small_data():
    """A small scenario for optimality / feasibility validation."""
    return generate_scenario("normal", seed=1, total_daily_cu=4000.0)
