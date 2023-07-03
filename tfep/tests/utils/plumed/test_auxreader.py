#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in tfep.utils.plumed.auxreader.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

import MDAnalysis
import numpy as np
import pint
import pytest

from tfep.utils.plumed.auxreader import PLUMEDAuxReader


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

SCRIPT_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, '..', '..', 'data')
CHLOROMETHANE_PDB_FILE_PATH = os.path.join(DATA_DIR_PATH, 'chloro-fluoromethane.pdb')
AUXILIARY_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'auxiliary.xvg')

_U = pint.UnitRegistry()

# Reference data.
EXPECTED_AUX_DATA = np.genfromtxt(AUXILIARY_DATA_FILE_PATH, skip_header=1)


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================



# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('dt', [1, 5])
@pytest.mark.parametrize('col_names,expected_col_indices', [
    (None, None),
    (['time', 'col1'], [0, 1]),
    (['col2', 'col1'], [0, 2, 1]),
    (['col1', 'time'], [0, 1])
])
@pytest.mark.parametrize('units,expected_units_factors', [
    (None, None),
    ({'col1': 'nanometer'}, {'col1': 10}),
    ({'time': 'femtosecond'}, {'time': 1e-3}),
])
def test_plumed_auxreader(dt, col_names, expected_col_indices, units, expected_units_factors):
    """PLUMEDAuxReader reads the correct timesteps."""
    expected_data = EXPECTED_AUX_DATA.copy()[::dt]

    # Fix units of time step in PDB.
    if expected_units_factors is not None and 'time' in expected_units_factors:
        dt = dt * expected_units_factors['time']

    # Create trajectory with auxiliary information.
    universe = MDAnalysis.Universe(CHLOROMETHANE_PDB_FILE_PATH, dt=dt)
    aux_reader = PLUMEDAuxReader(
        file_path=AUXILIARY_DATA_FILE_PATH,
        col_names=col_names,
        units=units,
    )
    universe.trajectory.add_auxiliary('plumed', auxdata=aux_reader)

    # Re-order expected data. 'time' is always read and it's always in the first position.
    if col_names is not None:
        expected_data = expected_data[:, expected_col_indices]

    # Convert to MDAnalysis internal units (angstrom and picoseconds).
    if units is not None:
        for col_name, factor in expected_units_factors.items():
            col_idx = aux_reader.get_column_idx(col_name)
            expected_data[:, col_idx] *= factor

    # CHLOROMETHANE_PDB_FILE_PATH has only 5 frames and that what we should obtain.
    for ts_idx, ts in enumerate(universe.trajectory):
        assert np.allclose(ts.aux['plumed'], expected_data[ts_idx])
