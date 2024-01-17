#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.potentials.base``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pint
import pytest

from tfep.potentials.ase import PotentialBase


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Common unit registry for all tests.
_UREG = pint.UnitRegistry()


# =============================================================================
# TEST POTENTIAL BASE
# =============================================================================

@pytest.mark.parametrize('energy_unit', [None, _UREG.kcal])
@pytest.mark.parametrize('positions_unit', [None, _UREG.nanometer])
def test_default_units(energy_unit, positions_unit):
    """Test that default units work correctly."""
    class ExamplePotential(PotentialBase):
        DEFAULT_ENERGY_UNIT = 'hartree'
        DEFAULT_POSITIONS_UNIT = 'angstrom'

    potential = ExamplePotential(energy_unit=energy_unit, positions_unit=positions_unit)
    if energy_unit is None:
        assert str(potential.energy_unit) == ExamplePotential.DEFAULT_ENERGY_UNIT
    else:
        assert potential.energy_unit == energy_unit

    if positions_unit is None:
        assert str(potential.positions_unit) == ExamplePotential.DEFAULT_POSITIONS_UNIT
    else:
        assert potential.positions_unit == positions_unit
