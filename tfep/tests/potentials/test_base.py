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
import torch

from tfep.potentials.ase import PotentialBase
from tfep.potentials.base import MultiStatePotential


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


def test_multi_state_potential_dispatch_with_and_without_dimensions():
    class DimAwarePotential(PotentialBase):
        DEFAULT_ENERGY_UNIT = "kcal/mol"
        DEFAULT_POSITIONS_UNIT = "angstrom"

        def __init__(self, bias: float):
            super().__init__()
            self.bias = float(bias)

        def forward(self, x, dimensions=None):
            base = x.sum(dim=1) + self.bias
            if dimensions is None:
                return base
            return base + dimensions[:, 0]

    p0 = DimAwarePotential(1.0)
    p1 = DimAwarePotential(2.0)
    multi = MultiStatePotential(p0, p1)

    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    dims = torch.tensor([[10.0], [20.0]])

    e0 = multi.energy(0, x)
    e1 = multi.energy(1, x)
    e0d = multi.energy(0, x, dims)
    e1d = multi.energy(1, x, dims)

    assert torch.allclose(e0, torch.tensor([7.0, 2.5]))
    assert torch.allclose(e1, torch.tensor([8.0, 3.5]))
    assert torch.allclose(e0d, torch.tensor([17.0, 22.5]))
    assert torch.allclose(e1d, torch.tensor([18.0, 23.5]))
