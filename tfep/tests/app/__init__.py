
# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the package ``tfep.app``.
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from tfep.potentials.base import PotentialBase


# =============================================================================
# PACKAGE-WIDE TEST UTILITIES
# =============================================================================

class MockPotential(PotentialBase):
    """Mock potential to test TFEPMaps."""

    DEFAULT_ENERGY_UNIT = 'kcal'
    DEFAULT_POSITION_UNIT = 'angstrom'

    def forward(self, x):
        return x.sum(dim=1)
