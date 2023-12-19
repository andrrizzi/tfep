
# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function of the ``tfep`` library.
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

from tfep.potentials.base import PotentialBase


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Path to the test data.
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), 'data')


# =============================================================================
# PACKAGE-WIDE TEST UTILITIES
# =============================================================================

class MockPotential(PotentialBase):
    """Mock potential to test TFEPMaps."""

    DEFAULT_ENERGY_UNIT = 'kcal'
    DEFAULT_POSITION_UNIT = 'angstrom'

    def forward(self, x):
        return x.sum(dim=1)
