
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


# =============================================================================
# TEST SYSTEMS
# =============================================================================

def benzoic_acid_universe():
    """Return an MDAnalysis.Universe representing a benzoic acid molecule.

    The atom naming and order of the benzoic acid molecule were taken from
    https://www.rcsb.org/ligand/BEZ.

    """
    import MDAnalysis
    import numpy as np

    # Load data from disk.
    benzoic_acid_file_path = os.path.join(DATA_DIR_PATH, 'benzoic_acid.npz')
    benzoic_acid_data = np.load(benzoic_acid_file_path)

    atom_names = benzoic_acid_data['names']
    elements = [name[0] for name in atom_names]
    n_atoms = len(atom_names)

    # Create an empty universe.
    universe = MDAnalysis.Universe.empty(n_atoms=n_atoms, trajectory=True)

    # Add topology attributes.
    universe.add_TopologyAttr('name', atom_names)
    universe.add_TopologyAttr('type', elements)
    universe.add_TopologyAttr('element', elements)
    universe.add_TopologyAttr('resname', ['BEN'])
    universe.add_TopologyAttr('resid', [1])
    universe.add_TopologyAttr('segid', ['BEN'])

    # Add bonds.
    universe.add_TopologyAttr('bonds', benzoic_acid_data['bonds'])

    # Add positions.
    universe.atoms.positions = benzoic_acid_data['positions']  # In Angstrom.

    return universe


def water_universe(n_waters):
    """Return a universe with n_waters water molecules."""
    # Recipe adapted from https://userguide.mdanalysis.org/stable/examples/constructing_universe.html.
    import MDAnalysis
    import numpy as np

    # Create the empty Universe
    n_atoms = n_waters*3
    universe = MDAnalysis.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_waters,
        atom_resindex=np.repeat(range(n_waters), 3),
        residue_segindex=[0] * n_waters,  # Put all waters in one segment.
        trajectory=True, # necessary for adding coordinates
    )

    # Add topology attributes.
    universe.add_TopologyAttr('name', ['O', 'H1', 'H2'] * n_waters)
    universe.add_TopologyAttr('type', ['O', 'H', 'H'] * n_waters)
    universe.add_TopologyAttr('element', ['O', 'H', 'H'] * n_waters)
    universe.add_TopologyAttr('resname', ['WAT'] * n_waters)
    universe.add_TopologyAttr('resid', list(range(1, n_waters+1)))
    universe.add_TopologyAttr('segid', ['WAT'])

    # Add bonds.
    bonds = []
    for o in range(0, n_atoms, 3):
        bonds.extend([(o, o+1), (o, o+2)])
    universe.add_TopologyAttr('bonds', bonds)

    # Add positions. Repeat these coordinates on a grid of points.
    h2o = np.array([[ 0,        0,       0      ],   # oxygen
                    [ 0.95908, -0.02691, 0.03231],   # hydrogen
                    [-0.28004, -0.58767, 0.70556]])  # hydrogen
    grid_size = 10
    spacing = 8
    coordinates = []
    for i in range(n_waters):
        x = spacing * (i % grid_size)
        y = spacing * ((i // grid_size) % grid_size)
        z = spacing * (i // (grid_size * grid_size))
        xyz = np.array([x, y, z])
        coordinates.extend(h2o + xyz.T)

    universe.atoms.positions = np.array(coordinates)

    return universe
