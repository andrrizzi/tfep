#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Miscellanea utility functions."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import contextlib
import os

import pint
import torch

from tfep.utils.geometry import standard_to_flattened


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def energies_array_to_tensor(energies, energy_unit=None, dtype=None):
    """Helper function to convert the a batch of energies from numpy array to PyTorch tensor.

    Parameters
    ----------
    energies : pint.Quantity
        The energies with shape ``(batch_size,)`` with units.
    energy_unit : pint.Unit, optional
        The units of energy used in the returned energies. If ``None``, no conversion
        is performed, and the energy will be in the same units as the input.
    dtype : type, optional
        The ``torch`` data type to be used for the returned ``Tensor``.

    Returns
    -------
    energies : torch.Tensor
        The energies with shape ``(batch_size,)`` as a unitless ``Tensor``
        in units of ``energy_unit``.

    """
    if energy_unit is not None:
        try:
            # Convert to Hartree/mol.
            energies = (energies * energy_unit._REGISTRY.avogadro_constant).to(energy_unit)
        except pint.errors.DimensionalityError:
            energies = energies.to(energy_unit)

        # Reconvert Pint array to tensor.
    return torch.tensor(energies.magnitude, dtype=dtype)


def forces_array_to_tensor(forces, distance_unit=None, energy_unit=None, dtype=None):
    """Helper function to convert the a batch of forces from numpy array to PyTorch tensor.

    Parameters
    ----------
    forces : pint.Quantity
        The forces with shape ``(batch_size, n_atoms, 3)`` with units.
    distance_unit : pint.Unit, optional
        The units of distance used in the returned forces. If ``None``, no
        conversion is performed, and the forces will be in the same units as the
        input.
    energy_unit : pint.Unit, optional
        The units of energy used in the returned forces. If ``None``, no conversion
        is performed, and the energy will be in the same units as the input.
    dtype : type, optional
        The ``torch`` data type to be used for the returned ``Tensor``.

    Returns
    -------
    forces : torch.Tensor
        The forces with shape ``(batch_size, n_atoms*3)`` as a unitless ``Tensor``
        in units of ``energy_unit/distance_unit``.

    """
    if (energy_unit is not None) and (distance_unit is not None):
        force_unit = energy_unit / distance_unit
        try:
            # Convert to Hartree/(Bohr mol).
            forces = (forces * force_unit._REGISTRY.avogadro_constant).to(force_unit)
        except pint.errors.DimensionalityError:
            forces = forces.to(force_unit)

    # The tensor must be unitless and with shape (batch_size, n_atoms*3).
    forces = standard_to_flattened(forces)
    return torch.tensor(forces.magnitude, dtype=dtype)


@contextlib.contextmanager
def temporary_cd(dir_path):
    """Context manager that temporarily sets the working directory.

    Parameters
    ----------
    dir_path : str or None
        The path to the temporary working directory. If ``None``, the working
        directory is not changed. This might be useful to avoid branching code.

    """
    if dir_path is None:
        yield
    else:
        old_dir_path = os.getcwd()
        os.chdir(dir_path)
        try:
            yield
        finally:
            os.chdir(old_dir_path)
