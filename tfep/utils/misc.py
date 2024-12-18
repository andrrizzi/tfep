#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Miscellanea utility functions."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence
import contextlib
import os
from typing import Union

import numpy as np
import pint
import torch


# =============================================================================
# CONVERSION UTILITY FUNCTIONS
# =============================================================================

def flattened_to_atom(positions, space_dimension=3):
    """Compute a positions from flattened to standard atom format.

    The function takes a configuration (or a batch of configurations) with shape
    ``(n_atoms*space_dimension)`` and converts them into the standard shape
    ``(n_atoms, space_dimension)``.

    It converts both ``torch.Tensors`` and ``numpy.ndarray``, with and without
    ``pint`` units.

    Parameters
    ----------
    positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        The input can have the following shapes: ``(batch_size, n_atoms*space_dimension)``
        or ``(n_atoms * space_dimension,)``.
    space_dimension : int, optional
        The dimensionality of the phase space.

    Returns
    -------
    reshaped_positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        A view of the original tensor or array with shape ``(batch_size, n_atoms, 3)``
        or ``(n_atoms, 3)``.

    """
    n_atoms = positions.shape[-1] // space_dimension
    if len(positions.shape) > 1:
        batch_size = positions.shape[0]
        standard_shape = (batch_size, n_atoms, space_dimension)
    else:
        standard_shape = (n_atoms, space_dimension)
    return positions.reshape(standard_shape)


def atom_to_flattened(positions):
    """Compute a positions from standard atom to flattened format.

    The inverse operation of :func:`.flattened_to_atom`.

    Parameters
    ----------
    positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        The input can have the following shapes: ``(batch_size, n_atoms, N)`` or
        ``(n_atoms, N)``, where ``N`` is the dimensionality of the phase space.

    Returns
    -------
    reshaped_positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        A view of the original tensor or array with shape ``(batch_size, n_atoms*N)``
        or ``(n_atoms*N)``.

    See Also
    --------
    flattened_to_atom

    """
    n_atoms = positions.shape[-2]
    space_dimension = positions.shape[-1]
    if len(positions.shape) > 2:
        batch_size = positions.shape[0]
        flattened_shape = (batch_size, n_atoms*space_dimension)
    else:
        flattened_shape = (n_atoms*space_dimension,)
    return positions.reshape(flattened_shape)


def atom_to_flattened_indices(atom_indices, space_dimension=3):
    """Convert atom indices to the indices of the corresponding degrees of freedom in flattened format.

    Parameters
    ----------
    atom_indices : torch.Tensor or numpy.ndarray
        The input can have the following shapes: ``(batch_size, n_atoms)`` or
        ``(n_atoms,)``.
    space_dimension : int, optional
        The dimensionality of the coordinate space (default is 3).

    Returns
    -------
    flattened_indices : torch.Tensor, numpy.ndarray, or pint.Quantity
        The indices of the corresponding degrees of freedom in flattened format
        with shape ``(batch_size, n_atoms*3)`` or ``(n_atoms*3,)``.

    Examples
    --------

    The function works both with ``Tensor``s and numpy arrays.

    >>> atom_indices_np = np.array([0, 2])
    >>> list(atom_to_flattened_indices(atom_indices_np))
    [0, 1, 2, 6, 7, 8]

    >>> atom_indices_torch = torch.tensor(atom_indices_np)
    >>> atom_to_flattened_indices(atom_indices_torch, space_dimension=2).tolist()
    [0, 1, 4, 5]

    Batches of indices are supported.

    >>> atom_indices = torch.tensor([[0, 2], [1, 3]])
    >>> atom_to_flattened_indices(atom_indices).tolist()
    [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]

    """
    is_numpy = isinstance(atom_indices, np.ndarray)  # else is Tensor.
    is_not_batch = len(atom_indices.shape) == 1

    flattened_indices = atom_indices * space_dimension

    # Add fake dimension to avoid code branching if necessary.
    if is_not_batch:
        if is_numpy:
            flattened_indices = np.expand_dims(flattened_indices, axis=0)
        else:
            flattened_indices = torch.unsqueeze(flattened_indices, dim=0)

    # Each indices array has three times the number of indices.
    if is_numpy:
        flattened_indices = np.repeat(flattened_indices, space_dimension, axis=1)
    else:  # Tensor.
        flattened_indices = torch.repeat_interleave(flattened_indices, space_dimension, dim=1)

    # Update indices of other dimensions.
    for i in range(1, space_dimension):
        flattened_indices[:, i::space_dimension] += i

    if is_not_batch:
        return flattened_indices[0]
    return flattened_indices


def ensure_tensor_sequence(x: Union[str, int, float, Sequence], dtype=None) -> torch.Tensor:
    r"""If x is a sequence, return it as a torch.Tensor without copying the memory (if possible).

    Parameters
    ----------
    x : str, int, float, or Sequence
        The input. Sequences (that are not strings) are converted to ``torch.Tensor``\ s.
    dtype : dtype or None, optional
        If set, forces the tensor to a data type.

    Returns
    -------
    converted_x : str, int, float, or torch.Tensor
        The input, eventually converted to a tensor.

    """
    # as_tensor supports scalars but not None or strings.
    if not np.isscalar(x):
        try:
            x = torch.as_tensor(x, dtype=dtype)
        except (TypeError, RuntimeError):
            pass
    return x


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

    ``distance_unit`` and ``energy_unit`` must be passed together. If they are
     both ``None`` no conversion is performed. If only one of them is ``None``
     an error is raised.

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

    Raises
    ------
    ValueError
        If only one between ``distance_unit`` and ``energy_unit`` is passed.

    """
    if (energy_unit is not None) and (distance_unit is not None):
        force_unit = energy_unit / distance_unit
        try:
            # Convert to Hartree/(Bohr mol).
            forces = (forces * force_unit._REGISTRY.avogadro_constant).to(force_unit)
        except pint.errors.DimensionalityError:
            forces = forces.to(force_unit)
    elif not ((energy_unit is None) and (distance_unit is None)):
        raise ValueError('Both or neither energy_unit and distance_unit must be passed.')

    # The tensor must be unitless and with shape (batch_size, n_atoms*3).
    forces = atom_to_flattened(forces)
    return torch.tensor(forces.magnitude, dtype=dtype)


def remove_and_shift_sorted_indices(
        indices: torch.Tensor,
        removed_indices: torch.Tensor,
        remove: bool = True,
        shift: bool = True,
) -> torch.Tensor:
    """Remove from ``indices`` the indices in ``removed_indices`` (by value).

    Both ``indices`` and ``removed_indices`` must be sorted tensors of
    non-negative integers. The indices in ``indices`` are (optionally) shifted
    so that it can be used to point to elements of an array where
    ``removed_indices`` have been removed.

    Parameters
    ----------
    indices : torch.Tensor
        The tensor from which to remove indices.
    removed_indices : torch.Tensor
        The indices that must be removed from ``indices``.
    remove : bool, optional
        If ``indices`` and ``removed_indices`` do not overlap, and only
        shifting is necessary, this can be set to ``False``. Default ``True``.
    shift : bool, optional
        If ``False`` shifting the indices is not performed.

    Returns
    -------
    out : torch.Tensor
        The ``indices`` tensor after removing and shifting the elements.

    Examples
    --------
    >>> remove_and_shift_sorted_indices(
    ...     indices=torch.tensor([0, 3, 9, 13]),
    ...     removed_indices=torch.tensor([3, 12]),
    ...     shift=False,
    ... ).tolist()
    [0, 9, 13]

    >>> remove_and_shift_sorted_indices(
    ...     indices=torch.tensor([0, 3, 9, 13]),
    ...     removed_indices=torch.tensor([3, 12]),
    ...     shift=True,
    ... ).tolist()
    [0, 8, 11]

    """
    insert_indices = torch.searchsorted(removed_indices, indices)

    # Remove.
    if remove:
        # The maximum index returned by searchsorted is len(removed_indices) so
        # we pad to avoid IndexErrors. We use a -1 since all elements of
        # indices must be non-negative
        padded_removed_indices = torch.nn.functional.pad(
            removed_indices, pad=(0, 1), value=-1)
        mask = padded_removed_indices[insert_indices] != indices
        indices = indices[mask]
        insert_indices = insert_indices[mask]

    # Shift.
    if shift:
        indices = indices - insert_indices

    return indices


# =============================================================================
# I/O
# =============================================================================

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
