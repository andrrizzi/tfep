#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Utility functions to manipulate coordinates."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import torch

# =============================================================================
# CONVERT POSITIONS FROM FLATTENED TO STANDARD ATOM FORMAT AND VICEVERSA
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
