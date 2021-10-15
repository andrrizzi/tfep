#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Utility functions to manipulate coordinates."""


# =============================================================================
# CONVERT POSITIONS FROM FLATTENED TO STANDARD ATOM FORMAT AND VICEVERSA
# =============================================================================

def flattened_to_atom(positions):
    """Compute a positions from flattened to standard atom format.

    The function takes a configuration (or a batch of configurations) with shape
    ``(n_atoms*3)`` and converts them into the standard shape ``(n_atoms, 3)``.

    It converts both ``torch.Tensors`` and ``numpy.ndarray``, with and without
    ``pint`` units.

    Parameters
    ----------
    positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        The input can have the following shapes: ``(batch_size, n_atoms*3)`` or
        ``(n_atoms * 3,)``.

    Returns
    -------
    reshaped_positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        A view of the original tensor or array with shape ``(batch_size, n_atoms, 3)``
        or ``(n_atoms, 3)``.

    """
    n_atoms = positions.shape[-1] // 3
    if len(positions.shape) > 1:
        batch_size = positions.shape[0]
        standard_shape = (batch_size, n_atoms, 3)
    else:
        standard_shape = (n_atoms, 3)
    return positions.reshape(standard_shape)


def atom_to_flattened(positions):
    """Compute a positions from standard atom to flattened format.

    The inverse operation of :func:`.flattened_to_atom`.

    Parameters
    ----------
    positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        The input can have the following shapes: ``(batch_size, n_atoms, 3)`` or
        ``(n_atoms, 3)``..

    Returns
    -------
    reshaped_positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        A view of the original tensor or array with shape ``(batch_size, n_atoms*3)``
        or ``(n_atoms*3)``.

    See Also
    --------
    flattened_to_atom

    """
    n_atoms = positions.shape[-2]
    if len(positions.shape) > 2:
        batch_size = positions.shape[0]
        flattened_shape = (batch_size, n_atoms*3)
    else:
        flattened_shape = (n_atoms*3,)
    return positions.reshape(flattened_shape)
