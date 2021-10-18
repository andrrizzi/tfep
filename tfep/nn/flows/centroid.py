#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Transformation that constrains the (weighted) centroid of the DOFs.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch

from tfep.utils.geometry import (
    flattened_to_atom, atom_to_flattened, atom_to_flattened_indices)
from tfep.nn.flows import PartialFlow


# =============================================================================
# FRAME TRANSFORMATIONS
# =============================================================================

class CenteredCentroidFlow(PartialFlow):
    """A transformation constraining the (weighted) centroid of the DOFs.

    This flow performs the following operations:
    - It computes the (optionally weighted) centroid of the input features (or
      a subset of them).
    - Translates the system to move the centroid to the origin or a user-defined
      point.
    - Removes N degrees of freedom (DOFs) from the input features (where N is the
      dimensionality of a point in space) and runs the remaining features through
      the wrapped flow.
    - Re-adds the N constrained DOFs so that the flow preserves the centroid.
    - Optionally, translates the system to restore the original coordinates of
      the centroid.

    The flow assumes that the input features have shape ``(batch_size, n_dofs)``,
    and each batch sample represent a sequence of points in an N-dimensional
    space. The points must must be listed so that ``input[b][N*i:N*i+N]`` are
    the coordinates of the ``i``-th point for batch sample ``b``.

    Parameters
    ----------
    flow : torch.nn.Module
        The wrapped flow.
    space_dimension : int
        The dimensionality of a single point in space (e.g., ``3`` if the input
        features represent points in a 3D space).
    subset_point_indices : Tensor of int, optional
        A tensor of shape ``(n_points,)``. If passed, the centroid is computed
        over a subset of ``n_points`` points. If not passed, the centroid is
        computed using all the input features.

        Note that the indices must refer to the points, not the feature indices.
        For example, in a 3D space, ``subset_point_indices = [1, 3]`` will compute
        the centroid using the feature indices ``[3, 4, 5, 9, 10, 11]``.
    weights : Tensor of floats, optional
        A tensor of shape ``(n_points,)``, where ``n_points`` is the number of
        points used to compute the centroid (which depends on ``subset_point_indices``).
        ``weights[i]`` is the (unnormalized) weight used for the ``i``-th point.
        If not passed, the center of geometry is computed. This can be used for
        example to center the system on the center of mass.
    fixed_point_idx : int, optional
        The index of the point that is fixed and does not go through the flow.
        Note that this refers to the index of the point, not the input feature.
        Also, if ``subset_point_indices`` is passed, this index is relative to
        the subset of atoms used to compute the centroid.

        For example, if ``fixed_point_idx`` is 1 and ``subset_point_indices`` is
        ``None``, the fixed DOFs will be ``[3, 4, 5]`` in a 3D space. However,
        if ``subset_point_indices = [1, 2]`` then the fixed DOFs will be
        ``[6, 7, 8]``.
    origin : Tensor of floats, optional
        A tensor of shape ``(space_dimension,)``. If the centroid must be moved
        to a point different than zero.
    translate_back : bool, optional
        If ``False``, the output configuration has the centroid in the ``origin``.
        Otherwise, it the centroid is restored to the original position.

    Attributes
    ----------
    flow : torch.nn.Module
        The wrapped flow.
    origin : Tensor of floats
        The position of the origin where the centroid is translated before going
        through the flow.
    translate_back : bool
        Whether the centroid is restored to its original position in the output
        configuration or if it is left at the ``origin``.

    """

    def __init__(
            self,
            flow,
            space_dimension,
            subset_point_indices=None,
            weights=None,
            fixed_point_idx=0,
            origin=None,
            translate_back=True
    ):
        # Handle mutable defaults.
        if origin is None:
            origin = torch.zeros(space_dimension)
        elif len(origin) != space_dimension:
            raise ValueError("'origin' must have length equal to 'space_dimension'.")

        # Determine which atom is fixed. Save fixed_point_idx before modifying it.
        self._fixed_point_idx = fixed_point_idx
        if subset_point_indices is not None:
            fixed_point_idx = subset_point_indices[fixed_point_idx]

            # Check the dimension of weights. If subset_point_indices is None we
            # have no way to know in advance the number of points used to compute
            # the centroid.
            if (weights is not None) and (len(weights) != len(subset_point_indices)):
                raise ValueError("'weights' must have the same length as 'subset_point_indices'.")

        # Both atom_to_flattened_indices and PartialFlow take an array-like of indices.
        fixed_indices = atom_to_flattened_indices(torch.tensor([fixed_point_idx]), space_dimension)

        # Call PartialFlow constructor to fix the indices.
        super().__init__(flow, fixed_indices=fixed_indices)

        # Normalize the weights and shape them so that we can simply multiply them later.
        if weights is not None:
            weights = weights / torch.sum(weights)
            weights = weights.unsqueeze(dim=1)

        # Save rest of dimensions.
        self._space_dimension = space_dimension
        self._subset_point_indices = subset_point_indices
        self._weights = weights
        self.origin = origin
        self.translate_back = translate_back

    @property
    def space_dimension(self):
        """int: The dimensionality of a single point in space.

        For example, ``3`` if the input features represent points in a 3D space.
        """
        # Currently we don't allow modifying this. If we do, we also
        # need to re-check that origin have a consistent dimension.
        return self._space_dimension

    def forward(self, x):
        """Transform the input configuration."""
        return self._transform(x)

    def inverse(self, y):
        """Invert the forward transformation.

        This works only if the forward transformation was performed with
        ``translate_back`` set to ``True``.

        """
        if not self.translate_back:
            raise ValueError("The inverse of CenteredCentroidFlow can be computed"
                             " only if 'translate_back' is set to True during both"
                             " the forward and inverse transformations.")
        return self._transform(y, inverse=True)

    def _transform(self, x, inverse=False):
        """Apply the forward/inverse transformation."""
        # Reshape the coordinates to standard atom shape.
        x_atom_shape = flattened_to_atom(x, self.space_dimension)

        # Compute the centroid. Shape is (batch_size, space_dimension).
        x_centroid = self._compute_centroid(x_atom_shape)

        # Determine the translation vectors to move each batch configuration to the origin.
        translate_vector = self.origin - x_centroid
        # Reshape from (batch_size, space_dimension) to (batch_size, 1, space_dimension).
        translate_vector = translate_vector.unsqueeze(dim=1)

        # Translate the system so that the centroid is in the origin.
        x_atom_shape = x_atom_shape + translate_vector
        x_translated = atom_to_flattened(x_atom_shape)

        # Apply the transformation through the PartialFlow.
        if inverse:
            y, log_det_J = super().inverse(x_translated)
        else:
            y, log_det_J = super().forward(x_translated)

        # Modify the fixed degrees of freedom so that the centroid remains in the origin.
        y_atom_shape = flattened_to_atom(y, self.space_dimension)
        y_centroid, fixed_weight = self._compute_centroid(y_atom_shape, exclude_fixed_point=True)

        # Translate the constrained point to force the centroid in the origin.
        y[:, self._fixed_indices] = (self.origin - y_centroid) / fixed_weight

        # If we need to translate back, the new centroid must equal the original.
        if self.translate_back:
            y_atom_shape = flattened_to_atom(y, self.space_dimension)
            y_atom_shape = y_atom_shape - translate_vector
            y = atom_to_flattened(y_atom_shape)

        return y, log_det_J

    def _compute_centroid(self, x_atom_shape, exclude_fixed_point=False):
        """Return the centroid given the coordinates in standard atom shape.

        The returned centroid has shape (batch_size, space_dimension).
        """
        # Select the subset of atoms used to compute the centroid.
        if self._subset_point_indices is None:
            x_centroid_atom_shape = x_atom_shape
        else:
            x_centroid_atom_shape = x_atom_shape[:, self._subset_point_indices]

        # Compute the weighted centroid.
        if self._weights is None:
            centroid = torch.mean(x_centroid_atom_shape, dim=1)
            fixed_weight = 1 / x_centroid_atom_shape.shape[1]
        else:
            centroid = torch.sum(x_centroid_atom_shape*self._weights, dim=1)
            fixed_weight = self._weights[self._fixed_point_idx]

        # Remove fixed point.
        if exclude_fixed_point:
            centroid = centroid - x_centroid_atom_shape[:, self._fixed_point_idx] * fixed_weight
            return centroid, fixed_weight

        return centroid
