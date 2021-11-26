#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Normalizing flow mapping only a subset of the input degrees of freedom.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# PARTIAL FLOW
# =============================================================================

class PartialFlow(torch.nn.Module):
    """A normalizing flows mapping only a subset of the degrees of freedom.

    The layer wraps a sequence of normalizing flows and passes to them only a
    subset of the degrees of freedom (DOF) while maintaining the other constants.
    Note that the constant DOFs are not even seen by the wrapped flow and so they
    cannot condition the output.

    The wrapped flow must be configured correctly to take as input the subset
    of the DOFs that are not constant.

    Parameters
    ----------
    flow : torch.nn.Module
        The wrapped normalizing flows mapping the non-constant degrees of freedom.
    fixed_indices : array-like of int, optional
        The indices of the degrees of freedom that must be kept constant.

    """

    def __init__(self, flow, fixed_indices):
        super().__init__()
        self.flow = flow

        # Make sure we store the fixed_indices as a list as we'll later transform
        # it into a set and we need to make sure it's a set of integers rather
        # than a set of tensors.
        try:
            self._fixed_indices = fixed_indices.tolist()
        except AttributeError:
            self._fixed_indices = fixed_indices

        # We also need the indices that we are not factoring out.
        # This is initialized lazily in self._pass() because we
        # need the dimension of the input.
        self._propagated_indices = None

    def n_parameters(self):
        """int: The total number of parameters that can be optimized."""
        try:
            return self.flow.n_parameters()
        except AttributeError:
            # Handle the case in which the flows have been encapsulated
            # in a Sequential module.
            return sum(f.n_parameters() for f in self.flow)

    def forward(self, x):
        return self._pass(x, inverse=False)

    def inverse(self, y):
        return self._pass(y, inverse=True)

    def _pass(self, x, inverse):
        # Check if we have already cached the propagated indices. The
        # _fixed_indices attribute is private so caching is "safe".
        if self._propagated_indices is None:
            fixed_indices_set = set(self._fixed_indices)
            self._propagated_indices =  list(i for i in range(x.size(1))
                                             if i not in fixed_indices_set)

        # This will be the returned tensors.
        y = torch.empty_like(x)
        y[:, self._fixed_indices] = x[:, self._fixed_indices]

        # This tensor goes through the flow.
        x = x[:, self._propagated_indices]

        # Now go through the flow layers.
        if inverse:
            x, log_det_J = self.flow.inverse(x)
        else:
            x, log_det_J = self.flow(x)

        # Add to the factored out dimensions.
        y[:, self._propagated_indices] = x

        return y, log_det_J
