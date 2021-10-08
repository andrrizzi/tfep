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
    """A sequence of normalizing flows.

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
    constant_indices : List[int], optional
        The indices of the degrees of freedom that must be kept constant.

    """

    def __init__(self, flow, constant_indices):
        super().__init__()
        self.flow = flow
        self._constant_indices = constant_indices

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
        # _constant_indices attribute is private so caching is "safe".
        if self._propagated_indices is None:
            constant_indices_set = set(self._constant_indices)
            self._propagated_indices =  list(i for i in range(x.size(1))
                                             if i not in constant_indices_set)

        # This will be the returned tensors.
        y = torch.empty_like(x)
        y[:, self._constant_indices] = x[:, self._constant_indices]

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
