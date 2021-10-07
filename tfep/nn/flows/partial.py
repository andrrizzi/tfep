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
    *flows : torch.nn.Module
        One or more normalizing flows mapping the non-constant degrees of freedom
        that must be executed in the given order.
    constant_indices : List[int], optional
        The indices of the degrees of freedom that must be kept constant.

    """

    def __init__(self, *flows, constant_indices):
        super().__init__()
        self.flows = torch.nn.ModuleList(flows)
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
        batch_size = x.size(0)
        cumulative_log_det_J = torch.zeros(batch_size, dtype=x.dtype)

        # Check if we need to traverse the flows in forward or inverse pass.
        layer_indices = range(len(self.flows))
        if inverse:
            flow_func_name = 'inverse'
            layer_indices = reversed(layer_indices)
        else:
            flow_func_name = 'forward'

        # Take care of the constant dimensions.
        if self._constant_indices is not None:
            # Check that we have already cached the propagated indices.
            if self._propagated_indices is None:
                constant_indices_set = set(self._constant_indices)
                self._propagated_indices =  list(i for i in range(x.size(1))
                                                 if i not in constant_indices_set)

            # This will be the returned tensors.
            final_x = torch.empty_like(x)
            final_x[:, self._constant_indices] = x[:, self._constant_indices]

            # This tensor goes through the flow.
            x = x[:, self._propagated_indices]

        # Now go through the flow layers.
        for layer_idx in layer_indices:
            flow = self.flows[layer_idx]
            # flow_func_name can be 'forward' or 'inv'.
            x, log_det_J = getattr(flow, flow_func_name)(x)
            cumulative_log_det_J += log_det_J

        # Add to the factored out dimensions.
        if self._constant_indices is not None:
            final_x[:, self._propagated_indices] = x
            x = final_x

        return x, cumulative_log_det_J
