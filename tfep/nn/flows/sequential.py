#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Normalizing flow concatenating multiple normalizing flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# PARTIAL FLOW
# =============================================================================

class SequentialFlow(torch.nn.Sequential):
    """A sequence of normalizing flows.

    The layer wraps a sequence of normalizing flows and returns the toal
    transformed coordinate with the cumulative log absolute determinant of
    the Jacobian. It also expose methods/properties that are shared to all
    flows in this library such as ``inverse()``.

    Parameters
    ----------
    *flows : torch.nn.Module
        One or more normalizing flows that must be executed in the given order
        in the forward direction.

    """

    def n_parameters(self):
        """int: The total number of parameters that can be optimized."""
        return sum(flow.n_parameters() for flow in self)

    def forward(self, x):
        return self._pass(x, inverse=False)

    def inverse(self, y):
        return self._pass(y, inverse=True)

    def _pass(self, x, inverse):
        batch_size = x.size(0)
        cumulative_log_det_J = torch.zeros(batch_size).to(x)

        # Check if we need to traverse the flows in forward or inverse pass.
        if inverse:
            flows = reversed(self)
            flow_func_name = 'inverse'
        else:
            flows = self
            flow_func_name = 'forward'

        # Now go through the flow layers.
        for flow in flows:
            # flow_func_name can be 'forward' or 'inverse'.
            x, log_det_J = getattr(flow, flow_func_name)(x)
            cumulative_log_det_J += log_det_J

        return x, cumulative_log_det_J
