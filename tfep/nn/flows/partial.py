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

from collections.abc import Sequence

import torch

from tfep.utils.misc import ensure_tensor_sequence


# =============================================================================
# PARTIAL FLOW
# =============================================================================

class PartialFlow(torch.nn.Module):
    """A normalizing flows mapping only a subset of the degrees of freedom.

    The layer wraps a normalizing flows and passes to it only a subset of the
    degrees of freedom (DOF) while maintaining the other constants. Note that the
    constant DOFs are not even seen by the wrapped flow and so they cannot
    condition the output.

    The wrapped flow must be configured correctly to take as input the subset
    of the DOFs that are not constant.

    Parameters
    ----------
    flow : torch.nn.Module
        The wrapped normalizing flows mapping the non-constant degrees of freedom.
    fixed_indices : array-like of int, optional
        The indices of the degrees of freedom that must be kept constant.
    return_partial : bool, optional
        If ``True``, only the propagated indices are returned.

    Attributes
    ----------
    return_partial : bool
        If ``True``, only the propagated indices are returned.

    """

    def __init__(
            self,
            flow: torch.nn.Module,
            fixed_indices: Sequence[int],
            return_partial: bool = False,
    ):
        super().__init__()
        self.flow = flow
        self.return_partial = return_partial

        # Convert to tensor.
        self._fixed_indices = ensure_tensor_sequence(fixed_indices)

        # We also need the indices that we are not factoring out.
        # This is initialized lazily in self._pass() because we
        # need the dimension of the input.
        self._propagated_indices = None

    def n_parameters(self):
        """int: The total number of parameters that can be optimized."""
        return self.flow.n_parameters()

    def forward(self, x):
        return self._pass(x, inverse=False)

    def inverse(self, y):
        return self._pass(y, inverse=True)

    def _pass(self, x, inverse):
        # There's no need to slice if fixed_indices is empty.
        if len(self._fixed_indices) > 0:
            # Check if we have already cached the propagated indices. The
            # _fixed_indices attribute is private so caching is "safe".
            if self._propagated_indices is None:
                fixed_indices_set = set(self._fixed_indices.tolist())
                self._propagated_indices =  torch.tensor([i for i in range(x.size(1))
                                                          if i not in fixed_indices_set])

            # This will be the returned tensors.
            y = torch.empty_like(x)
            y[:, self._fixed_indices] = x[:, self._fixed_indices]

            # This tensor goes through the flow.
            x = x[:, self._propagated_indices]

        # Now go through the flow layers. Continuous flow may return also a
        # regularization term in addition to the mapped x and the log_det_J.
        if inverse:
            out = self.flow.inverse(x)
        else:
            out = self.flow(x)

        if self.return_partial:
            return out

        # Add to the factored out dimensions.
        if len(self._fixed_indices) > 0:
            y[:, self._propagated_indices] = out[0]
        else:
            y = out[0]

        return (y, *out[1:])
