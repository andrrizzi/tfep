#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Estimators for tfep that are compatible with the bootstrap analysis.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# (T)FEP ESTIMATOR
# =============================================================================

def fep_estimator(data, kT=1.0, vectorized=False):
    """FEP estimator.

    Computes the (weighted) free energy difference given a set of work values
    and, optionally, a set of associated sample log-weights (i.e. bias potential
    values).

    Parameters
    ----------
    data : torch.Tensor
        Shape ``(n_samples,)`` or ``(2, n_samples)``. In the first case, ``data``
        is an tensor of work values (in ``kT`` units). In the second, ``data[0]``
        contain the work values, and ``data[1]`` the log-weights for each samples
        (also in ``kT`` units).

        If ``vectorized`` is ``True``, an extra dimension is expected so that
        the shape should be either ``(n_bootstraps, n_samples,)`` or
        ``(n_bootstraps, 2, n_samples)``.
    kT : float, optional
        The function assumes that the work and log-weights values are already
        divided by kT. If this is not the case, this should be set to the value
        of kT in the same unit of energy passed in ``data``.
    vectorized : bool, optional
        Whether the estimate should be vectorized for bootstrap analysis.

    Returns
    -------
    df : torch.Tensor
        Shape ``(1,)``. The free energy difference in the same units of the work
        values.

         If ``vectorized`` is ``True``, this has shape ``(n_bootstraps,)``.

    """
    # Separate work and bias.
    if vectorized:
        if len(data.shape) == 2:
            work, bias = data, None
        else:
            work, bias = data.permute(2, 0, 1)
    else:
        if len(data.shape) == 1:
            work, bias = data, None
        else:
            work, bias = data.T

    # Compute the log_weights.
    if bias is None:
        n_samples = torch.tensor(work.shape[-1])
        log_weights = -torch.log(n_samples)
    else:
        # Normalize the weights.
        log_weights = torch.nn.functional.log_softmax(bias/kT, dim=-1)

    return - kT * torch.logsumexp(-work/kT + log_weights, dim=-1)
