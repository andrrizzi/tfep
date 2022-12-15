#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Loss functions to train PyTorch normalizing flows for reweighting.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class BoltzmannKLDivLoss(torch.nn.Module):
    """KL divergence between two Boltzmann distributions.

    The loss function assumes the sampling is done in the reference distribution
    A. The KL divergence between two Boltzmann distribution is then given by

        :math:`D_{KL}[p_A||p_B] = \int p_A(x) \Delta u_{AB}(x) dx - \Delta f_{AB}`

    where :math:`p_A(x)` is distribution A, :math:`\Delta u_{AB}(x) = u_B(x) - u_A(x)`
    is the difference between the reduced potential energies B and A for configuration
    x (in units of :math:`k_B T`), and :math:`\Delta f_{AB} = f_B - f_A` is the
    reduced free energy difference (also in units of :math:`k_B T`).

    In TFEP, the KL divergence of interest is between A and the mapped
    distribution B', whose potential energy includes the logarithm of the
    absolute value of the Jacobian of the map M

        :math:`u_{B'}(x) = u_B(x) - log|det J_M(x)|`

    Moreover, because the free energy difference and reference potential energies
    do not depend on the map, they can be ignored, and the loss function can be
    optimized by minimizing

        :math:`\frac{1}{N} \sum_i u_{B'}(x_i)`

    Finally, if the samples were not sampled from A, the mean must be weighted.
    If log-weights are passed to the function, the loss is

        :math:`\frac{1}{N} \sum_i \frac{e^{w_i}}{\sum_i e^{w_i}} u_{B'}(x_i)`

    where :math:`w_i` is the log-weight for the i-th sample, and correspond to
    potential energy difference between the sampled and A distributions.

    """

    def forward(self, target_potentials, log_det_J=None, log_weights=None, ref_potentials=None):
        """Compute the loss.

        .. warning::
            Because ``Tensor``s are unit-less you need to make sure all arguments
            are passed using consistent units.

            Typically, the ``log_det_J`` obtained as output of the normalizing
            flow will be in units of :math:`k_BT` so potentials and log-weights
            should be divided by :math:`k_BT` as well.

        Parameters
        ----------
        target_potentials : torch.Tensor
            ``target_potentials[i]`` is the reduced potential energy of the i-th
            (mapped) sample in units of kT evaluated using target potential B.
            The shape is ``(batch_size,)``.
        log_det_J : torch.Tensor, optional
            ``log_det_J[i]`` is the logarithm of the absolute value of the
            determinant of the Jacobian of the map (in units of kT) for the i-th
            sample. The shape is ``(batch_size,)``.

            If not passed, it is assumed the samples were not mapped or,
            equivalently, that the Jacobian contribution has been already included
            in ``potentials_B``.
        log_weights : torch.Tensor, optional
            ``log_weights[i]`` is the log-weight for the i-th sample (in units
            of kT) that can be used to reweight the loss function if the samples
            were not sampled from A. The shape is ``(batch_size,)``.
        ref_potentials : torch.Tensor, optional
            ``ref_potentials_A[i]`` is the reduced potential energy of the i-th
            sample in units of kT evaluated using the reference potential A. The
            shape is ``(batch_size,)``.

            This is optional since it does not affect the optimization but only
            the value returned by the loss function.

        Returns
        -------
        loss : torch.Tensor
            The value of the loss function.

        """
        reduced_work = target_potentials
        if log_det_J is not None:
            reduced_work = reduced_work - log_det_J
        if ref_potentials is not None:
            reduced_work = reduced_work - ref_potentials

        # Check if this must be a weighted or unweighted mean.
        if log_weights is not None:
            weights = torch.nn.functional.softmax(log_weights)
            return torch.sum(weights * reduced_work)
        return torch.mean(reduced_work)
