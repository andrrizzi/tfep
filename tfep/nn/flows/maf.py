#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Masked autoregressive flow layer for PyTorch.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence
from typing import Optional, Union

import torch

from tfep.nn.conditioners.made import MADE
from tfep.nn.transformers.affine import AffineTransformer
from tfep.utils.misc import ensure_tensor_sequence


# =============================================================================
# MAF
# =============================================================================

class MAF(torch.nn.Module):
    """Masked Autoregressive Flow.

    This implements an autoregressive flow in which the :class:`tfep.nn.conditioners.MADE` [1]
    network is used for the conditioner. The class supports arbitrary
    transformers.

    When the transformer is the :class:`AffineTransformer`, this is
    equivalent to MAF and IAF [2-3]. These two differ only in the direction
    of the conditional dependence, effectively determining which between
    forward and inverse evaluation is faster.

    See Also
    --------
    :class:`tfep.nn.conditioners.MADE` : The autoregressive layer used as conditioner.

    References
    ----------
    [1] Germain M, Gregor K, Murray I, Larochelle H. Made: Masked autoencoder
        for distribution estimation. In International Conference on Machine
        Learning 2015 Jun 1 (pp. 881-889).
    [2] Kingma DP, Salimans T, Jozefowicz R, Chen X, Sutskever I, Welling M.
        Improved variational inference with inverse autoregressive flow.
        In Advances in neural information processing systems 2016 (pp. 4743-4751).
    [3] Papamakarios G, Pavlakou T, Murray I. Masked autoregressive flow for
        density estimation. In Advances in Neural Information Processing
        Systems 2017 (pp. 2338-2347).

    Examples
    --------

    A masked autoregressive flow using a linear transformer. This uses two MAF
    layers inverting the dependencies between inputs. This is a standard strategy
    in autoregressive flows to ensure every output depends on every input.

    >>> from tfep.nn.conditioners.made import generate_degrees
    >>> flow = torch.nn.Sequential(
    ...     MAF(degrees_in=generate_degrees(n_features=5, order='ascending')),
    ...     MAF(degrees_in=generate_degrees(n_features=5, order='descending')),
    ... )

    Multiple inputs can be assigned the same degree. Further, it is possible to
    specify "conditioning" inputs (in this case, the first 3) which affect all
    outputs but that are not mapped by assigning them degree -1.

    >>> maf = MAF(degrees_in=[-1, -1, -1, 0, 0, 1, 2])

    """

    def __init__(
            self,
            degrees_in: Sequence[int],
            transformer: Optional[torch.nn.Module] = None,
            hidden_layers: Union[int, Sequence[int], Sequence[Sequence[int]]] = 2,
            periodic_indices: Optional[Sequence[int]] = None,
            periodic_limits: Optional[Sequence[int]] = None,
            weight_norm: bool = True,
            initialize_identity: bool = True,
    ):
        """Constructor.

        Parameters
        ----------
        degrees_in : Sequence[int]
            Shape: ``(n_inputs,)``. ``degrees_in[i]`` is the degree assigned to
            the ``i``-th input. The degrees must assume consecutive values starting
            from 0 or -1. Input features assigned a -1 degree are labeled as "conditioning"
            and affect the output without being mapped.
        transformer : torch.nn.Module or None, optional
            The transformer used to map the input features. By default, the
            :class:`tfep.nn.transformers.affine.AffineTransformer` is used.
        hidden_layers : Union[int, Sequence[int], Sequence[Sequence[int]]], optional
            If an integer, this is the number of hidden layers. In this case,
            the number of nodes in each layer is set to
            ``max(n_inputs, ceil((n_inputs * n_outputs)**0.5))``
            where ``n_inputs`` is the number of input features that affect the
            output, and ``n_outputs`` is the number of output features.

            If a sequence of integers, ``hidden_layers[l]`` is the number of
            nodes in the l-th hidden layer. The degrees of each node are assigned
            in a round-robin fashion by tiling ``degrees_in`` until the requested
            number of nodes is covered.

            Otherwise, ``degrees_hidden[l][i]`` is the degree assigned to the
            ``i``-th node of the ``l``-th hidden layer.

            Default is 2.
        periodic_indices : Sequence[int] or None, optional
            Shape (n_periodic,). The (ordered) indices of the input features that
            are periodic. When passed to the conditioner, these are transformed
            to ``(cos(a), sin(a))``, where ``a`` is a shifted/rescaled feature
            to be in the interval [0, 2pi]. This way the conditioner will have
            periodic input.
        periodic_limits : Sequence[int] or None, optional
            A pair ``(lower, upper)`` defining the limits of the periodic input
            features  (e.g. ``[-pi, pi]``). The period is ``upper - lower``.
        weight_norm : bool, optional
            If ``True``, weight normalization is applied to the masked linear
            modules. Default is ``True``.
        initialize_identity : bool, optional
            If ``True``, the parameters are initialized in such a way that
            the flow initially performs the identity function.

        """
        super().__init__()

        # By default, use an affine transformer.
        if transformer is None:
            transformer = AffineTransformer()
        self._transformer = transformer

        # Convert all sequences to Tensors to simplify the code.
        degrees_in = ensure_tensor_sequence(degrees_in)
        periodic_indices = ensure_tensor_sequence(periodic_indices)
        periodic_limits = ensure_tensor_sequence(periodic_limits)

        # Check that degrees_in satisfy the requirements.
        min_degree_in = degrees_in.min().tolist()
        max_degree_in = degrees_in.max().tolist()
        if ((set(degrees_in.tolist()) != set(range(min_degree_in, max_degree_in+1))) or
                (min_degree_in not in {-1, 0})):
            raise ValueError('degrees_in must assume consecutive values starting '
                             'from 0 (or -1 for conditioning input features).')

        # Create the lifter used to map the periodic degrees of freedom.
        if periodic_indices is None:
            self._lifter = None
            degrees_in_embedded = degrees_in
        elif periodic_limits is None:
            raise ValueError('periodic_limits must be given if periodic_indices is passed.')
        else:
            # Initialize Module lifting the periodic features.
            self._lifter = _LiftPeriodic(
                dimension_in=len(degrees_in),
                periodic_indices=periodic_indices,
                limits=periodic_limits
            )

            # Each periodic feature is transformed to (cosa, sina). We assign
            # the same degree to each lifted feature.
            repeats = torch.ones_like(degrees_in)
            repeats[periodic_indices] = 2
            degrees_in_embedded = torch.repeat_interleave(degrees_in, repeats)

        # Keep track of all mapped and conditioning input features.
        mapped_mask = degrees_in != -1
        n_mapped_features = mapped_mask.count_nonzero()
        if n_mapped_features == len(degrees_in):
            # We set them to None to avoid some indexing operations in forward/inverse.
            self.register_buffer('_conditioning_mask', None)
            self.register_buffer('_mapped_mask', None)
        else:
            self.register_buffer('_conditioning_mask', ~mapped_mask)
            self.register_buffer('_mapped_mask', mapped_mask)

        # Create a map: mapped_degree_in -> mask which will be useful for inverse().
        self.register_buffer('_mapped_degree_to_mask', torch.empty(max_degree_in+1, len(degrees_in), dtype=bool))
        for degree in range(max_degree_in+1):
            self._mapped_degree_to_mask[degree] = degrees_in == degree

        # The output degrees.
        if self._mapped_mask is not None:
            degrees_out = degrees_in[self._mapped_mask]
        else:
            degrees_out = degrees_in
        degrees_out = degrees_out.tile((self._transformer.n_parameters_per_input,))

        # We need two MADE layers for the scaling and the shifting.
        self._conditioner = MADE(
            degrees_in=degrees_in_embedded,
            degrees_out=degrees_out,
            hidden_layers=hidden_layers,
            weight_norm=weight_norm,
        )

        # Initialize the log_scale and shift nets to 0.0 so that at
        # the beginning the MAF layer performs the identity function.
        if initialize_identity:
            # Determine the conditioner that will make the transformer the identity function.
            identity_conditioner = self._transformer.get_identity_parameters(n_mapped_features)

            # There is a single output bias parameter vector so we need to
            # convert from shape (batch_size, n_parameters_per_input, n_features)
            # to (batch_size, n_parameters_per_input * n_features).
            identity_conditioner = torch.reshape(
                identity_conditioner,
                (self._transformer.n_parameters_per_input * n_mapped_features,)
            )

            # Setting to 0.0 only the last layer suffices.
            if weight_norm:
                self._conditioner.layers[-1].weight_g.data.fill_(0.0)
            else:
                self._conditioner.layers[-1].weight.data.fill_(0.0)
            self._conditioner.layers[-1].bias.data = identity_conditioner

    def n_parameters(self) -> int:
        """The total number of (unmasked) parameters."""
        return self._conditioner.n_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map the input data.

        Parameters
        ----------
        x : torch.Tensor
            An input batch of data of shape ``(batch_size, dimension_in)``.

        Returns
        -------
        y : torch.Tensor
            The mapped data of shape ``(batch_size, dimension_in)``.
        log_det_J : torch.Tensor
            The log absolute value of the Jacobian of the flow as a tensor of
            shape ``(batch_size,)``.

        """
        parameters = self._run_conditioner(x)

        # Make sure the conditioning dimensions are not altered.
        if self._mapped_mask is None:
            y, log_det_J = self._transformer(x, parameters)
        else:
            # There are conditioning dimensions.
            y = torch.empty_like(x)

            y[:, self._conditioning_mask] = x[:, self._conditioning_mask]
            y[:, self._mapped_mask], log_det_J = self._transformer(
                x[:, self._mapped_mask], parameters)

        return y, log_det_J

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # This is slower than forward because to evaluate x_i we need
        # all x_<i. For the algorithm, see Eq 39 in arXiv:1912.02762.


        # Initialize x to an arbitrary value.
        x = torch.zeros_like(y)

        if self._mapped_mask is not None:
            # All outputs of the nets depend on the conditioning features,
            # which are not transformed by the MAF.
            x[:, self._conditioning_mask] = y[:, self._conditioning_mask]

            # Isolate the features that are mapped.
            y = y[:, self._mapped_mask]

            # Isolate also the mapped features on the masks.
            mapped_degree_to_mapped_mask = self._mapped_degree_to_mask[:, self._mapped_mask]
        else:
            mapped_degree_to_mapped_mask = self._mapped_degree_to_mask

        # Compute the inverse.
        for degree, (mask, mask_mapped) in enumerate(zip(
                self._mapped_degree_to_mask, mapped_degree_to_mapped_mask)):
            # Compute the inversion with the current x.
            # Cloning, allows to compute gradients on inverse.
            parameters = self._run_conditioner(x.clone())

            # The log_det_J that we compute with the last pass is the total log_det_J.
            # x_temp has shape (batch_size, n_mapped_features).
            x_temp, log_det_J = self._transformer.inverse(y, parameters)

            # No need to update all the xs, but only those we can update at this point.
            x[:, mask] = x_temp[:, mask_mapped]

        return x, log_det_J

    def _run_conditioner(self, x):
        """Return the conditioning parameters with shape (batch_size, n_parameters, n_features)."""
        batch_size, n_features = x.shape

        # Lift periodic features.
        if self._lifter is not None:
            x = self._lifter(x)

        # The conditioner return the parameters with shape (batch_size, n_features*n_parameters).
        return self._conditioner(x).reshape(
            batch_size,
            self._transformer.n_parameters_per_input,
            -1
        )


# =============================================================================
# HELPER FUNCTIONS FOR PERIODIC DOFs
# =============================================================================

class _LiftPeriodic(torch.nn.Module):
    """Lift periodic DOFs into a periodic representation (cos, sin).

    Parameters
    ----------
    dimension_in : int
        Dimension of the input.
    periodic_indices : torch.Tensor[int]
        Shape (n_periodic,). The (ordered) indices of the input features that
        are periodic and must be lifted to the (cos, sin) representation.
    limits : torch.Tensor[float]
        A pair ``(lower, upper)`` defining the limits of the periodic variables.
        The period is given by ``upper - lower``.

    """

    def __init__(
            self,
            dimension_in : int,
            periodic_indices : torch.Tensor,
            limits : torch.Tensor,
    ):
        super().__init__()
        self.register_buffer('limits', limits)

        # Cache a set of periodic/nonperiodic indices BEFORE and AFTER the input has been lifted.
        periodic_indices_set = set(periodic_indices.tolist())
        self.register_buffer('_periodic_indices', periodic_indices)
        self.register_buffer('_nonperiodic_indices', torch.tensor([i for i in range(dimension_in)
                                                                   if i not in periodic_indices_set]))

        periodic_indices_lifted = []  # Shape (n_periodic,).
        nonperiodic_indices_lifted = []  # Shape (n_non_periodic,).

        shift_idx = 0
        for i in range(dimension_in):
            if i in periodic_indices_set:
                periodic_indices_lifted.append(i+shift_idx)
                shift_idx += 1
            else:
                nonperiodic_indices_lifted.append(i + shift_idx)

        # Cache as Tensor.
        self.register_buffer('_periodic_indices_lifted', torch.tensor(periodic_indices_lifted))
        self.register_buffer('_nonperiodic_indices_lifted', torch.tensor(nonperiodic_indices_lifted))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lift each periodic degree of freedom x into a periodic representation (cosx, sinx).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_features)``. Input tensor.

        Returns
        -------
        y : torch.Tensor
            Shape ``(batch_size, n_features + n_periodic)``. The input with the
            periodic DOFs transformed. The cosx, sinx representation is placed
            contiguously where the original DOF was. E.g., if ``2`` is the first
            element in ``periodic_indices``, then cos and sin will be placed at
            ``y[:, 2]`` and ``y[:, 3]`` respectively.
        """
        batch_size, n_features = x.shape

        # Transform periodic interval to [0, 2pi].
        period_scale = 2*torch.pi / (self.limits[1] - self.limits[0])
        x_periodic = (x[:, self._periodic_indices] - self.limits[0]) * period_scale
        cosx = torch.cos(x_periodic)
        sinx = torch.sin(x_periodic)

        # Fill output.
        n_periodic = len(self._periodic_indices)
        y = torch.empty((batch_size, n_features+n_periodic)).to(x)
        y[:, self._periodic_indices_lifted] = cosx
        y[:, self._periodic_indices_lifted+1] = sinx
        y[:, self._nonperiodic_indices_lifted] = x[:, self._nonperiodic_indices]

        return y
