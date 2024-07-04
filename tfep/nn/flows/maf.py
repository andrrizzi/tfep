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

from tfep.nn.conditioners.made import MADE, degrees_in_from_str
from tfep.nn.transformers.affine import AffineTransformer
from tfep.utils.misc import ensure_tensor_sequence


# =============================================================================
# MAF
# =============================================================================

class MAF(torch.nn.Module):
    """Masked Autoregressive Flow.

    This implements an autoregressive flow in which the :class:`tfep.nn.conditioners.MADE`
    network is used for the conditioner. The class supports arbitrary
    transformers.

    When the transformer is the :class:`AffineTransformer`, this is
    equivalent to MAF and IAF [1-2]. These two differ only in the direction
    of the conditional dependence, effectively determining which between
    forward and inverse evaluation is faster.

    References
    ----------
    [1] Kingma DP, Salimans T, Jozefowicz R, Chen X, Sutskever I, Welling M.
        Improved variational inference with inverse autoregressive flow.
        In Advances in neural information processing systems 2016 (pp. 4743-4751).
    [2] Papamakarios G, Pavlakou T, Murray I. Masked autoregressive flow for
        density estimation. In Advances in Neural Information Processing
        Systems 2017 (pp. 2338-2347).
    [3] Papamakarios G, Nalisnick E, Rezende DJ, Mohamed S, Lakshminarayanan B.
        Normalizing Flows for Probabilistic Modeling and Inference. arXiv
        preprint arXiv:1912.02762. 2019 Dec 5.

    """

    def __init__(
            self,
            dimension_in : int,
            dimensions_hidden: Union[int, Sequence[int]] = 2,
            conditioning_indices: Optional[Sequence[int]] = None,
            periodic_indices: Optional[Sequence[int]] = None,
            periodic_limits: Optional[Sequence[int]] = None,
            degrees_in: Union[str, Sequence[int]] = 'input',
            degrees_hidden_motif: Optional[Sequence[int]] = None,
            weight_norm: bool = True,
            blocks: Union[int, Sequence[int]] = 1,
            shorten_last_block: bool = False,
            transformer: Optional[torch.nn.Module] = None,
            initialize_identity: bool = True,
    ):
        """Constructor.

        Parameters
        ----------
        dimension_in : int
            The number of features of a single input vector. This includes
            the number of conditioning features.
        dimensions_hidden : int or Sequence[int], optional
            Control the number of layers and nodes of the hidden layers in
            the MADE networks that implements the conditioner. If an int,
            this is the number of hidden layers, and the number of nodes in
            each hidden layer will be set to ``(dimension_in - 1) * floor(out_per_dimension**(1/2))``
            where ``out_per_dimension`` is the number of output nodes for each
            input feature. If a list, ``dimensions_hidden[l]`` must be the number
            of nodes in the l-th hidden layer. This means that the number of
            trainable parameters will roughly depend quadratically on the conditioner
            input and linearly on the conditioner output.
        conditioning_indices : None or Sequence[int], optional
            The indices of the input features corresponding to the conditioning
            features. These features affect the output, but they are not mapped
            by the flow.
        periodic_indices : Sequence[int] or None, optional
            Shape (n_periodic,). The (ordered) indices of the input features that
            are periodic. When passed to the conditioner, these are transformed
            to ``(cos(a), sin(a))``, where ``a`` is a shifted/rescaled feature
            to be in the interval [0, 2pi]. This way the conditioner will have
            periodic input.
        periodic_limits : Sequence[int] or None, optional
            A pair ``(lower, upper)`` defining the limits of the periodic input
            features  (e.g. ``[-pi, pi]``). The period is ``upper - lower``.
        degrees_in : str or Sequence[int], optional
            The degrees to assign to the input/output nodes. Effectively this
            controls the dependencies between variables in the conditioner.
            If ``'input'``/``'reversed'``, the degrees are assigned in
            the same/reversed order they are passed. If an array, this must
            be a permutation of ``numpy.arange(0, n_blocks)``, where ``n_blocks``
            is the number of blocks passed to the constructor. If blocks are
            not used, this corresponds to the number of non-conditioning features
            (i.e., ``dimension_in - len(conditioning_indices)``). Default is
            ``'input'``.
        degrees_hidden_motif : Sequence[int], optional
            The degrees of the hidden nodes of the conditioner are assigned
            using this array in a round-robin fashion. If not given, they
            are assigned in the same order used for the input nodes. This
            must be at least as large as the dimension of the smallest hidden
            layer.
        weight_norm : bool, optional
            If True, weight normalization is applied to the masked linear
            modules.
        blocks : int or Sequence[int], optional
            If an integer, the non-conditioning input features are divided
            into contiguous blocks of size ``blocks`` that are assigned the
            same degree in the MADE conditioner. If a list, ``blocks[i]``
            must represent the size of the i-th block. The default, ``1``,
            correspond to a fully autoregressive network.
        shorten_last_block : bool, optional
            If ``blocks`` is an integer that is not a divisor of the number
            of non-conditioning  features, this option controls whether the
            last block is shortened (``True``) or an exception is raised
            (``False``). Default is ``False``.
        transformer : torch.nn.Module or None, optional
            The transformer used to map the input features. By default, the
            ``AffineTransformer`` is used.
        initialize_identity : bool, optional
            If ``True``, the parameters are initialized in such a way that
            the flow initially performs the identity function.

        """
        super().__init__()

        # Convert all sequences to Tensors to simplify the code.
        dimensions_hidden = ensure_tensor_sequence(dimensions_hidden)
        conditioning_indices = ensure_tensor_sequence(conditioning_indices)
        periodic_indices = ensure_tensor_sequence(periodic_indices)
        periodic_limits = ensure_tensor_sequence(periodic_limits)
        degrees_in = ensure_tensor_sequence(degrees_in)
        degrees_hidden_motif = ensure_tensor_sequence(degrees_hidden_motif)
        blocks = ensure_tensor_sequence(blocks)

        # We'll need this set later.
        if conditioning_indices is None:
            conditioning_indices_set = set()
        else:
            conditioning_indices_set = set(conditioning_indices.tolist())

        # Create the lifter used to map the periodic degrees of freedom.
        if periodic_indices is None:
            dimension_in_made = dimension_in
            conditioning_indices_made = conditioning_indices
            degrees_per_out = None
            self._lifter = None
        else:
            if periodic_limits is None:
                raise ValueError('periodic_limits must be given if periodic_indices is passed.')
            if not (isinstance(blocks, int) and blocks == 1):
                raise ValueError('periodic features are not supported with blocks != 1.')

            # MADE takes as input the output of self._lifter which doubles
            # the number of inputs for periodic features.
            dimension_in_made = dimension_in + len(periodic_indices)

            # Initialize Module lifting the periodic features.
            self._lifter = _LiftPeriodic(
                dimension_in=dimension_in,
                periodic_indices=periodic_indices,
                limits=periodic_limits
            )

            # The periodic features are transformed to (cosa, sina) and need
            # blocks of dimensions 2 in MADE (except for the conditioning DOFs
            # which are always assigned the same degree and do not enter blocks).
            # In practice, this assigns them the same autoregressive degree.
            # TODO: Map cosa and sina using different degrees?
            periodic_indices_set = set(periodic_indices.tolist())
            blocks = [2 if i in periodic_indices_set else 1
                      for i in range(dimension_in) if i not in conditioning_indices_set]

            # If conditioning indices are greater than periodic indices, these
            # must be shifted since extra features are inserted in the input.
            conditioning_indices_made = []
            if conditioning_indices is not None:
                # We traverse the list in reverse in case a conditioning indices
                # is also a periodic index and we need to insert a new one.
                for i, cond_idx in enumerate(conditioning_indices.tolist()):
                    new_cond_idx = cond_idx + torch.sum(periodic_indices < cond_idx).tolist()
                    conditioning_indices_made.append(new_cond_idx)
                    if cond_idx in periodic_indices_set:
                        conditioning_indices_made.append(new_cond_idx+1)

            # The conditioner takes as input dimension_in+n_periodic features
            # but it must output dimension_in-n_conditioning features.
            degrees_per_out = degrees_in_from_str(degrees_in, len(blocks))

        # By default, use an affine transformer.
        if transformer is None:
            transformer = AffineTransformer()
        self._transformer = transformer

        # We need two MADE layers for the scaling and the shifting.
        self._conditioner = MADE(
            dimension_in=dimension_in_made,
            dimensions_hidden=dimensions_hidden,
            out_per_dimension=self._transformer.n_parameters_per_input,
            conditioning_indices=conditioning_indices_made,
            degrees_in=degrees_in,
            degrees_hidden_motif=degrees_hidden_motif,
            degrees_per_out=degrees_per_out,
            weight_norm=weight_norm,
            blocks=blocks,
            shorten_last_block=shorten_last_block,
        )

        # We cache the indices of the conditioning/mapped degrees of freedom
        # which is necessary to forward() and inverse(). The conditioning are
        # stored only if there are periodic DOFs as they can be read directly
        # from the conditioner otherwise.
        if conditioning_indices is None or len(conditioning_indices) == 0:
            n_conditioning_dofs = 0
            mapped_indices = None
            conditioning_indices = None
        else:
            n_conditioning_dofs = len(conditioning_indices)
            mapped_indices = torch.tensor([i for i in range(dimension_in)
                                           if i not in conditioning_indices_set])
            conditioning_indices = conditioning_indices
        self.register_buffer('_mapped_indices', mapped_indices)
        self.register_buffer('_conditioning_indices', conditioning_indices)

        # Initialize the log_scale and shift nets to 0.0 so that at
        # the beginning the MAF layer performs the identity function.
        if initialize_identity:
            dimension_out = dimension_in - n_conditioning_dofs

            # Determine the conditioner that will make the transformer the identity function.
            identity_conditioner = self._transformer.get_identity_parameters(dimension_out)

            # There is a single output bias parameter vector so we need to
            # convert from shape (batch_size, n_parameters_per_input, n_features)
            # to (batch_size, n_parameters_per_input * n_features).
            identity_conditioner = torch.reshape(
                identity_conditioner,
                (self._transformer.n_parameters_per_input * dimension_out,)
            )

            # Setting to 0.0 only the last layer suffices.
            if weight_norm:
                self._conditioner.layers[-1].weight_g.data.fill_(0.0)
            else:
                self._conditioner.layers[-1].weight.data.fill_(0.0)
            self._conditioner.layers[-1].bias.data = identity_conditioner

    @property
    def conditioning_indices(self) -> torch.Tensor:
        """The indices of the conditioning degrees of freedom."""
        # This is stored only if there are periodic conditioning indices.
        if self._conditioning_indices is None:
            return self._conditioner.conditioning_indices
        return self._conditioning_indices

    @property
    def degrees_in(self) -> torch.Tensor:
        """``degrees_in[i]`` is the degree assigned to the i-th input feature."""
        degrees_in = self._conditioner.degrees_in
        # The degree_in of periodic features is duplicated.
        if self._lifter is not None:
            mask = torch.full(degrees_in.shape, fill_value=True)
            mask[self._lifter._periodic_indices_lifted] = False
            degrees_in = degrees_in[mask]
        return degrees_in

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
        if self._mapped_indices is None:
            y, log_det_J = self._transformer(x, parameters)
        else:
            # There are conditioning dimensions.
            y = torch.empty_like(x)

            y[:, self.conditioning_indices] = x[:, self.conditioning_indices]
            y[:, self._mapped_indices], log_det_J = self._transformer(
                x[:, self._mapped_indices], parameters)

        return y, log_det_J

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # This is slower than forward because to evaluate x_i we need
        # all x_<i. For the algorithm, see Eq 39 in reference [3] above.
        degrees_in = self.degrees_in

        # Initialize x to an arbitrary value.
        x = torch.zeros_like(y)

        if self._mapped_indices is not None:
            # All outputs of the nets depend on the conditioning features,
            # which are not transformed by the MAF.
            x[:, self.conditioning_indices] = y[:, self.conditioning_indices]

            # Isolate the features that are mapped.
            y = y[:, self._mapped_indices]
            degrees_in = degrees_in[self._mapped_indices]

        # We need to process each block in the order given by their degree to
        # respect the dependencies.
        blocks = self._conditioner.blocks

        # With periodic indices, only block 1 is supported. blocks might be
        # different from 1 only because the periodic DOFs is expanded into 2
        # different inputs with the same degree.
        if self._lifter is not None:
            blocks = torch.ones(len(blocks)).to(blocks)

        # Determine the start index of each block.
        blocks_start_indices = torch.empty(len(blocks)).to(blocks)
        blocks_start_indices[0] = 0.
        blocks_start_indices[1:] = torch.cumsum(blocks[:-1], dim=0)

        # Order the block by their degree.
        blocks_order = torch.argsort(degrees_in[blocks_start_indices])

        # Now compute the inverse.
        for block_idx in blocks_order:
            # Compute the inversion with the current x.
            # Cloning, allows to compute gradients on inverse.
            parameters = self._run_conditioner(x.clone())

            # The log_det_J that we compute with the last pass is the total log_det_J.
            x_temp, log_det_J = self._transformer.inverse(y, parameters)

            # Determine the indices of x_temp that correspond to the computed block.
            block_size = blocks[block_idx]
            block_start_idx = blocks_start_indices[block_idx]
            block_end_idx = block_start_idx + block_size

            # Determine the indices of the computed DOFs in the output
            # tensor (which also include the conditioning DOFs).
            if self._mapped_indices is None:
                input_indices = slice(block_start_idx, block_end_idx)
            else:
                input_indices = self._mapped_indices[block_start_idx:block_end_idx]

            # No need to update all the xs, but only those we can update at this point.
            x[:, input_indices] = x_temp[:, block_start_idx:block_end_idx]

        return x, log_det_J

    def _run_conditioner(self, x):
        """Return the conditioning parameters with shape (batch_size, n_parameters, n_features)."""
        batch_size, n_features = x.shape

        # Check how many of the input features are conditioning and are not mapped.
        if self._mapped_indices is None:
            n_mapped_features = n_features
        else:
            n_mapped_features = len(self._mapped_indices)

        # Lift periodic features.
        if self._lifter is not None:
            x = self._lifter(x)

        # The conditioner return the parameters with shape (batch_size, n_features*n_parameters).
        return self._conditioner(x).reshape(
            batch_size,
            self._transformer.n_parameters_per_input,
            n_mapped_features
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
