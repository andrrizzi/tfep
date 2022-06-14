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

import numpy as np
import torch

from tfep.nn.conditioners.made import MADE, degrees_in_from_str
from tfep.nn.transformers.affine import AffineTransformer


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

    Parameters
    ----------
    dimension_in : int
        The number of features of a single input vector. This includes
        the number of conditioning features.
    dimensions_hidden : int or List[int], optional
        Control the number of layers and nodes of the hidden layers in
        the MADE networks that implements the conditioner. If an int,
        this is the number of hidden layers, and the number of nodes in
        each hidden layer will be set to ``(dimension_in - 1) * out_per_dimension``
        where ``out_per_dimension`` is the number of output nodes for each
        input feature. If a list, ``dimensions_hidden[l]`` must be the number
        of nodes in the l-th hidden layer.
    conditioning_indices : List[int], optional
        The indices of the input features corresponding to the conditioning
        features. These features affect the output, but they are not mapped
        by the flow.
    periodic_indices : List, optional
        Shape (n_periodic,). The (ordered) indices of the input features that
        are periodic. When passed to the conditioner, these are transformed to
        ``(cos(a), sin(a))``, where ``a`` is a shifted/rescaled feature to be in
        the interval [0, 2pi]. This way the conditioner will have periodic input.
    periodic_limits : List, optional
        A pair ``(lower, upper)`` defining the limits of the periodic input
        features  (e.g. ``[-pi, pi]``). The period is ``upper - lower``.
    degrees_in : str or numpy.ndarray, optional
        The degrees to assign to the input/output nodes. Effectively this
        controls the dependencies between variables in the conditioner.
        If ``'input'``/``'reversed'``, the degrees are assigned in
        the same/reversed order they are passed. If an array, this must
        be a permutation of ``numpy.arange(0, n_blocks)``, where ``n_blocks``
        is the number of blocks passed to the constructor. If blocks are
        not used, this corresponds to the number of non-conditioning features
        (i.e., ``dimension_in - len(conditioning_indices)``). Default is ``'input'``.
    degrees_hidden_motif : numpy.ndarray, optional
        The degrees of the hidden nodes of the conditioner are assigned
        using this array in a round-robin fashion. If not given, they
        are assigned in the same order used for the input nodes. This
        must be at least as large as the dimension of the smallest hidden
        layer.
    weight_norm : bool, optional
        If True, weight normalization is applied to the masked linear
        modules.
    blocks : int or List[int], optional
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
    split_conditioner : bool, optional
        If ``True``, separate MADE networks are used to compute separately
        each parameter of the transformer (e.g., for affine transformers
        which require scale and shift parameters, two networks are used).
        Otherwise, a single network is used to implement the conditioner,
        and all parameters are generated in a single pass.
    transformer : torch.nn.Module
        The transformer used to map the input features. By default, the
        ``AffineTransformer`` is used.
    initialize_identity : bool, optional
        If ``True``, the parameters are initialized in such a way that
        the flow initially performs the identity function.

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
            dimension_in,
            dimensions_hidden=2,
            conditioning_indices=None,
            periodic_indices=None,
            periodic_limits=None,
            degrees_in='input',
            degrees_hidden_motif=None,
            weight_norm=True,
            blocks=1,
            shorten_last_block=False,
            split_conditioner=True,
            transformer=None,
            initialize_identity=True
    ):
        super().__init__()

        # We'll need this set later.
        if conditioning_indices is None:
            conditioning_indices_set = set()
        else:
            conditioning_indices_set = set(conditioning_indices)

        # Create the lifter used to map the periodic degrees of freedom.
        if periodic_indices is None:
            dimension_in_made = dimension_in
            conditioning_indices_made = conditioning_indices
            degrees_per_out = None
            self._lifter = None
        else:
            if periodic_limits is None:
                raise ValueError('periodic_limits must be given if periodic_indices is passed.')
            if blocks != 1:
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
            # blocks of dimensions 2 in MADE (except for the conditioning DOFs).
            # In practice, this assigns them the same autoregressive degree.
            # TODO: Map cosa and sina using different degrees?
            periodic_indices_set = set(periodic_indices)
            blocks = [2 if i in periodic_indices_set else 1
                      for i in range(dimension_in) if i not in conditioning_indices_set]

            # If conditioning indices are greater than periodic indices, these
            # must be shifted since extra features are inserted in the input.
            conditioning_indices_made = []
            periodic_indices_tensor = torch.tensor(periodic_indices)
            if conditioning_indices is not None:
                # We traverse the list in reverse in case a conditioning indices
                # is also a periodic index and we need to insert a new one.
                for i, cond_idx in enumerate(conditioning_indices):
                    new_cond_idx = cond_idx + torch.sum(periodic_indices_tensor < cond_idx).tolist()
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

        if split_conditioner:
            n_conditioners = self._transformer.n_parameters_per_input
            out_per_dimension = 1
        else:
            n_conditioners = 1
            out_per_dimension = self._transformer.n_parameters_per_input

        # We need two MADE layers for the scaling and the shifting.
        self._conditioners = torch.nn.ModuleList()
        for i in range(n_conditioners):
            self._conditioners.append(MADE(
                dimension_in=dimension_in_made,
                dimensions_hidden=dimensions_hidden,
                out_per_dimension=out_per_dimension,
                conditioning_indices=conditioning_indices_made,
                degrees_in=degrees_in,
                degrees_hidden_motif=degrees_hidden_motif,
                degrees_per_out=degrees_per_out,
                weight_norm=weight_norm,
                blocks=blocks,
                shorten_last_block=shorten_last_block,
            ))

        # We cache the indices of the conditioning/mapped degrees of freedom
        # which is necessary to forward() and inverse(). The conditioning are
        # stored only if there are periodic DOFs as they can be read directly
        # from the conditioners otherwise.
        if conditioning_indices is None or len(conditioning_indices) == 0:
            n_conditioning_dofs = 0
            self._mapped_indices = None
            self._conditioning_indices = None
        else:
            n_conditioning_dofs = len(conditioning_indices)
            self._mapped_indices = [i for i in range(dimension_in) if i not in conditioning_indices_set]
            self._conditioning_indices = conditioning_indices

        # Initialize the log_scale and shift nets to 0.0 so that at
        # the beginning the MAF layer performs the identity function.
        if initialize_identity:
            dimension_out = dimension_in - n_conditioning_dofs

            # Determine the conditioner that will make the transformer the identity function.
            identity_conditioner = self._transformer.get_identity_parameters(dimension_out)

            # If we have not split the conditioners over multiple networks,
            # there is a single output bias parameter vector so we need to
            # convert from shape (batch_size, n_parameters_per_input, n_features)
            # to (batch_size, n_parameters_per_input * n_features).
            if not split_conditioner:
                identity_conditioner = torch.reshape(
                    identity_conditioner,
                    (self._transformer.n_parameters_per_input * dimension_out,)
                )
                identity_conditioner = [identity_conditioner]

            for net, id_cond in zip(self._conditioners, identity_conditioner):
                # Setting to 0.0 only the last layer suffices.
                if weight_norm:
                    net.layers[-1].weight_g.data.fill_(0.0)
                else:
                    net.layers[-1].weight.data.fill_(0.0)
                net.layers[-1].bias.data = id_cond

    @property
    def conditioning_indices(self):
        """List[int]: The indices of the conditioning degrees of freedom."""
        # This is stored only if there are periodic conditioning indices.
        if self._conditioning_indices is None:
            return self._conditioners[0].conditioning_indices
        return self._conditioning_indices

    @property
    def degrees_in(self):
        """numpy.ndarray: ``degrees_in[i]`` is the degree assigned to the i-th input feature."""
        degrees_in = self._conditioners[0].degrees_in
        # The degree_in of periodic features is duplicated.
        if self._lifter is not None:
            degrees_in = np.delete(degrees_in, self._lifter._periodic_indices_lifted)
        return degrees_in

    def n_parameters(self):
        """int: The total number of (unmasked) parameters."""
        return sum(c.n_parameters() for c in self._conditioners)

    def forward(self, x):
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
        parameters = self._run_conditioners(x)

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

    def inverse(self, y):
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
        if self._lifter is None:
            blocks = self._conditioners[0].blocks
        else:
            # With periodic indices, only block 1 is supported. blocks might be
            # different from 1 only because the periodic DOFs is expanded into
            # 2 different inputs with the same degree.
            blocks = [1 for _ in self._conditioners[0].blocks]

        block_start_idx = 0
        blocks_start_indices = []
        blocks_degrees = []
        for block_size in blocks:
            blocks_start_indices.append(block_start_idx)
            blocks_degrees.append(degrees_in[block_start_idx])
            block_start_idx += block_size

        # Order the block by their degree.
        blocks_order = np.argsort(blocks_degrees)

        # Now compute the inverse.
        for block_idx in blocks_order:
            # Compute the inversion with the current x.
            # Cloning, allows to compute gradients on inverse.
            parameters = self._run_conditioners(x.clone())

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

            block_start_idx += block_size

        return x, log_det_J

    def _run_conditioners(self, x):
        """Return the conditioning parameters with shape (batch_size, n_parameters, n_features)."""
        batch_size, n_features = x.shape
        n_conditioners = len(self._conditioners)

        # Check how many of the input features are conditioning and are not mapped.
        if self._mapped_indices is None:
            n_mapped_features = n_features
        else:
            n_mapped_features = len(self._mapped_indices)

        # The shape of the returned array.
        returned_shape = (
            batch_size,
            self._transformer.n_parameters_per_input,
            n_mapped_features
        )

        # Lift periodic features.
        if self._lifter is not None:
            x = self._lifter(x)

        if n_conditioners == 1:
            # A single conditioner for all parameters. The conditioners
            # return the parameters with shape (batch_size, n_features*n_parameters).
            conditioning_parameters = self._conditioners[0](x)
            conditioning_parameters = torch.reshape(
                conditioning_parameters, shape=returned_shape)
        else:
            # The conditioners are split into independent NNs.
            # conditioning_parameters has shape (batch_size, n_features*n_parameters_per_input).
            conditioning_parameters = torch.empty(
                size=returned_shape, dtype=x.dtype)
            for conditioner_idx, conditioner in enumerate(self._conditioners):
                conditioning_parameters[:, conditioner_idx] = conditioner(x)

        return conditioning_parameters


# =============================================================================
# HELPER FUNCTIONS FOR PERIODIC DOFs
# =============================================================================

class _LiftPeriodic(torch.nn.Module):
    """Lift periodic DOFs into a periodic representation (cos, sin).

    Parameters
    ----------
    periodic_indices : List
        Shape (n_periodic,). The (ordered) indices of the input features that
        are periodic and must be lifted to the (cos, sin) representation.
    limits : List
        A pair ``(lower, upper)`` defining the limits of the periodic variables.
        The period is given by ``upper - lower``.

    """

    def __init__(self, dimension_in, periodic_indices, limits):
        super().__init__()
        self.limits = limits


        # Cache a set of periodic/nonperiodic indices BEFORE and AFTER the input has been lifted.
        periodic_indices_set = set(periodic_indices)
        self._periodic_indices = torch.tensor(periodic_indices)
        self._nonperiodic_indices = torch.tensor([i for i in range(dimension_in) if i not in periodic_indices_set])

        self._periodic_indices_lifted = []  # Shape (n_periodic,).
        self._nonperiodic_indices_lifted = []  # Shape (n_non_periodic,).

        shift_idx = 0
        for i in range(dimension_in):
            if i in periodic_indices_set:
                self._periodic_indices_lifted.append(i+shift_idx)
                shift_idx += 1
            else:
                self._nonperiodic_indices_lifted.append(i + shift_idx)

        # Cache as Tensor.
        self._periodic_indices_lifted = torch.tensor(self._periodic_indices_lifted)
        self._nonperiodic_indices_lifted = torch.tensor(self._nonperiodic_indices_lifted)

    def forward(self, x):
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
        y = torch.empty((batch_size, n_features+n_periodic), dtype=x.dtype)
        y[:, self._periodic_indices_lifted] = cosx
        y[:, self._periodic_indices_lifted+1] = sinx
        y[:, self._nonperiodic_indices_lifted] = x[:, self._nonperiodic_indices]

        return y
