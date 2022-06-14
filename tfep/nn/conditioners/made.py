#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Masked Autoregressive layer for Density Estimation (MADE) module for PyTorch.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import torch

from tfep.nn import masked
from tfep.nn.utils import generate_block_sizes


# =============================================================================
# LAYERS
# =============================================================================

def degrees_in_from_str(degrees_in, n_blocks):
    """Helper function to create an array degrees_in from string representation.

    Parameters
    ----------
    degrees_in : str or numpy.ndarray
        Either ``'input'`` or ``'reversed'``, which assigns a increasing/decreasing
        degree to the input nodes. If an array, this is returned as it is.
    n_blocks : int
        The number of blocks in the input. ``degrees_in`` is created to assign
        all degrees from 0 to ``n_blocks-1``.

    Returns
    -------
    degrees_in : numpy.ndarray
        Return the input degrees for each block.

    """
    if isinstance(degrees_in, str):
        if degrees_in == 'input':
            degrees_in = list(range(n_blocks))
        elif degrees_in == 'reversed':
            degrees_in = list(range(n_blocks-1, -1, -1))
        else:
            raise ValueError("Accepted string values for 'degrees_in' "
                             "are 'input' and 'reversed'.")
    return degrees_in


class MADE(torch.nn.Module):
    """
    An autoregressive layer implemented through masked affine layers.

    The current implementation supports arbitrary dependencies between
    input features, while the degrees of the hidden layers are assigned
    in a round-robin fashion until the number of requested nodes in that
    layer has been generated. Each layer is a :class:`MaskedLinear`,
    which, in hidden layers, is followed an ``ELU`` nonlinearity. Inputs
    and outputs have the same dimension.

    The module supports a number of "conditioning features" which affect
    the output but are not modified. This can be used, for example, to
    implement the coupling layers flow. Currently, only the initial
    features of the input vector can be used as conditioning features.

    It is possible to divide the input into contiguous "blocks" which
    are assigned the same degree. In practice, this enables to implement
    architectures in between coupling layers and a fully autoregressive
    flow.

    The original paper used this as a Masked Autoregressive network for
    Distribution Estimation (MADE) [1], while the Inverse/Masked Autoregressive
    Flow (IAF/MAF) [2]/[3] used this as a layer to stack to create normalizing
    flows with the autoregressive property.

    An advantage of using masks over the naive implementation of an
    autoregressive layer, which use a different neural network for each
    parameter of the affine transformation, is that it generates all the
    affine parameters in a single pass, with much less parameters to train,
    and can be parallelized trivially.

    Parameters
    ----------
    dimension_in : int
        The number of features of a single input vector. This includes
        the number of conditioning features.
    dimensions_hidden : int or List[int], optional
        If an int, this is the number of hidden layers, and the number
        of nodes in each hidden layer will be set to
        ``(dimension_in - len(last_block)) * out_per_dimension`` where
        ``len(last_block)`` is the dimension of the block with the highest
        assigned degree (which depends on the values of ``blocks`` and
        ``shorten_last_block``).

        If a list, ``dimensions_hidden[l]`` must be the number of nodes in
        the l-th hidden layer.
    out_per_dimension : int, optional
        The dimension of the output layer in terms of a multiple of the
        number of non-conditioning input features. Since MADE is typically
        used for the conditioner, this usually correspond to the number of
        parameters per degree of freedom used in the transformer. Default
        is 1.
    conditioning_indices : List[int], optional
        The indices of the input features corresponding to the conditioning
        features. These features affect the output, but they are not mapped
        by the flow.
    degrees_in : str or numpy.ndarray, optional
        The degrees to assign to the input/output nodes. Effectively this
        controls the dependencies between variables in the autoregressive
        model. If ``'input'``/``'reversed'``, the degrees are assigned in
        the same/reversed order they are passed. If an array, this must
        be a permutation of ``numpy.arange(0, n_blocks)``, where ``n_blocks``
        is the number of blocks passed to the constructor. If blocks are
        not used, this corresponds to the number of non-conditioning features
        (i.e., ``dimension_in - len(conditioning_indices)``).
    degrees_hidden_motif : numpy.ndarray, optional
        The degrees of the hidden nodes are assigned using this array in
        a round-robin fashion. If not given, they are assigned in the
        same order used for the input nodes. This must be at least as
        large as the dimension of the smallest hidden layer.
    degrees_per_out : numpy.ndarray, optional
        The degrees of the output nodes for each ``out_per_dimension``. Note
        that this can be different from ``degrees_in``. For example, if
        ``degrees_per_out=[1, 2, 0]`` and ``out_per_dimension=2`` will result
        in an output of dimension 6 using a scrambled order of degrees even if
        ``degrees_in=[0, 0, 1, 2]``. By default this is set equal to
        ``self.degrees_in``, which automatically accounts for block sizes.
    weight_norm : bool, optional
        If True, weight normalization is applied to the masked linear
        modules.
    blocks : int or List[int], optional
        If an integer, the non-conditioning input features are divided
        into contiguous blocks of size ``blocks`` that are assigned the
        same degree. If a list, ``blocks[i]`` must represent the size
        of the i-th block. The default, ``1``, correspond to a fully
        autoregressive network.
    shorten_last_block : bool, optional
        If ``blocks`` is an integer that is not a divisor of the number
        of non-conditioning  features, this option controls whether the
        last block is shortened (``True``) or an exception is raised (``False``).
        Default is ``False``.

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

    """

    def __init__(
        self,
        dimension_in,
        dimensions_hidden=1,
        out_per_dimension=1,
        conditioning_indices=None,
        degrees_in='input',
        degrees_hidden_motif=None,
        degrees_per_out=None,
        weight_norm=False,
        blocks=1,
        shorten_last_block=False
    ):
        super().__init__()
        # Mutable defaults.
        if conditioning_indices is None:
            conditioning_indices =[]

        self._out_per_dimension = out_per_dimension
        self._conditioning_indices = conditioning_indices

        # Get the number of dimensions in list format.
        n_hidden_layers, dimensions_hidden, expanded_blocks = self._get_dimensions(
            dimension_in, dimensions_hidden, out_per_dimension, conditioning_indices,
            degrees_in, blocks, shorten_last_block)

        # Store the verified/expanded blocks size.
        self._blocks = expanded_blocks

        # Determine the degrees of all input nodes.
        self._degrees_in = self._assign_degrees_in(dimension_in, conditioning_indices, degrees_in)

        # Generate the degrees to assign to the hidden nodes in a round-robin fashion.
        self._degrees_hidden_motif = self._generate_degrees_hidden_motif(degrees_hidden_motif)

        # Find the mapped indices.
        conditioning_indices_set = set(conditioning_indices)
        mapped_indices = [i for i in range(dimension_in) if i not in conditioning_indices_set]

        # Verify output layer configuration.
        if degrees_per_out is None:
            degrees_per_out = self.degrees_in[mapped_indices]
        else:
            self._check_degrees(degrees_per_out, check_len=False, name='degrees_per_out')

        # Create a sequence of MaskedLinear + nonlinearity layers.
        layers = []
        degrees_previous = self.degrees_in
        for layer_idx in range(n_hidden_layers+1):
            is_output_layer = layer_idx == n_hidden_layers

            # Determine the degrees of the layer's nodes.
            if is_output_layer:
                # The output layer doesn't have the conditioning features.
                degrees_current = np.tile(degrees_per_out, out_per_dimension)
            else:
                n_round_robin, n_remaining = divmod(dimensions_hidden[layer_idx], len(self.degrees_hidden_motif))
                if n_round_robin == 0:
                    err_msg = (f'Hidden layer {layer_idx} is too small '
                               f'to fit the motif {self.degrees_hidden_motif}.'
                               ' Increase the size of the layer or explicitly '
                               'pass a different motif for its degrees.')
                    raise ValueError(err_msg)

                degrees_current = np.tile(self.degrees_hidden_motif, n_round_robin)
                if n_remaining != 0:
                    degrees_current = np.concatenate([degrees_current, self.degrees_hidden_motif[:n_remaining]])

            # We transpose the mask from shape (in, out) to (out, in) because
            # the mask must have the same shape of the weights in MaskedLinear.
            mask = masked.create_autoregressive_mask(degrees_previous, degrees_current,
                                                     strictly_less=is_output_layer, transpose=True)

            # Add the linear layer with or without weight normalization.
            masked_linear = masked.MaskedLinear(
                in_features=len(degrees_previous),
                out_features=len(degrees_current),
                bias=True, mask=mask
            )
            if weight_norm:
                masked_linear = masked.masked_weight_norm(masked_linear, name='weight')

            layers.extend([masked_linear, torch.nn.ELU()])

            # Update for next iteration.
            degrees_previous = degrees_current

        # Remove the nonlinearity from the output layer.
        layers.pop()

        # Create a forwardable module from the sequence of modules.
        self.layers = torch.nn.Sequential(*layers)

    @property
    def dimension_in(self):
        "int: Dimension of the input vector."
        return self.layers[0].in_features

    @property
    def dimension_out(self):
        "int: Dimension of the output vector (excluding the conditioning dimensions)."
        return self.layers[-1].out_features

    @property
    def n_layers(self):
        "int: Total number of layers (hidden layers + output)."
        # Each layer is a masked linear + activation function, except the output layer.
        return (len(self.layers) + 1) // 2

    @property
    def dimensions_hidden(self):
        """List[int]: The number of nodes in each hidden layer."""
        return [l.out_features for l in self.layers[:-1:2]]

    @property
    def out_per_dimension(self):
        """int: Multiple of non-conditioning features determining the output dimension."""
        return self._out_per_dimension

    @property
    def dimension_conditioning(self):
        """int: Number of conditioning features."""
        return len(self._conditioning_indices)

    @property
    def conditioning_indices(self):
        return self._conditioning_indices

    @property
    def degrees_in(self):
        """numpy.ndarray: The degrees assigned to each input node."""
        return self._degrees_in

    @property
    def degrees_hidden_motif(self):
        """numpy.ndarray: The degrees assigned to the hidden nodes in a round-robin fashion."""
        return self._degrees_hidden_motif

    @property
    def blocks(self):
        """List[int]: The sizes of each blocks or ``None`` otherwise."""
        return self._blocks

    def n_parameters(self):
        """int: The total number of (unmasked) parameters."""
        return sum(l.n_parameters() for l in self.layers[::2])

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def _get_dimensions(
            dimension_in,
            dimensions_hidden,
            out_per_dimension,
            conditioning_indices,
            degrees_in,
            blocks,
            shorten_last_block
    ):
        """Process the input arguments and return the dimensions of all layers in list format.

        By default, when only the depth of the hidden layers is specified,
        the number of nodes in each hidden layer is assigned so that there
        is one node for each output connected to the network (i.e., ignoring
        the last output/block).

        Returns
        -------
        n_hidden_layers : int
            The number of hidden layers.
        dimensions_hidden : List[int]
            ``dimensions_hidden[i]`` is the dimension of the i-th hidden layer.
        expanded_blocks : List[int]
            ``expanded_blocks[i]`` is the size of the i-th block of features.

        """
        n_conditioning_dofs = len(conditioning_indices)

        # Validate/expand block sizes.
        expanded_blocks = generate_block_sizes(dimension_in-n_conditioning_dofs, blocks,
                                               shorten_last_block=shorten_last_block)

        # Find the dimension of the last block.
        if blocks == 1:
            len_last_block = 1
        elif isinstance(degrees_in, str):
            if degrees_in == 'input':
                len_last_block = expanded_blocks[-1]
            elif degrees_in == 'reversed':
                len_last_block = expanded_blocks[0]
        else:
            # Search for the block assigned to the last degree.
            last_block_idx = np.argmax(degrees_in)
            len_last_block = expanded_blocks[last_block_idx]

        if isinstance(dimensions_hidden, int):
            dimensions_hidden = [(dimension_in - len_last_block) * out_per_dimension
                                 for _ in range(dimensions_hidden)]

        return len(dimensions_hidden), dimensions_hidden, expanded_blocks

    def _check_degrees(self, degrees, check_len, name):
        """Check that degrees_in/degrees_per_out passed as arrays are configured correctly."""
        n_blocks = len(self.blocks)

        err_msg = (" When 'blocks' is not explicitly passed. The number of "
                   "blocks corresponds to the number of non-conditioning "
                   "dimensions (dimension_in - len(conditioning_indices))")
        if len(degrees) != n_blocks:
            raise ValueError('len('+name+') must be equal to the number '
                             'of blocks.' + err_msg)
        if set(degrees) != set(range(n_blocks)):
            raise ValueError(name + ' must contain all degrees between '
                             '0 and the number of blocks minus 1.' + err_msg)


    def _assign_degrees_in(self, dimension_in, conditioning_indices, degrees_in):
        """Assign the degrees of all input nodes.

        The self._blocks variable must be assigned before calling this function.

        Returns
        -------
        assigned_degrees_in : numpy.array
            degrees_in[i] is the degree assigned to the i-th input node.

        """
        # Shortcut for the number of (non-conditioning) blocks.
        n_blocks = len(self.blocks)

        # Eventually convert "input/reversed" to an array of degrees.
        degrees_in = degrees_in_from_str(degrees_in, n_blocks)

        # Verify that the parameters are consistent.
        self._check_degrees(degrees_in, check_len=True, name='degrees_in')

        # Initialize the return value.
        assigned_degrees_in = np.empty(dimension_in)

        # The conditioning features are always assigned the lowest
        # degree regardless of the value of degrees_in so that
        # all output features depends on them.
        assigned_degrees_in[conditioning_indices] = -1

        # Now assign the degrees to each input node.
        conditioning_indices_set = set(conditioning_indices)
        dof_idx_pointer = 0
        for block_idx, block_size in enumerate(self.blocks):
            for block_dof_idx in range(block_size):
                # Skip to the next non-conditioning block.
                while dof_idx_pointer in conditioning_indices_set:
                    dof_idx_pointer += 1

                # Add the degree of this block to this DOF.
                assigned_degrees_in[dof_idx_pointer] = degrees_in[block_idx]
                dof_idx_pointer += 1

        return assigned_degrees_in

    def _generate_degrees_hidden_motif(self, degrees_hidden_motif):
        """Generate the degrees to assign to the hidden nodes in a round-robin fashion.

        The self.blocks and self.degrees_in attributes must be initialized.
        """
        if degrees_hidden_motif is None:
            # There is no need to add the degree of the last block since
            # it won't be connected to the output layer anyway.
            last_degree = len(self.blocks)-1
            indices_last_block = np.argwhere(self.degrees_in == last_degree).flatten()
            degrees_hidden_motif = np.delete(self.degrees_in, indices_last_block)
        elif np.any(degrees_hidden_motif >= len(self.blocks)-1):
            raise ValueError('degrees_hidden_motif cannot contain degrees '
                             'greater than the number of blocks minus 1 '
                             'as they would be ignored by the output layer.')
        return degrees_hidden_motif
