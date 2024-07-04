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

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import numpy as np
import torch

from tfep.nn import masked
from tfep.utils.misc import ensure_tensor_sequence


# =============================================================================
# LAYERS
# =============================================================================

def degrees_in_from_str(degrees_in: Union[str, Sequence], n_blocks: int) -> torch.Tensor:
    """Helper function to create an array degrees_in from string representation.

    Parameters
    ----------
    degrees_in : str or Sequence
        Either ``'input'`` or ``'reversed'``, which assigns a increasing/decreasing
        degree to the input nodes. If a ``Sequence``, this is returned as a
        ``torch.Tensor``.
    n_blocks : int
        The number of blocks in the input. ``degrees_in`` is created to assign
        all degrees from 0 to ``n_blocks-1``.

    Returns
    -------
    degrees_in : torch.Tensor
        Return the input degrees for each block.

    """
    if isinstance(degrees_in, str):
        if degrees_in == 'input':
            degrees_in = torch.tensor(range(n_blocks))
        elif degrees_in == 'reversed':
            degrees_in = torch.tensor(range(n_blocks-1, -1, -1))
        else:
            raise ValueError("Accepted string values for 'degrees_in' "
                             "are 'input' and 'reversed'.")
    else:
        degrees_in = torch.as_tensor(degrees_in)
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
        dimension_in: int,
        dimensions_hidden: Union[int, Sequence[int]] = 1,
        out_per_dimension: int = 1,
        conditioning_indices: Optional[Sequence[int]] = None,
        degrees_in: Union[str, Sequence[int]] = 'input',
        degrees_hidden_motif: Optional[Sequence[int]] = None,
        degrees_per_out: Optional[Sequence[int]] = None,
        weight_norm: bool = False,
        blocks: Union[int, Sequence[int]] = 1,
        shorten_last_block: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        dimension_in : int
            The number of features of a single input vector. This includes
            the number of conditioning features.
        dimensions_hidden : int or Sequence[int], optional
            If an int, this is the number of hidden layers, and the number
            of nodes in each hidden layer will be set to
            ``(dimension_in - len(last_block)) * floor(out_per_dimension**(1/2))``
            where ``len(last_block)`` is the dimension of the block with the
            highest assigned degree (which depends on the values of ``blocks``
            and ``shorten_last_block``).  This means that the number of trainable
            parameters will roughly depend quadratically on the conditioner input
            and linearly on the conditioner output.

            If a ``Sequence``, ``dimensions_hidden[l]`` must be the number of
            nodes in the l-th hidden layer.
        out_per_dimension : int, optional
            The dimension of the output layer in terms of a multiple of the
            number of non-conditioning input features. Since MADE is typically
            used for the conditioner, this usually correspond to the number of
            parameters per degree of freedom used in the transformer. Default
            is 1.
        conditioning_indices : None or Sequence[int], optional
            The indices of the input features corresponding to the conditioning
            features. These features affect the output, but they are not mapped
            by the flow.
        degrees_in : str or Sequence[int], optional
            The degrees to assign to the input/output nodes. Effectively this
            controls the dependencies between variables in the autoregressive
            model. If ``'input'``/``'reversed'``, the degrees are assigned in
            the same/reversed order they are passed. If a ``Sequence``, this must
            be a permutation of ``range(0, n_blocks)``, where ``n_blocks`` is the
            number of blocks passed to the constructor. If blocks are not used,
            this corresponds to the number of non-conditioning features (i.e.,
            ``dimension_in - len(conditioning_indices)``).
        degrees_hidden_motif : Sequence[int], optional
            The degrees of the hidden nodes are assigned using this array in
            a round-robin fashion. If not given, they are assigned in the
            same order used for the input nodes. This must be at least as
            large as the dimension of the smallest hidden layer.
        degrees_per_out : Sequence[int], optional
            The degrees of the output nodes for each ``out_per_dimension``. Note
            that this can be different from ``degrees_in``. For example, if
            ``degrees_per_out=[1, 2, 0]`` and ``out_per_dimension=2`` will result
            in an output of dimension 6 using a scrambled order of degrees even
            if ``degrees_in=[0, 0, 1, 2]``. By default this is set equal to
            ``self.degrees_in``, which automatically accounts for block sizes.
        weight_norm : bool, optional
            If ``True``, weight normalization is applied to the masked linear
            modules.
        blocks : int or Sequence[int], optional
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

        """
        super().__init__()

        # Mutable defaults.
        if conditioning_indices is None:
            conditioning_indices = torch.tensor([], dtype=int)

        # Validate values.
        if isinstance(degrees_in, str) and not degrees_in in ('input', 'reversed'):
            raise ValueError('degrees_in must be one between "input" and "reversed"')

        # Convert all list of indices to tensors (without copying memory if possible).
        dimensions_hidden = ensure_tensor_sequence(dimensions_hidden, dtype=int)
        conditioning_indices = ensure_tensor_sequence(conditioning_indices, dtype=int)
        degrees_in = ensure_tensor_sequence(degrees_in, dtype=int)
        degrees_hidden_motif = ensure_tensor_sequence(degrees_hidden_motif, dtype=int)
        degrees_per_out = ensure_tensor_sequence(degrees_per_out, dtype=int)
        blocks = ensure_tensor_sequence(blocks, dtype=int)

        # Store variabless.
        self._out_per_dimension = out_per_dimension
        self.register_buffer('_conditioning_indices', conditioning_indices)

        # Get the number of dimensions in array format.
        n_hidden_layers, dimensions_hidden, expanded_blocks = self._get_dimensions(
            dimension_in, dimensions_hidden, out_per_dimension, conditioning_indices,
            degrees_in, blocks, shorten_last_block)

        # Store the verified/expanded blocks size.
        self.register_buffer('_blocks', expanded_blocks)

        # Determine the degrees of all input nodes.
        self.register_buffer('_degrees_in', None)
        self._assign_degrees_in(dimension_in, conditioning_indices, degrees_in)

        # Generate the degrees to assign to the hidden nodes in a round-robin fashion.
        self._assign_degrees_hidden_motif(degrees_hidden_motif)

        # Find the mapped indices.
        conditioning_indices_set = set(conditioning_indices.tolist())
        mapped_indices = [i for i in range(dimension_in) if i not in conditioning_indices_set]

        # Verify output layer configuration.
        if degrees_per_out is None:
            degrees_per_out = self.degrees_in[mapped_indices]
        else:
            self._check_degrees(degrees_per_out, name='degrees_per_out')

        # Create a sequence of MaskedLinear + nonlinearity layers.
        layers = []
        degrees_previous = self.degrees_in
        for layer_idx in range(n_hidden_layers+1):
            is_output_layer = layer_idx == n_hidden_layers

            # Determine the degrees of the layer's nodes.
            if is_output_layer:
                # The output layer doesn't have the conditioning features.
                degrees_current = torch.tile(degrees_per_out, (out_per_dimension,))
            else:
                # TODO: torch doesn't support divmod at the moment (see #90820)
                n_round_robin, n_remaining = divmod(dimensions_hidden[layer_idx].tolist(), len(self.degrees_hidden_motif))
                if n_round_robin == 0:
                    err_msg = (f'Hidden layer {layer_idx} is too small '
                               f'to fit the motif {self.degrees_hidden_motif}.'
                               ' Increase the size of the layer or explicitly '
                               'pass a different motif for its degrees.')
                    raise ValueError(err_msg)

                degrees_current = torch.tile(self.degrees_hidden_motif, (n_round_robin,))
                if n_remaining != 0:
                    degrees_current = torch.cat([degrees_current, self.degrees_hidden_motif[:n_remaining]])

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
    def dimension_in(self) -> int:
        "Dimension of the input vector."
        return self.layers[0].in_features

    @property
    def dimension_out(self) -> int:
        "Dimension of the output vector (excluding the conditioning dimensions)."
        return self.layers[-1].out_features

    @property
    def n_layers(self) -> int:
        "Total number of layers (hidden layers + output)."
        # Each layer is a masked linear + activation function, except the output layer.
        return (len(self.layers) + 1) // 2

    @property
    def dimensions_hidden(self) -> torch.Tensor:
        """dimensions_hidden[i] is the number of nodes in the i-th hidden layer."""
        return torch.tensor([l.out_features for l in self.layers[:-1:2]])

    @property
    def out_per_dimension(self) -> int:
        """Multiple of non-conditioning features determining the output dimension."""
        return self._out_per_dimension

    @property
    def dimension_conditioning(self) -> int:
        """Number of conditioning features."""
        return len(self._conditioning_indices)

    @property
    def conditioning_indices(self) -> torch.Tensor:
        """conditioning_indices[i] is the index of the input feature corresponding to the i-th conditioning degree of freedom."""
        return self._conditioning_indices

    @property
    def degrees_in(self) -> torch.Tensor:
        """degrees_in[i] is the MADE degree assigned to the i-th input feature."""
        return self._degrees_in

    @property
    def degrees_hidden_motif(self) -> torch.Tensor:
        """degrees_hidden_motif[i] is the degrees assigned to i-th hidden node."""
        return self._degrees_hidden_motif

    @property
    def blocks(self) -> torch.Tensor:
        """blocks[i] is the size of the i-th block."""
        return self._blocks

    def n_parameters(self) -> int:
        """The total number of (unmasked) parameters."""
        return sum(l.n_parameters() for l in self.layers[::2])

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def _get_dimensions(
            dimension_in: int,
            dimensions_hidden: torch.Tensor,
            out_per_dimension: int,
            conditioning_indices: torch.Tensor,
            degrees_in: Union[str, torch.Tensor],
            blocks: Union[int, torch.Tensor],
            shorten_last_block: bool,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Process the input arguments and return the dimensions of all layers in list format.

        By default, when only the depth of the hidden layers is specified,
        the number of nodes in each hidden layer is assigned so that there
        is one node for each output connected to the network (i.e., ignoring
        the last output/block).

        Returns
        -------
        n_hidden_layers : int
            The number of hidden layers.
        dimensions_hidden : torch.Tensor[int]
            ``dimensions_hidden[i]`` is the dimension of the i-th hidden layer.
        expanded_blocks : torch.Tensor[int]
            ``expanded_blocks[i]`` is the size of the i-th block of features.

        """
        n_conditioning_dofs = len(conditioning_indices)

        # Validate/expand block sizes.
        expanded_blocks = _generate_block_sizes(dimension_in-n_conditioning_dofs, blocks,
                                                shorten_last_block=shorten_last_block)

        # Find the dimension of the last block.
        if isinstance(blocks, int) and (blocks == 1):
            len_last_block = 1
        elif isinstance(degrees_in, str):
            if degrees_in == 'input':
                len_last_block = expanded_blocks[-1]
            else:  # == 'reversed'. degrees_in has been validated in __init__().
                len_last_block = expanded_blocks[0]
        else:
            # Search for the block assigned to the last degree.
            last_block_idx = torch.argmax(degrees_in)
            len_last_block = expanded_blocks[last_block_idx]

        if isinstance(dimensions_hidden, int):
            dimensions_hidden = torch.tensor([(dimension_in - len_last_block) * int(np.floor(out_per_dimension**(1/2)))
                                              for _ in range(dimensions_hidden)])

        return len(dimensions_hidden), dimensions_hidden, expanded_blocks

    def _check_degrees(self, degrees: torch.Tensor, name: str):
        """Check that degrees_in/degrees_per_out are configured correctly.

        Parameters
        ----------
        degrees : torch.Tensor[int]
            Array of indices.
        name : str
            Name of the array for logging purposes.

        Raises
        ------
        ValueError
            If the checks don't pass.
        """
        n_blocks = len(self.blocks)

        err_msg = (" When 'blocks' is not explicitly passed. The number of "
                   "blocks corresponds to the number of non-conditioning "
                   "dimensions (dimension_in - len(conditioning_indices))")
        if len(degrees) != n_blocks:
            raise ValueError('len('+name+') must be equal to the number '
                             'of blocks.' + err_msg)
        if set(degrees.tolist()) != set(range(n_blocks)):
            raise ValueError(name + ' must contain all degrees between '
                             '0 and the number of blocks minus 1.' + err_msg)

    def _assign_degrees_in(
            self,
            dimension_in: int,
            conditioning_indices: torch.Tensor,
            degrees_in: Union[str, torch.Tensor],
    ):
        """Assign the degrees of all input nodes to self._degrees_in.

        self._degrees_in[i] is the degree assigned to the i-th input node.

        The self._blocks variable must be assigned before calling this function.

        """
        # Shortcut for the number of (non-conditioning) blocks.
        n_blocks = len(self.blocks)

        # Eventually convert "input/reversed" to an array of degrees.
        degrees_in = degrees_in_from_str(degrees_in, n_blocks)

        # Verify that the parameters are consistent.
        self._check_degrees(degrees_in, name='degrees_in')

        # Initialize the degrees_in tensor.
        self._degrees_in = torch.empty(dimension_in, dtype=int)

        # The conditioning features are always assigned the lowest
        # degree regardless of the value of degrees_in so that
        # all output features depends on them.
        self._degrees_in[conditioning_indices] = -1

        # Now assign the degrees to each input node.
        conditioning_indices_set = set(conditioning_indices.tolist())
        dof_idx_pointer = 0
        for block_idx, block_size in enumerate(self.blocks):
            for block_dof_idx in range(block_size):
                # Skip to the next non-conditioning block.
                while dof_idx_pointer in conditioning_indices_set:
                    dof_idx_pointer += 1

                # Add the degree of this block to this DOF.
                self._degrees_in[dof_idx_pointer] = degrees_in[block_idx]
                dof_idx_pointer += 1

    def _assign_degrees_hidden_motif(self, degrees_hidden_motif: Optional[torch.Tensor]):
        """Assign the degrees of the hidden nodes to self._degrees_hidden_motif.

        self._degrees_hidden_motif[i] is the degree assigned to the i-th hidden node.
        If len(_degrees_hidden_motif) < n_hidden_nodes, the degrees will be assigned
        in a round-robin fashion.

        The self.blocks and self.degrees_in attributes must be initialized.
        """
        if degrees_hidden_motif is None:
            # There is no need to add the degree of the last block since
            # it won't be connected to the output layer anyway.
            last_degree = len(self.blocks)-1
            degrees_hidden_motif = self.degrees_in[self.degrees_in != last_degree]
        elif np.any(degrees_hidden_motif >= len(self.blocks)-1):
            raise ValueError('degrees_hidden_motif cannot contain degrees '
                             'greater than the number of blocks minus 1 '
                             'as they would be ignored by the output layer.')
        self._degrees_hidden_motif = degrees_hidden_motif


# =============================================================================
# INTERNAL-USAGE-ONLY UTILS
# =============================================================================

def _generate_block_sizes(
        n_features: int,
        blocks: Union[int, torch.Tensor],
        shorten_last_block: bool = False,
) -> torch.Tensor:
    """Divides the features into blocks.

    In case a constant block size is requested, the function can automatically
    make the last block smaller if the number of features is not divisible
    by the block size. The function also raises errors if it detects inconsistencies
    between the number of features and the blocks parameters.

    Parameters
    ----------
    n_features : int
        The number of features to be divided into blocks.
    blocks : int or torch.Tensor[int]
        The size of the blocks. If an integer, the features are divided
        into blocks of equal size (except eventually for the last block).
        If a list, it is interpreted as the return value, and the function
        simply checks that the block sizes divide exactly the number of
        features.
    shorten_last_block : bool, optional
        If ``blocks`` is an integer that is not a divisor of the number
        of features, this option controls whether the last block is
        shortened (``True``) or an exception is raised (``False``).
        Default is ``False``.

    Returns
    -------
    blocks : torch.Tensor[int]
        The features can be divided into ``len(blocks)`` blocks, with
        the i-th block having size ``blocks[i]``.

    """
    # If blocks is an int, divide in blocks of equal size.
    if isinstance(blocks, int):
        if n_features % blocks != 0 and not shorten_last_block:
            raise ValueError('The parameter "n_features" must be '
                             f'divisible by "blocks" ({blocks})')

        div, mod = divmod(n_features, blocks)
        blocks = [blocks] * div
        if mod != 0:
            blocks += [mod]
        blocks = torch.tensor(blocks)
    elif n_features != sum(blocks):
        raise ValueError('The sum of the block sizes must be equal to '
                         f'"n_features" ({n_features}).')

    return blocks
