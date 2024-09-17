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
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

from tfep.nn import masked
from tfep.nn.conditioners.conditioner import Conditioner
from tfep.utils.misc import ensure_tensor_sequence


# =============================================================================
# UTILS
# =============================================================================

def generate_degrees(
        n_features : int,
        order : Literal['ascending', 'descending', 'random'] = 'ascending',
        max_value : Optional[int] = None,
        conditioning_indices : Optional[Sequence[int]] = None,
        repeats : Union[int, Sequence[int]] = 1,
) -> torch.Tensor:
    """Generate node degrees for MADE layers.

    This generates a tensor representing degrees from 0 to ``max_value``.
    Conditioning degrees are set to -1.

    Parameters
    ----------
    n_features : int
        The length of the generated 1D tensor.
    order : Literal['ascending', 'descending', 'random'], optional
        In what order to generate the degrees. Default is ``'ascending'``.
    max_value : int, optional
        The maximum value to assign to the degree tensor. By default, this is
        determined automatically to be consistent to the value of the other
        parameters.
    conditioning_indices : None or Sequence[int], optional
        The indices of the output tensor whose degree must be set to -1. Default
        is ``None``.
    repeats : Union[int, Sequence[int]], optional
        How many time to repeat a degree. This is similar to the ``repeats``
        argument in ``torch.repeat_interleave()``. If this is a ``Sequence[int]``
        and ``max_value`` is passed, this must have length ``max_value+1``.

    Returns
    -------
    degrees : torch.Tensor
        Shape ``(n_features,)``. The degrees.

    Examples
    --------

    >>> generate_degrees(n_features=3).tolist()
    [0, 1, 2]
    >>> generate_degrees(7, order='descending').tolist()
    [6, 5, 4, 3, 2, 1, 0]

    If ``max_value`` is smaller than the requested number of elements, the
    array of degrees is obtained by tiling

    >>> generate_degrees(7, order='descending', max_value=2).tolist()
    [2, 1, 0, 2, 1, 0, 2]

    ``conditioning_indices`` can be used to set some elements as "conditioning"
    (see the documentation of :class:`.MADE` for details).

    >>> generate_degrees(7, max_value=2, conditioning_indices=[0, 2, 3]).tolist()
    [-1, 0, -1, -1, 1, 2, 0]

    ``repeats`` can be used to assign contiguous elements to the same
    degree

    >>> generate_degrees(6, repeats=2).tolist()
    [0, 0, 1, 1, 2, 2]
    >>> generate_degrees(7, repeats=[1, 3, 2], conditioning_indices=[2]).tolist()
    [0, 1, -1, 1, 1, 2, 2]

    """
    # Determine the number of nonconditioning features.
    n_nonconditioning_features = n_features
    if conditioning_indices is not None:
        n_nonconditioning_features -= len(conditioning_indices)

    # Determine the maximum value.
    if max_value is None:
        try:
            # repeats is a Sequence[int].
            max_value = len(repeats) - 1
        except TypeError:
            # repeats is an integer.
            max_value = int(np.ceil(n_nonconditioning_features / repeats)) - 1

    # Generate the sequence of degrees to tile.
    if order == 'ascending':
        degrees = torch.arange(max_value+1)
    elif order == 'descending':
        degrees = torch.arange(max_value, -1, -1)
    elif order == 'random':
        degrees = torch.randperm(max_value+1)
    else:
        raise ValueError("Accepted string values for 'order' "
                         "are 'ascending', 'descending', and 'random'.")

    # Now expand by repeats.
    repeats = ensure_tensor_sequence(repeats, dtype=int)
    degrees = torch.repeat_interleave(degrees, repeats)[:n_nonconditioning_features]

    # Tile until we generate the correct length.
    degrees = _round_robin(degrees, length=n_nonconditioning_features)

    # Now insert conditioning indices.
    if conditioning_indices is not None:
        # Make sure conditioning indices is not an array/tensor.
        try:
            conditioning_indices = conditioning_indices.tolist()
        except AttributeError:
            pass
        conditioning_indices_set = set(conditioning_indices)
        nonconditioning_indices = [i for i in range(n_features) if i not in conditioning_indices_set]
        assert len(nonconditioning_indices) == n_nonconditioning_features

        # Create final tensor.
        nonconditioning_degrees = degrees
        degrees = torch.empty(n_features).to(nonconditioning_degrees)
        degrees[conditioning_indices] = -1
        degrees[nonconditioning_indices] = nonconditioning_degrees

    return degrees


# =============================================================================
# MADE
# =============================================================================

class MADE(Conditioner):
    """
    An autoregressive layer implemented through masked affine layers.

    This implements the Masked Autoregressive network for Distribution Estimation
    (MADE) [1], which is used in the Inverse/Masked Autoregressive Flow (IAF/MAF)
    [2]/[3] as a conditioner network. MADE is a dense layer, where the connections
    between nodes are partially masked to satisfy the autoregressive property.
    The mask is built based on the values of the degrees assigned to each node
    in the network (see [1]). Very briefly, an output node with degree ``i`` will
    depend only on the inputs assigned to a degree strictly less than ``i``.

    An advantage of using masks over the naive implementation of an
    autoregressive layer, which use a different neural network for each
    parameter of the affine transformation, is that it generates all the
    affine parameters in a single pass, with much less parameters to train,
    and can be parallelized trivially.

    The current implementation supports arbitrary dependencies between input
    features so that this can be used as a conditioner to implement the full
    range between fully autoregressive and coupling layers flows.

    Each layer is a :class:`MaskedLinear`, which, in hidden layers, is followed
    by an ``ELU`` nonlinearity.

    See Also
    --------
    :func:`.generate_degrees` : Utility to generate sequences of degrees.

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

    A fully autoregressive MADE layer with 4 inputs and 8 outputs. Note that, in
    this example, because of how MADE masks are built from the degrees (see [1]),
    the first 2 outputs will not depend on any input (i.e., they will be trainable
    constant numbers). Similarly, the last input will not affect any output, and
    may be omitted from the model with no effect.

    >>> made = MADE(
    ...     degrees_in=[0, 1, 2, 3],
    ...     degrees_out=[0, 0, 1, 1, 2, 2, 3, 3],
    ... )

    The degrees do not have to be in any particular order, and multiple inputs
    can be assigned the same degree.

    >>> made = MADE(
    ...     degrees_in=[1, 1, 0, 2],
    ...     degrees_out=[0, 1, 2, 3, 3, 2, 1, 0],
    ... )

    It is possible to define "conditioning" inputs (in this case, the first 3).
    These are defined as features that affect all outputs and (consequently) for
    which no output is generated.

    >>> made = MADE(
    ...     degrees_in=[-1, -1, -1, 0, 1, 2,],
    ...     degrees_out=[0, 1, 2, 3, 0, 1, 2, 3],
    ... )

    The :func:`.generate_degrees` utility function can be used to generate the
    degrees for several common scenarios. You can also control, the hidden layers.

    >>> made = MADE(
    ...     degrees_in=generate_degrees(n_features=3, order='descending'),
    ...     degrees_out=generate_degrees(n_features=8, order='descending'),
    ...     hidden_layers=[
    ...         generate_degrees(n_features=6, max_value=2),
    ...         generate_degrees(n_features=8, max_value=2),
    ...     ],
    ... )

    A coupling flow layer with 2 inputs, 4 outputs, and 3 hidden layers.

    >>> made = MADE(
    ...     degrees_in=[-1, -1],
    ...     degrees_out=[0, 0, 0, 0],
    ...     hidden_layers=3,
    ... )

    """

    def __init__(
        self,
        degrees_in: Sequence[int],
        degrees_out: Sequence[int],
        hidden_layers: Union[int, Sequence[int], Sequence[Sequence[int]]] = 2,
        weight_norm: bool = True,
    ):
        """Constructor.

        Parameters
        ----------
        degrees_in : Sequence[int]
            Shape: ``(n_inputs,)``. ``degrees_in[i]`` is the degree assigned to
            the ``i``-th input.
        degrees_out : Sequence[int]
            Shape: ``(n_outputs,)``. ``degrees_out[i]`` is the degree assigned to
            the ``i``-th output.
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
        weight_norm : bool, optional
            If ``True``, weight normalization is applied to the masked linear
            modules. Default is ``True``.

        """
        super().__init__()

        # Convert all list of indices to tensors (without copying memory if possible).
        degrees_in = ensure_tensor_sequence(degrees_in, dtype=int)
        degrees_out = ensure_tensor_sequence(degrees_out, dtype=int)

        # Create the degrees to assign to the hidden layers.
        degrees_hidden = self._get_degrees_hidden(degrees_in, degrees_out, hidden_layers)
        n_hidden_layers = len(degrees_hidden)

        # Create a sequence of MaskedLinear + nonlinearity layers.
        layers = []
        degrees_previous = degrees_in
        for layer_idx in range(n_hidden_layers+1):
            is_output_layer = layer_idx == n_hidden_layers

            # Determine the degrees of the layer's nodes.
            if is_output_layer:
                degrees_current = degrees_out
            else:
                degrees_current = degrees_hidden[layer_idx]

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
        """Dimension of the input tensor."""
        return self.layers[0].in_features

    @property
    def dimension_out(self) -> int:
        """Dimension of the output tensor."""
        return self.layers[-1].out_features

    @property
    def dimensions_hidden(self) -> torch.Tensor:
        """Shape: ``(n_hidden_layers,)``. ``dimensions_hidden[i]`` is the number of nodes in the ``i``-th hidden layer."""
        return torch.tensor([l.out_features for l in self.layers[:-1:2]])

    @property
    def weight_norm(self):
        """bool: ```True``` if weight norm is used, ```False``` otherwise."""
        return hasattr(self.layers[-1], 'weight_g')

    def n_parameters(self) -> int:
        """The total number of (unmasked) parameters."""
        return sum(l.n_parameters() for l in self.layers[::2])

    def forward(self, x):
        return self.layers(x)

    def set_output(self, output: torch.Tensor):
        """Implement :func:`tfep.nn.flows.autoregressive.Conditioner.set_output`."""
        if self.weight_norm:
            self.layers[-1].weight_g.data.fill_(0.0)
        else:
            self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data = output

    @classmethod
    def _get_degrees_hidden(
            cls,
            degrees_in : torch.Tensor,
            degrees_out : torch.Tensor,
            hidden_layers: Union[int, Sequence[int], Sequence[Sequence[int]]],
    ) -> list[torch.Tensor]:
        """Return the degrees of the hidden layers.

        Returns
        -------
        degrees_hidden : list[torch.Tensor]
            ``degrees_hidden[l][i]`` is the degree assigned to the ``i``-th node
            of the ``l``-th hidden layer.

        """
        # Make sure hidden_layers is not a numpy/torch type.
        try:
            hidden_layers = hidden_layers.tolist()
        except AttributeError:
            pass

        # Mask selecting the input features that affect the output.
        max_degree_out = degrees_out.max()
        relevant_in_features_mask = degrees_in < max_degree_out

        # Convert integer to a list of layer widths (Sequence[int]) which will be
        # handled in the next if block.
        if isinstance(hidden_layers, int):
            # Compute default number of nodes per layer.
            n_relevant_in_features = relevant_in_features_mask.sum().tolist()
            n_outputs = len(degrees_out)
            n_nodes_per_layer = int(np.ceil((n_relevant_in_features * n_outputs)**0.5))
            n_nodes_per_layer = max(n_nodes_per_layer, n_relevant_in_features)

            hidden_layers = [n_nodes_per_layer for _ in range(hidden_layers)]

        # Convert list of layer widths (Sequence[int]) to a list of degrees
        # for each layer (Sequence[Sequence[int]]).
        if isinstance(hidden_layers[0], int):
            degrees_hidden = []

            for layer_idx, width in enumerate(hidden_layers):
                # There is no need to add degrees that won't be connected to the output layer.
                degrees_hidden_motif = degrees_in[relevant_in_features_mask]

                # Create the degrees.
                layer_degrees = _round_robin(
                    x=degrees_hidden_motif,
                    length=hidden_layers[layer_idx],
                    err_msg=(f'Hidden layer {layer_idx} is too small for the number'
                             ' of input features. Increase the size of the layer or'
                             ' explicitly pass the degrees for the hidden layers.')
                )

                # Append new layer.
                degrees_hidden.append(layer_degrees)
        else:
            # Convert from list[list[int]] to list[Tensor[int]].
            degrees_hidden = [ensure_tensor_sequence(x) for x in hidden_layers]

            # Check that the user-provided degrees are sound.
            for layer_idx, degrees in enumerate(degrees_hidden):
                if torch.any(degrees >= max_degree_out):
                    raise ValueError(f'The {layer_idx}-th hidden layer contain '
                                     'nodes with degrees that will be ignored '
                                     'by the output layer.')

        return degrees_hidden


# =============================================================================
# PRIVATE UTILS
# =============================================================================

def _round_robin(x: torch.Tensor, length: int, err_msg: Optional[str] = None) -> torch.Tensor:
    """Tile x in a round-robin fashion until a tensor of size ``length`` is created.

    ``x`` is a 1D tensor.

    Returns a 1D tensor of shape (length,).

    """
    # TODO: torch doesn't support divmod at the moment (see #90820)
    n_round_robin, n_remaining = divmod(length, len(x))
    if n_round_robin == 0:
        if err_msg is None:
            err_msg = f'Length {length} is smaller than the array (len={len(x)}).'
        raise ValueError(err_msg)

    # Create hidden layer degrees by tiling degrees_in.
    out = torch.tile(x, (n_round_robin,))
    if n_remaining != 0:
        out = torch.cat([out, x[:n_remaining]])

    return out
