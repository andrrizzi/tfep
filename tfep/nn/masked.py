#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Masked linear transformations for PyTorch.

The module include both functional (``masked_linear``) and ``Module`` API
(``MaskedLinear``) to implement a masked linear transformation.

It also contains functions to implement weight normalization in masked linear
layers (``masked_weight_norm``). Indeed, the mask may cause NaNs in the native
PyTorch implementation.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch.autograd
import torch.nn.functional
from torch import norm_except_dim
from torch.nn.parameter import Parameter
from torch.nn.utils.weight_norm import WeightNorm


# =============================================================================
# CREATE AUTOREGRESSIVE MASKS
# =============================================================================

def create_autoregressive_mask(
        degrees_in,
        degrees_out,
        strictly_less=False,
        transpose=False,
        dtype=None
):
    """Create an autoregressive mask between input and output connections.

    ``mask[i][j]`` is ``1`` if the i-th input is connected to th j-th output.
    The output nodes are connected to input nodes with a strictly less degree
    unless ``strictly_less`` is ``False``, in which case output nodes are connected
    to all input nodes with less or equal degree.

    This function can be used to implement masks as proposed in the MADE
    paper [1] by setting ``strictly_less=False`` for hidden layers and ``True``
    for the output layer (see Eq. 13 in the MADE paper).

    Parameters
    ----------
    degrees_in : numpy.ndarray[int] or torch.Tensor[int]
        Shape ``(n_input_nodes,)``. ``degrees_in[k]`` is the integer degree
        assigned to the ``k``-th input node (i.e., :math:`m^{l-1}(k)` in the
        MADE paper).
    degrees_out : numpy.ndarray[int] or torch.Tensor[int]
        Shape ``(n_output_nodes,)``. ``degrees_out[k]`` is the integer degree
        assigned to the ``k``-th output node (i.e., :math:`m^l(k)` in the MADE
        paper).
    strictly_less : bool, optional
        ``True`` if the output nodes must be connected to input node with a strictly
        less degree. Otherwise, nodes are connected if they have a less or equal
        degree.
    transpose : bool, optional
        If ``True``, the returned mask is transposed and input/output node indices
        are swapped.
    dtype : torch.dtype, optional
        The data type of the returned mask. By default, the default PyTorch type
        is used.

    Returns
    -------
    mask : torch.Tensor
        If ``transpose`` is ``False``, this has shape ``(n_input_nodes, n_output_nodes)``,
        otherwise ``(n_output_nodes, n_input_nodes)``. In the first(latter) case, ``mask[i][j]``
        is ``1`` if the i-th input(output) is connected to th j-th output(input).
        This corresponds to the :math:`W^l`, in the MADE paper.

    References
    ----------
    [1] Germain M, Gregor K, Murray I, Larochelle H. Made: Masked autoencoder
        for distribution estimation. In International Conference on Machine
        Learning 2015 Jun 1 (pp. 881-889).

    """
    if transpose:
        if strictly_less:
            mask = degrees_out[:, None] > degrees_in[None, :]
        else:
            mask = degrees_out[:, None] >= degrees_in[None, :]
    else:
        if strictly_less:
            mask = degrees_out[None, :] > degrees_in[:, None]
        else:
            mask = degrees_out[None, :] >= degrees_in[:, None]

    # Convert to tensor of default type before returning.
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.tensor(mask, dtype=dtype)


# =============================================================================
# MASKED LINEAR MODULE API
# =============================================================================

class MaskedLinear(torch.nn.Linear):
    r"""Implement the masked linear transformation: :math:`y = x \cdot (M \circ A)^T + b`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        If set to ``False``, the layer will not learn an additive bias.
        Default is ``True``.
    mask : torch.Tensor, optional
        The mask of zeros and ones of shape ``(out_features, in_features)``
        to apply to the scaling matrix. Default is ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weights of the module of shape ``(out_features, in_features)``.
        The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
        where :math:`k = \frac{1}{\text{in\_features}}`.
    bias : torch.Tensor
        The learnable bias of the module of shape ``(out_features)``.
        If :attr:`bias` is ``True``, the values are initialized from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{1}{\text{in\_features}}`.
    mask : torch.Tensor
        The mask passed during initialization.

    See Also
    --------
    functions.MaskedLinearFunc
        The autograd ``Function`` object used to implement the module.

    Examples
    --------

    >>> in_features, out_features, batch_size = 8, 5, 20
    >>> # Lower triangular mask.
    >>> mask = torch.tril(torch.ones(out_features, in_features, dtype=torch.bool))
    >>> m = MaskedLinear(in_features, out_features, mask=mask)
    >>> input = torch.randn(batch_size, in_features)
    >>> output = m(input)
    >>> print(output.size())
    torch.Size([20, 5])

    """

    def __init__(self, in_features, out_features, bias=True, mask=None):
        # Let nn.Linear register and initialize the parameters.
        super().__init__(in_features, out_features, bias=bias)

        # We don't need to propagate gradients through the mask so we
        # register it as a buffer.
        self.register_buffer('mask', mask)

        # Set the masked weights to 0.0. This effectively sets the
        # gradient of the masked parameters to zero even when weight
        # normalization (whose gradient has a component that depend
        # on the gradient w.r.t. g) is used.
        self.weight.data = self.weight.data * self.mask

    def n_parameters(self):
        """int: The total number of (unmasked) parameters."""
        if self.mask is None:
            n_parameters = self.weight.numel()
        else:
            n_parameters = (self.mask != 0).sum()
        if self.bias is not None:
            n_parameters += self.bias.numel()
        return n_parameters

    def forward(self, input):
        """
        Performs the forward computation.

        Parameters
        ----------
        input : torch.Tensor
            Input of shape ``(batch_size, *, in_features)`` where ``*``
            means any number of additional dimensions.

        Returns
        -------
        output : torch.Tensor
            Output of shape ``(batch_size, *, in_features)`` where ``*``
            is the same number number of additional dimensions in ``input``.

        """
        # If there is no mask, fall back to normal linear behavior.
        if self.mask is None:
            return super().forward(input)
        return masked_linear(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, mask={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.mask
        )


# =============================================================================
# MASKED LINEAR FUNCTIONAL API
# =============================================================================

class MaskedLinearFunc(torch.autograd.Function):
    r"""Implement the masked linear transformation: :math:`y = x \cdot (M \circ A)^T + b`.

    This is based on :func:`torch.nn.functional.linear`, but with an extra
    keyword argument ``mask`` having the same shape as ``weight``.

    Note that the function does not perform a sparse multiplication, but
    simply implements the mask with an element-wise multiplication of the
    weight matrix before evaluating the linear transformation.

    A functional shortcut to ``MaskedLinearFunc`` is available in this same
    module with ``masked_linear``.

    The return value is a ``Tensor`` of shape ``(batch_size, *, n_out_features)``,
    where ``*`` correspond to the same number of additional dimensions
    in the `input` argument.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor x of shape ``(batch_size, *, n_in_features)``, where
        ``*`` means any number of additional dimensions.
    weight : torch.Tensor
        Scaling tensor A of shape ``(n_out_features, n_in_features)``.
    bias : torch.Tensor, optional
        Shifting tensor b of shape ``(n_out_features)``.
    mask : torch.Tensor, optional
        Mask of A of shape ``(n_out_features, n_in_features)``.

    Examples
    --------

    >>> batch_size = 2
    >>> in_features = 3
    >>> out_features = 5
    >>> input = torch.randn(batch_size, in_features, dtype=torch.double)
    >>> weight = torch.randn(out_features, in_features, dtype=torch.double)
    >>> bias = torch.randn(out_features, dtype=torch.double)

    >>> # Lower triangular mask.
    >>> mask = torch.tril(torch.ones(out_features, in_features, dtype=torch.bool))
    >>> output = masked_linear(input, weight, bias, mask)

    """

    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        # Check if we need to mask the weights.
        if mask is not None:
            # Mask weight matrix.
            weight = weight * mask

        # We save the MASKED weights for backward propagation so that
        # we don't need to perform the element-wise multiplication.
        ctx.save_for_backward(input, weight, bias, mask)

        # Compute the linear transformation.
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Unpack previously stored tensors.
        input, masked_weight, bias, mask = ctx.saved_tensors

        # We still need to return None for grad_mask even if we don't
        # compute its gradient.
        grad_input = grad_weight = grad_bias = grad_mask = None

        # Compute gradients w.r.t. input parameters.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(masked_weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

            # Mask the gradients.
            if mask is not None:
                grad_weight.mul_(mask)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, grad_mask

# Functional notation.
masked_linear = MaskedLinearFunc.apply


# =============================================================================
# WEIGHT NORMALIZATION FOR MASKED LINEAR LAYER
# =============================================================================

def masked_weight_norm(module, name='weight', dim=0):
    """NaN-free implementation of weight normalization.

    Applying the normal weight normalization implemented with :func:`torch.nn.utils.weight_norm`
    results in NaN entries in the matrices when the mask covers an entire
    vector (thus making its norm zero). This takes care of this special
    case.

    See Also
    --------
    torch.nn.utils.weight_norm.weight_norm

    """
    try:
        mask = module.mask
    except AttributeError:
        mask = None
    MaskedWeightNorm.apply(module, name, dim, mask)
    return module


def remove_masked_weight_norm(module, name='weight'):
    """Remove masked weighed normalization hooks.

    See Also
    --------
    torch.nn.utils.weight_norm.remove_weight_norm

    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, MaskedWeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))


class MaskedWeightNorm(WeightNorm):
    """NaN-free implementation of weight normalization.

    Applying the normal weight normalization implemented with :func:`torch.nn.utils.weight_norm`
    results in NaN entries in the matrices when the mask covers an entire
    vector (thus making its norm zero). This takes care of this special
    case.

    See Also
    --------
    torch.nn.utils.weight_norm.WeightNorm

    """

    def __init__(self, name, dim, mask):
        super().__init__(name, dim)
        self.apply_mask = _ApplyMask(mask)

    def compute_weight(self, module):
        weight = super().compute_weight(module)
        return self.apply_mask(weight)

    @staticmethod
    def apply(module, name, dim, mask):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, MaskedWeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = MaskedWeightNorm(name, dim, mask)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        g = Parameter(norm_except_dim(weight, 2, dim).data)
        v = Parameter(weight.data)
        module.register_parameter(name + '_g', g)
        module.register_parameter(name + '_v', v)
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        # Register hook to zero out gradient in the masked weights.
        g.register_hook(_ApplyMask(mask, dim, norm=True))
        v.register_hook(_ApplyMask(mask))

        return fn


class _ApplyMask:
    """NaN-safe mask application.

    Parameters
    ----------
    norm : bool, optional
        If True, the mask is applied to a norm vector (i.e., g) rather
        than a matrix (i.e., v or w). Default is False.
    inplace : bool, optional
        If True, the tensor is modified in place when ApplyTask is called.
        Otherwise, a copy is created.

    """

    def __init__(self, mask, dim=0, norm=False, inplace=True):
        # Precompute the masked indices.
        self.inplace = inplace
        self._zero_indices = None
        if mask is not None:
            if norm:
                # For g, we need to zet to zero only those vectors
                # that have zero norm because of the mask.
                self._zero_indices = torch.nonzero(norm_except_dim(mask, 2, dim).flatten() == 0.0)
            else:
                self._zero_indices = mask == 0.0

    def __call__(self, w):
        # An element-wise multiplication doesn't work if there are NaNs.
        if self._zero_indices is not None:
            if not self.inplace:
                w = w.clone()
            w.data[self._zero_indices] = 0.0
            return w
        return None
