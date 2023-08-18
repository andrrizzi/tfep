#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Moebius transformation for autoregressive normalizing flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
import torch.autograd

from tfep.nn.utils import generate_block_sizes
from tfep.utils.math import batchwise_outer


# =============================================================================
# MOEBIUS TRANSFORMER
# =============================================================================

class MoebiusTransformer(torch.nn.Module):
    """Moebius transformer module for autoregressive normalizing flows.

    This is a variant of the transformation proposed in [1], and used
    in [2] to create flows on tori and spheres manifolds. The difference
    with [1], is that we always project the input vector on the unit
    sphere before applying the transformation so that it always preserve
    the distance with the center (which in the current implementation
    is hardcoded to be the origin). The difference with [2] is that the
    transformation contracts points close to the parameter vector rather
    than expanding it.

    The transformer applies the Moebius transformation in "blocks". Blocks
    of size up to 3 are supported so that the transformation can be used to
    transform the positions of atoms (which have a 3-dimensional position)
    without changing their distance to the center of the unit sphere.

    Parameters
    ----------
    blocks : int or List[int]
        The size of the blocks. If an integer, the input and parameter tensors
        are divided into blocks of equal size. If a list, it is divided into
        ``len(blocks)`` blocks, with the i-th block having size ``blocks[i]``.
    shorten_last_block : bool, optional
        If ``True`` and ``blocks`` is an integer that is not a divisor of
        the number of features, the last block is shortened automatically.
        Otherwise, an exception is raised if ``blocks`` is an integer
        that does not divide the number of features.

    See Also
    --------
    nets.functions.transformer.moebius_transformer

    References
    ----------
    [1] Kato S, McCullagh P. Moebius transformation and a Cauchy family
        on the sphere. arXiv preprint arXiv:1510.07679. 2015 Oct 26.
    [2] Rezende DJ, Papamakarios G, Racanière S, Albergo MS, Kanwar G,
        Shanahan PE, Cranmer K. Normalizing Flows on Tori and Spheres.
        arXiv preprint arXiv:2002.02428. 2020 Feb 6.

    """
    # Number of parameters needed by the transformer for each input dimension.
    n_parameters_per_input = 1

    def __init__(self, blocks, shorten_last_block=False):
        super().__init__()
        self.blocks = blocks
        self.shorten_last_block = shorten_last_block

    def forward(self, x, w):
        """Apply the transformation to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor x of shape ``(batch_size, n_features)``.
        w : torch.Tensor
            The vectors determining the point to be contracted/expanded with shape
            ``(batch_size, n_features)``. This vector is first projected on the
            unit sphere before performing the transformation.

        Returns
        -------
        y : torch.Tensor
            Output tensor of shape ``(batch_size, n_features)``.

        """
        w = self._map_to_sphere(w)
        return moebius_transformer(x, w, self.blocks, self.shorten_last_block)

    def inverse(self, y, w):
        """Reverse the affine transformation.

        Parameters
        ----------
        y : torch.Tensor
            Input tensor x of shape ``(batch_size, n_features)``.
        w : torch.Tensor
            The vectors determining the point to be contracted/expanded with shape
            ``(batch_size, n_features)``. This vector is first projected on the
            unit sphere before performing the transformation.

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape ``(batch_size, n_features)``.

        """
        w = self._map_to_sphere(w)
        return moebius_transformer(y, -w, self.blocks, self.shorten_last_block)

    def get_identity_parameters(self, dimension_out):
        """Return the value of the parameters that makes this the identity function.

        This can be used to initialize the normalizing flow to perform the identity
        transformation.

        Parameters
        ----------
        n_features : int
            The dimension of the input vector of the transformer.

        Returns
        -------
        w : torch.Tensor
            A tensor of shape ``(n_features,)`` representing the parameter
            vector to perform the identity function with a Moebius transformer.

        """
        return torch.zeros(size=(self.n_parameters_per_input, dimension_out))

    def _map_to_sphere(self, w):
        """Map w from the real hypervolume to the unit hypersphere."""
        # MAF passes the parameters in shape (batch_size, n_parameters, n_features).
        w = w[:, 0]
        # Tanh maps the real hypervolume to the unit hypercube.
        w = torch.tanh(w)
        # Finally we map the unit hypercube to the unit hypersphere.
        return unit_cube_to_inscribed_sphere(w, self.blocks, self.shorten_last_block)


# =============================================================================
# FUNCTIONAL API
# =============================================================================

class MoebiusTransformerFunc(torch.autograd.Function):
    r"""Implement the Moebius transformation.

    This provide a function API to the ``MoebiusTransformer`` layer. It is
    a variant of the transformation proposed in [1], and used in [2] to
    create flows on tori and spheres manifolds. The difference with [1],
    is that we always project the input vector on the unit sphere before
    applying the transformation so that it always preserve the distance
    with the center (which in the current implementation is hardcoded to be
    the origin). The difference with [2] is that the transformation contracts
    points close to the parameter vector rather than expanding it.

    The transformer applies the Moebius transformation in "blocks". Blocks
    of size up to 3 are supported so that the transformation can be used to
    transform the positions of atoms (which have a 3-dimensional position)
    without changing their distance to the center of the unit sphere.

    The function returns the transformed feature as a ``Tensor`` of shape
    ``(batch_size, n_features)`` and the log absolute determinant of its
    Jacobian as a ``Tensor`` of shape ``(batch_size,)``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor x of shape ``(batch_size, n_features)``.
    w : torch.Tensor
        The vectors determining the point to be contracted/expanded.
        These vectors has shape ``batch_size, n_features)`` and they
        must be within the unit sphere for their block.
    blocks : int or List[int]
        The size of the blocks. If an integer, ``x`` and ``w`` are
        divided into blocks of equal size. Otherwise, it is divided
        into ``len(blocks)`` blocks, with the i-th block having size
        ``blocks[i]``.
    shorten_last_block : bool, optional
        If ``True`` and ``blocks`` is an integer that is not a divisor of
        the number of features, the last block is shortened automatically.
        Otherwise, an exception is raised if ``blocks`` is an integer
        that does not divide the number of features.

    References
    ----------
    [1] Kato S, McCullagh P. Moebius transformation and a Cauchy family
        on the sphere. arXiv preprint arXiv:1510.07679. 2015 Oct 26.
    [2] Rezende DJ, Papamakarios G, Racanière S, Albergo MS, Kanwar G,
        Shanahan PE, Cranmer K. Normalizing Flows on Tori and Spheres.
        arXiv preprint arXiv:2002.02428. 2020 Feb 6.

    """

    @staticmethod
    def forward(ctx, x, w, blocks, shorten_last_block=False):
        batch_size, n_features = x.shape

        # Eventually convert a constant block size to a list of block sizes.
        blocks = generate_block_sizes(n_features, blocks, shorten_last_block)

        # The input vectors normalized by their own norm.
        x_normalized = torch.empty_like(x)

        # x_normalized + w.
        x_normalized_plus_w = torch.empty_like(x)

        # Compute the norms of the x vectors for each block.
        # norm_squared has shape (batch_size, n_features).
        x_norm = torch.empty_like(w)
        w_norm_squared = torch.empty_like(w)
        xw_norm_squared = torch.empty_like(x)

        # The pointer to the index where the current block starts.
        block_pointer = 0

        for block_size in blocks:
            x_block = x[:, block_pointer:block_pointer+block_size]
            w_block = w[:, block_pointer:block_pointer+block_size]

            # Compute normalized input vector for this block.
            x_norm_block = torch.sqrt(torch.sum(x_block**2, dim=1, keepdim=True))
            x_normalized_block = x_block / x_norm_block
            x_norm[:, block_pointer:block_pointer+block_size] = x_norm_block
            x_normalized[:, block_pointer:block_pointer+block_size] = x_normalized_block

            x_normalized_plus_w_block = x_normalized_block + w_block
            x_normalized_plus_w[:, block_pointer:block_pointer+block_size] = x_normalized_plus_w_block

            # Compute norms.
            w_norm_squared_block = torch.sum(w_block**2, dim=1, keepdim=True)
            xw_norm_squared_block = torch.sum(x_normalized_plus_w_block**2, dim=1, keepdim=True)
            w_norm_squared[:, block_pointer:block_pointer+block_size] = w_norm_squared_block
            xw_norm_squared[:, block_pointer:block_pointer+block_size] = xw_norm_squared_block

            block_pointer += block_size

        # Compute the transformation.
        y_normalized = (1 - w_norm_squared) / xw_norm_squared * (x_normalized + w) + w
        y = x_norm * y_normalized

        # Compute the gradient for backprop and the Jacobian determinant.
        grad_x = torch.zeros(batch_size, n_features, n_features, dtype=x.dtype)
        log_det_J = torch.zeros(batch_size, dtype=x.dtype)

        block_pointer = 0
        for block_size in blocks:
            x_block = x[:, block_pointer:block_pointer+block_size]
            x_normalized_plus_w_block = x_normalized_plus_w[:, block_pointer:block_pointer+block_size]
            y_normalized_block = y_normalized[:, block_pointer:block_pointer+block_size]

            # Add two fake dimensions to the norms so that they can be broadcasted correctly.
            x_norm_block = x_norm[:, block_pointer, None, None]
            w_norm_squared_block = w_norm_squared[:, block_pointer, None, None]
            xw_norm_squared_block = xw_norm_squared[:, block_pointer, None, None]

            # d||x||/dx_j = x_j / ||x||
            dxnorm_dx = x_normalized[:, block_pointer:block_pointer+block_size]

            # Compute dx_normalized/dx = I/x_norm - (x_block X x_block)/x_norm**3
            # where "I" is the identity matrix and "X" is the outer product.
            # dxnormalized_dx[i][j] = dx_normalized_i/dx_j.
            x_block_outer_x_block = batchwise_outer(x_block, x_block)
            batch_eye = torch.diag_embed(torch.ones(batch_size, block_size, dtype=x.dtype))
            dxnormalized_dx = (batch_eye - x_block_outer_x_block / x_norm_block**2) / x_norm_block

            # Compute the block Jacobian.
            grad_x_block = torch.matmul(dxnormalized_dx, x_normalized_plus_w_block[:, :, None])[:, :, 0]
            grad_x_block = 2 / xw_norm_squared_block * batchwise_outer(x_normalized_plus_w_block, grad_x_block)
            grad_x_block = x_norm_block * (1 - w_norm_squared_block) / xw_norm_squared_block * (dxnormalized_dx - grad_x_block)
            grad_x_block += batchwise_outer(y_normalized_block, dxnorm_dx)
            grad_x[:, block_pointer:block_pointer+block_size, block_pointer:block_pointer+block_size] = grad_x_block

            # Compute the determinant.
            log_det_J += torch.log(torch.abs(_det(grad_x_block)))

            block_pointer += block_size

        # Save tensors used for backward() before returning.
        ctx.save_for_backward(grad_x, w, x_normalized_plus_w, x_norm, w_norm_squared, xw_norm_squared)
        ctx.blocks = blocks

        # We don't need to compute gradients of log_det_J.
        ctx.mark_non_differentiable(log_det_J)
        return y, log_det_J

    @staticmethod
    def backward(ctx, grad_y, grad_log_det_J):
        grad_x = grad_w = grad_blocks = grad_shorten_last_block = None

        # Read the saved tensors.
        saved_grad_x, w, x_normalized_plus_w, x_norm, w_norm_squared, xw_norm_squared = ctx.saved_tensors
        batch_size, n_features = w.shape

        # Compute gradients w.r.t. input parameters.
        if ctx.needs_input_grad[0]:
            grad_x = grad_y[:, None, :].matmul(saved_grad_x)[:, 0, :]

        # Initialize gradient tensors.
        if ctx.needs_input_grad[1]:
            # grad_w is block-diagonal so most of the entries are zero.
            grad_w = torch.zeros_like(saved_grad_x)

            # Compute the gradient for each block.
            block_pointer = 0
            for block_size in ctx.blocks:
                w_block = w[:, block_pointer:block_pointer+block_size]
                batch_eye = torch.diag_embed(torch.ones(batch_size, block_size, dtype=w.dtype))

                x_normalized_plus_w_block = x_normalized_plus_w[:, block_pointer:block_pointer+block_size]
                x_norm_block = x_norm[:, block_pointer]
                w_norm_squared_block = w_norm_squared[:, block_pointer]
                xw_norm_squared_block = xw_norm_squared[:, block_pointer]

                # Compute common terms between the two.
                factor1 = 1 - w_norm_squared_block + xw_norm_squared_block
                factor2 = 2 * (1 - w_norm_squared_block) / xw_norm_squared_block
                factor3 = x_norm_block / xw_norm_squared_block

                # Compute the gradients.
                grad_w_block = -2 * batchwise_outer(x_normalized_plus_w_block, w_block)
                grad_w_block += factor1[:, None, None] * batch_eye
                grad_w_block -= factor2[:, None, None] * batchwise_outer(x_normalized_plus_w_block, x_normalized_plus_w_block)
                grad_w_block *= factor3[:, None, None]

                grad_w[:, block_pointer:block_pointer+block_size, block_pointer:block_pointer+block_size] = grad_w_block

                # Next block.
                block_pointer += block_size

            # Batchwise matrix-vector product.
            grad_w = grad_y[:, None, :].matmul(grad_w)[:, 0, :]

        return grad_x, grad_w, grad_blocks, grad_shorten_last_block

# Functional notation.
moebius_transformer = MoebiusTransformerFunc.apply


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def unit_cube_to_inscribed_sphere(w, blocks, shorten_last_block=False):
    r"""Utility function mapping vectors from the cube to its inscribed sphere.

    The mapping is supported only for dimensions of blocks up to three.

    Parameters
    ----------
    w : torch.Tensor
        The vectors within the hypercubes to be mapped to the hyperspheres.
        This has shape ``(batch_size, n_features)``.
    blocks : int or List[int]
        The size of the blocks. If an integer, ``w`` is divided into
        blocks of equal size. Otherwise, it is divided into ``len(blocks)``
        blocks, with the i-th block having size ``blocks[i]``.
    shorten_last_block : bool, optional
        If ``True`` and ``blocks`` is an integer that is not a divisor of
        the number of features, the last block is shortened automatically.
        Otherwise, an exception is raised if ``blocks`` is an integer
        that does not divide the number of features.

    Returns
    -------
    mapped_w : torch.Tensor
        The mapped vectors of the same shape of ``w``.

    """
    batch_size, n_features = w.shape

    # Eventually convert a constant block size to a list of block sizes.
    blocks = generate_block_sizes(n_features, blocks, shorten_last_block)

    # Initialized the returned value.
    mapped_w = torch.empty_like(w)

    # The pointer to the index where the current block starts.
    block_pointer = 0

    for block_size in blocks:
        w_block = w[:, block_pointer:block_pointer+block_size]

        if block_size == 3:
            squared = w_block**2
            squared_norms = squared.sum(dim=1, keepdim=True)
            yxx = torch.index_select(squared, dim=1, index=torch.tensor([1, 0, 0]))
            zzy = torch.index_select(squared, dim=1, index=torch.tensor([2, 2, 1]))
            mapped_w_block = w_block * torch.sqrt(1 - (squared_norms - squared) / 2 + yxx * zzy / 3)
        elif block_size == 2:
            swapped = torch.index_select(w_block, dim=1, index=torch.tensor([1, 0]))
            mapped_w_block = w_block * torch.sqrt(1 - swapped**2/2)
        elif block_size == 1:
            mapped_w_block = w_block
        else:
            raise NotImplementedError('Hypercube to hypersphere mapping is not implemented')

        mapped_w[:, block_pointer:block_pointer+block_size] = mapped_w_block

        block_pointer += block_size

    return mapped_w


def _det(a):
    """
    Batch determinant.
    """
    if a.shape[1:] == (3, 3):
        return a[:, 0, 0] * _det(a[:, 1:, 1:]) - a[:, 0, 1] * _det(a[:, 1:, 0::2]) + a[:, 0, 2] * _det(a[:, 1:, :2])
    elif a.shape[1:] == (2, 2):
        return a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    elif a.shape[1:] == (1, 1):
        return a[:, 0, 0]
    return torch.det(a)

