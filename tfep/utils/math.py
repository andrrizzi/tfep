#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Math and geometry utility functions to manipulate coordinates.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# MATH
# =============================================================================

def batchwise_dot(x1, x2, keepdim=False):
    """Batchwise dot product between two batches of tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        A tensor of shape ``(*, N)``.
    x2 : torch.Tensor
        A tensor of shape ``(*, N)``.
    keepdim : bool, optional
        If ``True``, the return value has shape ``(*, 1)``.
        Otherwise ``(*,)``.

    Returns
    -------
    result : torch.Tensor
        ``result[i,j,...,k]`` is the dot product between ``x1[i,j,...,k]`` and
        ``x2[i,j,...,k]``

    """
    return (x1 * x2).sum(dim=-1, keepdim=keepdim)


def batchwise_outer(x1, x2):
    """Batchwise outer product between two 2D tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        A tensor of shape ``(batch_size, N)``.
    x2 : torch.Tensor
        A tensor of shape ``(batch_size, N)``.

    Returns
    -------
    result : torch.Tensor
        A tensor shape ``(batch_size, N, N)``, where ``result[b][i][j]`` is the
        outer product between ``x1[b][i]`` and ``x2[b][j]``.

    """
    # return torch.einsum('bi,bj->bij', x1, x2)
    return torch.matmul(x1[:, :, None], x2[:, None, :])


def cov(x, ddof=1, dim_sample=0, inplace=False, return_mean=False):
    """Return the covariance matrix of the data.

    .. note::

        Since PyTorch 1.10, there is also a ``torch.cov`` function. Compare to
        that function, this function currently does not offer weighting but
        allows selecting the sample dimension and returning also the mean.
        Nevertheless, we may drop or change name to this function when we stop
        supporting PyTorch 1.9.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape ``(n, m)``, where ``n`` is the number of samples
        used to estimate the covariance, and ``m`` is the dimension of
        the multivariate variable. If ``dim_sample`` is 1, then the expected
        shape is ``(m, n)``.
    ddof : int, optional
        The number of dependent degrees of freedom. The covariance will
        be estimated dividing by ``n - ddof``. Default is 1.
    dim_sample : int, optional
        The dimension of the features. Default is 0, which means each row of
        ``x`` is a sample and each column a different degree of freedom.
    inplace : bool, optional
        If ``True``, the input argument ``x`` is modified to be centered
        on its mean. Default is ``False``.
    return_mean : bool, optional
        If ``True``, the mean of degrees of freedom is also returned. This
        can save an operation if the mean is also required after computing
        the covariance matrix.

    Returns
    -------
    cov : torch.Tensor
        A tensor of shape ``(m, m)``.
    mean : torch.Tensor, optional
        A tensor of shape ``(m,)``.

    """
    if len(x.shape) != 2:
        raise ValueError('The function supports only 2D matrices')
    if dim_sample not in {0, 1}:
        raise ValueError('dim_sample must be either 0 or 1')

    # Center the data on the mean.
    if dim_sample == 0:
        keepdim = False
    else:
        keepdim = True
    mean = torch.mean(x, dim_sample, keepdim=keepdim)
    if inplace:
        x -= mean
    else:
        x = x - mean

    # Average normalization factor.
    n = x.shape[dim_sample] - ddof

    # Compute the covariance matrix
    if dim_sample == 0:
        c = torch.matmul(x.t(), x) / n
    else:
        c = torch.matmul(x, x.t()) / n

    if return_mean:
        return c, mean
    return c


# =============================================================================
# CALCULUS W/ AUTOGRAD
# =============================================================================

def batch_autograd_jacobian(x, y, **grad_kwargs):
    """Compute the batch jacobian of ``y`` w.r.t. ``x`` using ``torch.autograd.grad``.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch_size, *)``. Input.
    y : torch.Tensor
        Shape ``(batch_size, *)``. Output.
    **grad_kwargs
        Keyword arguments for ``autograd.grad`` (e.g., ``create_graph``,
        ``retain_graph``).

    Returns
    -------
    jac : torch.Tensor
        Shape ``(batch_size, size_y, size_x)``.

    """
    batch_size = x.shape[0]
    shape_x = x.shape[1:]
    shape_y = y.shape[1:]

    # Compute the jacobian with autograd.
    y_sum = y.sum(dim=0)
    # Each row vector grad_outputs must have the same shape of y_sum.
    n_features_y = torch.prod(torch.tensor(y_sum.shape), dtype=int)
    grad_outputs = torch.eye(n_features_y).reshape(n_features_y, *y_sum.shape)
    jacobian = torch.autograd.grad(y_sum, x, grad_outputs, is_grads_batched=True, **grad_kwargs)[0]

    # From shape (n_features_y, batch_size, *x_shape) to (batch_size, n_features_y, *x_shape).
    jacobian = jacobian.transpose(0, 1)

    # Now expand dimensions y.
    return jacobian.reshape((batch_size, *shape_y, *shape_x))


def batch_autograd_log_abs_det_J(x, y, **grad_kwargs):
    """Compute the batch log(abs(det(J))) of ``y`` w.r.t. ``x`` using ``torch.autograd.grad``.

    Note that for the determinant to make sense, ``x`` and ``y`` must be 2D
    tensors with identical dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch_size, N)``. Input.
    y : torch.Tensor
        Shape ``(batch_size, N)``. Output.
    **grad_kwargs
        Keyword arguments for ``autograd.grad`` (e.g., ``create_graph``,
        ``retain_graph``).

    Returns
    -------
    log_det_J : torch.Tensor
        Shape ``(batch_size,)``.

    """
    # First compute the batch jacobian.
    jacobian = batch_autograd_jacobian(x, y, **grad_kwargs)
    # Compute the log det J numerically.
    return torch.linalg.slogdet(jacobian)[1]
