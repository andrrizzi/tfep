#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
PyTorch-accelerated bootstrap analysis.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# BOOTSTRAP
# =============================================================================

def bootstrap(
        data, statistic, *,
        confidence_level=0.95,
        n_resamples=9999,
        bootstrap_sample_size=None,
        batch=None,
        method='percentile',
        generator=None
):
    r"""Compute the parameters of the bootstrap distribution of a statistic.

    The function API is inspired by ``scipy.stats.bootstrap``. Currently it
    computes confidence intervals, standard deviation, mean, and median of the
    bootstrap distribution of the statistic.

    It is also possible to compute the bootstrap statistics for multiple sample
    sizes that are randomly sampled from the data. For example, if ``data`` has
    1000 samples, sampling ``bootstrap_sample_size`` to 100 will randomly select
    only 100 samples out of the 1000 for each resampling.

    Currently only ``'percentile'`` and ``'basic'`` confidence interval methods
    are supported (i.e., no ``'bca'`` like in scipy). Moreover, only 1D data
    is supported.

    Parameters
    ----------
    data : torch.Tensor
        A 1D tensor of shape ``(n_samples,)``.
    statistic : Callable
        A function taking a 2D tensor of shape ``(batch_size, n_samples)`` and
        returning the value of the statistic for each batch as a tensor of shape
        ``(batch_size,)``.
    confidence_level : float, optional
        The confidence level of the confidence interval.
    n_resamples : int, optional
        The number of resamples performed to generate the bootstrap distribution
        of the statistic.
    bootstrap_sample_size : {int, List[int]}, optional
        If given, in each resample only ``bootstrap_sample_size`` will be drawn
        from the data (rather than the default ``n_samples``). If a ``list``, the
        function will perform multiple bootstrap analyses for each of the
        ``bootstrap_sample_size`` provided.
    batch : int, optional
        If given, the ``n_resamples`` resamples are performed into batches of
        size ``batch`` so that the memory consumption becomes ``batch * n_samples``.
    method : {'percentile', 'basic'}, optional
        Whether to return the percentile or reverse bootstrap confidence interval.
    generator : {int, `torch.Generator`}, optional
        If ``generator`` is an int, a new ``torch.Generator`` instance is used
        and seeded with ``generator``. If ``generator`` is already a ``torch.Generator``
        then that instance is used to generate random resamples.

    Returns
    -------
    result : {dict, List[dict]}
        If ``bootstrap_sample_size`` is given, then this is a ``list`` of ``dict``,
        and ``result[i]`` contains the result of the bootstrap analysis for
        ``bootstrap_sample_size[i]``. Otherwise, this is a single ``dict``. All
        ``dict`` have the following keys

        - ``'confidence_interval'``: Another ``dict`` including two keys ``'low'``
                                     and ``'high'`` having the lower and upper limits
                                     of the confidence interval respectively.
        - ``'standard_deviation'``: The standard deviation of the statistic bootstrap
                                    distribution.
        - ``'mean'``: The mean of the bootstrap distribution of the statistic.
        - ``'median'``: The median of the bootstrap distribution of the statistic.

    See Also
    --------
    ``scipy.stats.bootstrap``

    References
    ----------
    .. [1] Bootstrapping (statistics), Wikipedia,
       https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

    """
    if len(data.shape) > 1:
        raise ValueError('We currently support only 1D data.')
    n_samples = len(data)

    # Make sure the gradient tree is not traced.
    data = data.detach()

    # Check the number of samples in the bootstrap distribution.
    if bootstrap_sample_size is None:
        bootstrap_sample_size = [n_samples]

    # Check if we need to compute the vectorized statistic in batches.
    if batch is None:
        batch = n_resamples

    # Expand the data so that we can index it easily when resampling
    # since torch.gather does not broadcast (see pytorch#9407).
    data_expanded = data.expand((batch, n_samples))

    # bootstrap_statistics[i] contains the value of the statistic for the i-th resampling.
    bootstrap_statistics = torch.empty(n_resamples, dtype=data.dtype)

    # Compute for each sample size.
    results = []
    for sample_size in bootstrap_sample_size:
        _bootstrap_statistics(
            data_expanded, statistic, n_resamples, sample_size,
            batch, generator, bootstrap_statistics)

        # Calculate percentile confidence interval.
        alpha = (1 - confidence_level)/2
        quantiles = torch.tensor([alpha, 1-alpha], dtype=data.dtype)
        ci_l, ci_u = torch.quantile(bootstrap_statistics, q=quantiles)

        # Compute the "basic" confidence interval (see [1]).
        if method == 'basic':
            full_statistic = statistic(data.unsqueeze(0))
            ci_l, ci_u = 2*full_statistic - ci_u, 2*full_statistic - ci_l

        results.append(dict(
            confidence_interval=dict(low=ci_l, high=ci_u),
            standard_deviation=torch.std(bootstrap_statistics),
            mean=torch.mean(bootstrap_statistics),
            median=torch.median(bootstrap_statistics),
        ))

    if len(bootstrap_sample_size) == 1:
        return results[0]
    return results


def _bootstrap_statistics(
        data_expanded,
        statistic,
        n_resamples,
        bootstrap_sample_size,
        batch,
        generator,
        bootstrap_statistics
):
    """Generate a bootstrap distribution of the statistic.

    This modify bootstrap_statistics in place.
    """
    n_samples = data_expanded.shape[1]

    # Divide the bootstrap in batches to limit memory usage.
    for k in range(0, n_resamples, batch):
        # The last batch might be smaller.
        batch_actual = min(batch, n_resamples-k)
        if batch_actual != batch:
            data_expanded = data_expanded[:batch_actual]

        # Resample.
        bootstrap_sample_indices = torch.randint(
            low=0, high=n_samples,
            size=(batch_actual, bootstrap_sample_size),
            generator=generator
        )
        bootstrap_samples = torch.gather(
            data_expanded, dim=1, index=bootstrap_sample_indices)

        # Compute statistic for these batch.
        bootstrap_statistics[k:k+batch_actual] = statistic(bootstrap_samples, dim=1)

    return bootstrap_statistics
