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
        take_first_only=False,
        batch=None,
        method='percentile',
        bayesian=False,
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
        A tensor of shape ``(n_samples,)`` or ``(n_samples, data_dimension)``.
    statistic : Callable
        A function taking a tensor of shape ``(n_samples,)`` or ``(n_samples, data_dim)``
        and returning the value of the statistic.

        The function must accept a keyword argument ``vectorized``. If ``True``
        ``data`` has an extra dimension ``batch_size`` prepended and the returned
        value must have shape ``(batch_size,)``.

        If ``bayesian`` is ``True``, the function must also accept a ``weights``
        keyword argument (where weights sum to 1) to compute a sample-weighted
        statistic.
    confidence_level : float, optional
        The confidence level of the confidence interval.
    n_resamples : int, optional
        The number of resamples performed to generate the bootstrap distribution
        of the statistic.
    bootstrap_sample_size : {int, List[int]}, optional
        If given, in each resample only ``bootstrap_sample_size`` samples will
        be drawn from the data (rather than the default ``n_samples``). If a
        ``list``, the function will perform multiple bootstrap analyses for each
        of the ``bootstrap_sample_size`` provided.

        When ``bayesian`` is ``True``, this is supported only if ``take_first_only``
        is also ``True``.
    take_first_only : bool, optional
        If ``True`` and ``bootstrap_sample_size < n_samples``, the bootstrap samples
        will be drawn only from ``data[:bootstrap_sample_size]`` rather than the
        entire ``data`` tensor. This is useful, for example, if the data represent
        generalized work values coming from mapping functions that are progressively
        more trained, so that we expect the samples towards the end to be more
        accurate than the first ones.
    batch : int, optional
        If given, the ``n_resamples`` resamples are performed into batches of
        size ``batch`` so that the memory consumption becomes ``batch * n_samples``.
    method : {'percentile', 'basic'}, optional
        Whether to return the percentile or reverse bootstrap confidence interval.
    bayesian : bool, optional
        If ``True``, Bayesian rather than standard bootstrapping is performed. In
        this case, ``statistic`` must support weights.
    generator : {int, `torch.Generator`}, optional
        If ``generator`` is an int, a new ``torch.Generator`` instance is used
        and seeded with ``generator``. If ``generator`` is already a ``torch.Generator``
        then that instance is used to generate random resamples.

        This is not supported if ``bayesian`` is ``True``.

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
    n_samples = len(data)

    if bayesian and generator is not None:
        raise ValueError('Bayesian bootstrapping does not support random number generators.')

    # Check the number of samples in the bootstrap distribution.
    if bootstrap_sample_size is None:
        bootstrap_sample_size = [n_samples]
    elif bayesian and not take_first_only:
        raise ValueError('With Bayesian bootstrapping, specifying a bootstrap_sample_size '
                         'is supported only when take_first_only is True.')

    # Check if we need to compute the vectorized statistic in batches.
    if batch is None:
        batch = n_resamples

    # Make sure the computational graph is not kept in memory.
    with torch.no_grad():
        # Expand the data so that we can index it easily when resampling
        # since torch.gather does not broadcast (see pytorch#9407).
        data_expanded = data.expand((batch, *data.shape))

        # bootstrap_statistics[i] contains the value of the statistic for the i-th resampling.
        bootstrap_statistics = torch.empty(n_resamples, dtype=data.dtype)

        # Compute for each sample size.
        results = []
        for sample_size in bootstrap_sample_size:
            if bayesian:
                # With Bayesian, sample_size < n_samples means take_first_only is True.
                _bayesian_boostrap_statistics(
                    data_expanded[:, :sample_size], statistic, n_resamples, bootstrap_statistics)
            else:
                _bootstrap_statistics(
                    data_expanded, statistic, n_resamples, sample_size,
                    take_first_only, generator, bootstrap_statistics)

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
        take_first_only,
        generator,
        bootstrap_statistics
):
    """Generate a bootstrap distribution of the statistic.

    This modify bootstrap_statistics in place.
    """
    batch, n_samples = data_expanded.shape[:2]

    # Check if we need to sample only between the first bootstrap_sample_size samples.
    if take_first_only:
        max_data_idx = bootstrap_sample_size
    else:
        max_data_idx = n_samples

    # Divide the bootstrap in batches to limit memory usage.
    for k in range(0, n_resamples, batch):
        # The last batch might be smaller.
        batch_actual = min(batch, n_resamples-k)
        if batch_actual != batch:
            data_expanded = data_expanded[:batch_actual]

        # Generate random indices for resampling.
        bootstrap_sample_indices = torch.randint(
            low=0, high=max_data_idx,
            size=(batch_actual, bootstrap_sample_size),
            generator=generator
        )

        # If each sample is multidimensional the indices for gather() need an extra dimension.
        if len(data_expanded.shape) > 2:
            data_dimensionality = data_expanded.shape[2]
            bootstrap_sample_indices = bootstrap_sample_indices.repeat_interleave(
                repeats=data_dimensionality, dim=1).reshape(batch_actual, bootstrap_sample_size, data_dimensionality)

        # Resample.
        bootstrap_samples = torch.gather(
            data_expanded, dim=1, index=bootstrap_sample_indices)

        # Compute statistic for these batch.
        bootstrap_statistics[k:k+batch_actual] = statistic(bootstrap_samples, vectorized=True)

    return bootstrap_statistics


def _bayesian_boostrap_statistics(
        data_expanded,
        statistic,
        n_resamples,
        bootstrap_statistics
):
    """Generate a Bayesian bootstrap distribution of the statistic.

    This modify bootstrap_statistics in place.
    """
    batch, n_samples = data_expanded.shape[:2]

    # Initialize uniform Dirichlet distribution.
    dirichlet = torch.distributions.Dirichlet(torch.ones(n_samples))

    # Divide the bootstrap in batches to limit memory usage.
    for k in range(0, n_resamples, batch):
        # The last batch might be smaller.
        batch_actual = min(batch, n_resamples-k)
        if batch_actual != batch:
            data_expanded = data_expanded[:batch_actual]

        # Generate the weights.
        weights = dirichlet.sample((batch_actual,))

        # Compute the statistic.
        bootstrap_statistics[k:k+batch_actual] = statistic(data_expanded, weights=weights, vectorized=True)
