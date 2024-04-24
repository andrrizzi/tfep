#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module tfep.analysis.bootstrap.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import scipy.stats
import pytest
import torch

from tfep.analysis import bootstrap, fep_estimator


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Random number generator. Makes sure tests are reproducible from run to run.
GENERATOR = torch.Generator()
GENERATOR.manual_seed(0)

RANDOM_STATE = np.random.RandomState(0)


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================

_old_default_dtype = None

def setup_module(module):
    global _old_default_dtype
    _old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)


def teardown_module(module):
    torch.set_default_dtype(_old_default_dtype)


# =============================================================================
# UTILS
# =============================================================================

def std(data, vectorized=True):
    return torch.std(data, dim=-1)


def mean(data, weights=None, vectorized=True):
    """Weights must sum to 1 and have the same shape as data."""
    if weights is not None:
        return torch.sum(data * weights, dim=-1)
    return torch.mean(data, dim=-1)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('confidence_level', [0.95, 0.8])
@pytest.mark.parametrize('batch', [None, 100])
@pytest.mark.parametrize('method', ['percentile', 'basic'])
def test_against_scipy(confidence_level, batch, method):
    """Compare our implementation to scipy.stats.bootstrap implementation."""
    n_samples = 100
    n_resamples = 10000

    # Generate distribution.
    data = torch.randn(n_samples)

    # Compute bootstrap with tfep.
    tfep_result = bootstrap(
        data=data,
        statistic=std,
        n_resamples=n_resamples,
        method=method,
    )

    # Compute bootstrap with scipy.
    # bootstrap() expects a 2D array of data.
    scipy_result = scipy.stats.bootstrap(
        data.unsqueeze(0).numpy(),
        np.std,
        vectorized=False,
        n_resamples=n_resamples,
        method=method,
    )

    # Compare confidence intervals.
    assert np.isclose(tfep_result['confidence_interval']['low'],
                      scipy_result.confidence_interval.low, atol=1e-1)
    assert np.isclose(tfep_result['confidence_interval']['high'],
                      scipy_result.confidence_interval.high, atol=1e-1)
    assert np.isclose(tfep_result['standard_deviation'],
                      scipy_result.standard_error, atol=1e-2)


@pytest.mark.parametrize('confidence_level', [0.95, 0.8])
@pytest.mark.parametrize('batch', [None, 100])
@pytest.mark.parametrize('method', ['percentile', 'basic'])
@pytest.mark.parametrize('bayesian', [False, True])
def test_multiple_bootstrap_sample_size(confidence_level, batch, method, bayesian):
    """When bootstrap_sample_size is a list, bootstrap() gives multiple results."""
    n_samples = 1000
    bootstrap_sample_size = [10, 100, 1000]

    # Generate distribution.
    data = torch.randn(n_samples)

    # Compute bootstrap with tfep.
    results = bootstrap(
        data=data,
        statistic=mean,
        bootstrap_sample_size=bootstrap_sample_size,
        # Bayesian bootstrapping supports only take_first_only=True with bootstrap_sample_size
        take_first_only=bayesian,
        method=method,
        bayesian=bayesian,
    )

    # There should be one result for each bootstrap sample size.
    assert len(results) == len(bootstrap_sample_size)

    # If the sample size increase, the confidence interval
    # of the mean should decrease with sample size.
    ci_interval = [res['confidence_interval']['high'] - res['confidence_interval']['low']
                   for res in results]
    for i in range(1, len(ci_interval)):
        assert ci_interval[i-1] > ci_interval[i]


@pytest.mark.parametrize('bayesian', [False, True])
def test_multiple_inputs(bayesian):
    """bootstrap() handles statistics with multiple input arguments."""

    def _statistic(_data, weights=None, vectorized=True):
        """Sum of 3 numbers."""
        s = torch.sum(_data, dim=-1)
        if weights is not None:
            return torch.sum(s * weights, dim=-1)
        return torch.mean(s, dim=-1)

    # The triplets of inputs always sum to 5.
    data = torch.tensor([
        [0, 3, 2],
        [1, 4, 0],
        [3, 1, 1],
        [5, 0, 0],
    ], dtype=float)

    result = bootstrap(
        data=data,
        statistic=_statistic,
        n_resamples=100,
        method='percentile',
        bayesian=bayesian,
    )

    # The statistic should be constant and equal 5.
    assert np.isclose(result['confidence_interval']['low'], 5)
    assert np.isclose(result['confidence_interval']['high'], 5)
    assert np.isclose(result['standard_deviation'], 0)
    assert np.isclose(result['mean'], 5)
    assert np.isclose(result['median'], 5)


@pytest.mark.parametrize('bayesian', [False, True])
def test_fep_estimator(bayesian):
    """Test compatibility with fep_estimator."""
    # Work values normally distributed around 0.0 should yield free energy ~0.5.
    work = torch.randn(10000)
    result = bootstrap(
        data=work,
        statistic=fep_estimator,
        n_resamples=10000,
        method='percentile',
        bayesian=bayesian,
    )
    assert np.isclose(result['mean'], -0.5, rtol=0.0, atol=1e-1)


def test_error_bayesian_generator():
    """An error is raised if a generator is set with Bayesian bootstrapping."""
    with pytest.raises(ValueError, match='generator'):
        bootstrap(data=torch.randn(1000), statistic=mean, bayesian=True, generator=GENERATOR)


def test_error_bayesian_take_first_only():
    """With Bayesian bootstrapping, bootstrap_sample_size can be used only if take_first_only=True."""
    with pytest.raises(ValueError, match='bootstrap_sample_size'):
        bootstrap(data=torch.randn(1000), statistic=mean, bootstrap_sample_size=[10], bayesian=True)
