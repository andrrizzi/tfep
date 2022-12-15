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

from tfep.analysis import bootstrap


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


def mean(data, vectorized=True):
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
                      scipy_result.confidence_interval.low, atol=1e-2)
    assert np.isclose(tfep_result['confidence_interval']['high'],
                      scipy_result.confidence_interval.high, atol=1e-2)
    assert np.isclose(tfep_result['standard_deviation'],
                      scipy_result.standard_error, atol=1e-2)


@pytest.mark.parametrize('confidence_level', [0.95, 0.8])
@pytest.mark.parametrize('batch', [None, 100])
@pytest.mark.parametrize('method', ['percentile', 'basic'])
def test_multiple_bootstrap_sample_size(confidence_level, batch, method):
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
        method=method,
    )

    # There should be one result for each bootstrap sample size.
    assert len(results) == len(bootstrap_sample_size)

    # If the sample size increase, the confidence interval
    # of the mean should decrease with sample size.
    ci_interval = [res['confidence_interval']['high'] - res['confidence_interval']['low']
                   for res in results]
    for i in range(1, len(ci_interval)):
        assert ci_interval[i-1] > ci_interval[i]


def test_multiple_inputs():
    """bootstrap() handles statistics with multiple input arguments."""

    def _statistic(_data, vectorized=True):
        """Sum of 3 numbers."""
        s = torch.sum(_data, dim=-1)
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
    )

    # The statistic should be constant and equal 5.
    assert result['confidence_interval']['low'] == 5
    assert result['confidence_interval']['high'] == 5
    assert result['standard_deviation'] == 0
    assert result['mean'] == 5
    assert result['median'] == 5
