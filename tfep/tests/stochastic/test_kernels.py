import math

import pytest
import torch

from tfep.stochastic import GaussianRandomWalkKernel, KernelContext, UnadjustedLangevinKernel
from tfep.stochastic.cli import StochasticCLIConfig, add_stochastic_tfep_args


def test_gaussian_random_walk_log_prob_matches_manual_formula():
    torch.set_default_dtype(torch.double)
    kernel = GaussianRandomWalkKernel(sigma=0.5)
    x = torch.zeros(2, 3)
    x_next = torch.tensor([[0.5, 0.0, -0.5], [1.0, 0.5, 0.0]], dtype=torch.double)

    got = kernel.forward_log_prob(x_next, x)
    diff = x_next - x
    variance = 0.25
    manual = -0.5 * (diff.pow(2).sum(dim=1) / variance + 3 * math.log(2 * math.pi * variance))

    assert torch.allclose(got, manual)
    assert torch.allclose(kernel.reverse_log_prob(x, x_next), manual)


def test_gaussian_random_walk_selected_indices_leave_other_coordinates_fixed():
    torch.set_default_dtype(torch.double)
    kernel = GaussianRandomWalkKernel(sigma=0.1, selected_indices=[1, 3])
    x = torch.zeros(4, 5)
    generator = torch.Generator().manual_seed(123)

    x_next, logq, _ = kernel.forward(x, rng=generator)

    assert torch.allclose(x_next[:, [0, 2, 4]], x[:, [0, 2, 4]])
    assert logq.shape == (4,)


def test_gaussian_random_walk_seed_reproducibility():
    torch.set_default_dtype(torch.double)
    kernel = GaussianRandomWalkKernel(sigma=0.25)
    x = torch.zeros(3, 2)

    gen1 = torch.Generator().manual_seed(111)
    gen2 = torch.Generator().manual_seed(111)
    x1, logq1, _ = kernel.forward(x, rng=gen1)
    x2, logq2, _ = kernel.forward(x, rng=gen2)

    assert torch.allclose(x1, x2)
    assert torch.allclose(logq1, logq2)


def test_ula_log_prob_matches_manual_formula():
    torch.set_default_dtype(torch.double)

    def grad_u(x):
        return 2.0 * x

    context = KernelContext(reduced_potential_gradient=grad_u)
    kernel = UnadjustedLangevinKernel(step_size=0.1, diffusion=0.5)
    x = torch.tensor([[1.0, -1.0]], dtype=torch.double)
    x_next = torch.tensor([[0.8, -0.7]], dtype=torch.double)

    got = kernel.forward_log_prob(x_next, x, context=context)
    mean = x - 0.1 * 0.5 * grad_u(x)
    variance = 2 * 0.5 * 0.1
    manual = -0.5 * (((x_next - mean).pow(2).sum(dim=1) / variance) + 2 * math.log(2 * math.pi * variance))

    assert torch.allclose(got, manual)


def test_ula_can_compute_gradient_from_reduced_potential():
    torch.set_default_dtype(torch.double)

    def potential(x):
        return 0.5 * x.pow(2).sum(dim=1)

    context = KernelContext(reduced_potential=potential)
    kernel = UnadjustedLangevinKernel(step_size=0.1)
    x = torch.tensor([[2.0, -3.0]], dtype=torch.double)

    mean_logq = kernel.forward_log_prob(x, x, context=context)
    expected_mean = x - 0.1 * x
    expected = -0.5 * (((x - expected_mean).pow(2).sum(dim=1) / 0.2) + 2 * math.log(2 * math.pi * 0.2))

    assert torch.allclose(mean_logq, expected)


def test_ula_requires_context():
    kernel = UnadjustedLangevinKernel(step_size=0.1)
    with pytest.raises(ValueError, match="requires a KernelContext"):
        kernel.forward_log_prob(torch.zeros(1, 1), torch.zeros(1, 1))


def test_stochastic_cli_defaults_are_deterministic():
    import argparse

    parser = add_stochastic_tfep_args(argparse.ArgumentParser())
    namespace = parser.parse_args([])
    config = StochasticCLIConfig.from_namespace(namespace)

    assert config.estimator == "deterministic-tfep"
    assert config.snf_kernel == "gaussian-rw"
    assert config.snf_output_dir == "stochastic_tfep_outputs"
