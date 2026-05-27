import math

import torch

from tfep.stochastic import (
    AffineFlowBlock,
    GaussianRandomWalkKernel,
    IdentityFlowBlock,
    KernelContext,
    PathWeightedTFEP,
    StochasticFlowBlock,
    UnadjustedLangevinKernel,
    bar_delta_f,
    compute_path_work,
    jarzynski_delta_f_forward,
)


def harmonic_potential(mu: float, k: float):
    def _potential(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * k * (x - mu).pow(2).sum(dim=1)

    return _potential


def sample_harmonic(batch_size: int, mu: float, k: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return mu + torch.randn((batch_size, 1), generator=generator, dtype=torch.double) / math.sqrt(k)


def test_compute_path_work_deterministic_formula():
    torch.set_default_dtype(torch.double)
    u_source = torch.tensor([1.0, 2.0], dtype=torch.double)
    u_target = torch.tensor([3.0, 4.0], dtype=torch.double)
    logj = torch.tensor([0.5, -0.25], dtype=torch.double)
    zeros = torch.zeros(2, dtype=torch.double)

    got = compute_path_work(u_source, u_target, logj, zeros, zeros)

    assert torch.allclose(got, u_target - u_source - logj)


def test_identity_flow_no_stochasticity_reduces_to_fep():
    torch.set_default_dtype(torch.double)
    x = torch.tensor([[-1.0], [0.0], [2.0]], dtype=torch.double)
    u_a = harmonic_potential(0.0, 1.0)
    u_b = harmonic_potential(0.3, 1.4)
    estimator = PathWeightedTFEP(
        blocks=[StochasticFlowBlock(IdentityFlowBlock(), kernel=None)],
        source_potential=u_a,
        target_potential=u_b,
    )

    path = estimator.sample_forward(x)

    assert torch.allclose(path.path_work, u_b(x) - u_a(x))
    assert torch.allclose(path.sum_logJ, torch.zeros(x.shape[0], dtype=x.dtype))
    assert torch.allclose(path.sum_logq_forward, torch.zeros(x.shape[0], dtype=x.dtype))
    assert torch.allclose(path.sum_logq_reverse, torch.zeros(x.shape[0], dtype=x.dtype))


def test_affine_deterministic_limit_matches_tfep_formula():
    torch.set_default_dtype(torch.double)
    x = torch.tensor([[-1.0, 0.5], [0.2, 1.1]], dtype=torch.double)
    flow = AffineFlowBlock(scale=torch.tensor([2.0, 0.5], dtype=torch.double), shift=torch.tensor([0.1, -0.2], dtype=torch.double))
    block = StochasticFlowBlock(flow, kernel=None)
    u_a = lambda z: 0.5 * z.pow(2).sum(dim=1)
    u_b = lambda z: 0.25 * (z - 0.3).pow(2).sum(dim=1)
    estimator = PathWeightedTFEP([block], u_a, u_b)

    path = estimator.sample_forward(x)
    y, logj = flow.forward(x)
    expected = u_b(y) - u_a(x) - logj

    assert torch.allclose(path.xK, y)
    assert torch.allclose(path.path_work, expected)


def test_forward_and_reverse_paths_have_existing_bar_sign_convention():
    torch.set_default_dtype(torch.double)
    n = 6000
    u_a = harmonic_potential(0.0, 1.0)
    u_b = harmonic_potential(0.25, 1.0)
    x_a = sample_harmonic(n, 0.0, 1.0, seed=1)
    x_b = sample_harmonic(n, 0.25, 1.0, seed=2)
    estimator = PathWeightedTFEP([StochasticFlowBlock(IdentityFlowBlock(), kernel=None)], u_a, u_b)

    forward = estimator.sample_forward(x_a)
    reverse = estimator.sample_reverse(x_b)
    df = bar_delta_f(forward.path_work, reverse.path_work)

    assert abs(float(df)) < 0.05


def test_gaussian_random_walk_jarzynski_recovers_harmonic_free_energy():
    torch.set_default_dtype(torch.double)
    n = 30000
    k_a = 1.0
    k_b = 1.5
    exact_df = 0.5 * math.log(k_b / k_a)
    u_a = harmonic_potential(0.0, k_a)
    u_b = harmonic_potential(0.2, k_b)
    x_a = sample_harmonic(n, 0.0, k_a, seed=3)
    kernel = GaussianRandomWalkKernel(sigma=0.15)
    estimator = PathWeightedTFEP([StochasticFlowBlock(IdentityFlowBlock(), kernel=kernel)], u_a, u_b)

    forward = estimator.sample_forward(x_a, rng=torch.Generator().manual_seed(4))
    df = jarzynski_delta_f_forward(forward.path_work)

    assert abs(float(df) - exact_df) < 0.05


def test_reverse_path_stores_logq_terms_relative_to_generated_direction():
    torch.set_default_dtype(torch.double)
    kernel = GaussianRandomWalkKernel(sigma=0.2)
    u = harmonic_potential(0.0, 1.0)
    estimator = PathWeightedTFEP([StochasticFlowBlock(IdentityFlowBlock(), kernel=kernel)], u, u)
    x = torch.zeros(5, 1, dtype=torch.double)

    path = estimator.sample_reverse(x, rng=torch.Generator().manual_seed(8))

    assert path.direction == "B_to_A"
    assert torch.allclose(path.path_work, path.sum_logq_forward - path.sum_logq_reverse + path.u_target_xK - path.u_source_x0)


def test_ula_toy_path_terms_are_finite():
    torch.set_default_dtype(torch.double)
    u = harmonic_potential(0.0, 1.0)
    context = KernelContext(reduced_potential=u)
    kernel = UnadjustedLangevinKernel(step_size=0.01)
    estimator = PathWeightedTFEP(
        [StochasticFlowBlock(IdentityFlowBlock(), kernel=kernel)],
        source_potential=u,
        target_potential=u,
        kernel_contexts=[context],
    )
    x = sample_harmonic(128, 0.0, 1.0, seed=5)

    path = estimator.sample_forward(x, rng=torch.Generator().manual_seed(6))

    assert torch.all(torch.isfinite(path.path_work))
    assert torch.all(torch.isfinite(path.sum_logq_forward))
    assert torch.all(torch.isfinite(path.sum_logq_reverse))
