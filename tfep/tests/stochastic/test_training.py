import pytest
import torch

from tfep.stochastic.training import (
    StochasticTrainingConfig,
    compute_bidirectional_stochastic_training_works,
)


class ToyBidirectionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor([[0.15, -0.05, 0.10]], dtype=torch.double))
        self._kT = torch.ones((), dtype=torch.double)

    @property
    def device(self):
        return self.shift.device

    def get_mapped_indices(self, idx_type, remove_fixed):
        assert idx_type == "atom"
        assert remove_fixed is False
        return torch.tensor([0], dtype=torch.long)

    def forward(self, batch):
        x = batch["positions"]
        y = x + self.shift
        return {"positions": y, "log_det_J": torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)}

    def inverse(self, batch):
        x = batch["positions"]
        y = x - self.shift
        return {"positions": y, "log_det_J": torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)}

    def _eval_potential(self, state, positions, dimensions):
        center = 0.0 if int(state) == 0 else 0.4
        return 0.5 * (positions - center).pow(2).sum(dim=1)


def _batches():
    torch.manual_seed(11)
    x0 = torch.randn(5, 3, dtype=torch.double) * 0.2
    x1 = 0.4 + torch.randn(5, 3, dtype=torch.double) * 0.2
    return {"positions": x0}, {"positions": x1}


def test_stochastic_training_none_kernel_matches_deterministic_tfep_work():
    model = ToyBidirectionalModel()
    b0, b1 = _batches()
    mapped01 = model.forward(b0)
    mapped10 = model.inverse(b1)
    u0_x0 = model._eval_potential(0, b0["positions"], None)
    u1_x1 = model._eval_potential(1, b1["positions"], None)
    u1_y0 = model._eval_potential(1, mapped01["positions"], None)
    u0_y1 = model._eval_potential(0, mapped10["positions"], None)

    works = compute_bidirectional_stochastic_training_works(
        model=model,
        batch0=b0,
        batch1=b1,
        mapped01=mapped01,
        u0_x0=u0_x0,
        u1_x1=u1_x1,
        config=StochasticTrainingConfig(enabled=True, kernel="none"),
    )

    assert torch.allclose(works.w01, u1_y0 - u0_x0)
    assert torch.allclose(works.w10, u0_y1 - u1_x1)
    assert torch.allclose(works.sum_logq_forward_01, torch.zeros_like(works.w01))
    assert torch.allclose(works.sum_logq_reverse_10, torch.zeros_like(works.w10))


def test_stochastic_training_ula_requires_explicit_constrained_acknowledgement():
    cfg = StochasticTrainingConfig(enabled=True, kernel="ula", step_size=1e-4)
    with pytest.raises(RuntimeError, match="constrained/PBC"):
        cfg.validate()


def test_stop_gradient_ula_training_has_finite_first_order_gradients():
    torch.manual_seed(12)
    model = ToyBidirectionalModel()
    b0, b1 = _batches()
    mapped01 = model.forward(b0)
    u0_x0 = model._eval_potential(0, b0["positions"], None)
    u1_x1 = model._eval_potential(1, b1["positions"], None)
    cfg = StochasticTrainingConfig(
        enabled=True,
        kernel="ula",
        step_size=1e-4,
        diffusion=1.0,
        train_mc_samples=2,
        gradient_policy="stop-gradient",
        allow_constrained_cartesian_ula=True,
    )

    works = compute_bidirectional_stochastic_training_works(
        model=model,
        batch0=b0,
        batch1=b1,
        mapped01=mapped01,
        u0_x0=u0_x0,
        u1_x1=u1_x1,
        config=cfg,
    )
    loss = works.w01.mean() + works.w10.mean()
    loss.backward()

    assert torch.all(torch.isfinite(works.w01))
    assert torch.all(torch.isfinite(works.w10))
    assert model.shift.grad is not None
    assert torch.all(torch.isfinite(model.shift.grad))
