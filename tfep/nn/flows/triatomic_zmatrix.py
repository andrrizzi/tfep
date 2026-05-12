"""Triatomic internal-coordinate (Z-matrix) flow.

This is designed for *gas-phase* triatomics (e.g., O3).

Input/output:
- positions as flattened Cartesian coordinates (B, 9) corresponding to 3 atoms.
- returns (y, log_det_J)

The map preserves translation + orientation by factoring:
  x <-> (t, R, q)  where q=(r_a, r_b, theta)
Then it learns a bijection only in q (in an unconstrained z-space).

Jacobian correction:
  log|det dy/dx| = log|det dq'/dq| + log J_x(q') - log J_x(q)
where for a triatomic:
  J_x(q) ∝ r_a^2 r_b^2 sin(theta)

This substantially improves learning compared to pure Cartesian flows for triatomics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


def _safe_norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(v, dim=-1).clamp_min(eps)


def _logit(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


class _MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, n_hidden: int = 2):
        super().__init__()
        layers = []
        d = int(in_dim)
        for _ in range(int(n_hidden)):
            layers.append(torch.nn.Linear(d, int(hidden_dim)))
            layers.append(torch.nn.ReLU())
            d = int(hidden_dim)
        layers.append(torch.nn.Linear(d, int(out_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AffineCoupling(torch.nn.Module):
    """Simple affine coupling layer with a binary mask."""

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        scale: float = 0.8,
    ):
        super().__init__()
        dim = int(dim)
        mask = mask.to(dtype=torch.float32)
        if mask.ndim != 1 or mask.numel() != dim:
            raise ValueError("mask must be shape (dim,)")
        self.register_buffer("mask", mask)
        self.dim = dim
        self.scale = float(scale)
        self.cond = _MLP(dim, 2 * dim, hidden_dim=int(hidden_dim), n_hidden=int(n_hidden))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.mask
        x_m = x * m
        st = self.cond(x_m)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * self.scale
        y = x_m + (1.0 - m) * (x * torch.exp(s) + t)
        log_det = ((1.0 - m) * s).sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.mask
        y_m = y * m
        st = self.cond(y_m)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * self.scale
        x = y_m + (1.0 - m) * ((y - t) * torch.exp(-s))
        log_det = -((1.0 - m) * s).sum(dim=-1)
        return x, log_det


class VectorCouplingFlow(torch.nn.Module):
    """A small invertible flow in R^dim based on stacked affine couplings."""

    def __init__(
        self,
        dim: int = 3,
        n_layers: int = 6,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        scale: float = 0.8,
    ):
        super().__init__()
        self.dim = int(dim)
        n_layers = int(n_layers)
        if self.dim < 2:
            raise ValueError("dim must be >= 2")

        # For dim=3 we cycle 3 different masks.
        masks = [
            torch.tensor([1, 1, 0], dtype=torch.float32),
            torch.tensor([0, 1, 1], dtype=torch.float32),
            torch.tensor([1, 0, 1], dtype=torch.float32),
        ]
        self.layers = torch.nn.ModuleList(
            [
                _AffineCoupling(self.dim, masks[i % len(masks)], hidden_dim=hidden_dim, n_hidden=n_hidden, scale=scale)
                for i in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        y = x
        for layer in self.layers:
            y, ld = layer(y)
            log_det = log_det + ld
        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        x = y
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            log_det = log_det + ld
        return x, log_det


class TriatomicZMatrixFlow(torch.nn.Module):
    """Internal-coordinate flow for a triatomic.

    Parameters
    ----------
    z_flow
        Invertible flow in unconstrained z-space (R^3).
    origin_idx, axis_idx, plane_idx
        Atom indices (0..2) defining the internal-coordinate frame.
    """

    def __init__(
        self,
        z_flow: torch.nn.Module,
        origin_idx: int = 1,
        axis_idx: int = 0,
        plane_idx: int = 2,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.z_flow = z_flow
        self.origin_idx = int(origin_idx)
        self.axis_idx = int(axis_idx)
        self.plane_idx = int(plane_idx)
        self.eps = float(eps)

    def _as_B33(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if x.ndim == 2 and x.shape[1] == 9:
            return x.reshape(x.shape[0], 3, 3), True
        if x.ndim == 3 and x.shape[1:] == (3, 3):
            return x, False
        raise ValueError(
            f"TriatomicZMatrixFlow expects (B,9) or (B,3,3); got {tuple(x.shape)}. "
            "Note: for this flow you must map exactly 3 atoms (9 DOFs)."
        )

    def _restore_shape(self, x_b33: torch.Tensor, flat: bool) -> torch.Tensor:
        return x_b33.reshape(x_b33.shape[0], 9) if flat else x_b33

    def _decompose(self, x: torch.Tensor):
        o = x[:, self.origin_idx, :]
        a = x[:, self.axis_idx, :]
        p = x[:, self.plane_idx, :]

        v1 = a - o
        e1 = v1 / _safe_norm(v1, eps=self.eps).unsqueeze(-1)

        v2 = p - o
        v2p = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
        e2 = v2p / _safe_norm(v2p, eps=self.eps).unsqueeze(-1)

        e3 = torch.cross(e1, e2, dim=-1)
        e3 = e3 / _safe_norm(e3, eps=self.eps).unsqueeze(-1)

        R = torch.stack([e1, e2, e3], dim=-1)  # columns

        x_centered = x - o[:, None, :]
        x_can = torch.einsum("bij,bnj->bni", R.transpose(1, 2), x_centered)

        va = x_can[:, self.axis_idx, :]
        vb = x_can[:, self.plane_idx, :]
        r_a = _safe_norm(va, eps=self.eps)
        r_b = _safe_norm(vb, eps=self.eps)
        cos_th = (va * vb).sum(dim=-1) / (r_a * r_b).clamp_min(self.eps)
        cos_th = cos_th.clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.arccos(cos_th)
        theta = theta.clamp(self.eps, float(np.pi) - self.eps)

        return o, R, r_a, r_b, theta

    def _q_to_z(self, r_a: torch.Tensor, r_b: torch.Tensor, theta: torch.Tensor):
        r_a = r_a.clamp_min(self.eps)
        r_b = r_b.clamp_min(self.eps)
        theta = theta.clamp(self.eps, float(np.pi) - self.eps)

        z0 = torch.log(r_a)
        z1 = torch.log(r_b)
        s = (theta / float(np.pi)).clamp(self.eps, 1.0 - self.eps)
        z2 = _logit(s, eps=self.eps)

        log_det = -torch.log(r_a) - torch.log(r_b)
        log_det = log_det - torch.log(torch.tensor(float(np.pi), device=theta.device, dtype=theta.dtype))
        log_det = log_det - torch.log(s) - torch.log1p(-s)

        z = torch.stack([z0, z1, z2], dim=-1)
        return z, log_det

    def _z_to_q(self, z: torch.Tensor):
        z0, z1, z2 = z.unbind(dim=-1)
        r_a = torch.exp(z0).clamp_min(self.eps)
        r_b = torch.exp(z1).clamp_min(self.eps)
        s = torch.sigmoid(z2).clamp(self.eps, 1.0 - self.eps)
        theta = (float(np.pi) * s).clamp(self.eps, float(np.pi) - self.eps)

        log_det = torch.log(r_a) + torch.log(r_b)
        log_det = log_det + torch.log(torch.tensor(float(np.pi), device=z.device, dtype=z.dtype))
        log_det = log_det + torch.log(s) + torch.log1p(-s)

        return r_a, r_b, theta, log_det

    def _logJ(self, r_a: torch.Tensor, r_b: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        r_a = r_a.clamp_min(self.eps)
        r_b = r_b.clamp_min(self.eps)
        theta = theta.clamp(self.eps, float(np.pi) - self.eps)
        s = torch.sin(theta).clamp_min(self.eps)
        return 2.0 * torch.log(r_a) + 2.0 * torch.log(r_b) + torch.log(s)

    def _reconstruct(self, t: torch.Tensor, R: torch.Tensor, r_a: torch.Tensor, r_b: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        device, dtype = t.device, t.dtype

        x_can = torch.zeros((B, 3, 3), device=device, dtype=dtype)
        x_can[:, self.axis_idx, 0] = r_a
        x_can[:, self.plane_idx, 0] = r_b * torch.cos(theta)
        x_can[:, self.plane_idx, 1] = r_b * torch.sin(theta)

        x_lab = torch.einsum("bij,bnj->bni", R, x_can) + t[:, None, :]
        return x_lab

    def forward(self, x: torch.Tensor):
        x_b33, flat = self._as_B33(x)
        t, R, r_a, r_b, theta = self._decompose(x_b33)
        z, log_det_z_from_q = self._q_to_z(r_a, r_b, theta)

        z2, log_det_zflow = self.z_flow(z)
        r_a2, r_b2, theta2, log_det_q_from_z = self._z_to_q(z2)

        logJ1 = self._logJ(r_a, r_b, theta)
        logJ2 = self._logJ(r_a2, r_b2, theta2)

        log_det = log_det_q_from_z + log_det_zflow + log_det_z_from_q + (logJ2 - logJ1)
        y_b33 = self._reconstruct(t, R, r_a2, r_b2, theta2)
        y = self._restore_shape(y_b33, flat)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        y_b33, flat = self._as_B33(y)
        t, R, r_a2, r_b2, theta2 = self._decompose(y_b33)
        z2, log_det_z_from_q_2 = self._q_to_z(r_a2, r_b2, theta2)

        z, log_det_zflow_inv = self.z_flow.inverse(z2)
        r_a, r_b, theta, log_det_q_from_z_1 = self._z_to_q(z)

        logJ2 = self._logJ(r_a2, r_b2, theta2)
        logJ1 = self._logJ(r_a, r_b, theta)

        log_det = -log_det_q_from_z_1 + log_det_zflow_inv - log_det_z_from_q_2 + (logJ1 - logJ2)
        x_b33 = self._reconstruct(t, R, r_a, r_b, theta)
        x = self._restore_shape(x_b33, flat)
        return x, log_det
