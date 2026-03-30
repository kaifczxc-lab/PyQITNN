from __future__ import annotations

import torch
from torch import nn

from .ops import centered_simplex
from .ops import forward3
from .ops import forward3_packed_reference
from .precision import normalize_precision_mode as _normalize_precision_mode
from .precision import resolve_precision_mode as _resolve_precision_mode_args


#====================
# QITNNLinear: ternary Born-rule linear layer
#====================

class QITNNLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        ent_lambda: float = 0.0,
        init_std: float = 0.02,
        pack_output: bool = True,
        centered_simplex: bool = True,
        return_triplet: bool = False,
        mixed_precision: bool | None = None,
        precision_mode: str | None = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.ent_lambda = float(ent_lambda)
        self.init_std = float(init_std)
        self.pack_output = bool(pack_output)
        self.centered_simplex = bool(centered_simplex)
        self.return_triplet = bool(return_triplet)
        self.precision_mode, self.mixed_precision = _resolve_precision_mode_args(
            precision_mode,
            mixed_precision,
        )
        self._runtime_backend = "packed_reference"

        kw = {"device": device, "dtype": dtype}
        self.a_neg  = nn.Parameter(torch.empty((self.in_dim, self.out_dim), **kw))
        self.a_zero = nn.Parameter(torch.empty((self.in_dim, self.out_dim), **kw))
        self.a_pos  = nn.Parameter(torch.empty((self.in_dim, self.out_dim), **kw))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.a_neg,  mean=0.0, std=self.init_std)
        nn.init.normal_(self.a_zero, mean=0.0, std=self.init_std)
        nn.init.normal_(self.a_pos,  mean=0.0, std=self.init_std)

    def _packed_weight_view(self) -> torch.Tensor:
        # runtime-only ternary layout; public storage stays split for state_dict and diagnostics
        return torch.stack((self.a_neg, self.a_zero, self.a_pos), dim=-1).contiguous()

    def _set_runtime_backend(self, backend: str) -> None:
        normalized = str(backend).strip().lower()
        if normalized not in ("split", "packed_reference"):
            raise RuntimeError("backend must be 'split' or 'packed_reference'")
        self._runtime_backend = normalized

    #====================
    # forward passes
    #====================

    def _mixed_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def _maybe_cast_activation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.mixed_precision or not x.is_cuda or x.dtype != torch.float32:
            return x
        return x.to(dtype=self._mixed_dtype())

    def forward_raw(self, inp: torch.Tensor):
        if not inp.is_cuda:
            raise RuntimeError(
                "QITNNLinear requires CUDA tensors. "
                "Move the model and input to a CUDA device first: "
                "model.cuda(); inp = inp.cuda()"
            )
        if self.mixed_precision and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"precision_mode='{self.precision_mode}' currently targets CUDA bf16 visible activations. "
                "This GPU does not report bf16 support. "
                "Use precision_mode='fp32' to keep the trusted fp32 path."
            )
        if self.a_neg.dtype != torch.float32 or self.a_zero.dtype != torch.float32 or self.a_pos.dtype != torch.float32:
            raise RuntimeError(
                f"precision_mode='{self.precision_mode}' expects fp32 master weights. "
                "Do not call .half() or .bfloat16() on QITNN parameters."
            )
        inp = self._maybe_cast_activation(inp)
        if self._runtime_backend == "split":
            return forward3(
                inp,
                self.a_neg,
                self.a_zero,
                self.a_pos,
                ent_lambda=self.ent_lambda,
                mixed_precision=self.mixed_precision,
            )
        if self._runtime_backend == "packed_reference":
            return forward3_packed_reference(
                inp,
                self._packed_weight_view(),
                ent_lambda=self.ent_lambda,
                mixed_precision=self.mixed_precision,
            )
        raise RuntimeError(f"unsupported runtime backend: {self._runtime_backend}")

    def forward_visible(self, inp: torch.Tensor):
        u, v, cn, cz, cp = self.forward_raw(inp)
        if self.centered_simplex:
            x, y = centered_simplex(u, v, mixed_precision=self.mixed_precision)
            x = self._maybe_cast_activation(x)
            y = self._maybe_cast_activation(y)
            return x, y, cn, cz, cp
        u = self._maybe_cast_activation(u)
        v = self._maybe_cast_activation(v)
        return u, v, cn, cz, cp

    def forward(self, inp: torch.Tensor):
        left, right, cn, cz, cp = self.forward_visible(inp)

        if self.return_triplet:
            if self.pack_output:
                return torch.cat([left, right], dim=-1), (cn, cz, cp)
            return (left, right), (cn, cz, cp)

        if self.pack_output:
            return torch.cat([left, right], dim=-1)
        return left, right

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"ent_lambda={self.ent_lambda}, init_std={self.init_std}, "
            f"pack_output={self.pack_output}, centered_simplex={self.centered_simplex}, "
            f"return_triplet={self.return_triplet}, precision_mode={self.precision_mode}, "
            f"mixed_precision={self.mixed_precision}"
        )
