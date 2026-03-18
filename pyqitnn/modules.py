from __future__ import annotations

import torch
from torch import nn

from .ops import centered_simplex
from .ops import forward3


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

        kw = {"device": device, "dtype": dtype}
        self.a_neg  = nn.Parameter(torch.empty((self.in_dim, self.out_dim), **kw))
        self.a_zero = nn.Parameter(torch.empty((self.in_dim, self.out_dim), **kw))
        self.a_pos  = nn.Parameter(torch.empty((self.in_dim, self.out_dim), **kw))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.a_neg,  mean=0.0, std=self.init_std)
        nn.init.normal_(self.a_zero, mean=0.0, std=self.init_std)
        nn.init.normal_(self.a_pos,  mean=0.0, std=self.init_std)

    #====================
    # forward passes
    #====================

    def forward_raw(self, inp: torch.Tensor):
        if not inp.is_cuda:
            raise RuntimeError(
                "QITNNLinear requires CUDA tensors. "
                "Move the model and input to a CUDA device first: "
                "model.cuda(); inp = inp.cuda()"
            )
        return forward3(
            inp,
            self.a_neg,
            self.a_zero,
            self.a_pos,
            ent_lambda=self.ent_lambda,
        )

    def forward_visible(self, inp: torch.Tensor):
        u, v, cn, cz, cp = self.forward_raw(inp)
        if self.centered_simplex:
            x, y = centered_simplex(u, v)
            return x, y, cn, cz, cp
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
            f"return_triplet={self.return_triplet}"
        )
