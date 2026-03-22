from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from .diagnostics import render_qitnn_diag
from .modules import QITNNLinear
from .modules import _resolve_precision_mode_args
from .ops import attention2
from .ops import prior_


#====================
# shape helpers
#====================

def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.size(-1)).contiguous()


def _unflatten(x: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
    return x.reshape(batch, seq, x.size(-1))


def _runtime_mixed_dtype() -> torch.dtype:
    return torch.bfloat16


def _maybe_cast_activation(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled or not x.is_cuda or x.dtype != torch.float32:
        return x
    return x.to(dtype=_runtime_mixed_dtype())


#====================
# simplex gelu: gelu(x) | y  (NOT glu gating)
#====================

def simplex_gelu(xy: torch.Tensor) -> torch.Tensor:
    half = xy.size(-1) // 2
    x = F.gelu(xy[..., :half])
    y = xy[..., half:]
    return torch.cat([x, y], dim=-1)


#====================
# transformer block
#====================

class QITNNSimplexBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        proj_dim: int,
        ffn_dim: int,
        *,
        ent_lambda_qk: float = 0.0,
        ent_lambda_vo: float = 0.0,
        ent_lambda_ff: float = 0.0,
        init_std: float = 0.02,
        mixed_precision: bool | None = None,
        precision_mode: str | None = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.proj_dim = int(proj_dim)
        self.ffn_dim = int(ffn_dim)
        self.visible_dim = self.proj_dim * 2
        self.ff_visible_dim = self.ffn_dim * 2
        self.precision_mode, self.mixed_precision = _resolve_precision_mode_args(
            precision_mode,
            mixed_precision,
        )

        kw = {"device": device, "dtype": dtype}

        # attention
        self.ln1 = nn.LayerNorm(self.hidden_dim, **kw)
        self.q_proj = QITNNLinear(self.hidden_dim, self.proj_dim, ent_lambda=ent_lambda_qk, init_std=init_std, precision_mode=self.precision_mode, device=device, dtype=dtype)
        self.k_proj = QITNNLinear(self.hidden_dim, self.proj_dim, ent_lambda=ent_lambda_qk, init_std=init_std, precision_mode=self.precision_mode, device=device, dtype=dtype)
        self.v_proj = QITNNLinear(self.hidden_dim, self.proj_dim, ent_lambda=ent_lambda_vo, init_std=init_std, precision_mode=self.precision_mode, device=device, dtype=dtype)
        self.o_proj = QITNNLinear(self.visible_dim, self.proj_dim, ent_lambda=ent_lambda_vo, init_std=init_std, precision_mode=self.precision_mode, device=device, dtype=dtype)

        # feedforward
        self.ln2 = nn.LayerNorm(self.hidden_dim, **kw)
        self.ff1 = QITNNLinear(self.hidden_dim, self.ffn_dim, ent_lambda=ent_lambda_ff, init_std=init_std, precision_mode=self.precision_mode, device=device, dtype=dtype)
        self.ff2 = QITNNLinear(self.ff_visible_dim, self.proj_dim, ent_lambda=ent_lambda_ff, init_std=init_std, precision_mode=self.precision_mode, device=device, dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden.shape
        hidden = _maybe_cast_activation(hidden, self.mixed_precision)

        # self-attention
        norm = self.ln1(hidden)
        q = _unflatten(self.q_proj(_flatten(norm)), B, S)
        k = _unflatten(self.k_proj(_flatten(norm)), B, S)
        v = _unflatten(self.v_proj(_flatten(norm)), B, S)
        attn = attention2(q.contiguous(), k.contiguous(), v.contiguous(), mixed_precision=self.mixed_precision)
        hidden = hidden + _unflatten(self.o_proj(_flatten(attn)), B, S)
        hidden = _maybe_cast_activation(hidden, self.mixed_precision)

        # feedforward
        ff_in = self.ln2(hidden)
        ff_mid = _unflatten(self.ff1(_flatten(ff_in)), B, S)
        ff_mid = simplex_gelu(ff_mid)
        hidden = hidden + _unflatten(self.ff2(_flatten(ff_mid)), B, S)
        hidden = _maybe_cast_activation(hidden, self.mixed_precision)
        return hidden

    def iter_qitnn_layers(self) -> Iterator[tuple[str, str, QITNNLinear]]:
        yield "q_proj", "qk", self.q_proj
        yield "k_proj", "qk", self.k_proj
        yield "v_proj", "vo", self.v_proj
        yield "o_proj", "vo", self.o_proj
        yield "ff1",    "ff", self.ff1
        yield "ff2",    "ff", self.ff2


#====================
# language model
#====================

class QITNNSimplexTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 256,
        dim: int = 64,
        ffn_dim: int = 128,
        seq_len: int = 128,
        layers: int = 2,
        ent_lambda_qk: float = 0.0,
        ent_lambda_vo: float = 0.0,
        ent_lambda_ff: float = 0.0,
        init_std: float = 0.02,
        mixed_precision: bool | None = None,
        precision_mode: str | None = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.ffn_dim = int(ffn_dim)
        self.seq_len = int(seq_len)
        self.layers = int(layers)
        self.hidden_dim = self.dim * 2
        self.precision_mode, self.mixed_precision = _resolve_precision_mode_args(
            precision_mode,
            mixed_precision,
        )

        kw = {"device": device, "dtype": dtype}

        self.token_emb = nn.Embedding(self.vocab_size, self.hidden_dim, **kw)
        self.pos_emb = nn.Parameter(torch.empty(self.seq_len, self.hidden_dim, **kw))
        self.blocks = nn.ModuleList([
            QITNNSimplexBlock(
                self.hidden_dim, self.dim, self.ffn_dim,
                ent_lambda_qk=ent_lambda_qk,
                ent_lambda_vo=ent_lambda_vo,
                ent_lambda_ff=ent_lambda_ff,
                init_std=init_std,
                precision_mode=self.precision_mode,
                device=device, dtype=dtype,
            )
            for _ in range(self.layers)
        ])
        self.ln_f = nn.LayerNorm(self.hidden_dim, **kw)
        self.head = nn.Linear(self.hidden_dim, self.vocab_size, bias=True, **kw)

        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.pos_emb, mean=0.0, std=init_std)
        nn.init.normal_(self.head.weight, mean=0.0, std=init_std)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    #====================
    # forward
    #====================

    def forward(self, tokens: torch.Tensor, *, targets: torch.Tensor | None = None):
        if tokens.dim() != 2:
            raise RuntimeError("tokens must be [batch, seq]")

        B, S = tokens.shape
        if S > self.seq_len:
            raise RuntimeError("sequence is longer than configured seq_len")
        if self.mixed_precision:
            if tokens.is_cuda and not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    f"precision_mode='{self.precision_mode}' currently targets CUDA bf16 visible activations. "
                    "This GPU does not report bf16 support. "
                    "Use precision_mode='fp32' to keep the trusted fp32 path."
                )
            if self.token_emb.weight.dtype != torch.float32 or self.pos_emb.dtype != torch.float32:
                raise RuntimeError(
                    "mixed_precision expects fp32 master weights. "
                    "Do not call .half() or .bfloat16() on the model."
                )
            if self.head.weight.dtype != torch.float32 or (self.head.bias is not None and self.head.bias.dtype != torch.float32):
                raise RuntimeError(
                    "mixed_precision expects fp32 master weights. "
                    "Do not call .half() or .bfloat16() on the model."
                )

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
            if tokens.is_cuda and self.mixed_precision
            else (
                torch.amp.autocast(device_type="cuda", enabled=False)
                if tokens.is_cuda
                else nullcontext()
            )
        )
        with amp_ctx:
            hidden = self.token_emb(tokens) + self.pos_emb[:S].unsqueeze(0)
            hidden = _maybe_cast_activation(hidden, self.mixed_precision)
            for block in self.blocks:
                hidden = block(hidden)

            logits = self.head(self.ln_f(hidden))

            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    #====================
    # QTS layer iteration
    #====================

    def iter_qitnn_layers(self) -> Iterator[tuple[str, str, QITNNLinear]]:
        for idx, block in enumerate(self.blocks):
            prefix = f"blocks.{idx}"
            for name, role, layer in block.iter_qitnn_layers():
                yield f"{prefix}.{name}", role, layer

    #====================
    # optimizer param groups (SGD path)
    #====================

    def make_qitnn_optimizer_groups(
        self,
        *,
        lr: float,
        zero_boost_qk: float = 1.0,
        zero_boost_vo: float = 1.0,
        zero_boost_ff: float = 1.0,
        weight_decay: float = 0.0,
    ) -> list[dict]:
        main_params: list[torch.nn.Parameter] = []
        qk_zero: list[torch.nn.Parameter] = []
        vo_zero: list[torch.nn.Parameter] = []
        ff_zero: list[torch.nn.Parameter] = []
        seen: set[int] = set()

        for _, role, layer in self.iter_qitnn_layers():
            main_params.append(layer.a_neg)
            main_params.append(layer.a_pos)
            seen.add(id(layer.a_neg))
            seen.add(id(layer.a_zero))
            seen.add(id(layer.a_pos))

            if role == "qk":
                qk_zero.append(layer.a_zero)
            elif role == "vo":
                vo_zero.append(layer.a_zero)
            else:
                ff_zero.append(layer.a_zero)

        for param in self.parameters():
            if id(param) not in seen:
                main_params.append(param)

        groups = [
            {"params": main_params, "lr": lr,                    "lr_scale": 1.0,             "weight_decay": weight_decay},
            {"params": qk_zero,     "lr": lr * zero_boost_qk,   "lr_scale": zero_boost_qk,   "weight_decay": weight_decay},
            {"params": vo_zero,     "lr": lr * zero_boost_vo,   "lr_scale": zero_boost_vo,   "weight_decay": weight_decay},
            {"params": ff_zero,     "lr": lr * zero_boost_ff,   "lr_scale": zero_boost_ff,   "weight_decay": weight_decay},
        ]
        return [g for g in groups if g["params"]]

    #====================
    # trit-floor prior (called after optimizer.step)
    #====================

    @torch.no_grad()
    def apply_qitnn_prior(
        self,
        *,
        step_qk: float = 0.0,
        step_vo: float = 0.0,
        step_ff: float = 0.0,
        entropy_floor: float = 0.0,
    ) -> None:
        if entropy_floor <= 0.0:
            return

        step_map = {"qk": float(step_qk), "vo": float(step_vo), "ff": float(step_ff)}

        for _, role, layer in self.iter_qitnn_layers():
            s = step_map[role]
            if s <= 0.0:
                continue
            prior_(
                layer.a_neg,
                layer.a_zero,
                layer.a_pos,
                step=s,
                entropy_floor=entropy_floor,
                mixed_precision=self.mixed_precision,
            )

    #====================
    # diagnostics
    #====================

    @torch.no_grad()
    def format_qitnn_diagnostics(self, *, epoch: int, full: bool = False) -> list[str]:
        all_layers = list(self.iter_qitnn_layers())

        if full:
            picked = all_layers
        else:
            last = max(self.layers - 1, 0)
            wanted = [
                f"blocks.{last}.ff1",
                f"blocks.{last}.ff2",
                "blocks.0.v_proj",
                "blocks.0.o_proj",
                f"blocks.{last}.v_proj",
                f"blocks.{last}.o_proj",
            ]
            by_name = {name: (name, role, layer) for name, role, layer in all_layers}
            seen: set[str] = set()
            picked = []
            for w in wanted:
                if w not in seen and w in by_name:
                    seen.add(w)
                    picked.append(by_name[w])

        lines: list[str] = []
        for name, _, layer in picked:
            lines.extend(render_qitnn_diag(name, layer.a_neg, layer.a_zero, layer.a_pos, epoch=epoch))
        return lines

    #====================
    # generation (autoregressive byte-level)
    #====================

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 16,
        ascii_guard: bool = True,
    ) -> torch.Tensor:
        out = tokens
        for _ in range(max_new_tokens):
            idx = out[:, -self.seq_len:]
            logits, _ = self(idx)
            nxt = logits[:, -1, :]

            # mask non-printable bytes
            if ascii_guard:
                mask = torch.zeros_like(nxt, dtype=torch.bool)
                mask[:, 0] = True
                mask[:, 9] = True
                mask[:, 10] = True
                mask[:, 13] = True
                mask[:, 32:127] = True
                nxt = nxt.masked_fill(~mask, float("-inf"))

            if temperature <= 0.0:
                tok = torch.argmax(nxt, dim=-1, keepdim=True)
            else:
                nxt = nxt / temperature
                if 0 < top_k < nxt.size(-1):
                    top_vals, _ = torch.topk(nxt, top_k, dim=-1)
                    cutoff = top_vals[:, -1].unsqueeze(-1)
                    nxt = nxt.masked_fill(nxt < cutoff, float("-inf"))
                probs = torch.softmax(nxt, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, tok], dim=1)
        return out

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, dim={self.dim}, ffn_dim={self.ffn_dim}, "
            f"seq_len={self.seq_len}, layers={self.layers}, precision_mode={self.precision_mode}, "
            f"mixed_precision={self.mixed_precision}"
        )
