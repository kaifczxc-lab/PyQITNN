from __future__ import annotations

import torch

from .bridge import load_native

_PRECISION_MODE_ALIASES = {
    "mixed_bf16_native": "qts_fp32_rest_bf16",
    "bf16": "qts_fp32_rest_bf16",
}
_PRECISION_MODE_SET = {"fp32", "qts_fp32_rest_bf16"}


def _normalize_precision_mode(value: str) -> str:
    mode = value.strip().lower()
    mode = _PRECISION_MODE_ALIASES.get(mode, mode)
    if mode not in _PRECISION_MODE_SET:
        wanted = ", ".join(sorted(_PRECISION_MODE_SET))
        raise RuntimeError(f"precision_mode must be one of: {wanted}")
    return mode


def _resolve_precision_mode_args(
    precision_mode: str | None,
    mixed_precision: bool | None,
) -> tuple[str, bool]:
    mode = "fp32" if precision_mode is None else _normalize_precision_mode(str(precision_mode))
    if mixed_precision is not None:
        legacy_mode = "qts_fp32_rest_bf16" if bool(mixed_precision) else "fp32"
        if precision_mode is None:
            mode = legacy_mode
        elif legacy_mode != mode:
            raise RuntimeError(
                "mixed_precision and precision_mode conflict. "
                "Use precision_mode alone, or keep them aligned during migration."
            )
    return mode, mode != "fp32"


#====================
# input validation
#====================

def _require_f32_2d(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if t.dtype != torch.float32:
        raise RuntimeError(f"{name} must be float32")
    if t.dim() != 2:
        raise RuntimeError(f"{name} must be 2D")
    if not t.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def _require_amp_compatible_2d(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(f"{name} must be float16, bfloat16, or float32")
    if t.dim() != 2:
        raise RuntimeError(f"{name} must be 2D")
    if not t.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def _require_f32_packed(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if t.dtype != torch.float32:
        raise RuntimeError(f"{name} must be float32")
    if t.dim() not in (2, 3):
        raise RuntimeError(f"{name} must be 2D or 3D")
    if not t.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")
    if t.size(-1) % 2 != 0:
        raise RuntimeError(f"{name} last dim must be even")


def _require_amp_compatible_packed(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(f"{name} must be float16, bfloat16, or float32")
    if t.dim() not in (2, 3):
        raise RuntimeError(f"{name} must be 2D or 3D")
    if not t.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")
    if t.size(-1) % 2 != 0:
        raise RuntimeError(f"{name} last dim must be even")


#====================
# simplex channel helpers (internal)
#====================

def _split_packed(xy: torch.Tensor):
    half = xy.size(-1) // 2
    return xy[..., :half].contiguous(), xy[..., half:].contiguous()


def _pack_paired(x: torch.Tensor, y: torch.Tensor):
    return torch.cat([x, y], dim=-1)


#====================
# forward3: Born-rule projection (a_neg, a_zero, a_pos) -> (u, v)
#====================

class _Forward3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, a_neg, a_zero, a_pos, ent_lambda, mixed_precision):
        ext = load_native()
        u, v, cn, cz, cp = ext.forward3_cuda(inp, a_neg, a_zero, a_pos)
        ctx.save_for_backward(inp, a_neg, a_zero, a_pos, cn, cz, cp)
        ctx.ent_lambda = float(ent_lambda)
        ctx.input_dtype = inp.dtype
        ctx.mixed_precision = bool(mixed_precision)
        return u, v, cn, cz, cp

    @staticmethod
    def backward(ctx, grad_u, grad_v, grad_cn, grad_cz, grad_cp):
        inp, a_neg, a_zero, a_pos, cn, cz, cp = ctx.saved_tensors
        ext = load_native()

        grad_u = grad_u.contiguous() if grad_u is not None else torch.zeros_like(cn)
        grad_v = grad_v.contiguous() if grad_v is not None else torch.zeros_like(cn)

        dcn, dcz, dcp = ext.backnorm3_cuda(
            grad_u, grad_v, cn, cz, cp, ctx.ent_lambda,
        )

        if grad_cn is not None:
            dcn = dcn + grad_cn.contiguous().to(dtype=torch.float32)
        if grad_cz is not None:
            dcz = dcz + grad_cz.contiguous().to(dtype=torch.float32)
        if grad_cp is not None:
            dcp = dcp + grad_cp.contiguous().to(dtype=torch.float32)

        g_inp = None
        if ctx.needs_input_grad[0]:
            g_inp = dcn @ a_neg.t() + dcz @ a_zero.t() + dcp @ a_pos.t()
            if ctx.input_dtype != torch.float32:
                g_inp = g_inp.to(dtype=ctx.input_dtype)

        inp_fp32 = inp if inp.dtype == torch.float32 else inp.to(dtype=torch.float32)
        g_neg  = inp_fp32.t() @ dcn if ctx.needs_input_grad[1] else None
        g_zero = inp_fp32.t() @ dcz if ctx.needs_input_grad[2] else None
        g_pos  = inp_fp32.t() @ dcp if ctx.needs_input_grad[3] else None

        return g_inp, g_neg, g_zero, g_pos, None, None


#====================
# centered simplex: (u, v) -> (x, y) with y = sqrt(3)*v - 1/sqrt(3)
#====================

_SQRT3 = 1.7320508075688772


class _CenteredSimplexFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, mixed_precision):
        ext = load_native()
        x, y = ext.centered_simplex_cuda(u, v)
        ctx.u_dtype = u.dtype
        ctx.v_dtype = v.dtype
        return x, y

    @staticmethod
    def backward(ctx, grad_x, grad_y):
        grad_u = None
        grad_v = None
        if grad_x is not None:
            grad_u = grad_x.contiguous().to(dtype=torch.float32)
            if ctx.u_dtype != torch.float32:
                grad_u = grad_u.to(dtype=ctx.u_dtype)
        if grad_y is not None:
            grad_v = grad_y.contiguous().to(dtype=torch.float32) * _SQRT3
            if ctx.v_dtype != torch.float32:
                grad_v = grad_v.to(dtype=ctx.v_dtype)
        return grad_u, grad_v, None


#====================
# attention2: simplex-aware causal attention
#====================

class _Attention2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mixed_precision):
        ext = load_native()
        qx, qy = _split_packed(q)
        kx, ky = _split_packed(k)
        vx, vy = _split_packed(v)

        if q.dim() == 2:
            ox, oy = ext.attention2_cuda(qx, qy, kx, ky, vx, vy)
            out = _pack_paired(ox, oy)
        else:
            outs = []
            for b in range(q.size(0)):
                ox, oy = ext.attention2_cuda(
                    qx[b].contiguous(), qy[b].contiguous(),
                    kx[b].contiguous(), ky[b].contiguous(),
                    vx[b].contiguous(), vy[b].contiguous(),
                )
                outs.append(_pack_paired(ox, oy))
            out = torch.stack(outs, dim=0)

        ctx.save_for_backward(q, k, v)
        ctx.q_dtype = q.dtype
        ctx.k_dtype = k.dtype
        ctx.v_dtype = v.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        ext = load_native()
        grad_out = grad_out.contiguous()

        qx, qy = _split_packed(q)
        kx, ky = _split_packed(k)
        vx, vy = _split_packed(v)
        dox, doy = _split_packed(grad_out)

        if q.dim() == 2:
            dqx, dqy, dkx, dky, dvx, dvy = ext.attention_backward2_cuda(
                dox, doy, qx, qy, kx, ky, vx, vy,
            )
            gq = _pack_paired(dqx, dqy)
            gk = _pack_paired(dkx, dky)
            gv = _pack_paired(dvx, dvy)
            return (
                gq,
                gk,
                gv,
                None,
            )

        gq, gk, gv = [], [], []
        for b in range(q.size(0)):
            dqx, dqy, dkx, dky, dvx, dvy = ext.attention_backward2_cuda(
                dox[b].contiguous(), doy[b].contiguous(),
                qx[b].contiguous(), qy[b].contiguous(),
                kx[b].contiguous(), ky[b].contiguous(),
                vx[b].contiguous(), vy[b].contiguous(),
            )
            gq.append(_pack_paired(dqx, dqy))
            gk.append(_pack_paired(dkx, dky))
            gv.append(_pack_paired(dvx, dvy))

        out_gq = torch.stack(gq, 0)
        out_gk = torch.stack(gk, 0)
        out_gv = torch.stack(gv, 0)
        return out_gq, out_gk, out_gv, None


#====================
# public API
#====================

def forward3(
    inp: torch.Tensor,
    a_neg: torch.Tensor,
    a_zero: torch.Tensor,
    a_pos: torch.Tensor,
    *,
    ent_lambda: float = 0.0,
    mixed_precision: bool | None = None,
    precision_mode: str | None = None,
):
    _, use_mixed_precision = _resolve_precision_mode_args(precision_mode, mixed_precision)
    if use_mixed_precision:
        _require_amp_compatible_2d(inp, "inp")
    else:
        _require_f32_2d(inp, "inp")
    _require_f32_2d(a_neg, "a_neg")
    _require_f32_2d(a_zero, "a_zero")
    _require_f32_2d(a_pos, "a_pos")
    return _Forward3Fn.apply(inp, a_neg, a_zero, a_pos, float(ent_lambda), use_mixed_precision)


def prior_(
    a_neg: torch.Tensor,
    a_zero: torch.Tensor,
    a_pos: torch.Tensor,
    *,
    step: float,
    entropy_floor: float,
    mixed_precision: bool | None = None,
    precision_mode: str | None = None,
):
    _, use_mixed_precision = _resolve_precision_mode_args(precision_mode, mixed_precision)
    if use_mixed_precision:
        _require_amp_compatible_2d(a_neg, "a_neg")
        _require_amp_compatible_2d(a_zero, "a_zero")
        _require_amp_compatible_2d(a_pos, "a_pos")
    else:
        _require_f32_2d(a_neg, "a_neg")
        _require_f32_2d(a_zero, "a_zero")
        _require_f32_2d(a_pos, "a_pos")
    if a_neg.shape != a_zero.shape or a_neg.shape != a_pos.shape:
        raise RuntimeError("a_neg, a_zero, a_pos shapes must match")
    ext = load_native()
    ext.prior_cuda(a_neg, a_zero, a_pos, float(step), float(entropy_floor))
    return a_neg, a_zero, a_pos


def centered_simplex(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    mixed_precision: bool | None = None,
    precision_mode: str | None = None,
):
    _, use_mixed_precision = _resolve_precision_mode_args(precision_mode, mixed_precision)
    if use_mixed_precision:
        _require_amp_compatible_2d(u, "u")
        _require_amp_compatible_2d(v, "v")
    else:
        _require_f32_2d(u, "u")
        _require_f32_2d(v, "v")
    if u.shape != v.shape:
        raise RuntimeError("u and v shapes must match")
    if u.dtype != v.dtype:
        raise RuntimeError("u and v dtypes must match")
    return _CenteredSimplexFn.apply(u, v, use_mixed_precision)


def attention2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mixed_precision: bool | None = None,
    precision_mode: str | None = None,
):
    _, use_mixed_precision = _resolve_precision_mode_args(precision_mode, mixed_precision)
    if use_mixed_precision:
        _require_amp_compatible_packed(q, "q")
        _require_amp_compatible_packed(k, "k")
        _require_amp_compatible_packed(v, "v")
    else:
        _require_f32_packed(q, "q")
        _require_f32_packed(k, "k")
        _require_f32_packed(v, "v")
    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError("q, k, v shapes must match")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError("q, k, v dtypes must match")
    return _Attention2Fn.apply(q, k, v, use_mixed_precision)
