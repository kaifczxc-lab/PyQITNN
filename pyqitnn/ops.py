from __future__ import annotations

import torch

from .bridge import load_native


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
    def forward(ctx, inp, a_neg, a_zero, a_pos, ent_lambda):
        ext = load_native()
        u, v, cn, cz, cp = ext.forward3_cuda(inp, a_neg, a_zero, a_pos)
        ctx.save_for_backward(inp, a_neg, a_zero, a_pos, cn, cz, cp)
        ctx.ent_lambda = float(ent_lambda)
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
            dcn = dcn + grad_cn.contiguous()
        if grad_cz is not None:
            dcz = dcz + grad_cz.contiguous()
        if grad_cp is not None:
            dcp = dcp + grad_cp.contiguous()

        g_inp = None
        if ctx.needs_input_grad[0]:
            g_inp = dcn @ a_neg.t() + dcz @ a_zero.t() + dcp @ a_pos.t()

        g_neg  = inp.t() @ dcn if ctx.needs_input_grad[1] else None
        g_zero = inp.t() @ dcz if ctx.needs_input_grad[2] else None
        g_pos  = inp.t() @ dcp if ctx.needs_input_grad[3] else None

        return g_inp, g_neg, g_zero, g_pos, None


#====================
# centered simplex: (u, v) -> (x, y) with y = sqrt(3)*v - 1/sqrt(3)
#====================

_SQRT3 = 1.7320508075688772


class _CenteredSimplexFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        ext = load_native()
        x, y = ext.centered_simplex_cuda(u, v)
        return x, y

    @staticmethod
    def backward(ctx, grad_x, grad_y):
        grad_u = grad_x.contiguous() if grad_x is not None else None
        grad_v = grad_y.contiguous() * _SQRT3 if grad_y is not None else None
        return grad_u, grad_v


#====================
# attention2: simplex-aware causal attention
#====================

class _Attention2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
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
            return (
                _pack_paired(dqx, dqy),
                _pack_paired(dkx, dky),
                _pack_paired(dvx, dvy),
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

        return torch.stack(gq, 0), torch.stack(gk, 0), torch.stack(gv, 0)


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
):
    _require_f32_2d(inp, "inp")
    _require_f32_2d(a_neg, "a_neg")
    _require_f32_2d(a_zero, "a_zero")
    _require_f32_2d(a_pos, "a_pos")
    return _Forward3Fn.apply(inp, a_neg, a_zero, a_pos, float(ent_lambda))


def prior_(
    a_neg: torch.Tensor,
    a_zero: torch.Tensor,
    a_pos: torch.Tensor,
    *,
    step: float,
    entropy_floor: float,
):
    _require_f32_2d(a_neg, "a_neg")
    _require_f32_2d(a_zero, "a_zero")
    _require_f32_2d(a_pos, "a_pos")
    if a_neg.shape != a_zero.shape or a_neg.shape != a_pos.shape:
        raise RuntimeError("a_neg, a_zero, a_pos shapes must match")
    ext = load_native()
    ext.prior_cuda(a_neg, a_zero, a_pos, float(step), float(entropy_floor))
    return a_neg, a_zero, a_pos


def centered_simplex(u: torch.Tensor, v: torch.Tensor):
    _require_f32_2d(u, "u")
    _require_f32_2d(v, "v")
    if u.shape != v.shape:
        raise RuntimeError("u and v shapes must match")
    return _CenteredSimplexFn.apply(u, v)


def attention2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    _require_f32_packed(q, "q")
    _require_f32_packed(k, "k")
    _require_f32_packed(v, "v")
    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError("q, k, v shapes must match")
    return _Attention2Fn.apply(q, k, v)
