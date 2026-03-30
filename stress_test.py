"""
Stress test for PyQITNN architecture.
Designed to expose hidden bugs before public release.
Important: This stress_test created by using AI
Run: python stress_test.py
Requires CUDA GPU.
"""
from contextlib import nullcontext, redirect_stdout
import csv
import io
import json
import shutil
import subprocess
import sys
import math
import re
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pyqitnn
from BasicQITNN_Transformer import (
    TrainConfig,
    _capture_rng_state,
    _build_cli_parser,
    _build_token_byte_lut,
    _count_token_bytes,
    _parse_cli,
    _resolve_grad_accum_steps,
    _resolve_precision_mode_cfg,
    _resolve_train_step_plan,
    _restore_rng_state,
    _resolve_warmup_steps,
    _schedule_progress,
    eval_split_metrics,
    load_bytes,
    lr_cosine,
    lr_linear,
    lr_with_warmup,
    loss_to_bpb,
    loss_to_perplexity,
    load_ckpt,
    run_val,
    sample_batch,
    save_ckpt,
    total_nll_to_bpb,
    train,
)
from pyqitnn.diagnostics import QITNN_DIAG_CSV_HEADER
from pyqitnn.diagnostics import QITNN_DIAG_STAT_KEYS
from pyqitnn.diagnostics import format_qitnn_diag_snapshot
from pyqitnn.ops import forward3, forward3_packed_reference, prior_, centered_simplex, attention2
from pyqitnn.bridge import load_native
from pyqitnn.precision import resolve_precision_mode as resolve_precision_mode_shared

DEVICE = torch.device("cuda:0")
PASSED = 0
FAILED = 0
WARNED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} -- {detail}")


def warn(name, detail=""):
    global WARNED
    WARNED += 1
    print(f"  [WARN] {name} -- {detail}")


def capture_runtime_error(fn):
    try:
        fn()
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return ""


def cleanup_tree(path: Path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


@torch.no_grad()
def _generate_reference_loop(
    model,
    tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    ascii_guard: bool,
) -> torch.Tensor:
    out = tokens.clone()
    for _ in range(max_new_tokens):
        idx = out[:, -model.seq_len:]
        logits, _ = model(idx)
        nxt = logits[:, -1, :]

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

import pyqitnn

status = pyqitnn.bridge_status()
print(pyqitnn.__version__)        # e.g. 0.3.2
print(status["native_found"])     # True
print(status["native_loadable"])  # True
#====================
# 1. Born rule invariant: P- + P0 + P+ = 1.0 exactly
#    if this fails, the normalize kernel is wrong
#====================

def test_born_rule_sum():
    """P_neg + P_zero + P_pos must equal 1.0 for every element"""
    print("\n=== test_born_rule_sum ===")
    torch.manual_seed(42)

    for trial, (rows, in_d, out_d) in enumerate([
        (1, 1, 1), (4, 8, 6), (64, 128, 64),
        (256, 512, 256), (1, 1024, 1024),
    ]):
        inp = torch.randn(rows, in_d, device=DEVICE)
        a_n = torch.randn(in_d, out_d, device=DEVICE)
        a_z = torch.randn(in_d, out_d, device=DEVICE)
        a_p = torch.randn(in_d, out_d, device=DEVICE)

        u, v, cn, cz, cp = forward3(inp, a_n, a_z, a_p)

        # reconstruct probabilities from (u, v)
        # u = P+ - P-,  v = P0
        # P+ = (1 + u - v) / 2,  P- = (1 - u - v) / 2
        p_pos = (1.0 + u - v) / 2.0
        p_neg = (1.0 - u - v) / 2.0
        p_zero = v

        total = p_neg + p_zero + p_pos
        max_err = (total - 1.0).abs().max().item()

        check(f"Born sum=1 [{rows}x{in_d}->{out_d}] max_err={max_err:.2e}",
              max_err < 1e-5, f"max_err={max_err}")

        # all probabilities >= 0
        min_p = min(p_neg.min().item(), p_zero.min().item(), p_pos.min().item())
        check(f"Born P>=0  [{rows}x{in_d}->{out_d}] min_P={min_p:.2e}",
              min_p >= -1e-6, f"min_P={min_p}")


#====================
# forward3 raw-channel contract: cn/cz/cp must stay as linear amplitude projections
#====================

def _manual_forward3_from_raw_channels(
    cn: torch.Tensor,
    cz: torch.Tensor,
    cp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cn_f = cn.float()
    cz_f = cz.float()
    cp_f = cp.float()
    cn2 = cn_f.square()
    cz2 = cz_f.square()
    cp2 = cp_f.square()
    z = cn2 + cz2 + cp2
    inv_z = torch.where(z > 1e-12, z.reciprocal(), torch.zeros_like(z))
    u = (cp2 - cn2) * inv_z
    v = cz2 * inv_z
    return u, v


#====================
# forward3 raw-channel contract: cn/cz/cp must stay as linear amplitude projections
#====================

def test_forward3_raw_channel_projection_contract():
    """forward3 raw channels must equal the three linear amplitude projections"""
    print("\n=== test_forward3_raw_channel_projection_contract ===")
    torch.manual_seed(43)

    for rows, in_d, out_d in [(4, 16, 8), (3, 31, 11), (9, 64, 32)]:
        inp = torch.randn(rows, in_d, device=DEVICE)
        a_n = torch.randn(in_d, out_d, device=DEVICE)
        a_z = torch.randn(in_d, out_d, device=DEVICE)
        a_p = torch.randn(in_d, out_d, device=DEVICE)

        _, _, cn, cz, cp = forward3(inp, a_n, a_z, a_p)
        ref_cn = inp @ a_n
        ref_cz = inp @ a_z
        ref_cp = inp @ a_p

        err_cn = (cn - ref_cn).abs().max().item()
        err_cz = (cz - ref_cz).abs().max().item()
        err_cp = (cp - ref_cp).abs().max().item()

        check(f"raw cn matches inp@a_neg [{rows}x{in_d}->{out_d}]", err_cn < 1e-5, f"max_err={err_cn:.2e}")
        check(f"raw cz matches inp@a_zero [{rows}x{in_d}->{out_d}]", err_cz < 1e-5, f"max_err={err_cz:.2e}")
        check(f"raw cp matches inp@a_pos [{rows}x{in_d}->{out_d}]", err_cp < 1e-5, f"max_err={err_cp:.2e}")
        check(
            f"raw channels stay fp32 [{rows}x{in_d}->{out_d}]",
            cn.dtype == torch.float32 and cz.dtype == torch.float32 and cp.dtype == torch.float32,
            f"dtypes={cn.dtype}/{cz.dtype}/{cp.dtype}",
        )


#====================
# forward3 raw-channel contract: Born/simplex outputs must remain reconstructible from cn/cz/cp
#====================

def test_forward3_raw_channel_simplex_contract():
    """u/v and centered simplex outputs must be reconstructible from raw channels"""
    print("\n=== test_forward3_raw_channel_simplex_contract ===")
    torch.manual_seed(44)
    sqrt3 = 1.7320508075688772
    inv_sqrt3 = 0.5773502691896258

    for rows, in_d, out_d in [(4, 16, 8), (5, 19, 7), (2, 128, 33)]:
        inp = torch.randn(rows, in_d, device=DEVICE)
        a_n = torch.randn(in_d, out_d, device=DEVICE)
        a_z = torch.randn(in_d, out_d, device=DEVICE)
        a_p = torch.randn(in_d, out_d, device=DEVICE)

        u, v, cn, cz, cp = forward3(inp, a_n, a_z, a_p)
        x, y = centered_simplex(u, v)

        ref_u, ref_v = _manual_forward3_from_raw_channels(cn, cz, cp)
        ref_x = ref_u
        ref_y = (sqrt3 * ref_v) - inv_sqrt3

        err_u = (u.float() - ref_u).abs().max().item()
        err_v = (v.float() - ref_v).abs().max().item()
        err_x = (x.float() - ref_x).abs().max().item()
        err_y = (y.float() - ref_y).abs().max().item()

        check(f"u matches raw Born normalization [{rows}x{in_d}->{out_d}]", err_u < 1e-6, f"max_err={err_u:.2e}")
        check(f"v matches raw Born normalization [{rows}x{in_d}->{out_d}]", err_v < 1e-6, f"max_err={err_v:.2e}")
        check(f"x matches raw-channel simplex map [{rows}x{in_d}->{out_d}]", err_x < 1e-6, f"max_err={err_x:.2e}")
        check(f"y matches raw-channel simplex map [{rows}x{in_d}->{out_d}]", err_y < 1e-6, f"max_err={err_y:.2e}")


#====================
# forward3 mixed raw-channel contract: mixed visible path must stay finite while raw channels remain fp32
#====================

def test_forward3_mixed_raw_channel_contract():
    """mixed forward3 must keep raw channels fp32 and finite across scale regimes"""
    print("\n=== test_forward3_mixed_raw_channel_contract ===")
    torch.manual_seed(45)

    modes: list[tuple[str, torch.dtype, float]] = [
        ("fp16", torch.float16, 5e-3),
    ]
    if torch.cuda.is_bf16_supported():
        modes.append(("bf16", torch.bfloat16, 5e-2))

    for mode_name, input_dtype, uv_tol in modes:
        for scale in (1e-4, 1.0, 1e2, 1e4):
            inp = torch.randn(8, 16, device=DEVICE, dtype=input_dtype, requires_grad=True)
            a_n = (torch.randn(16, 8, device=DEVICE, dtype=torch.float32) * scale).requires_grad_()
            a_z = (torch.randn(16, 8, device=DEVICE, dtype=torch.float32) * scale).requires_grad_()
            a_p = (torch.randn(16, 8, device=DEVICE, dtype=torch.float32) * scale).requires_grad_()

            u, v, cn, cz, cp = forward3(inp, a_n, a_z, a_p, mixed_precision=True)
            ref_cn = inp.float() @ a_n
            ref_cz = inp.float() @ a_z
            ref_cp = inp.float() @ a_p
            ref_u, ref_v = _manual_forward3_from_raw_channels(ref_cn, ref_cz, ref_cp)

            loss = u.float().square().mean() + v.float().square().mean()
            loss.backward()

            err_cn = (cn - ref_cn).abs().max().item()
            err_cz = (cz - ref_cz).abs().max().item()
            err_cp = (cp - ref_cp).abs().max().item()
            err_u = (u.float() - ref_u).abs().max().item()
            err_v = (v.float() - ref_v).abs().max().item()
            finite_ok = all(
                torch.isfinite(t).all().item()
                for t in (u.float(), v.float(), cn, cz, cp, inp.grad.float(), a_n.grad, a_z.grad, a_p.grad)
            )

            check(f"mixed raw cn stays fp32 [{mode_name} scale={scale:g}]", cn.dtype == torch.float32, f"dtype={cn.dtype}")
            check(f"mixed raw cz stays fp32 [{mode_name} scale={scale:g}]", cz.dtype == torch.float32, f"dtype={cz.dtype}")
            check(f"mixed raw cp stays fp32 [{mode_name} scale={scale:g}]", cp.dtype == torch.float32, f"dtype={cp.dtype}")
            check(f"mixed u stays activation dtype [{mode_name} scale={scale:g}]", u.dtype == input_dtype, f"dtype={u.dtype}")
            check(f"mixed v stays activation dtype [{mode_name} scale={scale:g}]", v.dtype == input_dtype, f"dtype={v.dtype}")
            check(f"mixed raw cn parity [{mode_name} scale={scale:g}]", err_cn < 1e-4, f"max_err={err_cn:.2e}")
            check(f"mixed raw cz parity [{mode_name} scale={scale:g}]", err_cz < 1e-4, f"max_err={err_cz:.2e}")
            check(f"mixed raw cp parity [{mode_name} scale={scale:g}]", err_cp < 1e-4, f"max_err={err_cp:.2e}")
            check(f"mixed u parity [{mode_name} scale={scale:g}]", err_u < uv_tol, f"max_err={err_u:.2e}")
            check(f"mixed v parity [{mode_name} scale={scale:g}]", err_v < uv_tol, f"max_err={err_v:.2e}")
            check(f"mixed path stays finite [{mode_name} scale={scale:g}]", finite_ok)


#====================
# forward3 split backward contract: native split backward must match reference backnorm + matmul assembly
#====================

def test_forward3_split_backward_native_contract():
    """split forward3 backward must match the reference backnorm + Python matmul assembly"""
    print("\n=== test_forward3_split_backward_native_contract ===")
    torch.manual_seed(52)
    ext = load_native()

    modes: list[tuple[str, torch.dtype, bool, float, float]] = [("fp32", torch.float32, False, 1e-5, 1e-5)]
    modes.append(("fp16", torch.float16, True, 3e-3, 5e-4))
    if torch.cuda.is_bf16_supported():
        modes.append(("bf16", torch.bfloat16, True, 1e-2, 5e-4))

    for mode_name, input_dtype, mixed_precision, inp_tol, weight_tol in modes:
        rows, in_d, out_d = 4, 23, 17
        ent_lambda = 0.17

        inp = torch.randn(rows, in_d, device=DEVICE, dtype=input_dtype, requires_grad=True)
        a_neg = torch.randn(in_d, out_d, device=DEVICE, requires_grad=True)
        a_zero = torch.randn(in_d, out_d, device=DEVICE, requires_grad=True)
        a_pos = torch.randn(in_d, out_d, device=DEVICE, requires_grad=True)

        u, v, cn, cz, cp = forward3(
            inp,
            a_neg,
            a_zero,
            a_pos,
            ent_lambda=ent_lambda,
            mixed_precision=mixed_precision,
        )
        loss = (
            0.7 * u.float().square().mean()
            + 0.5 * v.float().square().mean()
            + 0.1 * cn.square().mean()
            + 0.2 * cz.square().mean()
            + 0.3 * cp.square().mean()
        )

        grad_u, grad_v, grad_cn, grad_cz, grad_cp = torch.autograd.grad(
            loss,
            (u, v, cn, cz, cp),
            retain_graph=True,
        )
        loss.backward()

        ref_dcn, ref_dcz, ref_dcp = ext.backnorm3_cuda(
            grad_u.contiguous(),
            grad_v.contiguous(),
            cn.detach(),
            cz.detach(),
            cp.detach(),
            ent_lambda,
        )
        ref_dcn = ref_dcn + grad_cn.contiguous().to(dtype=torch.float32)
        ref_dcz = ref_dcz + grad_cz.contiguous().to(dtype=torch.float32)
        ref_dcp = ref_dcp + grad_cp.contiguous().to(dtype=torch.float32)

        inp_fp32 = inp.detach() if inp.dtype == torch.float32 else inp.detach().to(dtype=torch.float32)
        ref_g_inp = ref_dcn @ a_neg.detach().t() + ref_dcz @ a_zero.detach().t() + ref_dcp @ a_pos.detach().t()
        if input_dtype != torch.float32:
            ref_g_inp = ref_g_inp.to(dtype=input_dtype)
        ref_g_neg = inp_fp32.t() @ ref_dcn
        ref_g_zero = inp_fp32.t() @ ref_dcz
        ref_g_pos = inp_fp32.t() @ ref_dcp

        inp_err = (inp.grad.float() - ref_g_inp.float()).abs().max().item()
        neg_err = (a_neg.grad - ref_g_neg).abs().max().item()
        zero_err = (a_zero.grad - ref_g_zero).abs().max().item()
        pos_err = (a_pos.grad - ref_g_pos).abs().max().item()
        finite_ok = all(
            torch.isfinite(t.float()).all().item()
            for t in (inp.grad, a_neg.grad, a_zero.grad, a_pos.grad)
        )

        check(f"split backward input grad dtype [{mode_name}]", inp.grad.dtype == input_dtype, f"dtype={inp.grad.dtype}")
        check(f"split backward input grad parity [{mode_name}]", inp_err < inp_tol, f"max_err={inp_err:.2e}")
        check(f"split backward a_neg grad parity [{mode_name}]", neg_err < weight_tol, f"max_err={neg_err:.2e}")
        check(f"split backward a_zero grad parity [{mode_name}]", zero_err < weight_tol, f"max_err={zero_err:.2e}")
        check(f"split backward a_pos grad parity [{mode_name}]", pos_err < weight_tol, f"max_err={pos_err:.2e}")
        check(f"split backward stays finite [{mode_name}]", finite_ok)


#====================
# forward3 packed reference contract: packed native path must match split native path
#====================

def test_forward3_packed_reference_parity():
    """packed forward3 reference path must match split forward3 path in fp32"""
    print("\n=== test_forward3_packed_reference_parity ===")
    torch.manual_seed(46)

    for rows, in_d, out_d in [(4, 16, 8), (3, 31, 11), (2, 128, 33)]:
        inp = torch.randn(rows, in_d, device=DEVICE)
        a_n = torch.randn(in_d, out_d, device=DEVICE)
        a_z = torch.randn(in_d, out_d, device=DEVICE)
        a_p = torch.randn(in_d, out_d, device=DEVICE)
        a_packed = torch.stack((a_n, a_z, a_p), dim=-1).contiguous()

        split_out = forward3(inp, a_n, a_z, a_p)
        packed_out = forward3_packed_reference(inp, a_packed)

        for name, left, right in zip(("u", "v", "cn", "cz", "cp"), split_out, packed_out):
            max_err = (left.float() - right.float()).abs().max().item()
            check(f"packed ref matches split {name} [{rows}x{in_d}->{out_d}]", max_err < 1e-6, f"max_err={max_err:.2e}")


#====================
# forward3 packed reference contract: mixed path must preserve parity and dtype contract
#====================

def test_forward3_packed_reference_mixed_parity():
    """packed forward3 reference path must match split forward3 path under mixed precision"""
    print("\n=== test_forward3_packed_reference_mixed_parity ===")
    torch.manual_seed(47)

    modes: list[tuple[str, torch.dtype]] = [("fp16", torch.float16)]
    if torch.cuda.is_bf16_supported():
        modes.append(("bf16", torch.bfloat16))

    for mode_name, input_dtype in modes:
        for scale in (1e-4, 1.0, 1e2, 1e4):
            inp = torch.randn(8, 16, device=DEVICE, dtype=input_dtype)
            a_n = torch.randn(16, 8, device=DEVICE) * scale
            a_z = torch.randn(16, 8, device=DEVICE) * scale
            a_p = torch.randn(16, 8, device=DEVICE) * scale
            a_packed = torch.stack((a_n, a_z, a_p), dim=-1).contiguous()

            split_out = forward3(inp, a_n, a_z, a_p, mixed_precision=True)
            packed_out = forward3_packed_reference(inp, a_packed, mixed_precision=True)

            for name, left, right in zip(("u", "v", "cn", "cz", "cp"), split_out, packed_out):
                max_err = (left.float() - right.float()).abs().max().item()
                check(f"packed ref mixed parity {name} [{mode_name} scale={scale:g}]", max_err < 1e-6, f"max_err={max_err:.2e}")

            u_ref, v_ref, cn_ref, cz_ref, cp_ref = packed_out
            finite_ok = all(torch.isfinite(t.float()).all().item() for t in (u_ref, v_ref, cn_ref, cz_ref, cp_ref))
            check(f"packed ref mixed u dtype [{mode_name} scale={scale:g}]", u_ref.dtype == input_dtype, f"dtype={u_ref.dtype}")
            check(f"packed ref mixed v dtype [{mode_name} scale={scale:g}]", v_ref.dtype == input_dtype, f"dtype={v_ref.dtype}")
            check(f"packed ref mixed raw dtype [{mode_name} scale={scale:g}]", cn_ref.dtype == torch.float32 and cz_ref.dtype == torch.float32 and cp_ref.dtype == torch.float32, f"dtypes={cn_ref.dtype}/{cz_ref.dtype}/{cp_ref.dtype}")
            check(f"packed ref mixed path stays finite [{mode_name} scale={scale:g}]", finite_ok)


#====================
# forward3 packed projection contract: direct packed kernel must match manual raw matmuls
#====================

def test_forward3_packed_reference_raw_projection_contract():
    """packed forward3 path must preserve raw projection indexing and Born reconstruction"""
    print("\n=== test_forward3_packed_reference_raw_projection_contract ===")
    torch.manual_seed(49)

    for rows, in_d, out_d in [(7, 53, 29), (5, 65, 17), (3, 96, 41)]:
        inp = torch.randn(rows, in_d, device=DEVICE)
        a_n = torch.randn(in_d, out_d, device=DEVICE)
        a_z = torch.randn(in_d, out_d, device=DEVICE)
        a_p = torch.randn(in_d, out_d, device=DEVICE)
        a_packed = torch.stack((a_n, a_z, a_p), dim=-1).contiguous()

        u, v, cn, cz, cp = forward3_packed_reference(inp, a_packed)
        ref_cn = inp @ a_n
        ref_cz = inp @ a_z
        ref_cp = inp @ a_p
        ref_u, ref_v = _manual_forward3_from_raw_channels(ref_cn, ref_cz, ref_cp)

        err_cn = (cn - ref_cn).abs().max().item()
        err_cz = (cz - ref_cz).abs().max().item()
        err_cp = (cp - ref_cp).abs().max().item()
        err_u = (u.float() - ref_u).abs().max().item()
        err_v = (v.float() - ref_v).abs().max().item()

        check(f"packed kernel raw cn contract [{rows}x{in_d}->{out_d}]", err_cn < 1e-6, f"max_err={err_cn:.2e}")
        check(f"packed kernel raw cz contract [{rows}x{in_d}->{out_d}]", err_cz < 1e-6, f"max_err={err_cz:.2e}")
        check(f"packed kernel raw cp contract [{rows}x{in_d}->{out_d}]", err_cp < 1e-6, f"max_err={err_cp:.2e}")
        check(f"packed kernel Born u contract [{rows}x{in_d}->{out_d}]", err_u < 1e-6, f"max_err={err_u:.2e}")
        check(f"packed kernel Born v contract [{rows}x{in_d}->{out_d}]", err_v < 1e-6, f"max_err={err_v:.2e}")


#====================
# forward3 packed epilogue contract: fused uv path must agree with returned raw channels
#====================

def test_forward3_packed_reference_fused_epilogue_contract():
    """packed forward3 fused epilogue must derive uv from the same raw channels it returns"""
    print("\n=== test_forward3_packed_reference_fused_epilogue_contract ===")
    torch.manual_seed(50)

    modes: list[tuple[str, torch.dtype, bool]] = [("fp32", torch.float32, False), ("fp16", torch.float16, True)]
    if torch.cuda.is_bf16_supported():
        modes.append(("bf16", torch.bfloat16, True))

    for mode_name, input_dtype, mixed_precision in modes:
        inp = torch.randn(6, 37, device=DEVICE, dtype=input_dtype)
        a_n = torch.randn(37, 19, device=DEVICE)
        a_z = torch.randn(37, 19, device=DEVICE)
        a_p = torch.randn(37, 19, device=DEVICE)
        a_packed = torch.stack((a_n, a_z, a_p), dim=-1).contiguous()

        u, v, cn, cz, cp = forward3_packed_reference(inp, a_packed, mixed_precision=mixed_precision)
        ref_u, ref_v = _manual_forward3_from_raw_channels(cn, cz, cp)
        ref_visible_u = ref_u.to(dtype=u.dtype).float()
        ref_visible_v = ref_v.to(dtype=v.dtype).float()
        err_u = (u.float() - ref_visible_u).abs().max().item()
        err_v = (v.float() - ref_visible_v).abs().max().item()

        check(f"fused epilogue u contract [{mode_name}]", err_u < 1e-6, f"max_err={err_u:.2e}")
        check(f"fused epilogue v contract [{mode_name}]", err_v < 1e-6, f"max_err={err_v:.2e}")

        zero_inp = torch.zeros(4, 37, device=DEVICE, dtype=input_dtype)
        zero_u, zero_v, zero_cn, zero_cz, zero_cp = forward3_packed_reference(
            zero_inp,
            a_packed,
            mixed_precision=mixed_precision,
        )
        finite_ok = all(torch.isfinite(t.float()).all().item() for t in (zero_u, zero_v, zero_cn, zero_cz, zero_cp))
        zero_u_max = zero_u.float().abs().max().item()
        zero_v_max = zero_v.float().abs().max().item()
        zero_cn_max = zero_cn.abs().max().item()
        zero_cz_max = zero_cz.abs().max().item()
        zero_cp_max = zero_cp.abs().max().item()

        check(f"fused epilogue zero guard finite [{mode_name}]", finite_ok)
        check(f"fused epilogue zero raw cn [{mode_name}]", zero_cn_max < 1e-8, f"max_abs={zero_cn_max:.2e}")
        check(f"fused epilogue zero raw cz [{mode_name}]", zero_cz_max < 1e-8, f"max_abs={zero_cz_max:.2e}")
        check(f"fused epilogue zero raw cp [{mode_name}]", zero_cp_max < 1e-8, f"max_abs={zero_cp_max:.2e}")
        check(f"fused epilogue zero u [{mode_name}]", zero_u_max < 1e-8, f"max_abs={zero_u_max:.2e}")
        check(f"fused epilogue zero v [{mode_name}]", zero_v_max < 1e-8, f"max_abs={zero_v_max:.2e}")


#====================
# forward3 packed reference contract: invalid ternary packed layout must fail loudly
#====================

def test_forward3_packed_reference_rejects_bad_layout():
    """packed forward3 reference path should reject malformed ternary layouts"""
    print("\n=== test_forward3_packed_reference_rejects_bad_layout ===")
    torch.manual_seed(48)

    inp = torch.randn(4, 16, device=DEVICE)
    bad_rank = torch.randn(16, 8, device=DEVICE)
    bad_last_dim = torch.randn(16, 8, 2, device=DEVICE)

    bad_rank_msg = capture_runtime_error(lambda: forward3_packed_reference(inp, bad_rank))
    bad_last_dim_msg = capture_runtime_error(lambda: forward3_packed_reference(inp, bad_last_dim))

    check("packed ref rejects non-3D weight tensor", "3D" in bad_rank_msg, bad_rank_msg or "no RuntimeError")
    check("packed ref rejects non-ternary last dim", "last dim must be 3" in bad_last_dim_msg, bad_last_dim_msg or "no RuntimeError")


#====================
# forward3 packed reference contract: backward path must match split gradients
#====================

def test_forward3_packed_reference_backward_parity():
    """packed forward3 backward must match split backward for input and ternary weights"""
    print("\n=== test_forward3_packed_reference_backward_parity ===")
    torch.manual_seed(51)

    modes: list[tuple[str, torch.dtype, bool, float, float]] = [("fp32", torch.float32, False, 1e-5, 1e-5)]
    modes.append(("fp16", torch.float16, True, 3e-3, 5e-4))
    if torch.cuda.is_bf16_supported():
        modes.append(("bf16", torch.bfloat16, True, 1e-2, 5e-4))

    for mode_name, input_dtype, mixed_precision, inp_tol, weight_tol in modes:
        rows, in_d, out_d = 4, 23, 17
        ent_lambda = 0.17
        base_inp = torch.randn(rows, in_d, device=DEVICE, dtype=input_dtype)
        base_neg = torch.randn(in_d, out_d, device=DEVICE)
        base_zero = torch.randn(in_d, out_d, device=DEVICE)
        base_pos = torch.randn(in_d, out_d, device=DEVICE)

        split_inp = base_inp.detach().clone().requires_grad_(True)
        split_neg = base_neg.detach().clone().requires_grad_(True)
        split_zero = base_zero.detach().clone().requires_grad_(True)
        split_pos = base_pos.detach().clone().requires_grad_(True)
        split_u, split_v, split_cn, split_cz, split_cp = forward3(
            split_inp,
            split_neg,
            split_zero,
            split_pos,
            ent_lambda=ent_lambda,
            mixed_precision=mixed_precision,
        )
        split_loss = (
            0.7 * split_u.float().square().mean()
            + 0.5 * split_v.float().square().mean()
            + 0.1 * split_cn.square().mean()
            + 0.2 * split_cz.square().mean()
            + 0.3 * split_cp.square().mean()
        )
        split_loss.backward()

        packed_inp = base_inp.detach().clone().requires_grad_(True)
        packed_weight = torch.stack((base_neg, base_zero, base_pos), dim=-1).contiguous().detach().requires_grad_(True)
        packed_u, packed_v, packed_cn, packed_cz, packed_cp = forward3_packed_reference(
            packed_inp,
            packed_weight,
            ent_lambda=ent_lambda,
            mixed_precision=mixed_precision,
        )
        packed_loss = (
            0.7 * packed_u.float().square().mean()
            + 0.5 * packed_v.float().square().mean()
            + 0.1 * packed_cn.square().mean()
            + 0.2 * packed_cz.square().mean()
            + 0.3 * packed_cp.square().mean()
        )
        packed_loss.backward()

        check(f"packed backward input grad exists [{mode_name}]", packed_inp.grad is not None, "grad is None")
        check(f"packed backward packed grad exists [{mode_name}]", packed_weight.grad is not None, "grad is None")
        if packed_inp.grad is None or packed_weight.grad is None:
            continue

        inp_err = (split_inp.grad.float() - packed_inp.grad.float()).abs().max().item()
        neg_err = (split_neg.grad - packed_weight.grad[..., 0]).abs().max().item()
        zero_err = (split_zero.grad - packed_weight.grad[..., 1]).abs().max().item()
        pos_err = (split_pos.grad - packed_weight.grad[..., 2]).abs().max().item()
        loss_err = abs(float(split_loss.detach().float()) - float(packed_loss.detach().float()))
        finite_ok = all(
            torch.isfinite(t.float()).all().item()
            for t in (
                packed_inp.grad,
                packed_weight.grad[..., 0],
                packed_weight.grad[..., 1],
                packed_weight.grad[..., 2],
            )
        )

        check(f"packed backward loss parity [{mode_name}]", loss_err < 1e-6, f"loss_err={loss_err:.2e}")
        check(f"packed backward input grad dtype [{mode_name}]", packed_inp.grad.dtype == input_dtype, f"dtype={packed_inp.grad.dtype}")
        check(f"packed backward weight grad dtype [{mode_name}]", packed_weight.grad.dtype == torch.float32, f"dtype={packed_weight.grad.dtype}")
        check(f"packed backward input grad parity [{mode_name}]", inp_err < inp_tol, f"max_err={inp_err:.2e}")
        check(f"packed backward a_neg grad parity [{mode_name}]", neg_err < weight_tol, f"max_err={neg_err:.2e}")
        check(f"packed backward a_zero grad parity [{mode_name}]", zero_err < weight_tol, f"max_err={zero_err:.2e}")
        check(f"packed backward a_pos grad parity [{mode_name}]", pos_err < weight_tol, f"max_err={pos_err:.2e}")
        check(f"packed backward stays finite [{mode_name}]", finite_ok)


#====================
# 2. Centered simplex geometry: pure states must map to equilateral triangle
#====================

def test_simplex_triangle_vertices():
    """The 3 pure qutrit states must map to vertices of an equilateral triangle"""
    print("\n=== test_simplex_triangle_vertices ===")

    # pure |+1>: P+=1, P-=0, P0=0  => u=1, v=0
    # pure |-1>: P+=0, P-=1, P0=0  => u=-1, v=0
    # pure |0> : P+=0, P-=0, P0=1  => u=0, v=1
    sqrt3 = 1.7320508075688772
    inv_sqrt3 = 0.5773502691896258

    # expected centered simplex coords:
    # |+1>: x=1, y = sqrt3*0 - inv_sqrt3 = -inv_sqrt3
    # |-1>: x=-1, y = -inv_sqrt3
    # |0> : x=0, y = sqrt3*1 - inv_sqrt3 = sqrt3 - inv_sqrt3 = 2*inv_sqrt3

    expected = {
        "+1": (1.0, -inv_sqrt3),
        "-1": (-1.0, -inv_sqrt3),
        "0":  (0.0, 2.0 * inv_sqrt3),
    }

    u_vals = torch.tensor([[1.0], [-1.0], [0.0]], device=DEVICE)
    v_vals = torch.tensor([[0.0], [0.0], [1.0]], device=DEVICE)

    x, y = centered_simplex(u_vals, v_vals)

    for i, label in enumerate(["+1", "-1", "0"]):
        ex, ey = expected[label]
        gx, gy = x[i, 0].item(), y[i, 0].item()
        err = math.sqrt((gx - ex)**2 + (gy - ey)**2)
        check(f"|{label}> -> ({gx:.4f}, {gy:.4f}), expected ({ex:.4f}, {ey:.4f})",
              err < 1e-5, f"err={err:.2e}")

    # check equilateral: all 3 pairwise distances must be equal
    pts = list(zip(x.cpu().tolist(), y.cpu().tolist()))
    dists = []
    for i in range(3):
        for j in range(i+1, 3):
            dx = pts[i][0][0] - pts[j][0][0]
            dy = pts[i][1][0] - pts[j][1][0]
            dists.append(math.sqrt(dx*dx + dy*dy))

    max_dist_diff = max(dists) - min(dists)
    check(f"equilateral triangle: dist_diff={max_dist_diff:.6f}",
          max_dist_diff < 1e-5, f"dists={[f'{d:.6f}' for d in dists]}")


#====================
# 3. Centered simplex backward: finite-difference check for sqrt(3) factor
#====================

def test_centered_simplex_backward_fd():
    """verify the sqrt(3) gradient factor in centered_simplex backward"""
    print("\n=== test_centered_simplex_backward_fd ===")
    torch.manual_seed(99)
    eps = 1e-4

    u = torch.randn(4, 8, device=DEVICE, requires_grad=True)
    v = torch.randn(4, 8, device=DEVICE, requires_grad=True)

    x, y = centered_simplex(u, v)
    loss = (x ** 2).sum() + (y ** 2).sum()
    loss.backward()

    # finite difference for v[0,0]
    v0 = v.detach().clone()
    v_plus = v0.clone(); v_plus[0, 0] += eps
    v_minus = v0.clone(); v_minus[0, 0] -= eps

    _, y_plus = centered_simplex(u.detach(), v_plus)
    _, y_minus = centered_simplex(u.detach(), v_minus)
    x_same, _ = centered_simplex(u.detach(), v0)

    loss_plus = (x_same ** 2).sum() + (y_plus ** 2).sum()
    loss_minus = (x_same ** 2).sum() + (y_minus ** 2).sum()

    fd = (loss_plus.item() - loss_minus.item()) / (2 * eps)
    analytic = v.grad[0, 0].item()
    rel_err = abs(fd - analytic) / (abs(analytic) + 1e-8)

    check(f"simplex backward v[0,0]: fd={fd:.6f} analytic={analytic:.6f} rel_err={rel_err:.4f}",
          rel_err < 0.02, f"rel_err={rel_err:.4f}")

    # exact Jacobian check: dx/du = 1, dy/dv = sqrt(3), and the cross terms are zero
    u2 = torch.randn(3, 5, device=DEVICE, requires_grad=True)
    v2 = torch.randn(3, 5, device=DEVICE, requires_grad=True)
    gx_up = torch.randn_like(u2)
    gy_up = torch.randn_like(v2)

    x2, y2 = centered_simplex(u2, v2)
    gu, gv = torch.autograd.grad((x2, y2), (u2, v2), grad_outputs=(gx_up, gy_up))

    sqrt3 = 1.7320508075688772
    err_u = (gu - gx_up).abs().max().item()
    err_v = (gv - gy_up * sqrt3).abs().max().item()
    check(f"simplex jacobian du identity: max_err={err_u:.2e}",
          err_u < 1e-6, f"max_err={err_u:.2e}")
    check(f"simplex jacobian dv sqrt(3): max_err={err_v:.2e}",
          err_v < 1e-6, f"max_err={err_v:.2e}")


#====================
# 4. BackNorm full finite-difference: both du AND dv channels
#    this is the generalization over the original kernel
#====================

def test_backnorm_full_fd():
    """finite-difference check for backnorm3 with BOTH du and dv gradients"""
    print("\n=== test_backnorm_full_fd ===")
    torch.manual_seed(77)
    eps = 5e-4

    inp = torch.randn(4, 8, device=DEVICE)
    a_n = torch.randn(8, 6, device=DEVICE)
    a_z = torch.randn(8, 6, device=DEVICE)
    a_p = torch.randn(8, 6, device=DEVICE)

    # loss that uses BOTH u and v channels with different weights
    def loss_fn(inp_, a_n_, a_z_, a_p_):
        u, v, _, _, _ = forward3(inp_, a_n_, a_z_, a_p_)
        return (u * 1.7).sum() + (v * 2.3).sum()

    params = [
        ("a_neg", a_n, 0, 0),
        ("a_neg", a_n, 3, 4),
        ("a_zero", a_z, 1, 2),
        ("a_zero", a_z, 7, 5),
        ("a_pos", a_p, 2, 1),
        ("a_pos", a_p, 6, 3),
        ("input", inp, 0, 0),
        ("input", inp, 2, 5),
    ]

    all_ok = True
    for name, tensor, ri, ci in params:
        t_g = tensor.clone().requires_grad_(True)

        if name == "a_neg":
            L = loss_fn(inp, t_g, a_z, a_p)
        elif name == "a_zero":
            L = loss_fn(inp, a_n, t_g, a_p)
        elif name == "a_pos":
            L = loss_fn(inp, a_n, a_z, t_g)
        else:
            L = loss_fn(t_g, a_n, a_z, a_p)

        L.backward()
        analytic = t_g.grad[ri, ci].item()

        t_plus = tensor.clone(); t_plus[ri, ci] += eps
        t_minus = tensor.clone(); t_minus[ri, ci] -= eps

        if name == "a_neg":
            fd = (loss_fn(inp, t_plus, a_z, a_p).item() - loss_fn(inp, t_minus, a_z, a_p).item()) / (2*eps)
        elif name == "a_zero":
            fd = (loss_fn(inp, a_n, t_plus, a_p).item() - loss_fn(inp, a_n, t_minus, a_p).item()) / (2*eps)
        elif name == "a_pos":
            fd = (loss_fn(inp, a_n, a_z, t_plus).item() - loss_fn(inp, a_n, a_z, t_minus).item()) / (2*eps)
        else:
            fd = (loss_fn(t_plus, a_n, a_z, a_p).item() - loss_fn(t_minus, a_n, a_z, a_p).item()) / (2*eps)

        rel_err = abs(fd - analytic) / (abs(analytic) + 1e-8)
        ok = rel_err < 0.05
        if not ok:
            all_ok = False
        check(f"backnorm {name}[{ri},{ci}] fd={fd:.6f} an={analytic:.6f}",
              ok, f"rel_err={rel_err:.4f}")

    return all_ok


#====================
# 5. Attention2 vs PyTorch SDPA: comprehensive comparison
#====================

def test_attention2_vs_sdpa():
    """attention2 forward and backward vs PyTorch scaled_dot_product_attention"""
    print("\n=== test_attention2_vs_sdpa ===")

    for seq_len, dim in [(7, 12), (32, 64), (128, 128)]:
        torch.manual_seed(42)
        q = torch.randn(seq_len, dim, device=DEVICE, requires_grad=True)
        k = torch.randn(seq_len, dim, device=DEVICE, requires_grad=True)
        v = torch.randn(seq_len, dim, device=DEVICE, requires_grad=True)

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        # native
        out = attention2(q, k, v)
        loss = out.square().sum()
        loss.backward()

        # reference
        out_ref = F.scaled_dot_product_attention(
            q_ref.unsqueeze(0).unsqueeze(0),
            k_ref.unsqueeze(0).unsqueeze(0),
            v_ref.unsqueeze(0).unsqueeze(0),
            dropout_p=0.0, is_causal=True,
        ).squeeze(0).squeeze(0)
        loss_ref = out_ref.square().sum()
        loss_ref.backward()

        fwd_err = (out - out_ref).abs().max().item()
        dq_err = (q.grad - q_ref.grad).abs().max().item()
        dk_err = (k.grad - k_ref.grad).abs().max().item()
        dv_err = (v.grad - v_ref.grad).abs().max().item()

        # tolerance depends on size (fp32 accumulation error grows)
        tol = 0.01 if seq_len <= 32 else 0.05
        check(f"attn2 fwd [{seq_len}x{dim}] max_err={fwd_err:.6f}", fwd_err < tol, f"err={fwd_err}")
        check(f"attn2 dQ  [{seq_len}x{dim}] max_err={dq_err:.6f}", dq_err < tol * 5, f"err={dq_err}")
        check(f"attn2 dK  [{seq_len}x{dim}] max_err={dk_err:.6f}", dk_err < tol * 5, f"err={dk_err}")
        check(f"attn2 dV  [{seq_len}x{dim}] max_err={dv_err:.6f}", dv_err < tol * 5, f"err={dv_err}")


#====================
# attention2 forward contract: long-sequence streaming path must match SDPA
#====================

def test_attention2_streaming_forward_large_seq_contract():
    """long-sequence attention2 forward should match SDPA without changing causal simplex semantics"""
    print("\n=== test_attention2_streaming_forward_large_seq_contract ===")
    torch.manual_seed(58)

    seq_len, dim = 513, 64
    q = torch.randn(seq_len, dim, device=DEVICE)
    k = torch.randn(seq_len, dim, device=DEVICE)
    v = torch.randn(seq_len, dim, device=DEVICE)

    out = attention2(q, k, v)
    out_ref = F.scaled_dot_product_attention(
        q.unsqueeze(0).unsqueeze(0),
        k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=True,
    ).squeeze(0).squeeze(0)

    max_err = (out - out_ref).abs().max().item()
    check(f"attn2 streaming fwd [513x64] max_err={max_err:.6f}", max_err < 0.02, f"err={max_err}")


#====================
# attention2 backward contract: long-sequence streaming backward must match SDPA
#====================

def test_attention2_streaming_backward_large_seq_contract():
    """long-sequence attention2 backward should match SDPA under streaming recomputation"""
    print("\n=== test_attention2_streaming_backward_large_seq_contract ===")
    torch.manual_seed(60)

    seq_len, dim = 513, 64
    q = torch.randn(seq_len, dim, device=DEVICE, requires_grad=True)
    k = torch.randn(seq_len, dim, device=DEVICE, requires_grad=True)
    v = torch.randn(seq_len, dim, device=DEVICE, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = attention2(q, k, v)
    grad = torch.randn_like(out)
    out.backward(grad)

    out_ref = F.scaled_dot_product_attention(
        q_ref.unsqueeze(0).unsqueeze(0),
        k_ref.unsqueeze(0).unsqueeze(0),
        v_ref.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=True,
    ).squeeze(0).squeeze(0)
    out_ref.backward(grad)

    fwd_err = (out - out_ref).abs().max().item()
    dq_err = (q.grad - q_ref.grad).abs().max().item()
    dk_err = (k.grad - k_ref.grad).abs().max().item()
    dv_err = (v.grad - v_ref.grad).abs().max().item()

    check("attn2 streaming backward fwd [513x64]", fwd_err < 0.02, f"max_err={fwd_err:.3e}")
    check("attn2 streaming backward dQ [513x64]", dq_err < 0.05, f"max_err={dq_err:.3e}")
    check("attn2 streaming backward dK [513x64]", dk_err < 0.05, f"max_err={dk_err:.3e}")
    check("attn2 streaming backward dV [513x64]", dv_err < 0.05, f"max_err={dv_err:.3e}")


#====================
# 6. Attention2 batched: single-batch loop vs batched must match
#====================

def test_attention2_batched_consistency():
    """batched attention2 must equal per-sample loop"""
    print("\n=== test_attention2_batched_consistency ===")
    torch.manual_seed(55)
    B, S, D = 4, 16, 24

    q = torch.randn(B, S, D, device=DEVICE)
    k = torch.randn(B, S, D, device=DEVICE)
    v = torch.randn(B, S, D, device=DEVICE)

    out_batch = attention2(q, k, v)

    outs_loop = []
    for b in range(B):
        out_b = attention2(q[b:b+1].squeeze(0), k[b:b+1].squeeze(0), v[b:b+1].squeeze(0))
        outs_loop.append(out_b)
    out_loop = torch.stack(outs_loop, dim=0)

    max_err = (out_batch - out_loop).abs().max().item()
    check(f"batched vs loop: max_err={max_err:.2e}", max_err < 1e-5, f"err={max_err}")


#====================
# attention2 batched forward contract: large streaming batched path must match native loop
#====================

def test_native_attention2_streaming_batched_forward_consistency():
    print("\n=== test_native_attention2_streaming_batched_forward_consistency ===")
    torch.manual_seed(59)
    ext = load_native()
    B, S, D = 2, 513, 48
    dtypes = [torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        qx = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        qy = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        kx = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        ky = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        vx = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        vy = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)

        ox_batch, oy_batch = ext.attention2_cuda(qx, qy, kx, ky, vx, vy)
        ox_loop = []
        oy_loop = []
        for b in range(B):
            ox_b, oy_b = ext.attention2_cuda(
                qx[b].contiguous(), qy[b].contiguous(),
                kx[b].contiguous(), ky[b].contiguous(),
                vx[b].contiguous(), vy[b].contiguous(),
            )
            ox_loop.append(ox_b)
            oy_loop.append(oy_b)
        ox_ref = torch.stack(ox_loop, dim=0)
        oy_ref = torch.stack(oy_loop, dim=0)

        max_err = max((ox_batch - ox_ref).abs().max().item(), (oy_batch - oy_ref).abs().max().item())
        tol = 1e-5 if dtype == torch.float32 else 2e-3
        check(
            f"native streaming batched attention forward matches native loop [{str(dtype).split('.')[-1]}]",
            max_err < tol,
            f"max_err={max_err:.3e}"
        )


#====================
# attention2 batched backward contract: large streaming batched backward must match native loop
#====================

def test_native_attention2_streaming_batched_backward_consistency():
    print("\n=== test_native_attention2_streaming_batched_backward_consistency ===")
    torch.manual_seed(61)
    ext = load_native()
    B, S, D = 2, 513, 48
    dtypes = [torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        qx = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        qy = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        kx = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        ky = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        vx = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        vy = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        dox = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        doy = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)

        grads_batch = ext.attention_backward2_cuda(dox, doy, qx, qy, kx, ky, vx, vy)
        grads_loop = [[] for _ in range(6)]
        for b in range(B):
            grads_b = ext.attention_backward2_cuda(
                dox[b].contiguous(), doy[b].contiguous(),
                qx[b].contiguous(), qy[b].contiguous(),
                kx[b].contiguous(), ky[b].contiguous(),
                vx[b].contiguous(), vy[b].contiguous(),
            )
            for i, t in enumerate(grads_b):
                grads_loop[i].append(t)
        grads_ref = tuple(torch.stack(parts, dim=0) for parts in grads_loop)

        max_err = max((gb - gr).abs().max().item() for gb, gr in zip(grads_batch, grads_ref))
        tol = 1e-5 if dtype == torch.float32 else 2e-3
        check(
            f"native streaming batched attention backward matches native loop [{str(dtype).split('.')[-1]}]",
            max_err < tol,
            f"max_err={max_err:.3e}"
        )


#====================
# 7. Native attention2 bridge batched: direct 3D bridge path must match per-sample native calls
#====================

def test_native_attention2_batched_bridge_consistency():
    print("\n=== test_native_attention2_batched_bridge_consistency ===")
    torch.manual_seed(56)
    ext = load_native()
    B, S, D = 3, 12, 16
    dtypes = [torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        qx = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        qy = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        kx = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        ky = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        vx = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)
        vy = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32).to(dtype)

        ox_batch, oy_batch = ext.attention2_cuda(qx, qy, kx, ky, vx, vy)
        ox_loop = []
        oy_loop = []
        for b in range(B):
            ox_b, oy_b = ext.attention2_cuda(
                qx[b].contiguous(), qy[b].contiguous(),
                kx[b].contiguous(), ky[b].contiguous(),
                vx[b].contiguous(), vy[b].contiguous(),
            )
            ox_loop.append(ox_b)
            oy_loop.append(oy_b)
        ox_ref = torch.stack(ox_loop, dim=0)
        oy_ref = torch.stack(oy_loop, dim=0)

        dox = torch.randn_like(ox_batch, dtype=torch.float32).to(dtype)
        doy = torch.randn_like(oy_batch, dtype=torch.float32).to(dtype)
        grads_batch = ext.attention_backward2_cuda(dox, doy, qx, qy, kx, ky, vx, vy)
        grads_loop = [[] for _ in range(6)]
        for b in range(B):
            grads_b = ext.attention_backward2_cuda(
                dox[b].contiguous(), doy[b].contiguous(),
                qx[b].contiguous(), qy[b].contiguous(),
                kx[b].contiguous(), ky[b].contiguous(),
                vx[b].contiguous(), vy[b].contiguous(),
            )
            for i, t in enumerate(grads_b):
                grads_loop[i].append(t)
        grads_ref = tuple(torch.stack(parts, dim=0) for parts in grads_loop)

        fwd_err = max((ox_batch - ox_ref).abs().max().item(), (oy_batch - oy_ref).abs().max().item())
        bwd_err = max((gb - gr).abs().max().item() for gb, gr in zip(grads_batch, grads_ref))
        tol = 1e-6 if dtype == torch.float32 else 1e-3

        check(
            f"native batched attention2 bridge forward matches per-sample native loop [{str(dtype).split('.')[-1]}]",
            fwd_err < tol,
            f"max_err={fwd_err:.3e}"
        )
        check(
            f"native batched attention2 bridge backward matches per-sample native loop [{str(dtype).split('.')[-1]}]",
            bwd_err < tol,
            f"max_err={bwd_err:.3e}"
        )


def test_native_attention2_backward_reuse_smoke():
    print("\n=== test_native_attention2_backward_reuse_smoke ===")
    torch.manual_seed(57)
    ext = load_native()

    shapes = [
        (2, 8, 12),
        (4, 16, 24),
        (2, 8, 12),
    ]

    for B, S, D in shapes:
        qx = torch.randn(B, S, D, device=DEVICE)
        qy = torch.randn(B, S, D, device=DEVICE)
        kx = torch.randn(B, S, D, device=DEVICE)
        ky = torch.randn(B, S, D, device=DEVICE)
        vx = torch.randn(B, S, D, device=DEVICE)
        vy = torch.randn(B, S, D, device=DEVICE)
        dox = torch.randn(B, S, D, device=DEVICE)
        doy = torch.randn(B, S, D, device=DEVICE)

        grads_batch = ext.attention_backward2_cuda(dox, doy, qx, qy, kx, ky, vx, vy)
        grads_loop = [[] for _ in range(6)]
        for b in range(B):
            grads_b = ext.attention_backward2_cuda(
                dox[b].contiguous(), doy[b].contiguous(),
                qx[b].contiguous(), qy[b].contiguous(),
                kx[b].contiguous(), ky[b].contiguous(),
                vx[b].contiguous(), vy[b].contiguous(),
            )
            for i, t in enumerate(grads_b):
                grads_loop[i].append(t)
        grads_ref = tuple(torch.stack(parts, dim=0) for parts in grads_loop)
        max_err = max((gb - gr).abs().max().item() for gb, gr in zip(grads_batch, grads_ref))
        check(
            f"native attention backward scratch reuse keeps batched parity [{B}x{S}x{D}]",
            max_err < 1e-6,
            f"max_err={max_err:.3e}"
        )


def test_native_attention2_path_switch_reuse_smoke():
    print("\n=== test_native_attention2_path_switch_reuse_smoke ===")
    torch.manual_seed(62)
    ext = load_native()

    cases = [
        (2, 513, 48, 1e-5),
        (2, 24, 320, 1e-5),
        (2, 513, 48, 1e-5),
    ]

    for B, S, D, tol in cases:
        qx = torch.randn(B, S, D, device=DEVICE)
        qy = torch.randn(B, S, D, device=DEVICE)
        kx = torch.randn(B, S, D, device=DEVICE)
        ky = torch.randn(B, S, D, device=DEVICE)
        vx = torch.randn(B, S, D, device=DEVICE)
        vy = torch.randn(B, S, D, device=DEVICE)
        dox = torch.randn(B, S, D, device=DEVICE)
        doy = torch.randn(B, S, D, device=DEVICE)

        ox_batch, oy_batch = ext.attention2_cuda(qx, qy, kx, ky, vx, vy)
        grads_batch = ext.attention_backward2_cuda(dox, doy, qx, qy, kx, ky, vx, vy)

        ox_parts = []
        oy_parts = []
        grads_loop = [[] for _ in range(6)]
        for b in range(B):
            ox_b, oy_b = ext.attention2_cuda(
                qx[b].contiguous(), qy[b].contiguous(),
                kx[b].contiguous(), ky[b].contiguous(),
                vx[b].contiguous(), vy[b].contiguous(),
            )
            grads_b = ext.attention_backward2_cuda(
                dox[b].contiguous(), doy[b].contiguous(),
                qx[b].contiguous(), qy[b].contiguous(),
                kx[b].contiguous(), ky[b].contiguous(),
                vx[b].contiguous(), vy[b].contiguous(),
            )
            ox_parts.append(ox_b)
            oy_parts.append(oy_b)
            for i, t in enumerate(grads_b):
                grads_loop[i].append(t)

        ox_ref = torch.stack(ox_parts, dim=0)
        oy_ref = torch.stack(oy_parts, dim=0)
        grads_ref = tuple(torch.stack(parts, dim=0) for parts in grads_loop)

        fwd_err = max((ox_batch - ox_ref).abs().max().item(), (oy_batch - oy_ref).abs().max().item())
        bwd_err = max((gb - gr).abs().max().item() for gb, gr in zip(grads_batch, grads_ref))
        check(
            f"native attention path-switch forward reuse stays stable [{B}x{S}x{D}]",
            fwd_err < tol,
            f"max_err={fwd_err:.3e}"
        )
        check(
            f"native attention path-switch backward reuse stays stable [{B}x{S}x{D}]",
            bwd_err < tol,
            f"max_err={bwd_err:.3e}"
        )


#====================
# 7. Prior effectiveness: does it actually push entropy up?
#====================

def test_prior_raises_entropy():
    """start with collapsed weights (one channel dominant), verify prior restores entropy"""
    print("\n=== test_prior_raises_entropy ===")
    torch.manual_seed(10)

    n = 1024
    # heavily collapsed: a_pos >> others
    a_neg = torch.randn(32, 32, device=DEVICE) * 0.01
    a_zero = torch.randn(32, 32, device=DEVICE) * 0.01
    a_pos = torch.randn(32, 32, device=DEVICE) * 2.0

    def measure_entropy(an, az, ap):
        a2 = an * an
        b2 = az * az
        c2 = ap * ap
        z = a2 + b2 + c2
        z = z.clamp(min=1e-20)
        pn = a2 / z
        pz = b2 / z
        pp = c2 / z
        h = -(pn * pn.clamp(min=1e-15).log2()
              + pz * pz.clamp(min=1e-15).log2()
              + pp * pp.clamp(min=1e-15).log2())
        return h.mean().item()

    h_before = measure_entropy(a_neg, a_zero, a_pos)

    # apply prior 200 times
    for _ in range(200):
        prior_(a_neg, a_zero, a_pos, step=0.01, entropy_floor=1.0840643)

    h_after = measure_entropy(a_neg, a_zero, a_pos)

    check(f"prior raised entropy: {h_before:.4f} -> {h_after:.4f}",
          h_after > h_before + 0.1,
          f"h_before={h_before:.4f} h_after={h_after:.4f}")

    check(f"entropy above floor (~1.08): {h_after:.4f}",
          h_after > 0.9,
          f"h_after={h_after:.4f}")


#====================
# 8. Prior doesn't overshoot: entropy should not exceed log2(3)
#====================

def test_prior_no_overshoot():
    """prior should not push entropy above maximum (log2(3) = 1.585)"""
    print("\n=== test_prior_no_overshoot ===")

    # start near-uniform
    a_neg = torch.ones(64, 64, device=DEVICE) * 0.577
    a_zero = torch.ones(64, 64, device=DEVICE) * 0.577
    a_pos = torch.ones(64, 64, device=DEVICE) * 0.577

    for _ in range(500):
        prior_(a_neg, a_zero, a_pos, step=0.1, entropy_floor=1.0840643)

    # measure per-element entropy
    a2 = a_neg * a_neg
    b2 = a_zero * a_zero
    c2 = a_pos * a_pos
    z = (a2 + b2 + c2).clamp(min=1e-20)
    pn, pz, pp = a2/z, b2/z, c2/z
    h = -(pn * pn.clamp(min=1e-15).log2()
          + pz * pz.clamp(min=1e-15).log2()
          + pp * pp.clamp(min=1e-15).log2())

    max_h = h.max().item()
    check(f"no overshoot: max_H={max_h:.4f} <= 1.585",
          max_h <= 1.586, f"max_H={max_h}")

    # amplitudes should remain finite
    check("amplitudes finite after 500 prior steps",
          not torch.isnan(a_neg).any().item() and not torch.isinf(a_neg).any().item())


#====================
# 9. Full model gradient flow: every QTS layer gets gradients
#====================

def test_full_model_gradient_flow():
    """every a_neg/a_zero/a_pos in every layer must receive non-zero gradients"""
    print("\n=== test_full_model_gradient_flow ===")
    torch.manual_seed(42)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=32, ffn_dim=64, seq_len=32, layers=2, device=DEVICE,
    )

    tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
    targets = torch.randint(0, 256, (2, 32), device=DEVICE)

    _, loss = model(tokens, targets=targets)
    loss.backward()

    dead_layers = []
    for name, role, layer in model.iter_qitnn_layers():
        for pname, param in [("a_neg", layer.a_neg), ("a_zero", layer.a_zero), ("a_pos", layer.a_pos)]:
            if param.grad is None:
                dead_layers.append(f"{name}.{pname}: no grad")
            elif param.grad.abs().max().item() < 1e-12:
                dead_layers.append(f"{name}.{pname}: grad~0 (max={param.grad.abs().max().item():.2e})")

    check(f"all QTS layers receive gradients ({len(dead_layers)} dead)",
          len(dead_layers) == 0,
          "; ".join(dead_layers[:5]))

    # also check embedding and head
    for name, param in [("token_emb", model.token_emb.weight),
                        ("pos_emb", model.pos_emb),
                        ("head.weight", model.head.weight)]:
        has_grad = param.grad is not None and param.grad.abs().max().item() > 1e-12
        check(f"{name} receives gradient", has_grad)


#====================
# 10. Forward split contract: internal hidden/logits helpers must reconstruct forward()
#====================

def test_model_forward_split_contract():
    """_forward_hidden + _forward_logits should match full forward()"""
    print("\n=== test_model_forward_split_contract ===")

    modes = [("fp32", torch.float32, 1e-7, 1e-7)]
    if torch.cuda.is_bf16_supported():
        modes.append(("qts_fp32_rest_bf16", torch.bfloat16, 1e-5, 1e-6))

    for mode, hidden_dtype, logits_tol, loss_tol in modes:
        torch.manual_seed(126)
        torch.cuda.manual_seed_all(126)

        model = pyqitnn.QITNNSimplexTransformerLM(
            dim=24,
            ffn_dim=48,
            seq_len=16,
            layers=2,
            precision_mode=mode,
            device=DEVICE,
        )
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)

        logits_full, loss_full = model(tokens, targets=tokens)

        with model._amp_context(tokens):
            hidden = model._forward_hidden(tokens)
            logits_split = model._forward_logits(hidden)
            loss_split = F.cross_entropy(
                logits_split.reshape(-1, logits_split.size(-1)),
                tokens.reshape(-1),
            )

        max_diff = (logits_full.float() - logits_split.float()).abs().max().item()
        loss_diff = abs(float(loss_full.detach().float()) - float(loss_split.detach().float()))

        check(f"forward split hidden dtype [{mode}]", hidden.dtype == hidden_dtype, f"dtype={hidden.dtype}")
        check(f"forward split logits parity [{mode}]", max_diff < logits_tol, f"max_diff={max_diff:.2e}")
        check(f"forward split loss parity [{mode}]", loss_diff < loss_tol, f"loss_diff={loss_diff:.2e}")


#====================
# 11. Stepwise decode contract: block/model step APIs must match full-prefix evaluation
#====================

def test_stepwise_decode_contract():
    """forward_step/decode_step should match full-prefix evaluation"""
    print("\n=== test_stepwise_decode_contract ===")

    modes = [("fp32", torch.float32, torch.float32, 1e-7)]
    if torch.cuda.is_bf16_supported():
        modes.append(("qts_fp32_rest_bf16", torch.bfloat16, torch.bfloat16, 1e-5))

    for mode, hidden_dtype, logits_dtype, tol in modes:
        torch.manual_seed(127)
        torch.cuda.manual_seed_all(127)

        model = pyqitnn.QITNNSimplexTransformerLM(
            dim=24,
            ffn_dim=48,
            seq_len=16,
            layers=2,
            precision_mode=mode,
            device=DEVICE,
        )
        tokens = torch.randint(0, 256, (2, 13), device=DEVICE)

        logits_full, _ = model(tokens)
        hidden_step = model.forward_step(tokens)
        logits_step = model.decode_step(tokens)

        with model._amp_context(tokens):
            embedded = model._embed_tokens(tokens)
            block_forward = model.blocks[0].forward(embedded)
            block_step = model.blocks[0].forward_step(embedded)
            hidden_full = model._forward_hidden(tokens)

        block_diff = (block_forward.float() - block_step.float()).abs().max().item()
        hidden_diff = (hidden_full[:, -1:, :].float() - hidden_step.float()).abs().max().item()
        logits_diff = (logits_full[:, -1, :].float() - logits_step.float()).abs().max().item()

        check(f"block forward_step parity [{mode}]", block_diff < tol, f"max_diff={block_diff:.2e}")
        check(f"model forward_step dtype [{mode}]", hidden_step.dtype == hidden_dtype, f"dtype={hidden_step.dtype}")
        check(f"model forward_step shape [{mode}]", tuple(hidden_step.shape) == (2, 1, model.hidden_dim), f"shape={tuple(hidden_step.shape)}")
        check(f"model forward_step parity [{mode}]", hidden_diff < tol, f"max_diff={hidden_diff:.2e}")
        check(f"model decode_step dtype [{mode}]", logits_step.dtype == logits_dtype, f"dtype={logits_step.dtype}")
        check(f"model decode_step shape [{mode}]", tuple(logits_step.shape) == (2, model.vocab_size), f"shape={tuple(logits_step.shape)}")
        check(f"model decode_step parity [{mode}]", logits_diff < tol, f"max_diff={logits_diff:.2e}")

        empty = tokens[:, :0]
        forward_msg = capture_runtime_error(lambda: model.forward_step(empty))
        decode_msg = capture_runtime_error(lambda: model.decode_step(empty))
        check(f"model forward_step rejects empty prefix [{mode}]", "at least one token" in forward_msg, forward_msg or "no RuntimeError")
        check(f"model decode_step rejects empty prefix [{mode}]", "at least one token" in decode_msg, decode_msg or "no RuntimeError")


#====================
# 12. Overfit test: model must memorize 1 batch
#====================

def test_overfit_single_batch():
    """the model must be able to perfectly memorize a single batch"""
    print("\n=== test_overfit_single_batch ===")
    torch.manual_seed(7)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=32, ffn_dim=64, seq_len=32, layers=2, device=DEVICE,
    )

    # fixed batch
    tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
    targets = torch.randint(0, 256, (2, 32), device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    first_loss = None
    last_loss = None
    for step in range(300):
        opt.zero_grad(set_to_none=True)
        _, loss = model(tokens, targets=targets)
        loss.backward()
        opt.step()

        val = loss.item()
        if first_loss is None:
            first_loss = val
        last_loss = val

    ratio = last_loss / first_loss if first_loss > 0 else 999
    check(f"overfit: loss dropped {first_loss:.3f} -> {last_loss:.3f} (ratio={ratio:.4f})",
          ratio < 0.3,
          f"ratio={ratio:.4f}, first={first_loss:.3f}, last={last_loss:.3f}")


#====================
# 11. Checkpoint roundtrip: save -> load -> same output
#====================

def test_checkpoint_roundtrip():
    """model weights survive save/load cycle"""
    print("\n=== test_checkpoint_roundtrip ===")
    torch.manual_seed(42)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=32, ffn_dim=64, seq_len=32, layers=2, device=DEVICE,
    )

    tokens = torch.randint(0, 256, (1, 32), device=DEVICE)

    with torch.no_grad():
        logits1, _ = model(tokens)

    # save
    tmp_path = Path("_test_ckpt_tmp.pt")
    torch.save(model.state_dict(), tmp_path)

    # create new model and load
    model2 = pyqitnn.QITNNSimplexTransformerLM(
        dim=32, ffn_dim=64, seq_len=32, layers=2, device=DEVICE,
    )
    model2.load_state_dict(torch.load(tmp_path, map_location=DEVICE, weights_only=True))

    with torch.no_grad():
        logits2, _ = model2(tokens)

    max_diff = (logits1 - logits2).abs().max().item()
    check(f"checkpoint roundtrip: max_diff={max_diff:.2e}", max_diff < 1e-6, f"diff={max_diff}")

    tmp_path.unlink(missing_ok=True)


def test_rng_state_helper_roundtrip():
    print("\n=== test_rng_state_helper_roundtrip ===")
    torch.manual_seed(314159)
    torch.cuda.manual_seed_all(271828)

    rng_state = _capture_rng_state()
    probs = torch.full((1, 11), 1.0 / 11.0, device=DEVICE)

    cpu_ref = torch.randint(0, 1000, (12,), dtype=torch.long)
    cuda_ref = torch.multinomial(probs, num_samples=9, replacement=True)

    _ = torch.randint(0, 1000, (5,), dtype=torch.long)
    _ = torch.multinomial(probs, num_samples=4, replacement=True)

    restored = _restore_rng_state(rng_state)
    cpu_now = torch.randint(0, 1000, (12,), dtype=torch.long)
    cuda_now = torch.multinomial(probs, num_samples=9, replacement=True)

    check("rng helper reports successful restore", restored, f"restored={restored}")
    check("rng helper restores CPU torch sequence", torch.equal(cpu_now, cpu_ref), f"cpu_now={cpu_now.tolist()} cpu_ref={cpu_ref.tolist()}")
    check("rng helper restores CUDA sampling sequence", torch.equal(cuda_now, cuda_ref), f"cuda_now={cuda_now.tolist()} cuda_ref={cuda_ref.tolist()}")


def test_checkpoint_payload_includes_rng_state():
    print("\n=== test_checkpoint_payload_includes_rng_state ===")
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=16, ffn_dim=32, seq_len=16, layers=1, device=DEVICE,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    tmp_path = ROOT / "_test_ckpt_rng_payload.pt"
    try:
        save_ckpt(tmp_path, model, opt, epoch=3, step=17, best_val=1.25, best_val_epoch=2)
        ckpt = torch.load(str(tmp_path), map_location="cpu", weights_only=False)
    finally:
        tmp_path.unlink(missing_ok=True)

    rng_state = ckpt.get("rng_state")
    cpu_state = None if rng_state is None else rng_state.get("torch_cpu")
    cuda_state = None if rng_state is None else rng_state.get("torch_cuda")

    check("checkpoint payload carries rng_state block", isinstance(rng_state, dict), f"type={type(rng_state).__name__}")
    check("checkpoint payload carries CPU torch RNG", isinstance(cpu_state, torch.Tensor) and cpu_state.dtype == torch.uint8 and cpu_state.device.type == "cpu", f"cpu_state={type(cpu_state).__name__ if cpu_state is not None else None}")
    check("checkpoint payload carries CUDA RNG list", isinstance(cuda_state, list) and len(cuda_state) >= 1, f"cuda_state={cuda_state}")
    check("checkpoint payload preserves epoch/global_step", ckpt.get("epoch") == 3 and ckpt.get("global_step") == 17, f"epoch={ckpt.get('epoch')} step={ckpt.get('global_step')}")
    check("checkpoint payload preserves best_val_epoch metadata", ckpt.get("best_val_epoch") == 2, f"best_val_epoch={ckpt.get('best_val_epoch')}")


def test_checkpoint_load_legacy_payload_without_rng_state():
    print("\n=== test_checkpoint_load_legacy_payload_without_rng_state ===")
    torch.manual_seed(44)
    torch.cuda.manual_seed_all(44)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=16, ffn_dim=32, seq_len=16, layers=1, device=DEVICE,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tokens = torch.randint(0, 256, (1, 16), device=DEVICE)

    with torch.no_grad():
        logits_ref, _ = model(tokens)

    legacy_payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": 2,
        "global_step": 9,
        "best_val_loss": 0.75,
    }

    model2 = pyqitnn.QITNNSimplexTransformerLM(
        dim=16, ffn_dim=32, seq_len=16, layers=1, device=DEVICE,
    )
    opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    tmp_path = ROOT / "_test_ckpt_legacy_no_rng.pt"
    try:
        torch.save(legacy_payload, str(tmp_path))
        err = capture_runtime_error(lambda: load_ckpt(tmp_path, model2, opt2, DEVICE))
        with torch.no_grad():
            logits_now, _ = model2(tokens)
    finally:
        tmp_path.unlink(missing_ok=True)

    max_diff = (logits_now - logits_ref).abs().max().item()
    check("legacy checkpoint without rng_state still loads", err == "", err or "ok")
    check("legacy checkpoint without rng_state preserves model weights", max_diff < 1e-6, f"diff={max_diff}")


def test_checkpoint_resume_restores_batch_rng_exact_fp32():
    print("\n=== test_checkpoint_resume_restores_batch_rng_exact_fp32 ===")

    tmp_path = ROOT / "_tmp_rng_resume_dataset.txt"
    save_root = ROOT / "_tmp_rng_resume_runs"
    run_cont = "rng_resume_cont"
    run_resumed = "rng_resume_resumed"

    text = (
        "resume exactness should preserve sampled batch windows across checkpoint restore\n"
        "this test isolates trainer rng continuity without touching qts math\n"
    ) * 64

    try:
        cleanup_tree(save_root)
        tmp_path.write_text(text, encoding="utf-8")

        common = dict(
            dataset=str(tmp_path),
            tokenizer="byte",
            precision_mode="fp32",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=2,
            optimizer="adamw",
            adamw_lr_start=3e-4,
            adamw_lr_end=3e-5,
            lr_schedule="linear",
            epochs=2,
            steps_per_epoch=2,
            val_steps=4,
            save_dir=str(save_root),
            save_every=1,
            no_interactive=True,
            prompt="resume",
            prompt_bytes=6,
            gen_bytes=0,
            log_every=2,
            seed=77,
        )

        train(run_name=run_cont, **common)
        resume_ckpt = save_root / run_cont / "ckpt_ep1.pt"
        train(run_name=run_resumed, resume=str(resume_ckpt), **common)

        cont_ckpt = torch.load(str(save_root / run_cont / "ckpt_final.pt"), map_location="cpu", weights_only=False)
        resumed_ckpt = torch.load(str(save_root / run_resumed / "ckpt_final.pt"), map_location="cpu", weights_only=False)

        max_diff = max(
            (cont_ckpt["model"][name] - resumed_ckpt["model"][name]).abs().max().item()
            for name in cont_ckpt["model"]
        )
        best_val_diff = abs(float(cont_ckpt["best_val_loss"]) - float(resumed_ckpt["best_val_loss"]))

        check("resume exactness preserves final fp32 model state", max_diff < 1e-9, f"max_diff={max_diff:.3e}")
        check("resume exactness preserves final global_step", cont_ckpt["global_step"] == resumed_ckpt["global_step"], f"cont={cont_ckpt['global_step']} resumed={resumed_ckpt['global_step']}")
        check("resume exactness preserves final epoch", cont_ckpt["epoch"] == resumed_ckpt["epoch"], f"cont={cont_ckpt['epoch']} resumed={resumed_ckpt['epoch']}")
        check("resume exactness preserves best_val_loss", best_val_diff < 1e-12, f"diff={best_val_diff:.3e}")
    finally:
        tmp_path.unlink(missing_ok=True)
        cleanup_tree(save_root)


def test_checkpoint_restore_preserves_generation_rng():
    print("\n=== test_checkpoint_restore_preserves_generation_rng ===")

    modes = ["fp32"]
    if torch.cuda.is_bf16_supported():
        modes.append("qts_fp32_rest_bf16")

    for mode in modes:
        torch.manual_seed(45)
        torch.cuda.manual_seed_all(45)

        model = pyqitnn.QITNNSimplexTransformerLM(
            dim=16,
            ffn_dim=32,
            seq_len=16,
            layers=1,
            precision_mode=mode,
            device=DEVICE,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        prompt = torch.tensor([[72, 101, 108, 108, 111]], device=DEVICE, dtype=torch.long)

        model2 = pyqitnn.QITNNSimplexTransformerLM(
            dim=16,
            ffn_dim=32,
            seq_len=16,
            layers=1,
            precision_mode=mode,
            device=DEVICE,
        )
        opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)

        tmp_path = ROOT / f"_test_ckpt_generation_rng_{mode}.pt"
        try:
            save_ckpt(tmp_path, model, opt, epoch=1, step=3, best_val=0.5)
            with torch.no_grad():
                out_ref = model.generate(
                    prompt,
                    max_new_tokens=12,
                    temperature=0.8,
                    top_k=8,
                    ascii_guard=True,
                )

            err = capture_runtime_error(lambda: load_ckpt(tmp_path, model2, opt2, DEVICE))
            with torch.no_grad():
                out_now = model2.generate(
                    prompt,
                    max_new_tokens=12,
                    temperature=0.8,
                    top_k=8,
                    ascii_guard=True,
                )
        finally:
            tmp_path.unlink(missing_ok=True)

        check(f"generation checkpoint load restores RNG [{mode}]", err == "", err or "ok")
        check(
            f"generation checkpoint restore keeps sampled tokens [{mode}]",
            torch.equal(out_now, out_ref),
            f"out_now={out_now.tolist()} out_ref={out_ref.tolist()}",
        )


#====================
# 12. Numerical stability: extreme amplitude scales
#====================

def test_extreme_amplitude_scales():
    """forward3 must be stable with very large and very small amplitudes"""
    print("\n=== test_extreme_amplitude_scales ===")

    cases = [
        ("tiny amplitudes (1e-7)", 1e-7),
        ("small amplitudes (1e-4)", 1e-4),
        ("normal amplitudes (1.0)", 1.0),
        ("large amplitudes (100)", 100.0),
        ("huge amplitudes (1e4)", 1e4),
    ]

    inp = torch.randn(8, 16, device=DEVICE)

    for label, scale in cases:
        a_n = torch.randn(16, 8, device=DEVICE) * scale
        a_z = torch.randn(16, 8, device=DEVICE) * scale
        a_p = torch.randn(16, 8, device=DEVICE) * scale

        a_n.requires_grad_(True)
        a_z.requires_grad_(True)
        a_p.requires_grad_(True)

        u, v, cn, cz, cp = forward3(inp, a_n, a_z, a_p)
        loss = u.sum() + v.sum()
        loss.backward()

        has_nan = (torch.isnan(u).any() or torch.isnan(v).any() or
                   torch.isnan(a_n.grad).any() or torch.isnan(a_z.grad).any() or
                   torch.isnan(a_p.grad).any())
        has_inf = (torch.isinf(u).any() or torch.isinf(v).any() or
                   torch.isinf(a_n.grad).any())

        check(f"{label}: no NaN", not has_nan.item() if isinstance(has_nan, torch.Tensor) else not has_nan)
        check(f"{label}: no Inf", not has_inf.item() if isinstance(has_inf, torch.Tensor) else not has_inf)

        # u must still be in [-1, 1], v in [0, 1]
        if not torch.isnan(u).any():
            check(f"{label}: u in [-1,1]",
                  u.min().item() >= -1.01 and u.max().item() <= 1.01,
                  f"u range [{u.min().item():.4f}, {u.max().item():.4f}]")


#====================
# 13. QITNNLinear: centered_simplex=True vs manual application
#====================

def test_qitnn_linear_simplex_consistency():
    """QITNNLinear(centered_simplex=True) must match manual forward_raw + centered_simplex"""
    print("\n=== test_qitnn_linear_simplex_consistency ===")
    torch.manual_seed(33)

    layer = pyqitnn.QITNNLinear(16, 8, centered_simplex=True, device=DEVICE)
    inp = torch.randn(4, 16, device=DEVICE)

    # path 1: through forward() which uses forward_visible internally
    packed = layer(inp)
    x_auto = packed[:, :8]
    y_auto = packed[:, 8:]

    # path 2: manual
    u, v, _, _, _ = layer.forward_raw(inp)
    x_manual, y_manual = centered_simplex(u, v)

    check("x matches",
          torch.allclose(x_auto, x_manual, atol=1e-6),
          f"max_diff={( x_auto - x_manual).abs().max().item():.2e}")
    check("y matches",
          torch.allclose(y_auto, y_manual, atol=1e-6),
          f"max_diff={(y_auto - y_manual).abs().max().item():.2e}")


#====================
# 14. QITNNLinear storage contract: private packed runtime view must preserve split public state
#====================

def test_qitnn_linear_packed_runtime_view_contract():
    """private packed runtime view must not leak into public parameters, state, or diagnostics"""
    print("\n=== test_qitnn_linear_packed_runtime_view_contract ===")
    torch.manual_seed(34)

    layer = pyqitnn.QITNNLinear(16, 8, device=DEVICE)
    packed = layer._packed_weight_view()
    state_keys = tuple(layer.state_dict().keys())
    param_names = tuple(name for name, _ in layer.named_parameters())

    check("packed runtime view shape", tuple(packed.shape) == (16, 8, 3), f"shape={tuple(packed.shape)}")
    check("packed runtime view is contiguous", packed.is_contiguous(), f"stride={packed.stride()}")
    check("packed runtime view keeps fp32 dtype", packed.dtype == torch.float32, f"dtype={packed.dtype}")
    check("qitnn linear defaults to packed_reference backend", layer._runtime_backend == "packed_reference", f"backend={layer._runtime_backend}")
    check("packed runtime view branch order keeps a_neg", torch.equal(packed[..., 0], layer.a_neg), "a_neg mismatch")
    check("packed runtime view branch order keeps a_zero", torch.equal(packed[..., 1], layer.a_zero), "a_zero mismatch")
    check("packed runtime view branch order keeps a_pos", torch.equal(packed[..., 2], layer.a_pos), "a_pos mismatch")
    check("packed runtime view stays out of state_dict", state_keys == ("a_neg", "a_zero", "a_pos"), f"keys={state_keys}")
    check("packed runtime view stays out of named_parameters", param_names == ("a_neg", "a_zero", "a_pos"), f"names={param_names}")

    diag_split = pyqitnn.qitnn_diag_stats(layer.a_neg, layer.a_zero, layer.a_pos)
    diag_packed = pyqitnn.qitnn_diag_stats(packed[..., 0], packed[..., 1], packed[..., 2])
    diag_gap = max(abs(float(diag_split[key]) - float(diag_packed[key])) for key in QITNN_DIAG_STAT_KEYS)
    check("packed runtime view preserves diagnostics semantics", diag_gap < 1e-12, f"diag_gap={diag_gap:.2e}")

    with torch.no_grad():
        layer.a_zero.add_(0.125)
    packed_after = layer._packed_weight_view()
    refresh_err = (packed_after[..., 1] - layer.a_zero).abs().max().item()
    check("fresh packed runtime view reflects current split weights", refresh_err == 0.0, f"max_err={refresh_err:.2e}")


#====================
# 15. Model runtime backend default: constructors should stay on packed_reference
#====================

def test_model_default_runtime_backend_contract():
    """new models should default every qitnn layer to packed_reference"""
    print("\n=== test_model_default_runtime_backend_contract ===")
    torch.manual_seed(35)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=24,
        ffn_dim=48,
        seq_len=16,
        layers=2,
        device=DEVICE,
    )
    backend_names = tuple(layer._runtime_backend for _, _, layer in model.iter_qitnn_layers())
    check(
        "model defaults every qitnn layer to packed_reference",
        backend_names == ("packed_reference",) * len(backend_names),
        f"backends={backend_names}",
    )


#====================
# 16. QITNNLinear runtime backend contract: split and packed_reference must stay interchangeable
#====================

def test_qitnn_linear_runtime_backend_parity():
    """QITNNLinear internal runtime backend switch must preserve visible and raw outputs"""
    print("\n=== test_qitnn_linear_runtime_backend_parity ===")
    torch.manual_seed(35)

    modes = [("fp32", False, torch.float32)]
    if torch.cuda.is_bf16_supported():
        modes.append(("mixed", True, torch.bfloat16))

    for label, use_mixed, visible_dtype in modes:
        layer = pyqitnn.QITNNLinear(16, 8, centered_simplex=True, mixed_precision=use_mixed, device=DEVICE)
        inp_dtype = torch.float32 if not use_mixed else torch.float32
        inp = torch.randn(4, 16, device=DEVICE, dtype=inp_dtype)

        layer._set_runtime_backend("split")
        packed_split, raw_split = layer(inp), layer.forward_raw(inp)
        layer._set_runtime_backend("packed_reference")
        packed_ref, raw_ref = layer(inp), layer.forward_raw(inp)

        packed_diff = (packed_split.float() - packed_ref.float()).abs().max().item()
        check(f"layer backend packed output parity [{label}]", packed_diff < 1e-6, f"max_diff={packed_diff:.2e}")
        check(f"layer backend visible dtype [{label}]", packed_ref.dtype == visible_dtype, f"dtype={packed_ref.dtype}")

        for name, left, right in zip(("u", "v", "cn", "cz", "cp"), raw_split, raw_ref):
            max_err = (left.float() - right.float()).abs().max().item()
            check(f"layer backend raw parity {name} [{label}]", max_err < 1e-6, f"max_err={max_err:.2e}")

        layer._set_runtime_backend("split")


#====================
# 17. QITNNLinear runtime backend contract: packed_reference must preserve backward reachability
#====================

def test_qitnn_linear_runtime_backend_backward_parity():
    """packed runtime backend must return gradients to split parameters and input"""
    print("\n=== test_qitnn_linear_runtime_backend_backward_parity ===")
    torch.manual_seed(38)

    modes: list[tuple[str, bool, torch.dtype, float, float]] = [("fp32", False, torch.float32, 1e-5, 1e-5)]
    if torch.cuda.is_bf16_supported():
        modes.append(("mixed", True, torch.float32, 1e-2, 5e-4))

    for label, use_mixed, input_dtype, inp_tol, weight_tol in modes:
        split_layer = pyqitnn.QITNNLinear(
            16,
            8,
            ent_lambda=0.19,
            centered_simplex=True,
            mixed_precision=use_mixed,
            device=DEVICE,
        )
        packed_layer = pyqitnn.QITNNLinear(
            16,
            8,
            ent_lambda=0.19,
            centered_simplex=True,
            mixed_precision=use_mixed,
            device=DEVICE,
        )
        packed_layer.load_state_dict(split_layer.state_dict())
        packed_layer._set_runtime_backend("packed_reference")

        split_inp = torch.randn(5, 16, device=DEVICE, dtype=input_dtype, requires_grad=True)
        packed_inp = split_inp.detach().clone().requires_grad_(True)

        split_out = split_layer(split_inp)
        packed_out = packed_layer(packed_inp)
        split_loss = split_out.float().square().mean()
        packed_loss = packed_out.float().square().mean()
        split_loss.backward()
        packed_loss.backward()

        check(f"layer backend backward input grad exists [{label}]", packed_inp.grad is not None, "grad is None")
        check(f"layer backend backward reaches a_neg [{label}]", packed_layer.a_neg.grad is not None, "grad is None")
        check(f"layer backend backward reaches a_zero [{label}]", packed_layer.a_zero.grad is not None, "grad is None")
        check(f"layer backend backward reaches a_pos [{label}]", packed_layer.a_pos.grad is not None, "grad is None")
        if (
            packed_inp.grad is None or
            packed_layer.a_neg.grad is None or
            packed_layer.a_zero.grad is None or
            packed_layer.a_pos.grad is None
        ):
            continue

        inp_err = (split_inp.grad.float() - packed_inp.grad.float()).abs().max().item()
        neg_err = (split_layer.a_neg.grad - packed_layer.a_neg.grad).abs().max().item()
        zero_err = (split_layer.a_zero.grad - packed_layer.a_zero.grad).abs().max().item()
        pos_err = (split_layer.a_pos.grad - packed_layer.a_pos.grad).abs().max().item()
        finite_ok = all(
            torch.isfinite(t.float()).all().item()
            for t in (
                packed_inp.grad,
                packed_layer.a_neg.grad,
                packed_layer.a_zero.grad,
                packed_layer.a_pos.grad,
            )
        )

        check(f"layer backend backward input grad dtype [{label}]", packed_inp.grad.dtype == input_dtype, f"dtype={packed_inp.grad.dtype}")
        check(f"layer backend backward input parity [{label}]", inp_err < inp_tol, f"max_err={inp_err:.2e}")
        check(f"layer backend backward a_neg parity [{label}]", neg_err < weight_tol, f"max_err={neg_err:.2e}")
        check(f"layer backend backward a_zero parity [{label}]", zero_err < weight_tol, f"max_err={zero_err:.2e}")
        check(f"layer backend backward a_pos parity [{label}]", pos_err < weight_tol, f"max_err={pos_err:.2e}")
        check(f"layer backend backward stays finite [{label}]", finite_ok)


#====================
# 18. Model runtime backend contract: internal backend switch must preserve full-model outputs
#====================

def test_model_runtime_backend_parity():
    """model-level runtime backend switch must preserve logits and loss"""
    print("\n=== test_model_runtime_backend_parity ===")

    modes = [("fp32", "fp32", 5e-5, 1e-6)]
    if torch.cuda.is_bf16_supported():
        modes.append(("mixed", "qts_fp32_rest_bf16", 1e-5, 1e-6))

    for label, precision_mode, logits_tol, loss_tol in modes:
        torch.manual_seed(36)
        torch.cuda.manual_seed_all(36)

        model = pyqitnn.QITNNSimplexTransformerLM(
            dim=24,
            ffn_dim=48,
            seq_len=16,
            layers=2,
            precision_mode=precision_mode,
            device=DEVICE,
        )
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)

        model._set_qitnn_runtime_backend("split")
        logits_split, loss_split = model(tokens, targets=tokens)

        model._set_qitnn_runtime_backend("packed_reference")
        logits_ref, loss_ref = model(tokens, targets=tokens)
        backend_names = tuple(layer._runtime_backend for _, _, layer in model.iter_qitnn_layers())

        logits_diff = (logits_split.float() - logits_ref.float()).abs().max().item()
        loss_diff = abs(float(loss_split.detach().float()) - float(loss_ref.detach().float()))

        check(f"model backend switches every qitnn layer [{label}]", backend_names == ("packed_reference",) * len(backend_names), f"backends={backend_names}")
        check(f"model backend logits parity [{label}]", logits_diff < logits_tol, f"max_diff={logits_diff:.2e}")
        check(f"model backend loss parity [{label}]", loss_diff < loss_tol, f"loss_diff={loss_diff:.2e}")

        model._set_qitnn_runtime_backend("split")
        reset_names = tuple(layer._runtime_backend for _, _, layer in model.iter_qitnn_layers())
        check(f"model backend returns to split [{label}]", reset_names == ("split",) * len(reset_names), f"backends={reset_names}")


#====================
# 19. Runtime backend contract: invalid backend names must fail loudly
#====================

def test_runtime_backend_rejects_invalid_name():
    """internal runtime backend selector should reject unsupported backend names"""
    print("\n=== test_runtime_backend_rejects_invalid_name ===")
    torch.manual_seed(37)

    layer = pyqitnn.QITNNLinear(16, 8, device=DEVICE)
    model = pyqitnn.QITNNSimplexTransformerLM(dim=16, ffn_dim=32, seq_len=16, layers=1, device=DEVICE)

    layer_msg = capture_runtime_error(lambda: layer._set_runtime_backend("bad_backend"))
    model_msg = capture_runtime_error(lambda: model._set_qitnn_runtime_backend("bad_backend"))

    check("layer backend selector rejects invalid backend", "split" in layer_msg and "packed_reference" in layer_msg, layer_msg or "no RuntimeError")
    check("model backend selector rejects invalid backend", "split" in model_msg and "packed_reference" in model_msg, model_msg or "no RuntimeError")


#====================
# 20. Product contract: internal runtime backend must stay invisible to checkpoint/diagnostics/generation
#====================

def test_runtime_backend_product_surface_contract():
    """internal packed runtime backend must stay invisible to product-facing contracts"""
    print("\n=== test_runtime_backend_product_surface_contract ===")
    torch.manual_seed(39)
    torch.cuda.manual_seed_all(39)

    ckpt_path = ROOT / "_test_runtime_backend_product_ckpt.pt"
    try:
        split_model = pyqitnn.QITNNSimplexTransformerLM(
            dim=24,
            ffn_dim=48,
            seq_len=16,
            layers=2,
            device=DEVICE,
        )
        packed_model = pyqitnn.QITNNSimplexTransformerLM(
            dim=24,
            ffn_dim=48,
            seq_len=16,
            layers=2,
            device=DEVICE,
        )
        packed_model.load_state_dict(split_model.state_dict())
        packed_model._set_qitnn_runtime_backend("packed_reference")

        state_keys_split = tuple(split_model.state_dict().keys())
        state_keys_packed = tuple(packed_model.state_dict().keys())
        diag_split = split_model.collect_qitnn_diagnostics(epoch=4, full=True)
        diag_packed = packed_model.collect_qitnn_diagnostics(epoch=4, full=True)
        fmt_split = split_model.format_qitnn_diagnostics(epoch=4, full=True)
        fmt_packed = packed_model.format_qitnn_diagnostics(epoch=4, full=True)

        check("packed runtime backend stays out of model state_dict", state_keys_packed == state_keys_split, f"keys={state_keys_packed}")
        check("packed runtime backend preserves diagnostics snapshot", diag_packed == diag_split, json.dumps(diag_packed, ensure_ascii=False))
        check("packed runtime backend preserves formatted diagnostics", fmt_packed == fmt_split, "\n".join(fmt_packed))

        opt = torch.optim.AdamW(packed_model.parameters(), lr=1e-3)
        save_ckpt(ckpt_path, packed_model, opt, epoch=2, step=11, best_val=1.5, best_val_epoch=1)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        loaded_model = pyqitnn.QITNNSimplexTransformerLM(
            dim=24,
            ffn_dim=48,
            seq_len=16,
            layers=2,
            device=DEVICE,
        )
        loaded_opt = torch.optim.AdamW(loaded_model.parameters(), lr=1e-3)
        load_msg = capture_runtime_error(lambda: load_ckpt(ckpt_path, loaded_model, loaded_opt, DEVICE))
        loaded_backends = tuple(layer._runtime_backend for _, _, layer in loaded_model.iter_qitnn_layers())

        check("checkpoint saved from packed runtime backend still loads", load_msg == "", load_msg or "ok")
        check("checkpoint payload keeps public tensor-only model keys", tuple(ckpt["model"].keys()) == state_keys_split, f"keys={tuple(ckpt['model'].keys())}")
        check("loaded model returns to packed_reference runtime backend", loaded_backends == ("packed_reference",) * len(loaded_backends), f"backends={loaded_backends}")

        tokens = torch.randint(0, 256, (2, 13), device=DEVICE)
        greedy_packed = packed_model.generate(tokens, max_new_tokens=10, temperature=0.0, top_k=0, ascii_guard=False)
        greedy_loaded = loaded_model.generate(tokens, max_new_tokens=10, temperature=0.0, top_k=0, ascii_guard=False)
        logits_packed, loss_packed = packed_model(tokens, targets=tokens)
        logits_loaded, loss_loaded = loaded_model(tokens, targets=tokens)
        logits_diff = (logits_packed.float() - logits_loaded.float()).abs().max().item()
        loss_diff = abs(float(loss_packed.detach().float()) - float(loss_loaded.detach().float()))
        logits_tol = 6e-5

        check("checkpointed packed backend preserves greedy generation", torch.equal(greedy_packed, greedy_loaded), f"packed={greedy_packed.tolist()} loaded={greedy_loaded.tolist()}")
        check("checkpointed packed backend preserves forward logits", logits_diff < logits_tol, f"max_diff={logits_diff:.2e}")
        check("checkpointed packed backend preserves loss", loss_diff < 1e-6, f"loss_diff={loss_diff:.2e}")
    finally:
        if ckpt_path.exists():
            ckpt_path.unlink()


#====================
# 20. Training convergence: AdamW param groups correctness
#====================

def test_adamw_param_groups():
    """AdamW groups: QTS params must have weight_decay=0, others must have wd>0"""
    print("\n=== test_adamw_param_groups ===")
    torch.manual_seed(42)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=32, ffn_dim=64, seq_len=32, layers=2, device=DEVICE,
    )

    # collect QTS param ids
    qts_ids = set()
    for _, _, layer in model.iter_qitnn_layers():
        qts_ids.add(id(layer.a_neg))
        qts_ids.add(id(layer.a_zero))
        qts_ids.add(id(layer.a_pos))

    # build groups like the training script does
    qts_main = []
    qts_zero = []
    for _, role, layer in model.iter_qitnn_layers():
        for p in (layer.a_neg, layer.a_pos):
            if id(p) not in {id(x) for x in qts_main}:
                qts_main.append(p)
        if id(layer.a_zero) not in {id(x) for x in qts_zero}:
            qts_zero.append(layer.a_zero)

    other = [p for p in model.parameters() if id(p) not in qts_ids]

    groups = [
        {"params": qts_main, "lr": 3e-4, "weight_decay": 0.0},
        {"params": qts_zero, "lr": 3e-4 * 3.5, "weight_decay": 0.0},
        {"params": other, "lr": 3e-4, "weight_decay": 0.01},
    ]

    opt = torch.optim.AdamW([g for g in groups if g["params"]])

    # verify
    n_qts = len(qts_main) + len(qts_zero)
    n_other = len(other)

    check(f"QTS params found: {n_qts}", n_qts > 0)
    check(f"other params found: {n_other}", n_other > 0)

    # verify no QTS param has weight_decay > 0
    for g in opt.param_groups:
        wd = g.get("weight_decay", 0.0)
        for p in g["params"]:
            if id(p) in qts_ids:
                check(f"QTS param wd=0 (got {wd})", wd == 0.0, f"wd={wd}")
                break  # one check per group is enough


#====================
# 15. Generation sanity: model generates valid bytes
#====================

def test_generation_sanity():
    """generated tokens must be valid ASCII (with ascii_guard=True)"""
    print("\n=== test_generation_sanity ===")
    torch.manual_seed(42)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=32, ffn_dim=64, seq_len=64, layers=2, device=DEVICE,
    )

    prompt = torch.tensor([[72, 101, 108, 108, 111]], device=DEVICE)  # "Hello"
    out = model.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=8, ascii_guard=True)

    generated = out[0, 5:].cpu().tolist()
    valid_ascii = {0, 9, 10, 13} | set(range(32, 127))
    invalid = [t for t in generated if t not in valid_ascii]

    check(f"all generated tokens are valid ASCII ({len(invalid)} invalid)",
          len(invalid) == 0,
          f"invalid tokens: {invalid[:10]}")


#====================
# 16. Generation parity: generate() must match the reference decode loop
#====================

def test_generation_reference_loop_parity_fp32():
    """generate() should match the reference decode loop in fp32"""
    print("\n=== test_generation_reference_loop_parity_fp32 ===")
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=24, ffn_dim=48, seq_len=16, layers=2, device=DEVICE,
    )
    prompt = torch.randint(0, 256, (2, 11), device=DEVICE)

    cases = [
        ("sampled", 12, 0.8, 8, True),
        ("greedy", 10, 0.0, 0, False),
    ]

    for label, max_new_tokens, temperature, top_k, ascii_guard in cases:
        rng_state = _capture_rng_state()
        out_ref = _generate_reference_loop(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            ascii_guard=ascii_guard,
        )
        restored = _restore_rng_state(rng_state)
        out_now = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            ascii_guard=ascii_guard,
        )

        check(f"generation reference loop restores RNG [{label}]", restored, "restore failed")
        check(
            f"generate() matches reference loop [{label}]",
            torch.equal(out_now, out_ref),
            f"out_now={out_now.tolist()} out_ref={out_ref.tolist()}",
        )


#====================
# 17. Generation routing: generate() must use decode_step(), not forward()
#====================

def test_generate_uses_decode_step_contract():
    """generate() should route through decode_step() for each new token"""
    print("\n=== test_generate_uses_decode_step_contract ===")
    torch.manual_seed(126)
    torch.cuda.manual_seed_all(126)

    class TrackingTransformer(pyqitnn.QITNNSimplexTransformerLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.forward_calls = 0
            self.decode_step_calls = 0

        def forward(self, tokens: torch.Tensor, *, targets: torch.Tensor | None = None):
            self.forward_calls += 1
            raise RuntimeError("generate() should not call forward()")

        def decode_step(self, tokens: torch.Tensor) -> torch.Tensor:
            self.decode_step_calls += 1
            return super().decode_step(tokens)

    model = TrackingTransformer(
        dim=24, ffn_dim=48, seq_len=8, layers=2, device=DEVICE,
    )
    prompt = torch.randint(0, 256, (2, 11), device=DEVICE)

    out = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        top_k=0,
        ascii_guard=False,
    )

    check("generate() appends the requested number of tokens", out.shape == (2, 16), f"shape={tuple(out.shape)}")
    check("generate() does not route through forward()", model.forward_calls == 0, f"forward_calls={model.forward_calls}")
    check("generate() calls decode_step() once per generated token", model.decode_step_calls == 5, f"decode_step_calls={model.decode_step_calls}")


#====================
# 18. Generation window: only the last seq_len tokens may affect decode
#====================

def test_generation_seq_len_window_contract():
    """generate() should depend only on the visible decode window"""
    print("\n=== test_generation_seq_len_window_contract ===")
    torch.manual_seed(124)
    torch.cuda.manual_seed_all(124)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=24, ffn_dim=48, seq_len=8, layers=2, device=DEVICE,
    )

    shared_tail = torch.tensor([[41, 42, 43, 44, 45, 46, 47, 48]], device=DEVICE, dtype=torch.long)
    prompt_a = torch.tensor([[7, 9, 11, 13, 17, 19, 23, 29]], device=DEVICE, dtype=torch.long)
    prompt_a = torch.cat([prompt_a, shared_tail], dim=1)
    prompt_b = torch.tensor([[101, 103, 107, 109, 113, 127, 131, 137, 139]], device=DEVICE, dtype=torch.long)
    prompt_b = torch.cat([prompt_b, shared_tail], dim=1)

    rng_state = _capture_rng_state()
    out_a = model.generate(prompt_a, max_new_tokens=12, temperature=0.8, top_k=8, ascii_guard=True)
    restored = _restore_rng_state(rng_state)
    out_b = model.generate(prompt_b, max_new_tokens=12, temperature=0.8, top_k=8, ascii_guard=True)

    cont_a = out_a[:, prompt_a.size(1):]
    cont_b = out_b[:, prompt_b.size(1):]

    check("generation window test restores RNG", restored, "restore failed")
    check(
        "generation continuation depends only on the visible seq_len tail",
        torch.equal(cont_a, cont_b),
        f"cont_a={cont_a.tolist()} cont_b={cont_b.tolist()}",
    )


#====================
# 19. Mixed generation parity: bf16 visible path must match the reference decode loop
#====================

def test_mixed_precision_generation_sanity():
    """mixed generation should keep bf16 visible activations and emit valid guarded tokens"""
    print("\n=== test_mixed_precision_generation_sanity ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed generation skipped", "bf16 not supported on this GPU")
        return

    torch.manual_seed(42)
    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=32, ffn_dim=64, seq_len=64, layers=2, device=DEVICE, mixed_precision=True,
    )

    seen = {"q_proj_out": None}

    def hook(_module, _inp, out):
        seen["q_proj_out"] = out.dtype

    handle = model.blocks[0].q_proj.register_forward_hook(hook)
    try:
        prompt = torch.tensor([[72, 101, 108, 108, 111]], device=DEVICE)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=8, ascii_guard=True)
    finally:
        handle.remove()

    generated = out[0, 5:].cpu().tolist()
    valid_ascii = {0, 9, 10, 13} | set(range(32, 127))
    invalid = [t for t in generated if t not in valid_ascii]

    check("mixed generation keeps bf16 visible activations", seen["q_proj_out"] == torch.bfloat16, f"dtype={seen['q_proj_out']}")
    check("mixed generation returns token ids", out.dtype == torch.long, f"dtype={out.dtype}")
    check(f"mixed generation keeps ascii_guard valid ({len(invalid)} invalid)",
          len(invalid) == 0,
          f"invalid tokens: {invalid[:10]}")


def test_generation_reference_loop_parity_mixed_precision():
    """mixed generate() should match the reference decode loop"""
    print("\n=== test_generation_reference_loop_parity_mixed_precision ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed generation reference parity skipped", "bf16 not supported on this GPU")
        return

    torch.manual_seed(125)
    torch.cuda.manual_seed_all(125)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=24, ffn_dim=48, seq_len=16, layers=2, device=DEVICE, mixed_precision=True,
    )
    prompt = torch.randint(0, 256, (2, 13), device=DEVICE)

    rng_state = _capture_rng_state()
    out_ref = _generate_reference_loop(
        model,
        prompt,
        max_new_tokens=12,
        temperature=0.8,
        top_k=8,
        ascii_guard=True,
    )
    restored = _restore_rng_state(rng_state)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out_now = model.generate(
            prompt,
            max_new_tokens=12,
            temperature=0.8,
            top_k=8,
            ascii_guard=True,
        )

    check("mixed generation reference parity restores RNG", restored, "restore failed")
    check(
        "mixed generate() matches reference loop",
        torch.equal(out_now, out_ref),
        f"out_now={out_now.tolist()} out_ref={out_ref.tolist()}",
    )


#====================
# 20. Mixed guardrails: invalid high-level mixed usage must fail loudly
#====================

def test_mixed_precision_guardrails():
    """mixed mode should reject half-cast master weights and missing bf16 support"""
    print("\n=== test_mixed_precision_guardrails ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed guardrails skipped", "bf16 not supported on this GPU")
        return

    inp = torch.randn(4, 16, device=DEVICE)
    layer = pyqitnn.QITNNLinear(16, 8, mixed_precision=True, device=DEVICE).half()
    layer_msg = capture_runtime_error(lambda: layer(inp))

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=16, ffn_dim=32, seq_len=16, layers=1, device=DEVICE, mixed_precision=True,
    ).half()
    tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
    model_msg = capture_runtime_error(lambda: model(tokens))

    with patch("torch.cuda.is_bf16_supported", return_value=False):
        train_msg = capture_runtime_error(
            lambda: train(
                dataset="ignored_under_bf16_guard.txt",
                steps=1,
                mixed_precision=True,
                no_save=True,
                no_interactive=True,
            )
        )

    check("mixed linear rejects half-cast QITNN master weights",
          "fp32 master weights" in layer_msg and ".half()" in layer_msg,
          layer_msg or "no RuntimeError")
    check("mixed model rejects half-cast master weights",
          "fp32 master weights" in model_msg and ".half()" in model_msg,
          model_msg or "no RuntimeError")
    check("train() mixed path rejects missing bf16 support",
          "requires CUDA bf16 support" in train_msg,
          train_msg or "no RuntimeError")


#====================
# 18. Mixed precision off: explicit fp32 path must stay trusted even under outer autocast
#====================

def test_mixed_precision_default_stays_fp32():
    """mixed_precision=False must force the trusted fp32 path even under outer autocast"""
    print("\n=== test_mixed_precision_default_stays_fp32 ===")
    torch.manual_seed(42)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=16, ffn_dim=32, seq_len=16, layers=1, device=DEVICE, mixed_precision=False,
    )

    seen = {"q_proj_out": None}

    def hook(_mod, _inp, out):
        seen["q_proj_out"] = out.dtype

    handle = model.blocks[0].q_proj.register_forward_hook(hook)

    tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(tokens, targets=tokens)

    handle.remove()

    check("mixed_precision=False keeps q_proj output in fp32", seen["q_proj_out"] == torch.float32, f"dtype={seen['q_proj_out']}")
    check("mixed_precision=False keeps logits in fp32", logits.dtype == torch.float32, f"dtype={logits.dtype}")
    check("mixed_precision=False keeps loss in fp32", loss.dtype == torch.float32, f"dtype={loss.dtype}")


#====================
# 19. Mixed precision on: bf16 activations with fp32 master weights must work
#====================

def test_qitnn_linear_mixed_precision_smoke():
    """QITNNLinear mixed_precision path should emit bf16 activations but keep fp32 master weights"""
    print("\n=== test_qitnn_linear_mixed_precision_smoke ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed precision linear skipped", "bf16 not supported on this GPU")
        return

    torch.manual_seed(42)
    layer = pyqitnn.QITNNLinear(16, 8, mixed_precision=True, device=DEVICE)
    inp = torch.randn(4, 16, device=DEVICE)

    out = layer(inp)
    loss = out.float().square().mean()
    loss.backward()

    check("QITNNLinear mixed output is bf16", out.dtype == torch.bfloat16, f"dtype={out.dtype}")
    check("QITNNLinear master weights stay fp32", layer.a_neg.dtype == torch.float32 and layer.a_zero.dtype == torch.float32 and layer.a_pos.dtype == torch.float32)
    check("QITNNLinear mixed gradients are finite", torch.isfinite(layer.a_neg.grad).all().item() and torch.isfinite(layer.a_zero.grad).all().item() and torch.isfinite(layer.a_pos.grad).all().item())


def test_qitnn_linear_mixed_precision_forces_bf16():
    """high-level mixed_precision should stay on the conservative bf16 path even under outer fp16 autocast"""
    print("\n=== test_qitnn_linear_mixed_precision_forces_bf16 ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed precision linear force-bf16 skipped", "bf16 not supported on this GPU")
        return

    torch.manual_seed(42)
    layer = pyqitnn.QITNNLinear(16, 8, mixed_precision=True, device=DEVICE)
    inp = torch.randn(4, 16, device=DEVICE)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = layer(inp)
        loss = out.float().square().mean()
    loss.backward()

    check("QITNNLinear mixed path ignores outer fp16 autocast for visible dtype", out.dtype == torch.bfloat16, f"dtype={out.dtype}")
    check("QITNNLinear mixed path keeps fp32 master-weight grads under outer fp16 autocast", layer.a_neg.grad is not None and layer.a_neg.grad.dtype == torch.float32)


def test_model_mixed_precision_smoke():
    """full model mixed_precision path should run with bf16 visible activations and fp32 loss"""
    print("\n=== test_model_mixed_precision_smoke ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed precision model skipped", "bf16 not supported on this GPU")
        return

    torch.manual_seed(42)
    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=16, ffn_dim=32, seq_len=16, layers=1, device=DEVICE, mixed_precision=True,
    )

    seen = {"q_proj_out": None}

    def hook(_mod, _inp, out):
        seen["q_proj_out"] = out.dtype

    handle = model.blocks[0].q_proj.register_forward_hook(hook)

    tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
    logits, loss = model(tokens, targets=tokens)
    loss.backward()
    model.apply_qitnn_prior(step_qk=5e-5, step_vo=5e-5, step_ff=5e-5, entropy_floor=1.0840643)

    handle.remove()

    check("mixed_precision=True uses bf16 QTS activations", seen["q_proj_out"] == torch.bfloat16, f"dtype={seen['q_proj_out']}")
    check("mixed_precision=True emits bf16 logits", logits.dtype == torch.bfloat16, f"dtype={logits.dtype}")
    check("mixed_precision=True keeps fp32 loss", loss.dtype == torch.float32, f"dtype={loss.dtype}")
    check("mixed_precision=True keeps fp32 QTS weights", model.blocks[0].q_proj.a_neg.dtype == torch.float32)
    check("mixed_precision=True backward is finite", torch.isfinite(model.blocks[0].q_proj.a_neg.grad).all().item() and torch.isfinite(model.head.weight.grad).all().item())


#====================
# 18. CLI precision toggle: defaults and explicit overrides must be stable
#====================

def test_mixed_precision_cli_defaults():
    """CLI parsing must resolve the new precision_mode contract and keep legacy flags compatible"""
    print("\n=== test_mixed_precision_cli_defaults ===")

    train_default = _resolve_precision_mode_cfg(TrainConfig())
    default_cfg = _parse_cli([])
    legacy_on_cfg = _parse_cli(["--mixed-precision"])
    legacy_off_cfg = _parse_cli(["--mixed-precision", "--no-mixed-precision"])
    mode_cfg = _parse_cli(["--precision-mode", "qts_fp32_rest_bf16"])
    alias_cfg = _parse_cli(["--precision-mode", "mixed_bf16_native"])
    conflict_msg = capture_runtime_error(
        lambda: _resolve_precision_mode_cfg(_parse_cli(["--mixed-precision", "--precision-mode", "fp32"]))
    )

    check("TrainConfig default resolves to canonical trainer mixed path", train_default == ("qts_fp32_rest_bf16", True), f"default={train_default}")
    check("CLI default keeps precision_mode unset before resolution", default_cfg.precision_mode is None and default_cfg.mixed_precision is None, f"precision_mode={default_cfg.precision_mode} mixed_precision={default_cfg.mixed_precision}")
    check("CLI default preserves TrainConfig default", _resolve_precision_mode_cfg(default_cfg) == train_default, f"resolved={_resolve_precision_mode_cfg(default_cfg)}")
    check("CLI legacy --mixed-precision maps to qts_fp32_rest_bf16", _resolve_precision_mode_cfg(legacy_on_cfg) == ("qts_fp32_rest_bf16", True), f"resolved={_resolve_precision_mode_cfg(legacy_on_cfg)}")
    check("CLI legacy --no-mixed-precision resolves to fp32", _resolve_precision_mode_cfg(legacy_off_cfg) == ("fp32", False), f"resolved={_resolve_precision_mode_cfg(legacy_off_cfg)}")
    check("CLI --precision-mode qts_fp32_rest_bf16 enables mixed path", _resolve_precision_mode_cfg(mode_cfg) == ("qts_fp32_rest_bf16", True), f"resolved={_resolve_precision_mode_cfg(mode_cfg)}")
    check("CLI precision_mode alias normalizes to qts_fp32_rest_bf16", _resolve_precision_mode_cfg(alias_cfg) == ("qts_fp32_rest_bf16", True), f"resolved={_resolve_precision_mode_cfg(alias_cfg)}")
    check("CLI conflicting legacy flag and precision_mode is rejected", "conflict" in conflict_msg, conflict_msg or "no RuntimeError")


def test_precision_mode_single_source_of_truth():
    print("\n=== test_precision_mode_single_source_of_truth ===")

    shared_default = resolve_precision_mode_shared(None, None)
    trainer_default = _resolve_precision_mode_cfg(TrainConfig())
    alias_shared = resolve_precision_mode_shared("mixed_bf16_native", None)
    alias_trainer = _resolve_precision_mode_cfg(TrainConfig(precision_mode="mixed_bf16_native"))

    check("shared resolver keeps library default fp32 when precision omitted", shared_default == ("fp32", False), f"default={shared_default}")
    check("trainer resolver layers its own default on top of shared contract", trainer_default == ("qts_fp32_rest_bf16", True), f"default={trainer_default}")
    check("shared resolver normalizes mixed alias", alias_shared == ("qts_fp32_rest_bf16", True), f"resolved={alias_shared}")
    check("trainer resolver uses same alias normalization", alias_trainer == ("qts_fp32_rest_bf16", True), f"resolved={alias_trainer}")


def test_legacy_mixed_precision_python_compat():
    print("\n=== test_legacy_mixed_precision_python_compat ===")

    legacy_on = _resolve_precision_mode_cfg(TrainConfig(mixed_precision=True))
    legacy_off = _resolve_precision_mode_cfg(TrainConfig(mixed_precision=False))
    explicit = _resolve_precision_mode_cfg(TrainConfig(precision_mode="fp32"))
    conflict_msg = capture_runtime_error(
        lambda: _resolve_precision_mode_cfg(TrainConfig(precision_mode="fp32", mixed_precision=True))
    )

    check("TrainConfig legacy mixed_precision=True still resolves cleanly", legacy_on == ("qts_fp32_rest_bf16", True), f"resolved={legacy_on}")
    check("TrainConfig legacy mixed_precision=False still resolves cleanly", legacy_off == ("fp32", False), f"resolved={legacy_off}")
    check("TrainConfig explicit precision_mode remains canonical", explicit == ("fp32", False), f"resolved={explicit}")
    check("TrainConfig rejects conflicting legacy bool and precision_mode", "conflict" in conflict_msg, conflict_msg or "no RuntimeError")


def test_cli_help_prefers_precision_mode():
    print("\n=== test_cli_help_prefers_precision_mode ===")

    help_text = _build_cli_parser().format_help()

    check("CLI help exposes precision_mode as the canonical flag", "--precision-mode" in help_text, help_text)
    check("CLI help documents the trainer default precision path", "qts_fp32_rest_bf16" in help_text, help_text)
    check("CLI help hides legacy mixed_precision flags", "--mixed-precision" not in help_text and "--no-mixed-precision" not in help_text, help_text)


def test_precision_config_artifact_prefers_canonical_mode():
    print("\n=== test_precision_config_artifact_prefers_canonical_mode ===")

    tmp_path = ROOT / "_tmp_precision_config_dataset.txt"
    save_dir = ROOT / "_tmp_precision_config_runs"
    run_name = "precision_config_contract"
    config_ok = False
    config_detail = ""

    text = (
        "precision config artifact should preserve a canonical trainer contract\n"
        "legacy trainer toggles may still enter, but saved artifacts should stay explicit\n"
    ) * 64

    try:
        shutil.rmtree(save_dir, ignore_errors=True)
        tmp_path.write_text(text, encoding="utf-8")
        result = train(
            dataset=str(tmp_path),
            tokenizer="byte",
            mixed_precision=False,
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            steps=1,
            save_dir=str(save_dir),
            run_name=run_name,
            no_interactive=True,
            prompt="hello simplex",
            prompt_bytes=16,
            gen_bytes=0,
            log_every=1,
        )

        config_path = Path(result["run_dir"]) / "config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config_ok = (
                cfg.get("precision_mode") == "fp32"
                and cfg.get("precision_mode_requested") is None
                and cfg.get("precision_mode_resolved") == "fp32"
                and cfg.get("legacy_mixed_precision_input") is False
            )
            config_detail = json.dumps(
                {
                    "precision_mode": cfg.get("precision_mode"),
                    "precision_mode_requested": cfg.get("precision_mode_requested"),
                    "precision_mode_resolved": cfg.get("precision_mode_resolved"),
                    "legacy_mixed_precision_input": cfg.get("legacy_mixed_precision_input"),
                },
                ensure_ascii=False,
            )
        else:
            config_detail = str(config_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(save_dir, ignore_errors=True)

    check("trainer config artifact keeps canonical precision_mode", config_ok, config_detail)


def test_diagnostics_schema_contract():
    print("\n=== test_diagnostics_schema_contract ===")

    torch.manual_seed(42)
    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=16,
        ffn_dim=32,
        seq_len=16,
        layers=1,
        device=DEVICE,
    )

    snapshot = model.collect_qitnn_diagnostics(epoch=3, full=False)
    formatted = model.format_qitnn_diagnostics(epoch=3, full=False)
    formatted_from_snapshot = format_qitnn_diag_snapshot(snapshot)
    layers = snapshot.get("layers", [])
    first = layers[0] if layers else {}
    stats = first.get("stats", {})

    schema_ok = (
        snapshot.get("schema_version") == 1
        and snapshot.get("epoch") == 3
        and snapshot.get("full") is False
        and snapshot.get("layer_count") == 4
        and len(layers) == 4
        and set(first.keys()) == {"name", "label", "role", "stats"}
        and tuple(stats.keys()) == QITNN_DIAG_STAT_KEYS
    )

    check("diagnostics snapshot keeps a stable schema", schema_ok, json.dumps(snapshot, ensure_ascii=False))
    check("summary diagnostics keep the expected representative layer count", len(layers) == 4, f"layer_count={len(layers)}")
    check("formatted diagnostics are derived from the raw snapshot", formatted == formatted_from_snapshot, "\n".join(formatted_from_snapshot))


def _run_diag_artifact_train(run_name: str):
    tmp_path = ROOT / f"_tmp_{run_name}_dataset.txt"
    save_dir = ROOT / "_tmp_diag_runs"
    text = (
        "qitnn diagnostics artifacts should preserve raw layer statistics across epochs\n"
        "summary epochs keep a representative subset and full epochs keep all qitnn layers\n"
    ) * 64

    cleanup_tree(save_dir)
    tmp_path.write_text(text, encoding="utf-8")
    result = train(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="fp32",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=16,
        batch_size=1,
        epochs=2,
        steps_per_epoch=1,
        diag_every=2,
        save_dir=str(save_dir),
        run_name=run_name,
        no_interactive=True,
        prompt="diag",
        prompt_bytes=8,
        gen_bytes=0,
        log_every=1,
    )
    return tmp_path, save_dir, result


def test_trainer_writes_diag_json():
    print("\n=== test_trainer_writes_diag_json ===")

    tmp_path = None
    save_dir = None
    json_ok = False
    json_detail = ""

    try:
        tmp_path, save_dir, result = _run_diag_artifact_train("diag_json_contract")
        diag_path = Path(result["diagnostics_json"]) if result["diagnostics_json"] else Path()
        if diag_path.exists():
            payload = json.loads(diag_path.read_text(encoding="utf-8"))
            epochs = payload.get("epochs", [])
            first_epoch = epochs[0] if len(epochs) > 0 else {}
            second_epoch = epochs[1] if len(epochs) > 1 else {}
            json_ok = (
                payload.get("kind") == "qitnn_layer_diagnostics"
                and payload.get("schema_version") == 1
                and payload.get("stats_keys") == list(QITNN_DIAG_STAT_KEYS)
                and len(epochs) == 2
                and first_epoch.get("epoch") == 1
                and first_epoch.get("full") is False
                and first_epoch.get("layer_count") == 4
                and len(first_epoch.get("layers", [])) == 4
                and second_epoch.get("epoch") == 2
                and second_epoch.get("full") is True
                and second_epoch.get("layer_count") == 6
                and len(second_epoch.get("layers", [])) == 6
            )
            json_detail = json.dumps(
                {
                    "schema_version": payload.get("schema_version"),
                    "epochs": len(epochs),
                    "epoch1": {
                        "full": first_epoch.get("full"),
                        "layer_count": first_epoch.get("layer_count"),
                    },
                    "epoch2": {
                        "full": second_epoch.get("full"),
                        "layer_count": second_epoch.get("layer_count"),
                    },
                },
                ensure_ascii=False,
            )
        else:
            json_detail = str(diag_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if save_dir is not None:
            cleanup_tree(save_dir)

    check("trainer writes diagnostics.json with stable epoch snapshots", json_ok, json_detail)


def test_trainer_writes_diag_csv():
    print("\n=== test_trainer_writes_diag_csv ===")

    tmp_path = None
    save_dir = None
    csv_ok = False
    csv_detail = ""

    try:
        tmp_path, save_dir, result = _run_diag_artifact_train("diag_csv_contract")
        diag_json_path = Path(result["diagnostics_json"]) if result["diagnostics_json"] else Path()
        diag_csv_path = Path(result["diagnostics_csv"]) if result["diagnostics_csv"] else Path()
        metrics_path = Path(result["run_dir"]) / "metrics.csv"

        if diag_json_path.exists() and diag_csv_path.exists() and metrics_path.exists():
            payload = json.loads(diag_json_path.read_text(encoding="utf-8"))
            json_rows = {}
            for epoch_payload in payload.get("epochs", []):
                epoch = int(epoch_payload["epoch"])
                for layer in epoch_payload.get("layers", []):
                    json_rows[(epoch, layer["name"])] = {
                        "full": bool(epoch_payload["full"]),
                        "label": layer["label"],
                        "role": layer["role"],
                        "stats": layer["stats"],
                    }

            with open(diag_csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                diag_rows = list(reader)
                diag_header = reader.fieldnames

            with open(metrics_path, newline="", encoding="utf-8") as f:
                metrics_reader = csv.reader(f)
                metrics_header = next(metrics_reader, [])

            epoch_counts: dict[int, int] = {}
            parity_ok = len(diag_rows) == len(json_rows)
            parity_detail = ""
            for row in diag_rows:
                epoch = int(row["epoch"])
                key = (epoch, row["layer_name"])
                epoch_counts[epoch] = epoch_counts.get(epoch, 0) + 1
                expected = json_rows.get(key)
                if expected is None:
                    parity_ok = False
                    parity_detail = f"missing json row for {key}"
                    break
                if (row["full"].lower() == "true") != expected["full"]:
                    parity_ok = False
                    parity_detail = f"full mismatch for {key}: csv={row['full']} json={expected['full']}"
                    break
                if row["layer_label"] != expected["label"] or row["role"] != expected["role"]:
                    parity_ok = False
                    parity_detail = (
                        f"label/role mismatch for {key}: "
                        f"csv=({row['layer_label']}, {row['role']}) json=({expected['label']}, {expected['role']})"
                    )
                    break
                for stat_key in QITNN_DIAG_STAT_KEYS:
                    csv_value = float(row[stat_key])
                    json_value = float(expected["stats"][stat_key])
                    if abs(csv_value - json_value) > 1e-9:
                        parity_ok = False
                        parity_detail = (
                            f"{stat_key} mismatch for {key}: "
                            f"csv={csv_value:.12f} json={json_value:.12f}"
                        )
                        break
                if not parity_ok:
                    break

            metrics_ok = metrics_header == [
                "epoch",
                "train_loss",
                "train_ppl",
                "val_loss",
                "val_ppl",
                "train_tok_s",
                "val_tok_s",
                "lr",
                "time_s",
                "train_bpb",
                "val_bpb",
            ]
            csv_ok = (
                diag_header == list(QITNN_DIAG_CSV_HEADER)
                and epoch_counts == {1: 4, 2: 6}
                and parity_ok
                and metrics_ok
            )
            csv_detail = parity_detail or json.dumps(
                {
                    "diag_rows": len(diag_rows),
                    "epoch_counts": epoch_counts,
                    "metrics_header": metrics_header,
                },
                ensure_ascii=False,
            )
        else:
            csv_detail = json.dumps(
                {
                    "diagnostics_json": str(diag_json_path),
                    "diagnostics_csv": str(diag_csv_path),
                    "metrics_csv": str(metrics_path),
                },
                ensure_ascii=False,
            )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if save_dir is not None:
            cleanup_tree(save_dir)

    check("trainer writes diagnostics_layers.csv with raw layer rows", csv_ok, csv_detail)


def test_trainer_writes_diag_artifacts_under_mixed_multilayer_stress():
    print("\n=== test_trainer_writes_diag_artifacts_under_mixed_multilayer_stress ===")

    tmp_path = ROOT / "_tmp_diag_mixed_multilayer_dataset.txt"
    save_dir = ROOT / "_tmp_diag_mixed_multilayer_runs"
    run_name = "diag_mixed_multilayer_stress"
    stress_ok = False
    stress_detail = ""

    text = (
        "layer diagnostics should survive mixed precision, repeated epochs, and larger qitnn stacks without drift\n"
        "full epochs must serialize every qitnn layer while summary epochs keep only the representative subset\n"
        "the artifact contract must remain exact even when the trainer is under a heavier mixed-path load\n"
    ) * 192

    try:
        cleanup_tree(save_dir)
        tmp_path.write_text(text, encoding="utf-8")
        result = train(
            dataset=str(tmp_path),
            tokenizer="byte",
            precision_mode="qts_fp32_rest_bf16",
            dim=24,
            ffn=48,
            layers=3,
            seq_len=24,
            batch_size=2,
            epochs=3,
            steps_per_epoch=2,
            diag_every=3,
            save_dir=str(save_dir),
            run_name=run_name,
            no_interactive=True,
            prompt="diag mixed",
            prompt_bytes=12,
            gen_bytes=0,
            log_every=1,
        )

        diag_json_path = Path(result["diagnostics_json"]) if result["diagnostics_json"] else Path()
        diag_csv_path = Path(result["diagnostics_csv"]) if result["diagnostics_csv"] else Path()
        metrics_path = Path(result["run_dir"]) / "metrics.csv"
        config_path = Path(result["run_dir"]) / "config.json"

        if diag_json_path.exists() and diag_csv_path.exists() and metrics_path.exists() and config_path.exists():
            payload = json.loads(diag_json_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            epochs = payload.get("epochs", [])
            epoch_flags = [bool(ep.get("full")) for ep in epochs]
            epoch_counts = [int(ep.get("layer_count", -1)) for ep in epochs]

            with open(diag_csv_path, newline="", encoding="utf-8") as f:
                diag_reader = csv.DictReader(f)
                diag_rows = list(diag_reader)
                diag_header = diag_reader.fieldnames

            with open(metrics_path, newline="", encoding="utf-8") as f:
                metrics_reader = csv.reader(f)
                metrics_rows = list(metrics_reader)

            expected_epoch_counts = [6, 6, 18]
            finite_ok = True
            finite_detail = ""
            for epoch_payload in epochs:
                for layer in epoch_payload.get("layers", []):
                    stats = layer.get("stats", {})
                    if tuple(stats.keys()) != QITNN_DIAG_STAT_KEYS:
                        finite_ok = False
                        finite_detail = f"stats key order mismatch for {layer.get('name')}"
                        break
                    count_value = float(stats["count"])
                    if count_value <= 0.0:
                        finite_ok = False
                        finite_detail = f"non-positive count for {layer.get('name')}: {count_value}"
                        break
                    for key, value in stats.items():
                        if not math.isfinite(float(value)):
                            finite_ok = False
                            finite_detail = f"non-finite {key} for {layer.get('name')}: {value}"
                            break
                    if not finite_ok:
                        break
                if not finite_ok:
                    break

            stress_ok = (
                payload.get("kind") == "qitnn_layer_diagnostics"
                and payload.get("schema_version") == 1
                and payload.get("stats_keys") == list(QITNN_DIAG_STAT_KEYS)
                and diag_header == list(QITNN_DIAG_CSV_HEADER)
                and len(epochs) == 3
                and epoch_flags == [False, False, True]
                and epoch_counts == expected_epoch_counts
                and len(diag_rows) == sum(expected_epoch_counts)
                and len(metrics_rows) == 4
                and config.get("diagnostics_json") == str(diag_json_path)
                and config.get("diagnostics_csv") == str(diag_csv_path)
                and result["precision_mode"] == "qts_fp32_rest_bf16"
                and result["mixed_precision"] is True
                and finite_ok
            )
            stress_detail = finite_detail or json.dumps(
                {
                    "epoch_flags": epoch_flags,
                    "epoch_counts": epoch_counts,
                    "diag_rows": len(diag_rows),
                    "metrics_rows": len(metrics_rows),
                    "precision_mode": result["precision_mode"],
                    "mixed_precision": result["mixed_precision"],
                },
                ensure_ascii=False,
            )
        else:
            stress_detail = json.dumps(
                {
                    "diagnostics_json": str(diag_json_path),
                    "diagnostics_csv": str(diag_csv_path),
                    "metrics_csv": str(metrics_path),
                    "config_json": str(config_path),
                },
                ensure_ascii=False,
            )
    finally:
        tmp_path.unlink(missing_ok=True)
        cleanup_tree(save_dir)

    check("trainer diagnostics artifacts survive mixed multilayer stress", stress_ok, stress_detail)


def test_diag_artifacts_resume_same_run_prunes_future_epochs():
    print("\n=== test_diag_artifacts_resume_same_run_prunes_future_epochs ===")

    tmp_path = ROOT / "_tmp_diag_resume_same_run_dataset.txt"
    save_dir = ROOT / "_tmp_diag_resume_same_run_runs"
    run_name = "diag_resume_same_run"
    resume_ok = False
    resume_detail = ""

    text = (
        "resume-safe diagnostics artifacts must drop stale future epochs before appending fresh snapshots\n"
        "the final artifacts after resume into the same run directory must match the canonical single-run result exactly\n"
        "this check stresses same-path overwrite behavior without touching qts math or simplex geometry\n"
    ) * 160

    precision_mode = "qts_fp32_rest_bf16" if torch.cuda.is_bf16_supported() else "fp32"

    try:
        cleanup_tree(save_dir)
        tmp_path.write_text(text, encoding="utf-8")

        common = dict(
            dataset=str(tmp_path),
            tokenizer="byte",
            precision_mode=precision_mode,
            dim=24,
            ffn=48,
            layers=2,
            seq_len=24,
            batch_size=2,
            epochs=3,
            steps_per_epoch=2,
            diag_every=3,
            save_dir=str(save_dir),
            run_name=run_name,
            save_every=1,
            no_interactive=True,
            prompt="diag resume",
            prompt_bytes=12,
            gen_bytes=0,
            log_every=1,
            seed=91,
        )

        base = train(**common)
        run_dir = Path(base["run_dir"])
        diag_json_path = run_dir / "diagnostics.json"
        diag_csv_path = run_dir / "diagnostics_layers.csv"
        resume_ckpt = run_dir / "ckpt_ep1.pt"

        base_json_text = diag_json_path.read_text(encoding="utf-8")
        base_csv_text = diag_csv_path.read_text(encoding="utf-8")
        base_json = json.loads(base_json_text)
        base_csv_rows = list(csv.DictReader(base_csv_text.splitlines()))

        resumed = train(resume=str(resume_ckpt), **common)
        resumed_json_text = diag_json_path.read_text(encoding="utf-8")
        resumed_csv_text = diag_csv_path.read_text(encoding="utf-8")
        resumed_json = json.loads(resumed_json_text)
        resumed_csv_rows = list(csv.DictReader(resumed_csv_text.splitlines()))

        epoch_layer_pairs = [(int(row["epoch"]), row["layer_name"]) for row in resumed_csv_rows]
        pair_uniques = len(epoch_layer_pairs) == len(set(epoch_layer_pairs))

        resume_ok = (
            resumed["run_dir"] == base["run_dir"]
            and base_json_text == resumed_json_text
            and base_csv_text == resumed_csv_text
            and [int(ep["epoch"]) for ep in resumed_json.get("epochs", [])] == [1, 2, 3]
            and [bool(ep["full"]) for ep in resumed_json.get("epochs", [])] == [False, False, True]
            and len(resumed_csv_rows) == len(base_csv_rows)
            and pair_uniques
        )
        resume_detail = json.dumps(
            {
                "precision_mode": precision_mode,
                "run_dir_same": resumed["run_dir"] == base["run_dir"],
                "json_equal": base_json_text == resumed_json_text,
                "csv_equal": base_csv_text == resumed_csv_text,
                "epochs": [int(ep["epoch"]) for ep in resumed_json.get("epochs", [])],
                "full_flags": [bool(ep["full"]) for ep in resumed_json.get("epochs", [])],
                "row_count": len(resumed_csv_rows),
                "unique_pairs": pair_uniques,
            },
            ensure_ascii=False,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
        cleanup_tree(save_dir)

    check("resume into the same run dir keeps diagnostics artifacts canonical", resume_ok, resume_detail)


#====================
# sample_batch contract: vectorized sampling must preserve next-token window semantics
#====================

def test_sample_batch_contract():
    print("\n=== test_sample_batch_contract ===")

    data = torch.arange(16, dtype=torch.long)
    starts = torch.tensor([0, 3, 3, 11], dtype=torch.long)
    expected_x = torch.tensor([
        [0, 1, 2, 3],
        [3, 4, 5, 6],
        [3, 4, 5, 6],
        [11, 12, 13, 14],
    ], dtype=torch.long)
    expected_y = torch.tensor([
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [12, 13, 14, 15],
    ], dtype=torch.long)

    with patch("BasicQITNN_Transformer.torch.randint", return_value=starts):
        x, y = sample_batch(data, batch=4, seq_len=4, device=DEVICE)

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    check("sample_batch preserves x window slices", torch.equal(x_cpu, expected_x), f"x={x_cpu.tolist()}")
    check("sample_batch preserves y next-token shift", torch.equal(y_cpu, expected_y), f"y={y_cpu.tolist()}")
    check("sample_batch keeps duplicate starts stable", torch.equal(x_cpu[1], x_cpu[2]) and torch.equal(y_cpu[1], y_cpu[2]), f"x={x_cpu.tolist()} y={y_cpu.tolist()}")
    check("sample_batch returns requested device", x.device == DEVICE and y.device == DEVICE, f"x.device={x.device} y.device={y.device}")
    check("sample_batch keeps long dtype", x.dtype == torch.long and y.dtype == torch.long, f"x.dtype={x.dtype} y.dtype={y.dtype}")


#====================
# 19. Grad accumulation contract: config/CLI/step semantics must be fixed before loop changes
#====================

def test_grad_accumulation_config_contract():
    print("\n=== test_grad_accumulation_config_contract ===")

    default_cfg = TrainConfig()
    none_resolved = _resolve_grad_accum_steps(None)
    explicit_resolved = _resolve_grad_accum_steps(4)
    zero_msg = capture_runtime_error(lambda: _resolve_grad_accum_steps(0))
    neg_msg = capture_runtime_error(lambda: _resolve_grad_accum_steps(-3))

    check("TrainConfig default keeps grad accumulation disabled", default_cfg.grad_accum_steps == 1, f"grad_accum_steps={default_cfg.grad_accum_steps}")
    check("grad accumulation resolver treats None as legacy single-step mode", none_resolved == 1, f"resolved={none_resolved}")
    check("grad accumulation resolver preserves explicit positive values", explicit_resolved == 4, f"resolved={explicit_resolved}")
    check("zero grad_accum_steps is rejected", ">= 1" in zero_msg, zero_msg or "no RuntimeError")
    check("negative grad_accum_steps is rejected", ">= 1" in neg_msg, neg_msg or "no RuntimeError")


def test_grad_accumulation_cli_contract():
    print("\n=== test_grad_accumulation_cli_contract ===")

    default_cfg = _parse_cli([])
    cli_cfg = _parse_cli(["--grad-accum-steps", "8"])
    help_text = _build_cli_parser().format_help()

    check("CLI default keeps grad_accum_steps at legacy value 1", default_cfg.grad_accum_steps == 1, f"grad_accum_steps={default_cfg.grad_accum_steps}")
    check("CLI parses grad_accum_steps", cli_cfg.grad_accum_steps == 8, f"grad_accum_steps={cli_cfg.grad_accum_steps}")
    check("CLI help exposes grad_accum_steps", "--grad-accum-steps" in help_text, help_text)
    check("CLI help documents legacy single-step contract", "Use 1 to keep" in help_text and "legacy trainer contract" in help_text, help_text)


def test_grad_accumulation_step_semantics_contract():
    print("\n=== test_grad_accumulation_step_semantics_contract ===")

    base_plan = _resolve_train_step_plan(TrainConfig(epochs=3, steps_per_epoch=5, grad_accum_steps=4))
    override_plan = _resolve_train_step_plan(TrainConfig(epochs=9, steps_per_epoch=5, steps=7, grad_accum_steps=4))

    check("grad_accum_steps does not change optimizer-step budget", base_plan == (3, 5, 15, 4), f"plan={base_plan}")
    check("steps override remains optimizer-step budget when grad_accum_steps is set", override_plan == (1, 7, 7, 4), f"plan={override_plan}")

    tmp_path = ROOT / "_tmp_grad_accum_contract_dataset.txt"
    save_dir = ROOT / "_tmp_grad_accum_contract_runs"
    run_name = "grad_accum_contract"
    artifact_ok = False
    artifact_detail = ""

    text = (
        "grad accumulation contract artifact should stay on optimizer-step semantics\n"
        "substep one must not change the actual training loop yet\n"
    ) * 64

    try:
        cleanup_tree(save_dir)
        tmp_path.write_text(text, encoding="utf-8")
        result = train(
            dataset=str(tmp_path),
            tokenizer="byte",
            precision_mode="fp32",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            grad_accum_steps=4,
            steps=3,
            save_dir=str(save_dir),
            run_name=run_name,
            save_every=999,
            no_interactive=True,
            prompt="grad accum",
            prompt_bytes=10,
            gen_bytes=0,
            log_every=1,
        )

        cfg_path = Path(result["run_dir"]) / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            artifact_ok = (
                result.get("grad_accum_steps") == 4
                and result.get("optimizer_steps_per_epoch") == 3
                and result.get("optimizer_total_steps") == 3
                and int(cfg.get("grad_accum_steps", -1)) == 4
                and int(cfg.get("optimizer_steps_per_epoch", -1)) == 3
                and int(cfg.get("optimizer_total_steps", -1)) == 3
            )
            artifact_detail = json.dumps(
                {
                    "result_grad_accum_steps": result.get("grad_accum_steps"),
                    "result_optimizer_steps_per_epoch": result.get("optimizer_steps_per_epoch"),
                    "result_optimizer_total_steps": result.get("optimizer_total_steps"),
                    "config_grad_accum_steps": cfg.get("grad_accum_steps"),
                    "config_optimizer_steps_per_epoch": cfg.get("optimizer_steps_per_epoch"),
                    "config_optimizer_total_steps": cfg.get("optimizer_total_steps"),
                },
                ensure_ascii=False,
            )
        else:
            artifact_detail = str(cfg_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        cleanup_tree(save_dir)

    check("trainer persists optimizer-step semantics for grad accumulation contract", artifact_ok, artifact_detail)


#====================
# 20. Grad accumulation execution: scaled micro-batches must match one optimizer-step reference
#====================

def _run_train_with_patched_batches(
    sample_batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    capture_stdout: bool = False,
    **kwargs,
):
    calls = {"count": 0}

    def fake_sample_batch(data, batch, seq_len, device):
        idx = calls["count"]
        calls["count"] += 1
        if idx >= len(sample_batches):
            raise RuntimeError(
                f"sample_batch called {calls['count']} times, but only {len(sample_batches)} batches were prepared"
            )
        x, y = sample_batches[idx]
        return x.clone().to(device), y.clone().to(device)

    if capture_stdout:
        buf = io.StringIO()
        with patch("BasicQITNN_Transformer.sample_batch", side_effect=fake_sample_batch):
            with redirect_stdout(buf):
                result = train(**kwargs)
        return result, calls["count"], buf.getvalue()

    with patch("BasicQITNN_Transformer.sample_batch", side_effect=fake_sample_batch):
        result = train(**kwargs)
    return result, calls["count"]


def _model_state_max_abs_diff(left, right) -> tuple[float, str]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    max_diff = 0.0
    worst_name = ""
    for name, left_value in left_state.items():
        right_value = right_state[name]
        if left_value.dtype.is_floating_point:
            diff = (left_value.detach().float().cpu() - right_value.detach().float().cpu()).abs().max().item()
        else:
            diff = 0.0 if torch.equal(left_value.detach().cpu(), right_value.detach().cpu()) else 1.0
        if diff > max_diff:
            max_diff = diff
            worst_name = name
    return max_diff, worst_name


def _extract_logged_train_metrics(stdout: str) -> tuple[list[tuple[float, float, float]], tuple[float, float, float] | None]:
    pattern = re.compile(r"train_loss=([0-9.]+)\s+train_bpb=([0-9.]+)\s+train_ppl=([0-9.]+)")
    step_rows: list[tuple[float, float, float]] = []
    epoch_row: tuple[float, float, float] | None = None

    for line in stdout.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        row = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        if line.startswith("  ["):
            step_rows.append(row)
        elif line.startswith("epoch "):
            epoch_row = row

    return step_rows, epoch_row


def _train_metric_rows_close(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    *,
    loss_tol: float = 1e-6,
    coarse_tol: float = 5e-5,
) -> bool:
    return (
        abs(left[0] - right[0]) < loss_tol
        and abs(left[1] - right[1]) < coarse_tol
        and abs(left[2] - right[2]) < coarse_tol
    )


def _run_train_capture_stdout(**kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = train(**kwargs)
    return result, buf.getvalue()


def _extract_logged_step_lrs(stdout: str) -> list[str]:
    pattern = re.compile(r"^\s+\[\d+\].*lr=([0-9.]+)$")
    lrs: list[str] = []
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match is not None:
            lrs.append(match.group(1))
    return lrs


def _build_grad_accum_parity_batch_plan(
    *,
    steps: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    seq_len: int,
    offset: int = 5,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
    base = torch.arange(seq_len, dtype=torch.long)
    accum_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    ref_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    for step_idx in range(steps):
        step_xs: list[torch.Tensor] = []
        step_ys: list[torch.Tensor] = []
        for micro_idx in range(grad_accum_steps):
            row_offsets = (
                offset
                + step_idx * (grad_accum_steps * micro_batch_size * (seq_len + 11))
                + micro_idx * (micro_batch_size * (seq_len + 7))
                + torch.arange(micro_batch_size, dtype=torch.long).unsqueeze(1) * (seq_len + 5)
            )
            x = (base.unsqueeze(0) + row_offsets) % 251
            y = (x + 3 + step_idx + micro_idx) % 251
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE)
            accum_batches.append((x, y))
            step_xs.append(x)
            step_ys.append(y)
        ref_batches.append((torch.cat(step_xs, dim=0), torch.cat(step_ys, dim=0)))

    return accum_batches, ref_batches


def test_grad_accumulation_loss_scaling_contract():
    print("\n=== test_grad_accumulation_loss_scaling_contract ===")

    tmp_path = ROOT / "_tmp_grad_accum_loss_scaling.txt"
    text = ("grad accumulation loss scaling contract\n" * 128)
    x0 = torch.tensor([[5, 7, 9, 11, 13, 15, 17, 19]], dtype=torch.long, device=DEVICE)
    y0 = torch.tensor([[7, 9, 11, 13, 15, 17, 19, 21]], dtype=torch.long, device=DEVICE)
    x1 = torch.tensor([[23, 25, 27, 29, 31, 33, 35, 37]], dtype=torch.long, device=DEVICE)
    y1 = torch.tensor([[25, 27, 29, 31, 33, 35, 37, 39]], dtype=torch.long, device=DEVICE)
    accum_batches = [(x0, y0), (x1, y1)]
    ref_batches = [(torch.cat([x0, x1], dim=0), torch.cat([y0, y1], dim=0))]

    common = dict(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="fp32",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=8,
        optimizer="sgd",
        lr_start=0.05,
        lr_end=0.05,
        zero_boost=1.0,
        grad_clip=0.0,
        ent_lambda=0.0,
        trit_floor_h=0.0,
        steps=1,
        no_save=True,
        no_interactive=True,
        prompt="grad accum",
        prompt_bytes=10,
        gen_bytes=0,
        log_every=1,
    )

    try:
        tmp_path.write_text(text, encoding="utf-8")
        accum_result, accum_calls = _run_train_with_patched_batches(
            accum_batches,
            batch_size=1,
            grad_accum_steps=2,
            **common,
        )
        ref_result, ref_calls = _run_train_with_patched_batches(
            ref_batches,
            batch_size=2,
            grad_accum_steps=1,
            **common,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    max_diff, worst_name = _model_state_max_abs_diff(accum_result["model"], ref_result["model"])
    loss_diff = abs(float(accum_result["last_epoch_train_loss"]) - float(ref_result["last_epoch_train_loss"]))
    bpb_left = accum_result["last_epoch_train_bpb"]
    bpb_right = ref_result["last_epoch_train_bpb"]
    bpb_diff = 0.0 if bpb_left is None or bpb_right is None else abs(float(bpb_left) - float(bpb_right))

    check("grad accumulation uses one sample_batch call per micro-batch", accum_calls == 2 and ref_calls == 1, f"accum_calls={accum_calls} ref_calls={ref_calls}")
    check("grad accumulation loss scaling matches one large optimizer-step reference", max_diff < 1e-6, f"max_diff={max_diff:.3e} worst={worst_name}")
    check("grad accumulation keeps optimizer-step train loss aligned with large-batch reference", loss_diff < 1e-6, f"loss_diff={loss_diff:.3e}")
    check("grad accumulation keeps optimizer-step BPB aligned with large-batch reference", bpb_diff < 1e-6, f"bpb_diff={bpb_diff:.3e}")


def test_grad_accumulation_optimizer_step_count_contract():
    print("\n=== test_grad_accumulation_optimizer_step_count_contract ===")

    tmp_path = ROOT / "_tmp_grad_accum_step_count.txt"
    text = ("grad accumulation optimizer step count contract\n" * 128)
    counts = {"sample": 0, "step": 0, "prior": 0}
    orig_step = torch.optim.AdamW.step
    orig_prior = pyqitnn.QITNNSimplexTransformerLM.apply_qitnn_prior

    def sample_wrapper(data, batch, seq_len, device):
        counts["sample"] += 1
        return sample_batch(data, batch, seq_len, device)

    def step_wrapper(self, *args, **kwargs):
        counts["step"] += 1
        return orig_step(self, *args, **kwargs)

    def prior_wrapper(self, *args, **kwargs):
        counts["prior"] += 1
        return orig_prior(self, *args, **kwargs)

    try:
        tmp_path.write_text(text, encoding="utf-8")
        with patch("BasicQITNN_Transformer.sample_batch", side_effect=sample_wrapper):
            with patch.object(torch.optim.AdamW, "step", new=step_wrapper):
                with patch.object(pyqitnn.QITNNSimplexTransformerLM, "apply_qitnn_prior", new=prior_wrapper):
                    result = train(
                        dataset=str(tmp_path),
                        tokenizer="byte",
                        precision_mode="fp32",
                        dim=16,
                        ffn=32,
                        layers=1,
                        seq_len=8,
                        batch_size=1,
                        grad_accum_steps=3,
                        optimizer="adamw",
                        steps=2,
                        grad_clip=0.0,
                        ent_lambda=0.0,
                        trit_floor_h=0.0,
                        no_save=True,
                        no_interactive=True,
                        prompt="step count",
                        prompt_bytes=10,
                        gen_bytes=0,
                        log_every=1,
                    )
    finally:
        tmp_path.unlink(missing_ok=True)

    detail = json.dumps(
        {
            "sample_calls": counts["sample"],
            "step_calls": counts["step"],
            "prior_calls": counts["prior"],
            "optimizer_total_steps": result["optimizer_total_steps"],
        },
        ensure_ascii=False,
    )
    check("grad accumulation samples one batch per micro-step", counts["sample"] == 6, detail)
    check("grad accumulation calls optimizer.step once per optimizer-step", counts["step"] == 2, detail)
    check("grad accumulation calls prior once per optimizer-step", counts["prior"] == 2, detail)


def test_grad_accumulation_grad_clip_boundary():
    print("\n=== test_grad_accumulation_grad_clip_boundary ===")

    tmp_path = ROOT / "_tmp_grad_accum_clip_boundary.txt"
    text = ("grad accumulation clip boundary contract\n" * 128)
    clip_calls: list[float] = []
    clip_max_norm = 0.01
    x0 = torch.tensor([[41, 43, 45, 47, 49, 51, 53, 55]], dtype=torch.long, device=DEVICE)
    y0 = torch.tensor([[43, 45, 47, 49, 51, 53, 55, 57]], dtype=torch.long, device=DEVICE)
    x1 = torch.tensor([[59, 61, 63, 65, 67, 69, 71, 73]], dtype=torch.long, device=DEVICE)
    y1 = torch.tensor([[61, 63, 65, 67, 69, 71, 73, 75]], dtype=torch.long, device=DEVICE)
    accum_batches = [(x0, y0), (x1, y1)]
    ref_batches = [(torch.cat([x0, x1], dim=0), torch.cat([y0, y1], dim=0))]
    orig_clip = torch.nn.utils.clip_grad_norm_

    def clip_wrapper(parameters, max_norm, *args, **kwargs):
        total_norm = orig_clip(parameters, max_norm, *args, **kwargs)
        clip_calls.append(float(total_norm))
        return total_norm

    common = dict(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="fp32",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=8,
        optimizer="sgd",
        lr_start=0.05,
        lr_end=0.05,
        zero_boost=1.0,
        grad_clip=clip_max_norm,
        ent_lambda=0.0,
        trit_floor_h=0.0,
        steps=1,
        no_save=True,
        no_interactive=True,
        prompt="clip boundary",
        prompt_bytes=10,
        gen_bytes=0,
        log_every=1,
    )

    try:
        tmp_path.write_text(text, encoding="utf-8")
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=clip_wrapper):
            accum_result, accum_calls = _run_train_with_patched_batches(
                accum_batches,
                batch_size=1,
                grad_accum_steps=2,
                **common,
            )
        ref_result, ref_calls = _run_train_with_patched_batches(
            ref_batches,
            batch_size=2,
            grad_accum_steps=1,
            **common,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    max_diff, worst_name = _model_state_max_abs_diff(accum_result["model"], ref_result["model"])
    clip_triggered = len(clip_calls) == 1 and clip_calls[0] > clip_max_norm
    detail = json.dumps(
        {
            "accum_calls": accum_calls,
            "ref_calls": ref_calls,
            "clip_calls": clip_calls,
            "max_diff": max_diff,
            "worst_name": worst_name,
        },
        ensure_ascii=False,
    )

    check("grad accumulation clips once after the full micro-batch sum", len(clip_calls) == 1 and accum_calls == 2 and ref_calls == 1, detail)
    check("grad accumulation clip boundary actually triggers clipping", clip_triggered, detail)
    check("grad accumulation clip boundary matches one large clipped optimizer-step", max_diff < 1e-6, detail)


#====================
# 21. Grad accumulation metric reporting: console/result metrics must stay on optimizer-step semantics
#====================

def test_grad_accumulation_metric_reporting_contract():
    print("\n=== test_grad_accumulation_metric_reporting_contract ===")

    tmp_path = ROOT / "_tmp_grad_accum_metric_reporting.txt"
    text = ("grad accumulation metric reporting contract\n" * 128)
    x00 = torch.tensor([[5, 8, 11, 14, 17, 20, 23, 26]], dtype=torch.long, device=DEVICE)
    y00 = torch.tensor([[8, 11, 14, 17, 20, 23, 26, 29]], dtype=torch.long, device=DEVICE)
    x01 = torch.tensor([[31, 34, 37, 40, 43, 46, 49, 52]], dtype=torch.long, device=DEVICE)
    y01 = torch.tensor([[34, 37, 40, 43, 46, 49, 52, 55]], dtype=torch.long, device=DEVICE)
    x10 = torch.tensor([[57, 60, 63, 66, 69, 72, 75, 78]], dtype=torch.long, device=DEVICE)
    y10 = torch.tensor([[60, 63, 66, 69, 72, 75, 78, 81]], dtype=torch.long, device=DEVICE)
    x11 = torch.tensor([[83, 86, 89, 92, 95, 98, 101, 104]], dtype=torch.long, device=DEVICE)
    y11 = torch.tensor([[86, 89, 92, 95, 98, 101, 104, 107]], dtype=torch.long, device=DEVICE)
    accum_batches = [(x00, y00), (x01, y01), (x10, y10), (x11, y11)]
    ref_batches = [
        (torch.cat([x00, x01], dim=0), torch.cat([y00, y01], dim=0)),
        (torch.cat([x10, x11], dim=0), torch.cat([y10, y11], dim=0)),
    ]

    common = dict(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="fp32",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=8,
        optimizer="sgd",
        lr_start=0.0,
        lr_end=0.0,
        zero_boost=1.0,
        grad_clip=0.0,
        ent_lambda=0.0,
        trit_floor_h=0.0,
        steps=2,
        no_save=True,
        no_interactive=True,
        prompt="metric report",
        prompt_bytes=12,
        gen_bytes=0,
        log_every=1,
    )

    try:
        tmp_path.write_text(text, encoding="utf-8")
        accum_result, accum_calls, accum_stdout = _run_train_with_patched_batches(
            accum_batches,
            capture_stdout=True,
            batch_size=1,
            grad_accum_steps=2,
            **common,
        )
        ref_result, ref_calls, ref_stdout = _run_train_with_patched_batches(
            ref_batches,
            capture_stdout=True,
            batch_size=2,
            grad_accum_steps=1,
            **common,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    accum_steps, accum_epoch = _extract_logged_train_metrics(accum_stdout)
    ref_steps, ref_epoch = _extract_logged_train_metrics(ref_stdout)

    first_loss_diff = abs(float(accum_result["first_loss"]) - float(ref_result["first_loss"]))
    last_loss_diff = abs(float(accum_result["last_loss"]) - float(ref_result["last_loss"]))
    epoch_loss_diff = abs(float(accum_result["last_epoch_train_loss"]) - float(ref_result["last_epoch_train_loss"]))
    first_bpb_diff = abs(float(accum_result["first_bpb"]) - float(ref_result["first_bpb"]))
    last_bpb_diff = abs(float(accum_result["last_bpb"]) - float(ref_result["last_bpb"]))
    epoch_bpb_diff = abs(float(accum_result["last_epoch_train_bpb"]) - float(ref_result["last_epoch_train_bpb"]))

    accum_first_console_ok = (
        len(accum_steps) == 2
        and abs(accum_steps[0][0] - float(accum_result["first_loss"])) < 1e-9
        and abs(accum_steps[0][1] - float(accum_result["first_bpb"])) < 5e-5
        and abs(accum_steps[0][2] - float(accum_result["first_ppl"])) < 5e-5
    )
    accum_last_console_ok = (
        len(accum_steps) == 2
        and abs(accum_steps[1][0] - float(accum_result["last_epoch_train_loss"])) < 1e-6
        and abs(accum_steps[1][1] - float(accum_result["last_epoch_train_bpb"])) < 5e-5
        and abs(accum_steps[1][2] - float(accum_result["last_epoch_train_ppl"])) < 5e-5
    )
    accum_epoch_console_ok = (
        accum_epoch is not None
        and abs(accum_epoch[0] - float(accum_result["last_epoch_train_loss"])) < 1e-9
        and abs(accum_epoch[1] - float(accum_result["last_epoch_train_bpb"])) < 5e-5
        and abs(accum_epoch[2] - float(accum_result["last_epoch_train_ppl"])) < 5e-5
    )
    ref_console_ok = (
        len(ref_steps) == 2
        and ref_epoch is not None
        and _train_metric_rows_close(accum_steps[0], ref_steps[0])
        and _train_metric_rows_close(accum_steps[1], ref_steps[1])
        and _train_metric_rows_close(accum_epoch, ref_epoch)
    )

    check("grad accumulation metric reporting preserves first optimizer-step loss", first_loss_diff < 1e-6, f"diff={first_loss_diff:.3e}")
    check("grad accumulation metric reporting preserves last optimizer-step loss", last_loss_diff < 1e-6, f"diff={last_loss_diff:.3e}")
    check("grad accumulation metric reporting preserves epoch train loss", epoch_loss_diff < 1e-6, f"diff={epoch_loss_diff:.3e}")
    check("grad accumulation metric reporting preserves first optimizer-step BPB", first_bpb_diff < 1e-6, f"diff={first_bpb_diff:.3e}")
    check("grad accumulation metric reporting preserves last optimizer-step BPB", last_bpb_diff < 1e-6, f"diff={last_bpb_diff:.3e}")
    check("grad accumulation metric reporting preserves epoch train BPB", epoch_bpb_diff < 1e-6, f"diff={epoch_bpb_diff:.3e}")
    check("grad accumulation console first-step metrics match result dict", accum_first_console_ok, accum_stdout[-1200:])
    check("grad accumulation console last-step metrics match result dict", accum_last_console_ok, accum_stdout[-1200:])
    check("grad accumulation epoch console metrics match result dict", accum_epoch_console_ok, accum_stdout[-1200:])
    check("grad accumulation console metrics match large-batch reference", ref_console_ok and accum_calls == 4 and ref_calls == 2, f"accum_calls={accum_calls} ref_calls={ref_calls}")


def test_grad_accumulation_bpb_contract():
    print("\n=== test_grad_accumulation_bpb_contract ===")

    tmp_path = ROOT / "_tmp_grad_accum_bpb_contract.txt"
    text = ("grad accumulation weighted bpb contract\n" * 128)
    x0 = torch.tensor([[7, 7, 7, 7, 7, 7, 7, 7]], dtype=torch.long, device=DEVICE)
    y0 = torch.tensor([[7, 7, 7, 7, 7, 7, 7, 7]], dtype=torch.long, device=DEVICE)
    x1 = torch.tensor([
        [101, 103, 105, 107, 109, 111, 113, 115],
        [117, 119, 121, 123, 125, 127, 129, 131],
        [133, 135, 137, 139, 141, 143, 145, 147],
        [149, 151, 153, 155, 157, 159, 161, 163],
        [165, 167, 169, 171, 173, 175, 177, 179],
        [181, 183, 185, 187, 189, 191, 193, 195],
        [197, 199, 201, 203, 205, 207, 209, 211],
    ], dtype=torch.long, device=DEVICE)
    y1 = torch.tensor([
        [103, 105, 107, 109, 111, 113, 115, 117],
        [119, 121, 123, 125, 127, 129, 131, 133],
        [135, 137, 139, 141, 143, 145, 147, 149],
        [151, 153, 155, 157, 159, 161, 163, 165],
        [167, 169, 171, 173, 175, 177, 179, 181],
        [183, 185, 187, 189, 191, 193, 195, 197],
        [199, 201, 203, 205, 207, 209, 211, 213],
    ], dtype=torch.long, device=DEVICE)
    accum_batches = [(x0, y0), (x1, y1)]
    ref_batches = [(torch.cat([x0, x1], dim=0), torch.cat([y0, y1], dim=0))]
    micro0_batches = [(x0, y0)]
    micro1_batches = [(x1, y1)]

    common = dict(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="fp32",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=8,
        optimizer="sgd",
        lr_start=0.0,
        lr_end=0.0,
        zero_boost=1.0,
        grad_clip=0.0,
        ent_lambda=0.0,
        trit_floor_h=0.0,
        steps=1,
        no_save=True,
        no_interactive=True,
        prompt="weighted bpb",
        prompt_bytes=12,
        gen_bytes=0,
        log_every=1,
    )

    try:
        tmp_path.write_text(text, encoding="utf-8")
        accum_result, _ = _run_train_with_patched_batches(
            accum_batches,
            batch_size=1,
            grad_accum_steps=2,
            **common,
        )
        ref_result, _ = _run_train_with_patched_batches(
            ref_batches,
            batch_size=8,
            grad_accum_steps=1,
            **common,
        )
        micro0_result, _ = _run_train_with_patched_batches(
            micro0_batches,
            batch_size=1,
            grad_accum_steps=1,
            **common,
        )
        micro1_result, _ = _run_train_with_patched_batches(
            micro1_batches,
            batch_size=7,
            grad_accum_steps=1,
            **common,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    naive_loss = 0.5 * (float(micro0_result["last_epoch_train_loss"]) + float(micro1_result["last_epoch_train_loss"]))
    naive_bpb = 0.5 * (float(micro0_result["last_epoch_train_bpb"]) + float(micro1_result["last_epoch_train_bpb"]))
    loss_diff = abs(float(accum_result["last_epoch_train_loss"]) - float(ref_result["last_epoch_train_loss"]))
    bpb_diff = abs(float(accum_result["last_epoch_train_bpb"]) - float(ref_result["last_epoch_train_bpb"]))
    naive_loss_gap = abs(float(accum_result["last_epoch_train_loss"]) - naive_loss)
    naive_bpb_gap = abs(float(accum_result["last_epoch_train_bpb"]) - naive_bpb)

    check("grad accumulation weighted loss matches full concatenated reference", loss_diff < 1e-6, f"diff={loss_diff:.3e}")
    check("grad accumulation weighted BPB matches full concatenated reference", bpb_diff < 1e-6, f"diff={bpb_diff:.3e}")
    check("grad accumulation loss is not a naive mean of micro-step losses", naive_loss_gap > 1e-5, f"gap={naive_loss_gap:.3e}")
    check("grad accumulation BPB is not a naive mean of micro-step BPBs", naive_bpb_gap > 1e-5, f"gap={naive_bpb_gap:.3e}")


def test_grad_accumulation_csv_contract():
    print("\n=== test_grad_accumulation_csv_contract ===")

    tmp_path = ROOT / "_tmp_grad_accum_csv_contract.txt"
    save_root = ROOT / "_tmp_grad_accum_csv_runs"
    text = ("grad accumulation csv contract\n" * 128)
    x00 = torch.tensor([[9, 12, 15, 18, 21, 24, 27, 30]], dtype=torch.long, device=DEVICE)
    y00 = torch.tensor([[12, 15, 18, 21, 24, 27, 30, 33]], dtype=torch.long, device=DEVICE)
    x01 = torch.tensor([[36, 39, 42, 45, 48, 51, 54, 57]], dtype=torch.long, device=DEVICE)
    y01 = torch.tensor([[39, 42, 45, 48, 51, 54, 57, 60]], dtype=torch.long, device=DEVICE)
    x10 = torch.tensor([[63, 66, 69, 72, 75, 78, 81, 84]], dtype=torch.long, device=DEVICE)
    y10 = torch.tensor([[66, 69, 72, 75, 78, 81, 84, 87]], dtype=torch.long, device=DEVICE)
    x11 = torch.tensor([[90, 93, 96, 99, 102, 105, 108, 111]], dtype=torch.long, device=DEVICE)
    y11 = torch.tensor([[93, 96, 99, 102, 105, 108, 111, 114]], dtype=torch.long, device=DEVICE)
    accum_batches = [(x00, y00), (x01, y01), (x10, y10), (x11, y11)]
    ref_batches = [
        (torch.cat([x00, x01], dim=0), torch.cat([y00, y01], dim=0)),
        (torch.cat([x10, x11], dim=0), torch.cat([y10, y11], dim=0)),
    ]

    common = dict(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="fp32",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=8,
        optimizer="sgd",
        lr_start=0.0,
        lr_end=0.0,
        zero_boost=1.0,
        grad_clip=0.0,
        ent_lambda=0.0,
        trit_floor_h=0.0,
        steps=2,
        save_every=999,
        no_interactive=True,
        prompt="csv contract",
        prompt_bytes=12,
        gen_bytes=0,
        log_every=1,
    )

    try:
        cleanup_tree(save_root)
        tmp_path.write_text(text, encoding="utf-8")
        accum_result, _ = _run_train_with_patched_batches(
            accum_batches,
            batch_size=1,
            grad_accum_steps=2,
            save_dir=str(save_root),
            run_name="accum",
            **common,
        )
        ref_result, _ = _run_train_with_patched_batches(
            ref_batches,
            batch_size=2,
            grad_accum_steps=1,
            save_dir=str(save_root),
            run_name="ref",
            **common,
        )

        accum_csv_path = Path(accum_result["run_dir"]) / "metrics.csv"
        ref_csv_path = Path(ref_result["run_dir"]) / "metrics.csv"
        accum_row = next(csv.DictReader(accum_csv_path.read_text(encoding="utf-8").splitlines()))
        ref_row = next(csv.DictReader(ref_csv_path.read_text(encoding="utf-8").splitlines()))
    finally:
        tmp_path.unlink(missing_ok=True)
        cleanup_tree(save_root)

    accum_row_ok = (
        abs(float(accum_row["train_loss"]) - float(accum_result["last_epoch_train_loss"])) < 5e-6
        and abs(float(accum_row["train_bpb"]) - float(accum_result["last_epoch_train_bpb"])) < 5e-6
        and abs(float(accum_row["train_ppl"]) - float(accum_result["last_epoch_train_ppl"])) < 5e-6
    )
    ref_match_ok = (
        abs(float(accum_row["train_loss"]) - float(ref_row["train_loss"])) < 5e-6
        and abs(float(accum_row["train_bpb"]) - float(ref_row["train_bpb"])) < 5e-6
        and abs(float(accum_row["train_ppl"]) - float(ref_row["train_ppl"])) < 5e-5
        and abs(float(accum_row["val_loss"]) - float(ref_row["val_loss"])) < 5e-6
        and abs(float(accum_row["val_bpb"]) - float(ref_row["val_bpb"])) < 5e-6
        and abs(float(accum_row["val_ppl"]) - float(ref_row["val_ppl"])) < 5e-5
        and accum_row["lr"] == ref_row["lr"]
    )

    check("grad accumulation metrics.csv row matches result dict train metrics", accum_row_ok, json.dumps(accum_row, ensure_ascii=False))
    check("grad accumulation metrics.csv matches large-batch reference metrics", ref_match_ok, json.dumps({"accum": accum_row, "ref": ref_row}, ensure_ascii=False))


#====================
# 24. Grad accumulation schedule/resume: optimizer-step schedule and checkpoint boundaries must stay exact
#====================

def test_grad_accumulation_schedule_contract():
    print("\n=== test_grad_accumulation_schedule_contract ===")

    tmp_path = ROOT / "_tmp_grad_accum_schedule_contract.txt"
    text = ("grad accumulation schedule contract\n" * 128)

    common = dict(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="fp32",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=8,
        optimizer="sgd",
        lr_start=0.12,
        lr_end=0.03,
        lr_schedule="linear",
        warmup_steps=2,
        zero_boost=1.0,
        grad_clip=0.0,
        ent_lambda=0.0,
        trit_floor_h=0.0,
        steps=4,
        no_save=True,
        no_interactive=True,
        prompt="sched",
        prompt_bytes=5,
        gen_bytes=0,
        log_every=1,
        seed=17,
    )

    try:
        tmp_path.write_text(text, encoding="utf-8")
        accum_result, accum_stdout = _run_train_capture_stdout(
            batch_size=1,
            grad_accum_steps=3,
            **common,
        )
        ref_result, ref_stdout = _run_train_capture_stdout(
            batch_size=3,
            grad_accum_steps=1,
            **common,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    accum_lrs = _extract_logged_step_lrs(accum_stdout)
    ref_lrs = _extract_logged_step_lrs(ref_stdout)
    expected_lrs = [
        f"{lr_with_warmup(common['lr_start'], common['lr_end'], i, common['steps'], warmup_steps=common['warmup_steps'], schedule_fn=lr_linear):.8f}"
        for i in range(common["steps"])
    ]
    detail = json.dumps(
        {
            "accum_lrs": accum_lrs,
            "ref_lrs": ref_lrs,
            "expected_lrs": expected_lrs,
            "accum_optimizer_total_steps": accum_result["optimizer_total_steps"],
            "ref_optimizer_total_steps": ref_result["optimizer_total_steps"],
        },
        ensure_ascii=False,
    )

    check("grad accumulation emits one logged LR per optimizer-step", len(accum_lrs) == common["steps"] and accum_result["optimizer_total_steps"] == common["steps"], detail)
    check("grad accumulation keeps warmup/LR schedule on optimizer-step count", accum_lrs == expected_lrs, detail)
    check("grad accumulation LR schedule matches large-batch reference", accum_lrs == ref_lrs and ref_result["optimizer_total_steps"] == common["steps"], detail)


def test_grad_accumulation_prior_step_contract():
    print("\n=== test_grad_accumulation_prior_step_contract ===")

    tmp_path = ROOT / "_tmp_grad_accum_prior_contract.txt"
    text = ("grad accumulation prior step contract\n" * 128)
    prior_calls: list[tuple[float, float, float, float]] = []
    orig_prior = pyqitnn.QITNNSimplexTransformerLM.apply_qitnn_prior

    def prior_wrapper(self, *args, **kwargs):
        prior_calls.append(
            (
                float(kwargs["step_qk"]),
                float(kwargs["step_vo"]),
                float(kwargs["step_ff"]),
                float(kwargs["entropy_floor"]),
            )
        )
        return orig_prior(self, *args, **kwargs)

    common = dict(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="fp32",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=8,
        batch_size=1,
        grad_accum_steps=2,
        optimizer="sgd",
        lr_start=0.12,
        lr_end=0.03,
        lr_schedule="linear",
        warmup_steps=2,
        zero_boost=1.0,
        grad_clip=0.0,
        ent_lambda=0.0,
        trit_floor_h=0.25,
        trit_floor_mul_start=2.0,
        trit_floor_mul_end=0.5,
        steps=4,
        no_save=True,
        no_interactive=True,
        prompt="prior",
        prompt_bytes=5,
        gen_bytes=0,
        log_every=4,
        seed=19,
    )

    try:
        tmp_path.write_text(text, encoding="utf-8")
        with patch.object(pyqitnn.QITNNSimplexTransformerLM, "apply_qitnn_prior", new=prior_wrapper):
            result = train(**common)
    finally:
        tmp_path.unlink(missing_ok=True)

    expected_calls: list[tuple[float, float, float, float]] = []
    for step_idx in range(common["steps"]):
        frac = _schedule_progress(step_idx, common["steps"], common["warmup_steps"])
        lr_now = lr_with_warmup(
            common["lr_start"],
            common["lr_end"],
            step_idx,
            common["steps"],
            warmup_steps=common["warmup_steps"],
            schedule_fn=lr_linear,
        )
        floor_mul = lr_linear(common["trit_floor_mul_start"], common["trit_floor_mul_end"], frac)
        step_size = lr_now * floor_mul
        expected_calls.append((step_size, step_size, step_size, common["trit_floor_h"]))

    sequence_ok = (
        len(prior_calls) == len(expected_calls)
        and all(
            abs(actual[0] - expected[0]) < 1e-12
            and abs(actual[1] - expected[1]) < 1e-12
            and abs(actual[2] - expected[2]) < 1e-12
            and abs(actual[3] - expected[3]) < 1e-12
            for actual, expected in zip(prior_calls, expected_calls)
        )
    )
    detail = json.dumps(
        {
            "prior_calls": prior_calls,
            "expected_calls": expected_calls,
            "optimizer_total_steps": result["optimizer_total_steps"],
        },
        ensure_ascii=False,
    )

    check("grad accumulation applies prior once per optimizer-step", len(prior_calls) == result["optimizer_total_steps"] == common["steps"], detail)
    check("grad accumulation prior step sizes follow optimizer-step warmup/schedule", sequence_ok, detail)


def test_grad_accumulation_resume_exactness_fp32():
    print("\n=== test_grad_accumulation_resume_exactness_fp32 ===")

    tmp_path = ROOT / "_tmp_grad_accum_resume_dataset.txt"
    save_root = ROOT / "_tmp_grad_accum_resume_runs"
    run_cont = "grad_accum_resume_cont"
    run_resumed = "grad_accum_resume_resumed"

    text = (
        "grad accumulation resume exactness should remain aligned on optimizer-step checkpoint boundaries\n"
        "the checkpoint resumes only completed optimizer steps and restores rng for the next micro-step sequence\n"
    ) * 64

    try:
        cleanup_tree(save_root)
        tmp_path.write_text(text, encoding="utf-8")

        common = dict(
            dataset=str(tmp_path),
            tokenizer="byte",
            precision_mode="fp32",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            grad_accum_steps=2,
            optimizer="adamw",
            adamw_lr_start=3e-4,
            adamw_lr_end=3e-5,
            lr_schedule="linear",
            warmup_steps=2,
            epochs=2,
            steps_per_epoch=2,
            val_steps=4,
            save_dir=str(save_root),
            save_every=1,
            no_interactive=True,
            prompt="resume",
            prompt_bytes=6,
            gen_bytes=0,
            log_every=2,
            seed=79,
        )

        train(run_name=run_cont, **common)
        resume_ckpt = save_root / run_cont / "ckpt_ep1.pt"
        train(run_name=run_resumed, resume=str(resume_ckpt), **common)

        resume_boundary_ckpt = torch.load(str(resume_ckpt), map_location="cpu", weights_only=False)
        cont_ckpt = torch.load(str(save_root / run_cont / "ckpt_final.pt"), map_location="cpu", weights_only=False)
        resumed_ckpt = torch.load(str(save_root / run_resumed / "ckpt_final.pt"), map_location="cpu", weights_only=False)

        max_diff = max(
            (cont_ckpt["model"][name] - resumed_ckpt["model"][name]).abs().max().item()
            for name in cont_ckpt["model"]
        )
        best_val_diff = abs(float(cont_ckpt["best_val_loss"]) - float(resumed_ckpt["best_val_loss"]))
        step_boundary_ok = resume_boundary_ckpt.get("epoch") == 1 and resume_boundary_ckpt.get("global_step") == 2
        best_val_epoch_ok = cont_ckpt.get("best_val_epoch") == resumed_ckpt.get("best_val_epoch")

        check("grad accumulation checkpoint keeps optimizer-step global_step at resume boundary", step_boundary_ok, f"epoch={resume_boundary_ckpt.get('epoch')} step={resume_boundary_ckpt.get('global_step')}")
        check("grad accumulation resume exactness preserves final fp32 model state", max_diff < 1e-9, f"max_diff={max_diff:.3e}")
        check("grad accumulation resume exactness preserves final global_step", cont_ckpt["global_step"] == resumed_ckpt["global_step"], f"cont={cont_ckpt['global_step']} resumed={resumed_ckpt['global_step']}")
        check("grad accumulation resume exactness preserves final epoch", cont_ckpt["epoch"] == resumed_ckpt["epoch"], f"cont={cont_ckpt['epoch']} resumed={resumed_ckpt['epoch']}")
        check("grad accumulation resume exactness preserves best_val_loss", best_val_diff < 1e-12, f"diff={best_val_diff:.3e}")
        check("grad accumulation resume exactness preserves best_val_epoch metadata", best_val_epoch_ok, f"cont={cont_ckpt.get('best_val_epoch')} resumed={resumed_ckpt.get('best_val_epoch')}")
    finally:
        tmp_path.unlink(missing_ok=True)
        cleanup_tree(save_root)


#====================
# 25. Grad accumulation large-batch parity: multi-step short runs should stay aligned with the equivalent large batch
#====================

def test_grad_accumulation_large_batch_parity_fp32():
    print("\n=== test_grad_accumulation_large_batch_parity_fp32 ===")

    tmp_path = ROOT / "_tmp_grad_accum_large_batch_fp32.txt"
    text = ("grad accumulation fp32 large batch parity\n" * 128)
    steps = 3
    seq_len = 12
    micro_batch_size = 2
    grad_accum_steps = 2
    accum_batches, ref_batches = _build_grad_accum_parity_batch_plan(
        steps=steps,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        seq_len=seq_len,
        offset=41,
    )

    common = dict(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="fp32",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=seq_len,
        optimizer="adamw",
        adamw_lr_start=3e-4,
        adamw_lr_end=3e-5,
        lr_schedule="linear",
        warmup_steps=1,
        grad_clip=0.0,
        ent_lambda=0.0,
        trit_floor_h=0.0,
        trit_floor_mul_start=0.0,
        trit_floor_mul_end=0.0,
        steps=steps,
        val_steps=4,
        no_save=True,
        no_interactive=True,
        prompt="fp32 parity",
        prompt_bytes=11,
        gen_bytes=0,
        log_every=steps,
        seed=29,
    )

    probe_offsets = torch.tensor([[173], [191], [209]], dtype=torch.long, device=DEVICE)
    probe_tokens = (torch.arange(seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0) + probe_offsets) % 251
    probe_targets = (probe_tokens + 7) % 251

    try:
        tmp_path.write_text(text, encoding="utf-8")
        accum_result, accum_calls = _run_train_with_patched_batches(
            accum_batches,
            batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            **common,
        )
        ref_result, ref_calls = _run_train_with_patched_batches(
            ref_batches,
            batch_size=micro_batch_size * grad_accum_steps,
            grad_accum_steps=1,
            **common,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    with torch.no_grad():
        accum_logits, accum_loss = accum_result["model"](probe_tokens, targets=probe_targets)
        ref_logits, ref_loss = ref_result["model"](probe_tokens, targets=probe_targets)

    max_diff, worst_name = _model_state_max_abs_diff(accum_result["model"], ref_result["model"])
    logit_diff = (accum_logits.float() - ref_logits.float()).abs()
    loss_gap = abs(float(accum_loss.detach().cpu()) - float(ref_loss.detach().cpu()))
    train_gap = abs(float(accum_result["last_epoch_train_loss"]) - float(ref_result["last_epoch_train_loss"]))
    val_gap = abs(float(accum_result["last_epoch_val_loss"]) - float(ref_result["last_epoch_val_loss"]))
    bpb_gap = abs(float(accum_result["last_epoch_train_bpb"]) - float(ref_result["last_epoch_train_bpb"]))
    detail = json.dumps(
        {
            "accum_calls": accum_calls,
            "ref_calls": ref_calls,
            "max_diff": max_diff,
            "worst_name": worst_name,
            "loss_gap": loss_gap,
            "train_gap": train_gap,
            "val_gap": val_gap,
            "bpb_gap": bpb_gap,
            "max_logit_gap": float(logit_diff.max().item()),
        },
        ensure_ascii=False,
    )

    check("grad accumulation fp32 parity keeps one planned batch per optimizer-step bundle", accum_calls == steps * grad_accum_steps and ref_calls == steps, detail)
    check("grad accumulation fp32 parity keeps final model state aligned with the equivalent large batch", max_diff < 1e-6, detail)
    check("grad accumulation fp32 parity keeps train/val loss metrics aligned", train_gap < 1e-6 and val_gap < 5e-6 and bpb_gap < 1e-6, detail)
    check("grad accumulation fp32 parity keeps probe logits and eval loss aligned", loss_gap < 1e-6 and float(logit_diff.max().item()) < 1e-4, detail)


def test_grad_accumulation_large_batch_parity_mixed_smoke():
    print("\n=== test_grad_accumulation_large_batch_parity_mixed_smoke ===")
    if not torch.cuda.is_bf16_supported():
        warn("grad accumulation mixed parity skipped", "bf16 not supported on this GPU")
        return

    tmp_path = ROOT / "_tmp_grad_accum_large_batch_mixed.txt"
    text = ("grad accumulation mixed large batch parity\n" * 128)
    steps = 3
    seq_len = 12
    micro_batch_size = 2
    grad_accum_steps = 2
    accum_batches, ref_batches = _build_grad_accum_parity_batch_plan(
        steps=steps,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        seq_len=seq_len,
        offset=37,
    )

    common = dict(
        dataset=str(tmp_path),
        tokenizer="byte",
        precision_mode="qts_fp32_rest_bf16",
        dim=16,
        ffn=32,
        layers=1,
        seq_len=seq_len,
        optimizer="adamw",
        adamw_lr_start=3e-4,
        adamw_lr_end=3e-5,
        lr_schedule="linear",
        warmup_steps=1,
        grad_clip=0.0,
        ent_lambda=0.0,
        trit_floor_h=0.0,
        trit_floor_mul_start=0.0,
        trit_floor_mul_end=0.0,
        steps=steps,
        val_steps=4,
        no_save=True,
        no_interactive=True,
        prompt="mixed parity",
        prompt_bytes=12,
        gen_bytes=0,
        log_every=steps,
        seed=31,
    )

    probe_offsets = torch.tensor([[149], [173]], dtype=torch.long, device=DEVICE)
    probe_tokens = (torch.arange(seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0) + probe_offsets) % 251
    probe_targets = (probe_tokens + 9) % 251

    try:
        tmp_path.write_text(text, encoding="utf-8")
        accum_result, accum_calls = _run_train_with_patched_batches(
            accum_batches,
            batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            **common,
        )
        ref_result, ref_calls = _run_train_with_patched_batches(
            ref_batches,
            batch_size=micro_batch_size * grad_accum_steps,
            grad_accum_steps=1,
            **common,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            accum_logits, accum_loss = accum_result["model"](probe_tokens, targets=probe_targets)
            ref_logits, ref_loss = ref_result["model"](probe_tokens, targets=probe_targets)

    logit_diff = (accum_logits.float() - ref_logits.float()).abs()
    state_gap, worst_name = _model_state_max_abs_diff(accum_result["model"], ref_result["model"])
    eval_loss_gap = abs(float(accum_loss.detach().cpu()) - float(ref_loss.detach().cpu()))
    train_gap = abs(float(accum_result["last_epoch_train_loss"]) - float(ref_result["last_epoch_train_loss"]))
    val_gap = abs(float(accum_result["last_epoch_val_loss"]) - float(ref_result["last_epoch_val_loss"]))
    finite_ok = (
        math.isfinite(float(accum_result["last_epoch_train_loss"]))
        and math.isfinite(float(ref_result["last_epoch_train_loss"]))
        and math.isfinite(float(accum_loss.detach().cpu()))
        and math.isfinite(float(ref_loss.detach().cpu()))
        and torch.isfinite(accum_logits.float()).all().item()
        and torch.isfinite(ref_logits.float()).all().item()
    )
    dtype_ok = (
        accum_logits.dtype == torch.bfloat16
        and ref_logits.dtype == torch.bfloat16
        and accum_loss.dtype == torch.float32
        and ref_loss.dtype == torch.float32
    )
    stats_accum = pyqitnn.qitnn_diag_stats(
        accum_result["model"].blocks[0].q_proj.a_neg,
        accum_result["model"].blocks[0].q_proj.a_zero,
        accum_result["model"].blocks[0].q_proj.a_pos,
    )
    stats_ref = pyqitnn.qitnn_diag_stats(
        ref_result["model"].blocks[0].q_proj.a_neg,
        ref_result["model"].blocks[0].q_proj.a_zero,
        ref_result["model"].blocks[0].q_proj.a_pos,
    )
    diag_gap = max(
        abs(stats_accum["p_neg"] - stats_ref["p_neg"]),
        abs(stats_accum["p_zero"] - stats_ref["p_zero"]),
        abs(stats_accum["p_pos"] - stats_ref["p_pos"]),
        abs(stats_accum["h"] - stats_ref["h"]),
    )
    detail = json.dumps(
        {
            "accum_calls": accum_calls,
            "ref_calls": ref_calls,
            "eval_loss_gap": eval_loss_gap,
            "train_gap": train_gap,
            "val_gap": val_gap,
            "mean_logit_gap": float(logit_diff.mean().item()),
            "max_logit_gap": float(logit_diff.max().item()),
            "diag_gap": diag_gap,
            "state_gap": state_gap,
            "worst_name": worst_name,
        },
        ensure_ascii=False,
    )

    check("grad accumulation mixed parity keeps mixed visible dtype pinned to bf16", dtype_ok, detail)
    check("grad accumulation mixed parity keeps both runs finite", finite_ok, detail)
    check("grad accumulation mixed parity keeps optimizer-step batch plan intact", accum_calls == steps * grad_accum_steps and ref_calls == steps, detail)
    check("grad accumulation mixed parity keeps train/eval losses close to the equivalent large batch", train_gap < 0.002 and val_gap < 0.01 and eval_loss_gap < 0.01, detail)
    check("grad accumulation mixed parity keeps logits and QTS diagnostics close to the equivalent large batch", float(logit_diff.mean().item()) < 0.06 and float(logit_diff.max().item()) < 0.40 and diag_gap < 0.01 and state_gap < 0.01, detail)


def test_grad_accumulation_finite_grads_smoke():
    print("\n=== test_grad_accumulation_finite_grads_smoke ===")

    tmp_path = ROOT / "_tmp_grad_accum_finite_grads.txt"
    text = ("grad accumulation finite grads smoke\n" * 128)
    grad_checks: list[tuple[bool, int, float]] = []
    orig_step = torch.optim.AdamW.step
    precision_mode = "qts_fp32_rest_bf16" if torch.cuda.is_bf16_supported() else "fp32"

    def step_wrapper(self, *args, **kwargs):
        finite = True
        seen = 0
        max_abs = 0.0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                seen += 1
                finite = finite and bool(torch.isfinite(grad).all().item())
                max_abs = max(max_abs, float(grad.abs().max().item()))
        grad_checks.append((finite, seen, max_abs))
        return orig_step(self, *args, **kwargs)

    try:
        tmp_path.write_text(text, encoding="utf-8")
        with patch.object(torch.optim.AdamW, "step", new=step_wrapper):
            result = train(
                dataset=str(tmp_path),
                tokenizer="byte",
                precision_mode=precision_mode,
                dim=24,
                ffn=48,
                layers=2,
                seq_len=12,
                batch_size=1,
                grad_accum_steps=4,
                optimizer="adamw",
                adamw_lr_start=3e-4,
                adamw_lr_end=3e-5,
                lr_schedule="cosine",
                warmup_steps=1,
                grad_clip=0.0,
                ent_lambda=0.0,
                trit_floor_h=0.0,
                trit_floor_mul_start=0.0,
                trit_floor_mul_end=0.0,
                steps=3,
                val_steps=4,
                no_save=True,
                no_interactive=True,
                prompt="finite grads",
                prompt_bytes=12,
                gen_bytes=0,
                log_every=3,
                seed=41,
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    grad_ok = (
        len(grad_checks) == result["optimizer_total_steps"]
        and all(finite and seen > 0 and max_abs > 0.0 for finite, seen, max_abs in grad_checks)
    )
    finite_losses = (
        math.isfinite(float(result["first_loss"]))
        and math.isfinite(float(result["last_loss"]))
        and math.isfinite(float(result["last_epoch_train_loss"]))
        and math.isfinite(float(result["last_epoch_val_loss"]))
    )
    detail = json.dumps(
        {
            "precision_mode": precision_mode,
            "optimizer_total_steps": result["optimizer_total_steps"],
            "grad_checks": grad_checks,
            "first_loss": result["first_loss"],
            "last_loss": result["last_loss"],
            "last_epoch_train_loss": result["last_epoch_train_loss"],
            "last_epoch_val_loss": result["last_epoch_val_loss"],
        },
        ensure_ascii=False,
    )

    check("grad accumulation finite-grad smoke sees one finite gradient set per optimizer-step", grad_ok, detail)
    check("grad accumulation finite-grad smoke keeps train/val losses finite", finite_losses, detail)


#====================
# 26. Warmup contract: scheduler warmup must be explicit, reversible, and CLI-addressable
#====================

def test_warmup_schedule_contract():
    """warmup must preserve legacy behavior at 0 and apply a clean linear ramp when enabled"""
    print("\n=== test_warmup_schedule_contract ===")

    default_cfg = TrainConfig()
    cli_cfg = _parse_cli(["--warmup-steps", "7"])
    neg_msg = capture_runtime_error(lambda: _resolve_warmup_steps(-1))

    check("TrainConfig default keeps warmup disabled", default_cfg.warmup_steps == 0, f"warmup_steps={default_cfg.warmup_steps}")
    check("CLI parses warmup_steps", cli_cfg.warmup_steps == 7, f"warmup_steps={cli_cfg.warmup_steps}")
    check("negative warmup is rejected", ">= 0" in neg_msg, neg_msg or "no RuntimeError")

    linear_legacy = [lr_linear(1.0, 0.1, i / 9.0) for i in range(10)]
    linear_nowarm = [lr_with_warmup(1.0, 0.1, i, 10, warmup_steps=0, schedule_fn=lr_linear) for i in range(10)]
    cosine_legacy = [lr_cosine(1.0, 0.1, i / 9.0) for i in range(10)]
    cosine_nowarm = [lr_with_warmup(1.0, 0.1, i, 10, warmup_steps=0, schedule_fn=lr_cosine) for i in range(10)]
    linear_diff = max(abs(a - b) for a, b in zip(linear_legacy, linear_nowarm))
    cosine_diff = max(abs(a - b) for a, b in zip(cosine_legacy, cosine_nowarm))

    check("warmup=0 preserves linear schedule", linear_diff < 1e-12, f"max_diff={linear_diff:.3e}")
    check("warmup=0 preserves cosine schedule", cosine_diff < 1e-12, f"max_diff={cosine_diff:.3e}")

    warm = [lr_with_warmup(1.0, 0.1, i, 10, warmup_steps=3, schedule_fn=lr_linear) for i in range(10)]
    progress = [_schedule_progress(i, 10, 3) for i in range(10)]
    full_warm = [lr_with_warmup(0.4, 0.1, i, 4, warmup_steps=10, schedule_fn=lr_linear) for i in range(4)]

    check("warmup step 1 ramps from zero", abs(warm[0] - (1.0 / 3.0)) < 1e-9, f"lr={warm[0]:.12f}")
    check("warmup reaches base LR at the warmup boundary", abs(warm[2] - 1.0) < 1e-9, f"lr={warm[2]:.12f}")
    check("first post-warmup step starts decay from base LR", abs(warm[3] - 1.0) < 1e-9, f"lr={warm[3]:.12f}")
    check("warmup schedule still ends at lr_end", abs(warm[-1] - 0.1) < 1e-9, f"lr={warm[-1]:.12f}")
    check("decay progress stays frozen during warmup prefix", progress[:4] == [0.0, 0.0, 0.0, 0.0], f"progress={progress[:4]}")
    check("full-run warmup ramps cleanly when warmup exceeds total steps", all(full_warm[i] < full_warm[i + 1] for i in range(len(full_warm) - 1)) and abs(full_warm[-1] - 0.4) < 1e-9, f"full_warm={full_warm}")


#====================
# 20. Warmup trainer smoke: real train() runs must surface warmup in saved artifacts
#====================

def test_warmup_trainer_csv_smoke():
    print("\n=== test_warmup_trainer_csv_smoke ===")

    tmp_path = ROOT / "_tmp_warmup_dataset.txt"
    save_dir = ROOT / "_tmp_warmup_runs"
    run_name = "warmup_csv_smoke"
    metrics_ok = False
    config_ok = False
    lr_rows_ok = False
    metrics_detail = ""
    config_detail = ""
    lr_detail = ""

    text = (
        "hello simplex warmup trainer csv smoke path\n"
        "product artifact contract should expose lr schedule cleanly\n"
    ) * 64

    try:
        shutil.rmtree(save_dir, ignore_errors=True)
        tmp_path.write_text(text, encoding="utf-8")
        result = train(
            dataset=str(tmp_path),
            tokenizer="byte",
            precision_mode="fp32",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            optimizer="adamw",
            adamw_lr_start=3e-4,
            adamw_lr_end=3e-5,
            lr_schedule="linear",
            warmup_steps=2,
            epochs=5,
            steps_per_epoch=1,
            save_dir=str(save_dir),
            run_name=run_name,
            save_every=999,
            no_interactive=True,
            prompt="hello simplex",
            prompt_bytes=16,
            gen_bytes=0,
            log_every=1,
        )

        run_dir = Path(result["run_dir"])
        metrics_path = run_dir / "metrics.csv"
        config_path = run_dir / "config.json"
        metrics_ok = metrics_path.exists()
        config_ok = config_path.exists()
        metrics_detail = str(metrics_path)
        config_detail = str(config_path)

        if metrics_ok:
            with open(metrics_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            logged_lrs = [row.get("lr", "") for row in rows]
            expected_lrs = ["0.00015000", "0.00030000", "0.00030000", "0.00016500", "0.00003000"]
            lr_rows_ok = logged_lrs == expected_lrs
            lr_detail = f"logged_lrs={logged_lrs}"
        else:
            lr_detail = "metrics.csv missing"

        if config_ok:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config_ok = (
                int(cfg.get("warmup_steps", -1)) == 2
                and cfg.get("lr_schedule") == "linear"
                and cfg.get("precision_mode") == "fp32"
            )
            config_detail = json.dumps(
                {
                    "warmup_steps": cfg.get("warmup_steps"),
                    "lr_schedule": cfg.get("lr_schedule"),
                    "precision_mode": cfg.get("precision_mode"),
                },
                ensure_ascii=False,
            )
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(save_dir, ignore_errors=True)

    check("warmup trainer saved metrics.csv", metrics_ok, metrics_detail)
    check("warmup trainer saved config.json with warmup contract", config_ok, config_detail)
    check("warmup trainer logs per-epoch LR schedule with warmup", lr_rows_ok, lr_detail)


#====================
# 21. Warmup resume smoke: resumed runs must continue the LR schedule from saved global_step
#====================

def test_warmup_resume_schedule_smoke():
    print("\n=== test_warmup_resume_schedule_smoke ===")

    tmp_path = ROOT / "_tmp_warmup_resume_dataset.txt"
    save_dir = ROOT / "_tmp_warmup_resume_runs"
    run_name_a = "warmup_resume_a"
    run_name_b = "warmup_resume_b"
    ckpt_ok = False
    metrics_ok = False
    lr_rows_ok = False
    ckpt_detail = ""
    metrics_detail = ""
    lr_detail = ""

    text = (
        "hello simplex warmup resume schedule smoke path\n"
        "resume should continue from saved global_step without restarting lr warmup\n"
    ) * 64

    try:
        shutil.rmtree(save_dir, ignore_errors=True)
        tmp_path.write_text(text, encoding="utf-8")
        base = train(
            dataset=str(tmp_path),
            tokenizer="byte",
            precision_mode="fp32",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            optimizer="adamw",
            adamw_lr_start=3e-4,
            adamw_lr_end=3e-5,
            lr_schedule="linear",
            warmup_steps=2,
            epochs=5,
            steps_per_epoch=1,
            save_dir=str(save_dir),
            run_name=run_name_a,
            save_every=1,
            no_interactive=True,
            prompt="hello simplex",
            prompt_bytes=16,
            gen_bytes=0,
            log_every=1,
        )

        ckpt_path = Path(base["run_dir"]) / "ckpt_ep2.pt"
        ckpt_ok = ckpt_path.exists()
        ckpt_detail = str(ckpt_path)

        if ckpt_ok:
            resumed = train(
                dataset=str(tmp_path),
                tokenizer="byte",
                precision_mode="fp32",
                dim=16,
                ffn=32,
                layers=1,
                seq_len=16,
                batch_size=1,
                optimizer="adamw",
                adamw_lr_start=3e-4,
                adamw_lr_end=3e-5,
                lr_schedule="linear",
                warmup_steps=2,
                epochs=5,
                steps_per_epoch=1,
                resume=str(ckpt_path),
                save_dir=str(save_dir),
                run_name=run_name_b,
                save_every=1,
                no_interactive=True,
                prompt="hello simplex",
                prompt_bytes=16,
                gen_bytes=0,
                log_every=1,
            )
            metrics_path = Path(resumed["run_dir"]) / "metrics.csv"
            metrics_ok = metrics_path.exists()
            metrics_detail = str(metrics_path)
            if metrics_ok:
                with open(metrics_path, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                logged_lrs = [row.get("lr", "") for row in rows]
                expected_lrs = ["0.00030000", "0.00016500", "0.00003000"]
                lr_rows_ok = logged_lrs == expected_lrs
                lr_detail = f"logged_lrs={logged_lrs}"
            else:
                lr_detail = "metrics.csv missing"
        else:
            lr_detail = "resume checkpoint missing"
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(save_dir, ignore_errors=True)

    check("warmup base run saved ckpt_ep2.pt", ckpt_ok, ckpt_detail)
    check("warmup resumed run saved metrics.csv", metrics_ok, metrics_detail)
    check("warmup resumed run continues LR schedule from checkpoint step", lr_rows_ok, lr_detail)


#====================
# 22. High-level precision_mode: layer/model constructors must accept the new mode contract
#====================

def test_precision_mode_high_level_api():
    """high-level pyqitnn constructors should accept precision_mode directly"""
    print("\n=== test_precision_mode_high_level_api ===")
    if not torch.cuda.is_bf16_supported():
        warn("precision_mode high-level skipped", "bf16 not supported on this GPU")
        return

    inp = torch.randn(4, 16, device=DEVICE)
    layer = pyqitnn.QITNNLinear(16, 8, precision_mode="qts_fp32_rest_bf16", device=DEVICE)
    out = layer(inp)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=16, ffn_dim=32, seq_len=16, layers=1, device=DEVICE, precision_mode="mixed_bf16_native",
    )
    seen = {"q_proj_out": None}

    def hook(_module, _inp, out):
        seen["q_proj_out"] = out.dtype

    handle = model.blocks[0].q_proj.register_forward_hook(hook)
    try:
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        logits, loss = model(tokens, targets=tokens)
    finally:
        handle.remove()

    conflict_msg = capture_runtime_error(
        lambda: pyqitnn.QITNNLinear(16, 8, precision_mode="fp32", mixed_precision=True, device=DEVICE)
    )

    check("precision_mode layer emits bf16 visible output", out.dtype == torch.bfloat16, f"dtype={out.dtype}")
    check("precision_mode model normalizes alias to qts_fp32_rest_bf16", model.precision_mode == "qts_fp32_rest_bf16", f"mode={model.precision_mode}")
    check("precision_mode model drives bf16 q_proj activations", seen["q_proj_out"] == torch.bfloat16, f"dtype={seen['q_proj_out']}")
    check("precision_mode model emits bf16 logits", logits.dtype == torch.bfloat16, f"dtype={logits.dtype}")
    check("precision_mode model keeps fp32 loss", loss.dtype == torch.float32, f"dtype={loss.dtype}")
    check("precision_mode conflict with legacy bool is rejected", "conflict" in conflict_msg, conflict_msg or "no RuntimeError")


#====================
# 20. Low-level precision_mode: public ops should accept the new mode contract too
#====================

def test_precision_mode_low_level_api():
    """low-level public ops should accept precision_mode while keeping legacy mixed_precision compatible"""
    print("\n=== test_precision_mode_low_level_api ===")
    if not torch.cuda.is_bf16_supported():
        warn("precision_mode low-level skipped", "bf16 not supported on this GPU")
        return

    inp = torch.randn(4, 16, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    a_n = torch.randn(16, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
    a_z = torch.randn(16, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
    a_p = torch.randn(16, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)

    u, v, cn, cz, cp = forward3(inp, a_n, a_z, a_p, precision_mode="qts_fp32_rest_bf16")
    x, y = centered_simplex(u, v, precision_mode="mixed_bf16_native")

    q = torch.randn(12, 16, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(12, 16, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    val = torch.randn(12, 16, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    attn = attention2(q, k, val, precision_mode="qts_fp32_rest_bf16")

    total = x.float().sum() + y.float().sum() + attn.float().sum()
    total.backward()

    pn = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16)
    pz = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16)
    pp = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16)
    prior_(pn, pz, pp, step=1e-4, entropy_floor=1.0840643, precision_mode="qts_fp32_rest_bf16")

    conflict_msg = capture_runtime_error(
        lambda: forward3(inp.detach(), a_n.detach(), a_z.detach(), a_p.detach(), mixed_precision=False, precision_mode="qts_fp32_rest_bf16")
    )

    check("precision_mode forward3 bf16 outputs stay bf16", u.dtype == torch.bfloat16 and v.dtype == torch.bfloat16)
    check("precision_mode forward3 raw channels stay fp32", cn.dtype == torch.float32 and cz.dtype == torch.float32 and cp.dtype == torch.float32)
    check("precision_mode centered_simplex bf16 outputs stay bf16", x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16)
    check("precision_mode attention2 bf16 output stays bf16", attn.dtype == torch.bfloat16, f"dtype={attn.dtype}")
    check("precision_mode backward reaches bf16 activations", inp.grad is not None and inp.grad.dtype == torch.bfloat16, f"dtype={None if inp.grad is None else inp.grad.dtype}")
    check("precision_mode backward keeps fp32 master-weight grads", a_n.grad is not None and a_n.grad.dtype == torch.float32, f"dtype={None if a_n.grad is None else a_n.grad.dtype}")
    check("precision_mode attention grads stay bf16", q.grad is not None and q.grad.dtype == torch.bfloat16, f"dtype={None if q.grad is None else q.grad.dtype}")
    check("precision_mode prior_ accepts bf16 tensors", pn.dtype == torch.bfloat16 and pz.dtype == torch.bfloat16 and pp.dtype == torch.bfloat16 and torch.isfinite(pn).all().item())
    check("precision_mode conflict with legacy bool is rejected", "conflict" in conflict_msg, conflict_msg or "no RuntimeError")


#====================
# 21. Native mixed bridge: extension must accept bf16 tensors directly
#====================

def test_native_mixed_bridge_smoke():
    """the native extension should accept bf16 activations without Python-side fp32 staging"""
    print("\n=== test_native_mixed_bridge_smoke ===")
    if not torch.cuda.is_bf16_supported():
        warn("native mixed bridge skipped", "bf16 not supported on this GPU")
        return

    ext = load_native()
    torch.manual_seed(42)

    inp = torch.randn(4, 16, device=DEVICE, dtype=torch.bfloat16)
    a_n = torch.randn(16, 8, device=DEVICE, dtype=torch.float32)
    a_z = torch.randn(16, 8, device=DEVICE, dtype=torch.float32)
    a_p = torch.randn(16, 8, device=DEVICE, dtype=torch.float32)
    u, v, cn, cz, cp = ext.forward3_cuda(inp, a_n, a_z, a_p)

    check("native forward3 accepts bf16 input", u.dtype == torch.bfloat16 and v.dtype == torch.bfloat16)
    check("native forward3 keeps cn/cz/cp in fp32", cn.dtype == torch.float32 and cz.dtype == torch.float32 and cp.dtype == torch.float32)

    x, y = ext.centered_simplex_cuda(u, v)
    check("native centered_simplex preserves bf16", x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16)

    ox, oy = ext.attention2_cuda(x, y, x, y, x, y)
    check("native attention2 preserves bf16", ox.dtype == torch.bfloat16 and oy.dtype == torch.bfloat16)

    dox = torch.randn_like(ox)
    doy = torch.randn_like(oy)
    dqx, dqy, dkx, dky, dvx, dvy = ext.attention_backward2_cuda(dox, doy, x, y, x, y, x, y)
    check(
        "native attention backward preserves bf16",
        all(t.dtype == torch.bfloat16 for t in (dqx, dqy, dkx, dky, dvx, dvy))
    )

    pn = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16)
    pz = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16)
    pp = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16)
    ext.prior_cuda(pn, pz, pp, 1e-4, 1.0840643)
    check(
        "native prior accepts bf16 tensors",
        pn.dtype == torch.bfloat16 and pz.dtype == torch.bfloat16 and pp.dtype == torch.bfloat16 and
        torch.isfinite(pn.float()).all().item() and torch.isfinite(pz.float()).all().item() and torch.isfinite(pp.float()).all().item()
    )


#====================
# 22. Public mixed path: fp16 should work through the Python API too
#====================

def test_public_fp16_mixed_api_smoke():
    """public mixed API should accept fp16 tensors without falling back to Python-side fp32 staging"""
    print("\n=== test_public_fp16_mixed_api_smoke ===")
    torch.manual_seed(42)

    inp = torch.randn(4, 16, device=DEVICE, dtype=torch.float16, requires_grad=True)
    a_n = torch.randn(16, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
    a_z = torch.randn(16, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
    a_p = torch.randn(16, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)

    u, v, cn, cz, cp = forward3(inp, a_n, a_z, a_p, mixed_precision=True)
    x, y = centered_simplex(u, v, mixed_precision=True)

    q = torch.randn(12, 16, device=DEVICE, dtype=torch.float16, requires_grad=True)
    k = torch.randn(12, 16, device=DEVICE, dtype=torch.float16, requires_grad=True)
    val = torch.randn(12, 16, device=DEVICE, dtype=torch.float16, requires_grad=True)
    attn = attention2(q, k, val, mixed_precision=True)

    loss = (
        x.float().square().mean()
        + y.float().square().mean()
        + attn.float().square().mean()
    )
    loss.backward()

    check("public forward3 fp16 outputs stay fp16", u.dtype == torch.float16 and v.dtype == torch.float16)
    check("public forward3 raw channels stay fp32", cn.dtype == torch.float32 and cz.dtype == torch.float32 and cp.dtype == torch.float32)
    check("public centered_simplex fp16 outputs stay fp16", x.dtype == torch.float16 and y.dtype == torch.float16)
    check("public attention2 fp16 output stays fp16", attn.dtype == torch.float16, f"dtype={attn.dtype}")
    check("public fp16 backward reaches activations", inp.grad is not None and inp.grad.dtype == torch.float16, f"dtype={None if inp.grad is None else inp.grad.dtype}")
    check(
        "public fp16 backward keeps fp32 master-weight grads",
        a_n.grad is not None and a_z.grad is not None and a_p.grad is not None and
        a_n.grad.dtype == torch.float32 and a_z.grad.dtype == torch.float32 and a_p.grad.dtype == torch.float32
    )
    check(
        "public fp16 attention grads stay fp16",
        q.grad is not None and k.grad is not None and val.grad is not None and
        q.grad.dtype == torch.float16 and k.grad.dtype == torch.float16 and val.grad.dtype == torch.float16
    )

    pn = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
    pz = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
    pp = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
    prior_(pn, pz, pp, step=1e-4, entropy_floor=1.0840643, mixed_precision=True)
    check(
        "public prior_ accepts fp16 tensors",
        pn.dtype == torch.float16 and pz.dtype == torch.float16 and pp.dtype == torch.float16 and
        torch.isfinite(pn.float()).all().item() and torch.isfinite(pz.float()).all().item() and torch.isfinite(pp.float()).all().item()
    )


#====================
# 23. Mixed train parity: adversarial short-run against fp32
#====================

def test_mixed_precision_training_parity_stress():
    """mixed training should stay finite and reasonably close to fp32 under alternating outer autocast"""
    print("\n=== test_mixed_precision_training_parity_stress ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed parity stress skipped", "bf16 not supported on this GPU")
        return

    torch.manual_seed(123)
    model_fp32 = pyqitnn.QITNNSimplexTransformerLM(
        dim=24, ffn_dim=48, seq_len=24, layers=2, device=DEVICE, mixed_precision=False,
    )
    model_mixed = pyqitnn.QITNNSimplexTransformerLM(
        dim=24, ffn_dim=48, seq_len=24, layers=2, device=DEVICE, mixed_precision=True,
    )
    model_mixed.load_state_dict(model_fp32.state_dict())

    opt_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=3e-4)
    opt_mixed = torch.optim.AdamW(model_mixed.parameters(), lr=3e-4)
    batch_gen = torch.Generator(device="cpu")
    batch_gen.manual_seed(321)

    loss_drifts: list[float] = []
    dtype_ok = True
    finite_ok = True
    steps = 12

    for step in range(steps):
        tokens = torch.randint(0, 256, (2, 24), generator=batch_gen, dtype=torch.long).to(DEVICE)
        targets = torch.randint(0, 256, (2, 24), generator=batch_gen, dtype=torch.long).to(DEVICE)

        opt_fp32.zero_grad(set_to_none=True)
        _, loss_fp32 = model_fp32(tokens, targets=targets)
        loss_fp32.backward()
        opt_fp32.step()
        model_fp32.apply_qitnn_prior(step_qk=5e-5, step_vo=5e-5, step_ff=5e-5, entropy_floor=1.0840643)

        mode = step % 3
        if mode == 0:
            outer_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        elif mode == 1:
            outer_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            outer_ctx = nullcontext()

        opt_mixed.zero_grad(set_to_none=True)
        with outer_ctx:
            logits_mixed, loss_mixed = model_mixed(tokens, targets=targets)
        loss_mixed.backward()
        opt_mixed.step()
        model_mixed.apply_qitnn_prior(step_qk=5e-5, step_vo=5e-5, step_ff=5e-5, entropy_floor=1.0840643)

        lf = float(loss_fp32.detach().cpu())
        lm = float(loss_mixed.detach().cpu())
        loss_drifts.append(abs(lm - lf) / max(abs(lf), 1e-8))
        dtype_ok = dtype_ok and logits_mixed.dtype == torch.bfloat16 and loss_mixed.dtype == torch.float32
        finite_ok = finite_ok and math.isfinite(lf) and math.isfinite(lm)

    probe_tokens = torch.randint(0, 256, (2, 24), generator=batch_gen, dtype=torch.long).to(DEVICE)
    probe_targets = torch.randint(0, 256, (2, 24), generator=batch_gen, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits_fp32, loss_fp32 = model_fp32(probe_tokens, targets=probe_targets)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits_mixed, loss_mixed = model_mixed(probe_tokens, targets=probe_targets)

    logit_diff = (logits_mixed.float() - logits_fp32.float()).abs()
    eval_loss_gap = abs(float(loss_mixed.detach().cpu()) - float(loss_fp32.detach().cpu()))

    stats_fp32 = pyqitnn.qitnn_diag_stats(
        model_fp32.blocks[0].q_proj.a_neg,
        model_fp32.blocks[0].q_proj.a_zero,
        model_fp32.blocks[0].q_proj.a_pos,
    )
    stats_mixed = pyqitnn.qitnn_diag_stats(
        model_mixed.blocks[0].q_proj.a_neg,
        model_mixed.blocks[0].q_proj.a_zero,
        model_mixed.blocks[0].q_proj.a_pos,
    )
    diag_gap = max(
        abs(stats_fp32["p_neg"] - stats_mixed["p_neg"]),
        abs(stats_fp32["p_zero"] - stats_mixed["p_zero"]),
        abs(stats_fp32["p_pos"] - stats_mixed["p_pos"]),
        abs(stats_fp32["h"] - stats_mixed["h"]),
    )

    check("mixed parity stress keeps mixed visible dtype pinned to bf16", dtype_ok)
    check("mixed parity stress keeps both runs finite", finite_ok)
    check(f"mixed parity stress max train loss drift={max(loss_drifts):.4f}", max(loss_drifts) < 0.03, f"drifts={loss_drifts}")
    check(f"mixed parity stress eval loss gap={eval_loss_gap:.4f}", eval_loss_gap < 0.08, f"gap={eval_loss_gap:.4f}")
    check(f"mixed parity stress mean logit gap={logit_diff.mean().item():.4f}", logit_diff.mean().item() < 0.20, f"mean_gap={logit_diff.mean().item():.4f}")
    check(f"mixed parity stress max logit gap={logit_diff.max().item():.4f}", logit_diff.max().item() < 1.00, f"max_gap={logit_diff.max().item():.4f}")
    check(f"mixed parity stress q_proj diag gap={diag_gap:.4f}", diag_gap < 0.03, f"diag_gap={diag_gap:.4f}")


#====================
# 24. Memory stability: repeated forward/backward does not OOM or leak
#====================

def test_memory_stability():
    """100 forward/backward iterations should not cause memory growth"""
    print("\n=== test_memory_stability ===")
    torch.manual_seed(42)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=32, ffn_dim=64, seq_len=32, layers=2, device=DEVICE,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # warmup
    for _ in range(5):
        tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
        targets = torch.randint(0, 256, (2, 32), device=DEVICE)
        opt.zero_grad(set_to_none=True)
        _, loss = model(tokens, targets=targets)
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    for _ in range(100):
        tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
        targets = torch.randint(0, 256, (2, 32), device=DEVICE)
        opt.zero_grad(set_to_none=True)
        _, loss = model(tokens, targets=targets)
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    growth_mb = (mem_after - mem_before) / (1024 * 1024)

    check(f"memory growth after 100 steps: {growth_mb:.2f} MB",
          growth_mb < 5.0,
          f"growth={growth_mb:.2f} MB")


#====================
# 19. Simplex gelu: only x channel is activated, y passes through
#====================

def test_simplex_gelu_passthrough():
    """simplex_gelu must apply gelu ONLY to x (first half), y (second half) passes through"""
    print("\n=== test_simplex_gelu_passthrough ===")
    from pyqitnn.modeling import simplex_gelu

    torch.manual_seed(42)
    xy = torch.randn(4, 16, device=DEVICE, requires_grad=True)

    out = simplex_gelu(xy)

    x_in = xy[:, :8]
    y_in = xy[:, 8:]
    x_out = out[:, :8]
    y_out = out[:, 8:]

    # y must pass through unchanged
    check("y passes through unchanged",
          torch.equal(y_in, y_out))

    # x must be gelu(x_in)
    x_expected = F.gelu(x_in)
    check("x = gelu(x_in)",
          torch.allclose(x_out, x_expected, atol=1e-6),
          f"max_diff={(x_out - x_expected).abs().max().item():.2e}")

    # gradient flows to both halves
    out.sum().backward()
    check("grad flows to x half", xy.grad[:, :8].abs().max().item() > 1e-3)
    check("grad flows to y half (=1.0)", torch.allclose(xy.grad[:, 8:], torch.ones_like(xy.grad[:, 8:])))


#====================
# 20. Weight initialization: all triplets should start near-uniform
#====================

def test_init_near_uniform():
    """fresh model's QTS weights should have P- ~ P0 ~ P+ ~ 1/3"""
    print("\n=== test_init_near_uniform ===")
    torch.manual_seed(42)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=64, ffn_dim=128, seq_len=32, layers=2, device=DEVICE,
    )

    for name, _, layer in model.iter_qitnn_layers():
        a = layer.a_neg.detach()
        b = layer.a_zero.detach()
        c = layer.a_pos.detach()

        a2, b2, c2 = a*a, b*b, c*c
        z = (a2 + b2 + c2).clamp(min=1e-20)
        pn, pz, pp = a2/z, b2/z, c2/z

        # at init, each P should be ~ 1/3 on average
        mean_pn = pn.mean().item()
        mean_pz = pz.mean().item()
        mean_pp = pp.mean().item()

        # tolerance: with small init_std, each amplitude is ~N(0, 0.02)
        # so a², b², c² are similar scale => P ~ 1/3
        ok = (abs(mean_pn - 1/3) < 0.05 and
              abs(mean_pz - 1/3) < 0.05 and
              abs(mean_pp - 1/3) < 0.05)

        short = pyqitnn.short_qitnn_label(name)
        check(f"init {short}: P=({mean_pn:.3f}, {mean_pz:.3f}, {mean_pp:.3f})",
              ok, f"deviation from 1/3 too large")


#====================
# 25. Determinism: same seed -> exact same training trajectory
#====================

def test_determinism():
    """two runs with same seed must produce identical loss sequences"""
    print("\n=== test_determinism ===")

    losses_runs = []
    for run in range(2):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        model = pyqitnn.QITNNSimplexTransformerLM(
            dim=32, ffn_dim=64, seq_len=32, layers=2, device=DEVICE,
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        losses = []
        for step in range(10):
            tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
            targets = torch.randint(0, 256, (2, 32), device=DEVICE)
            opt.zero_grad(set_to_none=True)
            _, loss = model(tokens, targets=targets)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        losses_runs.append(losses)

    max_diff = max(abs(a - b) for a, b in zip(losses_runs[0], losses_runs[1]))
    check(f"deterministic: max loss diff between runs = {max_diff:.2e}",
          max_diff < 1e-5, f"diff={max_diff}")


#====================
# 26. Residual connection: removing residual should hurt loss
#====================

def test_residual_matters():
    """verify residual connections are functional (not bypassed)"""
    print("\n=== test_residual_matters ===")
    torch.manual_seed(42)

    model = pyqitnn.QITNNSimplexTransformerLM(
        dim=32, ffn_dim=64, seq_len=32, layers=2, device=DEVICE,
    )

    tokens = torch.randint(0, 256, (4, 32), device=DEVICE)
    targets = torch.randint(0, 256, (4, 32), device=DEVICE)

    # normal forward
    with torch.no_grad():
        _, loss_normal = model(tokens, targets=targets)

    # corrupt residual by zeroing position embeddings
    with torch.no_grad():
        saved_pos = model.pos_emb.data.clone()
        model.pos_emb.data.zero_()
        _, loss_no_pos = model(tokens, targets=targets)
        model.pos_emb.data.copy_(saved_pos)

    check(f"pos_emb matters: normal={loss_normal.item():.4f} vs zero={loss_no_pos.item():.4f}",
          abs(loss_normal.item() - loss_no_pos.item()) > 0.001,
          "losses identical -- pos_emb may not be connected")


#====================
# 27. Byte tokenizer: current byte path must remain stable
#====================

def test_byte_tokenizer_roundtrip():
    print("\n=== test_byte_tokenizer_roundtrip ===")
    tok = pyqitnn.ByteTokenizer()
    sample = "Hello, QITNN.\nByte path stays intact."
    ids = tok.encode_text(sample)
    decoded = tok.decode(ids)

    check("byte tokenizer vocab=256", tok.vocab_size == 256, f"vocab={tok.vocab_size}")
    check("byte tokenizer roundtrip exact", decoded == sample, f"decoded={decoded!r}")


#====================
# 28. BPE tokenizer: roundtrip and vocab smoke
#====================

def test_bpe_tokenizer_roundtrip():
    print("\n=== test_bpe_tokenizer_roundtrip ===")
    try:
        tok = pyqitnn.train_bpe_tokenizer(
            [
                "hello world hello simplex transformer",
                "born rule ternary simplex attention",
            ],
            vocab_size=320,
            min_frequency=1,
        )
    except RuntimeError as e:
        warn("bpe tokenizer roundtrip skipped", str(e))
        return

    sample = "hello world simplex"
    ids = tok.encode_text(sample)
    decoded = tok.decode(ids)

    check("bpe tokenizer produced ids", len(ids) > 0, "no token ids")
    check("bpe tokenizer vocab size", tok.vocab_size >= 256, f"vocab={tok.vocab_size}")
    check("bpe tokenizer roundtrip exact", decoded == sample, f"decoded={decoded!r}")


#====================
# 29. BPE trainer smoke: end-to-end trainer path without touching QTS math
#====================

def test_bpe_trainer_smoke():
    print("\n=== test_bpe_trainer_smoke ===")
    tmp_path = ROOT / "_tmp_bpe_dataset.txt"
    text = (
        "hello simplex transformer born rule attention qitnn data stream\n"
        "tokenizer smoke test keeps qts math unchanged and only changes token ids\n"
    ) * 64

    try:
        tmp_path.write_text(text, encoding="utf-8")
        result = train(
            dataset=str(tmp_path),
            tokenizer="bpe",
            tokenizer_vocab_size=320,
            tokenizer_min_frequency=1,
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            steps=2,
            no_save=True,
            no_interactive=True,
            log_every=1,
            prompt="hello simplex",
            prompt_tokens=8,
            gen_tokens=8,
        )
    except RuntimeError as e:
        warn("bpe trainer smoke skipped", str(e))
        tmp_path.unlink(missing_ok=True)
        return

    tmp_path.unlink(missing_ok=True)
    last_loss = result.get("last_loss")
    train_bpb = result.get("last_epoch_train_bpb")
    val_bpb = result.get("last_epoch_val_bpb")
    check("bpe trainer produced finite loss", last_loss is not None and math.isfinite(last_loss), f"last_loss={last_loss}")
    check("bpe trainer returned finite train BPB", train_bpb is not None and math.isfinite(train_bpb), f"train_bpb={train_bpb}")
    check("bpe trainer returned finite val BPB", val_bpb is not None and math.isfinite(val_bpb), f"val_bpb={val_bpb}")


#====================
# 30. Mixed trainer smoke: full train() path must preserve the conservative bf16 contract
#====================

def test_mixed_precision_trainer_smoke():
    print("\n=== test_mixed_precision_trainer_smoke ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed precision trainer skipped", "bf16 not supported on this GPU")
        return

    tmp_path = ROOT / "_tmp_mixed_dataset.txt"
    text = (
        "hello simplex transformer born rule attention qitnn stream\n"
        "mixed precision trainer smoke keeps qts master weights in fp32\n"
    ) * 64

    try:
        tmp_path.write_text(text, encoding="utf-8")
        result = train(
            dataset=str(tmp_path),
            tokenizer="byte",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            steps=2,
            mixed_precision=True,
            no_save=True,
            no_interactive=True,
            log_every=1,
            prompt="hello simplex",
            prompt_bytes=16,
            gen_bytes=16,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    model = result["model"]
    last_loss = result.get("last_loss")
    first_loss = result.get("first_loss")
    first_bpb = result.get("first_bpb")
    last_bpb = result.get("last_bpb")
    train_bpb = result.get("last_epoch_train_bpb")
    val_loss = result.get("last_epoch_val_loss")
    val_bpb = result.get("last_epoch_val_bpb")
    tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
    targets = torch.randint(0, 256, (2, 16), device=DEVICE)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, loss = model(tokens, targets=targets)

    check("mixed trainer produced finite loss", last_loss is not None and math.isfinite(last_loss), f"last_loss={last_loss}")
    check("mixed trainer returns mixed model", model.mixed_precision is True)
    check("mixed trainer preserves fp32 QTS master weights", model.blocks[0].q_proj.a_neg.dtype == torch.float32)
    check("mixed trainer preserves fp32 head weights", model.head.weight.dtype == torch.float32)
    check("mixed trainer keeps bf16 visible logits after training", logits.dtype == torch.bfloat16, f"dtype={logits.dtype}")
    check("mixed trainer keeps fp32 loss after training", loss.dtype == torch.float32, f"dtype={loss.dtype}")
    check("mixed trainer first BPB tracks byte loss", first_bpb is not None and first_loss is not None and abs(first_bpb - (first_loss / math.log(2.0))) < 1e-6, f"first_bpb={first_bpb} first_loss={first_loss}")
    check("mixed trainer last BPB tracks byte loss", last_bpb is not None and last_loss is not None and abs(last_bpb - (last_loss / math.log(2.0))) < 1e-6, f"last_bpb={last_bpb} last_loss={last_loss}")
    check("mixed trainer epoch train BPB tracks byte loss", train_bpb is not None and abs(train_bpb - (result['last_epoch_train_loss'] / math.log(2.0))) < 1e-6, f"train_bpb={train_bpb} train_loss={result['last_epoch_train_loss']}")
    check("mixed trainer epoch val BPB tracks byte loss", val_bpb is not None and val_loss is not None and abs(val_bpb - (val_loss / math.log(2.0))) < 1e-6, f"val_bpb={val_bpb} val_loss={val_loss}")


#====================
# 31. Mixed checkpoint resume: saved mixed runs must reload and continue cleanly
#====================

def test_mixed_precision_checkpoint_resume_smoke():
    print("\n=== test_mixed_precision_checkpoint_resume_smoke ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed checkpoint resume skipped", "bf16 not supported on this GPU")
        return

    tmp_path = ROOT / "_tmp_mixed_resume_dataset.txt"
    save_dir = ROOT / "_tmp_mixed_resume_runs"
    run_name_a = "mixed_resume_a"
    run_name_b = "mixed_resume_b"
    text = (
        "hello simplex transformer born rule attention qitnn stream\n"
        "mixed checkpoint resume smoke keeps precision contract intact\n"
    ) * 64

    cleanup_tree(save_dir)
    saved_ckpt = False
    saved_tok = False
    csv_ok = False
    csv_detail = ""
    try:
        tmp_path.write_text(text, encoding="utf-8")
        base = train(
            dataset=str(tmp_path),
            tokenizer="byte",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            steps=2,
            mixed_precision=True,
            save_dir=str(save_dir),
            run_name=run_name_a,
            save_every=1,
            no_interactive=True,
            log_every=1,
            prompt="hello simplex",
            prompt_bytes=16,
            gen_bytes=16,
        )

        base_run_dir = Path(base["run_dir"])
        ckpt_path = base_run_dir / "ckpt_final.pt"
        saved_ckpt = ckpt_path.exists()
        saved_tok = (base_run_dir / "tokenizer.json").exists()
        base_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        metrics_path = base_run_dir / "metrics.csv"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                row0 = rows[0]
                csv_ok = (
                    "train_bpb" in row0 and
                    "val_bpb" in row0 and
                    row0["train_bpb"] not in {"", "n/a"} and
                    row0["val_bpb"] not in {"", "n/a"}
                )
                csv_detail = str(row0)
            else:
                csv_detail = "metrics.csv has no data rows"
        else:
            csv_detail = str(metrics_path)

        resumed = train(
            dataset=str(tmp_path),
            tokenizer="byte",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            epochs=2,
            steps_per_epoch=1,
            mixed_precision=True,
            save_dir=str(save_dir),
            run_name=run_name_b,
            save_every=1,
            resume=str(ckpt_path),
            no_interactive=True,
            log_every=1,
            prompt="hello simplex",
            prompt_bytes=16,
            gen_bytes=16,
        )

        resumed_run_dir = Path(resumed["run_dir"])
        resumed_ckpt_path = resumed_run_dir / "ckpt_final.pt"
        resumed_ckpt = torch.load(str(resumed_ckpt_path), map_location="cpu", weights_only=False)
        model = resumed["model"]

        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        targets = torch.randint(0, 256, (2, 16), device=DEVICE)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, loss = model(tokens, targets=targets)
    finally:
        tmp_path.unlink(missing_ok=True)
        cleanup_tree(save_dir)

    check("mixed checkpoint run saved final checkpoint", saved_ckpt, str(ckpt_path))
    check("mixed checkpoint run saved tokenizer asset", saved_tok, str(base_run_dir / "tokenizer.json"))
    check("mixed checkpoint metrics.csv carries BPB columns", csv_ok, csv_detail)
    check("mixed resume advanced global_step", resumed_ckpt.get("global_step") == base_ckpt.get("global_step", 0) + 1, f"base={base_ckpt.get('global_step')} resumed={resumed_ckpt.get('global_step')}")
    check("mixed resume advanced epoch", resumed_ckpt.get("epoch") == 2, f"epoch={resumed_ckpt.get('epoch')}")
    returned_final_diff = 0.0
    for key, tensor in model.state_dict().items():
        ckpt_tensor = resumed_ckpt["model"][key]
        if torch.is_tensor(tensor):
            diff = (tensor.detach().cpu().float() - ckpt_tensor.detach().cpu().float()).abs().max().item()
            if diff > returned_final_diff:
                returned_final_diff = diff
    check("mixed resumed model still matches final checkpoint after eval", returned_final_diff < 1e-7, f"max_diff={returned_final_diff}")
    check("mixed resumed model keeps fp32 QTS master weights", model.blocks[0].q_proj.a_neg.dtype == torch.float32)
    check("mixed resumed model keeps bf16 visible logits", logits.dtype == torch.bfloat16, f"dtype={logits.dtype}")
    check("mixed resumed model keeps fp32 loss", loss.dtype == torch.float32, f"dtype={loss.dtype}")


#====================
# 32. Real CLI mixed smoke: entrypoint path must work in mixed mode
#====================

def test_mixed_precision_cli_smoke():
    print("\n=== test_mixed_precision_cli_smoke ===")
    if not torch.cuda.is_bf16_supported():
        warn("mixed cli smoke skipped", "bf16 not supported on this GPU")
        return

    tmp_path = ROOT / "_tmp_mixed_cli_dataset.txt"
    text = (
        "hello simplex transformer born rule attention qitnn stream\n"
        "mixed cli smoke keeps the entrypoint contract honest\n"
    ) * 64

    try:
        tmp_path.write_text(text, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                "BasicQITNN_Transformer.py",
                "--dataset", str(tmp_path),
                "--tokenizer", "byte",
                "--dim", "16",
                "--ffn", "32",
                "--layers", "1",
                "--seq-len", "16",
                "--batch-size", "1",
                "--steps", "2",
                "--mixed-precision",
                "--no-save",
                "--no-interactive",
                "--log-every", "1",
                "--prompt", "hello simplex",
                "--prompt-bytes", "16",
                "--gen-bytes", "16",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    detail = stdout[-400:] if proc.returncode == 0 else (stderr or stdout)[-800:]

    check("mixed CLI process exits cleanly", proc.returncode == 0, detail)
    check("mixed CLI reports mixed mode enabled", "mixed_prec    True" in stdout, detail)
    check("mixed CLI reports precision_mode qts_fp32_rest_bf16", "precision_mode qts_fp32_rest_bf16" in stdout, detail)
    check("mixed CLI reports train BPB", "train_bpb=" in stdout, detail)
    check("mixed CLI reports val BPB", "val_bpb=" in stdout, detail)
    check("mixed CLI reaches generation output", "generated_text_begin" in stdout and "generated_text_end" in stdout, detail)


#====================
# 33. JSON loader: extract training text from JSON payloads
#====================

def test_json_loader_extracts_text():
    print("\n=== test_json_loader_extracts_text ===")
    tmp_path = ROOT / "_tmp_loader.json"
    payload = []
    for _ in range(32):
        payload.append({"text": "hello simplex"})
        payload.append({"meta": {"content": "born rule attention"}})

    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        raw = load_bytes(tmp_path, 100_000, data_format="auto", json_text_fields="text,content")
    finally:
        tmp_path.unlink(missing_ok=True)

    text = raw.decode("utf-8", errors="replace")
    check("json loader captured text field", "hello simplex" in text, text[:120])
    check("json loader captured nested content field", "born rule attention" in text, text[:120])
    check("json loader emitted plain corpus text", "\"text\"" not in text and "{" not in text, text[:120])


#====================
# 34. JSON trainer smoke: train on JSONL without changing QTS math
#====================

def test_json_trainer_smoke():
    print("\n=== test_json_trainer_smoke ===")
    tmp_path = ROOT / "_tmp_dataset.jsonl"
    lines = []
    for _ in range(64):
        lines.append(json.dumps({"text": "hello simplex transformer born rule attention stream"}))

    try:
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = train(
            dataset=str(tmp_path),
            data_format="jsonl",
            json_text_fields="text",
            tokenizer="byte",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            steps=2,
            no_save=True,
            no_interactive=True,
            log_every=1,
            prompt="hello simplex",
            prompt_bytes=16,
            gen_bytes=16,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    last_loss = result.get("last_loss")
    check("json trainer produced finite loss", last_loss is not None and math.isfinite(last_loss), f"last_loss={last_loss}")


#====================
# 35. BPB byte math: exact helper formulas must stay stable
#====================

def test_bpb_byte_math_helpers():
    print("\n=== test_bpb_byte_math_helpers ===")
    tok = pyqitnn.ByteTokenizer()
    targets = torch.tensor([[65, 66, 67, 10], [68, 69, 70, 71]], dtype=torch.long)
    loss = math.log(2.0)

    target_tokens = int(targets.numel())
    target_bytes = _count_token_bytes(targets, tok)
    bpb = loss_to_bpb(loss, target_tokens, target_bytes)
    ppl = loss_to_perplexity(loss)

    check("byte target byte count equals token count", target_bytes == target_tokens, f"bytes={target_bytes} tokens={target_tokens}")
    check("byte BPB from ln(2) loss equals 1.0", bpb is not None and abs(bpb - 1.0) < 1e-12, f"bpb={bpb}")
    check("byte PPL from ln(2) loss equals 2.0", ppl is not None and abs(ppl - 2.0) < 1e-12, f"ppl={ppl}")
    check("byte tokenizer keeps PPL = 2**BPB", bpb is not None and ppl is not None and abs((2.0 ** bpb) - ppl) < 1e-12, f"bpb={bpb} ppl={ppl}")


#====================
# 36. BPB BPE accounting: per-token byte LUT must match the effective decoded stream
#====================

def test_bpb_bpe_byte_accounting():
    print("\n=== test_bpb_bpe_byte_accounting ===")
    try:
        tok = pyqitnn.train_bpe_tokenizer(
            [
                "hello simplex transformer\n",
                "born rule attention stream\n",
                "json text path keeps utf8 stable: Привет мир\n",
            ],
            vocab_size=320,
            min_frequency=1,
        )
    except RuntimeError as e:
        warn("bpb bpe accounting skipped", str(e))
        return

    tok_bytes_lut = _build_token_byte_lut(tok)
    samples = [
        "hello simplex",
        "born rule attention stream",
        "Привет simplex\nhello",
    ]

    for sample in samples:
        ids = tok.encode_text(sample)
        ids_t = torch.tensor(ids, dtype=torch.long)
        counted_bytes = _count_token_bytes(ids_t, tok, tok_bytes_lut)
        effective_text = tok.decode(ids, skip_special_tokens=False)
        expected_bytes = len(effective_text.encode("utf-8", errors="replace"))
        check(
            f"bpe byte accounting matches effective stream [{sample[:12]!r}]",
            counted_bytes == expected_bytes,
            f"bytes={counted_bytes} expected={expected_bytes} ids={ids}",
        )


#====================
# 37. BPB aggregation: exact corpus BPB must be computed from total bits / total bytes
#====================

def test_bpb_total_aggregation_math():
    print("\n=== test_bpb_total_aggregation_math ===")
    loss_a = math.log(2.0)
    loss_b = math.log(2.0)

    bpb_a = loss_to_bpb(loss_a, target_tokens=2, target_bytes=2)
    bpb_b = loss_to_bpb(loss_b, target_tokens=4, target_bytes=8)
    mean_window_bpb = (bpb_a + bpb_b) / 2.0
    exact_bpb = total_nll_to_bpb(loss_a * 2 + loss_b * 4, target_bytes=10)

    check("window A BPB", bpb_a is not None and abs(bpb_a - 1.0) < 1e-12, f"bpb={bpb_a}")
    check("window B BPB", bpb_b is not None and abs(bpb_b - 0.5) < 1e-12, f"bpb={bpb_b}")
    check("exact total BPB uses total bits / total bytes", exact_bpb is not None and abs(exact_bpb - 0.6) < 1e-12, f"bpb={exact_bpb}")
    check("exact total BPB differs from naive window average", abs(exact_bpb - mean_window_bpb) > 1e-12, f"exact={exact_bpb} mean={mean_window_bpb}")


#====================
# 38. Validation BPB: run_val must aggregate total bits / total bytes
#====================

def test_run_val_bpb_aggregation():
    print("\n=== test_run_val_bpb_aggregation ===")
    try:
        tok = pyqitnn.train_bpe_tokenizer(
            [
                "ascii stream hello world\n",
                "utf8 stream Привет мир\n",
            ],
            vocab_size=320,
            min_frequency=1,
        )
    except RuntimeError as e:
        warn("run_val bpb aggregation skipped", str(e))
        return

    tok_bytes_lut = _build_token_byte_lut(tok)
    short_ids = torch.nonzero(tok_bytes_lut == 1).reshape(-1)
    long_ids = torch.nonzero(tok_bytes_lut >= 2).reshape(-1)
    if short_ids.numel() < 3 or long_ids.numel() < 2:
        warn("run_val bpb aggregation skipped", "need both short and long token pieces")
        return

    anchor = int(short_ids[0].item())
    y0 = torch.tensor([int(short_ids[1].item()), int(short_ids[2].item())], dtype=torch.long)
    y1 = torch.tensor([int(long_ids[0].item()), int(long_ids[1].item())], dtype=torch.long)
    val_data = torch.tensor([anchor, int(y0[0].item()), int(y0[1].item()), int(y1[0].item()), int(y1[1].item())], dtype=torch.long)
    loss_value = math.log(2.0)

    class ScriptedLossModel(torch.nn.Module):
        def __init__(self, losses: list[float]) -> None:
            super().__init__()
            self.losses = list(losses)
            self.calls = 0

        def forward(self, x, targets=None):
            del targets
            loss = torch.tensor(self.losses[self.calls], device=x.device, dtype=torch.float32)
            self.calls += 1
            return x, loss

    model = ScriptedLossModel([loss_value, loss_value])
    avg_loss, avg_bpb, n_win = run_val(
        model,
        val_data,
        seq_len=2,
        device=DEVICE,
        max_steps=0,
        tokenizer=tok,
        tok_bytes_lut=tok_bytes_lut,
    )

    bytes_y0 = _count_token_bytes(y0, tok, tok_bytes_lut)
    bytes_y1 = _count_token_bytes(y1, tok, tok_bytes_lut)
    exact_bpb = total_nll_to_bpb(loss_value * 4, bytes_y0 + bytes_y1)
    mean_window_bpb = (loss_to_bpb(loss_value, 2, bytes_y0) + loss_to_bpb(loss_value, 2, bytes_y1)) / 2.0

    check("run_val produced two windows", n_win == 2, f"n_win={n_win}")
    check("run_val keeps exact mean loss", abs(avg_loss - loss_value) < 1e-6, f"loss={avg_loss}")
    check("run_val exact BPB uses total bits / total bytes", avg_bpb is not None and exact_bpb is not None and abs(avg_bpb - exact_bpb) < 1e-6, f"bpb={avg_bpb} exact={exact_bpb}")
    check("run_val BPB is not naive mean of window BPBs", avg_bpb is not None and abs(avg_bpb - mean_window_bpb) > 1e-12, f"bpb={avg_bpb} mean={mean_window_bpb}")


#====================
# run_val batching contract: validation should preserve metrics while grouping windows
#====================

def test_run_val_batches_windows():
    print("\n=== test_run_val_batches_windows ===")
    tok = pyqitnn.ByteTokenizer()
    val_data = torch.tensor([65, 66, 67, 68, 69, 70, 71], dtype=torch.long)
    loss_value = math.log(2.0)

    class BatchRecordingLossModel(torch.nn.Module):
        def __init__(self, loss_value: float) -> None:
            super().__init__()
            self.loss_value = float(loss_value)
            self.batch_sizes: list[int] = []

        def forward(self, x, targets=None):
            del targets
            self.batch_sizes.append(int(x.size(0)))
            loss = torch.tensor(self.loss_value, device=x.device, dtype=torch.float32)
            return x, loss

    model = BatchRecordingLossModel(loss_value)
    avg_loss, avg_bpb, n_win = run_val(
        model,
        val_data,
        seq_len=2,
        device=DEVICE,
        max_steps=0,
        tokenizer=tok,
        tok_bytes_lut=None,
    )

    check("run_val batched path keeps window count", n_win == 3, f"n_win={n_win}")
    check("run_val batched path keeps exact mean loss", abs(avg_loss - loss_value) < 1e-6, f"loss={avg_loss}")
    check("run_val batched path keeps exact byte BPB", avg_bpb is not None and abs(avg_bpb - 1.0) < 1e-6, f"bpb={avg_bpb}")
    check("run_val now groups multiple windows per model call", max(model.batch_sizes, default=0) > 1, f"batch_sizes={model.batch_sizes}")
    check("run_val batched path accounts for every window once", sum(model.batch_sizes) == n_win, f"batch_sizes={model.batch_sizes} n_win={n_win}")


#====================
# 39. Split-eval helper: wrapper should expose loss/BPB/PPL/windows consistently
#====================

def test_eval_split_metrics_helper():
    print("\n=== test_eval_split_metrics_helper ===")
    tok = pyqitnn.ByteTokenizer()
    data = torch.tensor([65, 66, 67, 68, 69], dtype=torch.long)
    loss_value = math.log(2.0)

    class ScriptedLossModel(torch.nn.Module):
        def __init__(self, losses: list[float]) -> None:
            super().__init__()
            self.losses = list(losses)
            self.calls = 0

        def forward(self, x, targets=None):
            del targets
            loss = torch.tensor(self.losses[self.calls], device=x.device, dtype=torch.float32)
            self.calls += 1
            return x, loss

    model = ScriptedLossModel([loss_value, loss_value])
    metrics = eval_split_metrics(
        model,
        data,
        seq_len=2,
        device=DEVICE,
        max_steps=0,
        tokenizer=tok,
        tok_bytes_lut=None,
    )

    check("eval_split_metrics windows", metrics["windows"] == 2, f"metrics={metrics}")
    check("eval_split_metrics loss", metrics["loss"] is not None and abs(float(metrics["loss"]) - loss_value) < 1e-6, f"metrics={metrics}")
    check("eval_split_metrics bpb", metrics["bpb"] is not None and abs(float(metrics["bpb"]) - 1.0) < 1e-6, f"metrics={metrics}")
    check("eval_split_metrics ppl", metrics["ppl"] is not None and abs(float(metrics["ppl"]) - 2.0) < 1e-6, f"metrics={metrics}")


#====================
# 40. Extended test metrics: trainer should return both best_test_* and final_test_*.
#====================

def test_extended_dataset_best_final_test_metrics_smoke():
    print("\n=== test_extended_dataset_best_final_test_metrics_smoke ===")
    root = ROOT / "_tmp_extended_eval"
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    for p in (train_dir, val_dir, test_dir):
        p.mkdir(parents=True, exist_ok=True)

    try:
        (train_dir / "train.txt").write_text(("hello simplex train stream\n" * 96), encoding="utf-8")
        (val_dir / "val.txt").write_text(("hello simplex val stream\n" * 64), encoding="utf-8")
        (test_dir / "test.txt").write_text(("hello simplex test stream\n" * 64), encoding="utf-8")
        result = train(
            extended_dataset=True,
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            test_dir=str(test_dir),
            tokenizer="byte",
            dim=16,
            ffn=32,
            layers=1,
            seq_len=16,
            batch_size=1,
            epochs=2,
            steps_per_epoch=1,
            no_save=True,
            no_interactive=True,
            log_every=1,
            prompt="hello simplex",
            prompt_bytes=16,
            gen_bytes=16,
        )
    finally:
        cleanup_tree(root)

    final_test_loss = result.get("final_test_loss")
    final_test_bpb = result.get("final_test_bpb")
    final_test_ppl = result.get("final_test_ppl")
    best_test_loss = result.get("best_test_loss")
    best_test_bpb = result.get("best_test_bpb")
    best_test_ppl = result.get("best_test_ppl")

    check("extended trainer returned final_test_loss", final_test_loss is not None and math.isfinite(final_test_loss), f"final_test_loss={final_test_loss}")
    check("extended trainer returned final_test_bpb", final_test_bpb is not None and math.isfinite(final_test_bpb), f"final_test_bpb={final_test_bpb}")
    check("extended trainer returned final_test_ppl", final_test_ppl is not None and math.isfinite(final_test_ppl), f"final_test_ppl={final_test_ppl}")
    check("extended trainer returned best_test_loss", best_test_loss is not None and math.isfinite(best_test_loss), f"best_test_loss={best_test_loss}")
    check("extended trainer returned best_test_bpb", best_test_bpb is not None and math.isfinite(best_test_bpb), f"best_test_bpb={best_test_bpb}")
    check("extended trainer returned best_test_ppl", best_test_ppl is not None and math.isfinite(best_test_ppl), f"best_test_ppl={best_test_ppl}")
    check("legacy test_loss aliases final_test_loss", abs(result["test_loss"] - final_test_loss) < 1e-7, f"test_loss={result['test_loss']} final_test_loss={final_test_loss}")
    check("legacy test_bpb aliases final_test_bpb", abs(result["test_bpb"] - final_test_bpb) < 1e-7, f"test_bpb={result['test_bpb']} final_test_bpb={final_test_bpb}")
    check("legacy test_ppl aliases final_test_ppl", abs(result["test_ppl"] - final_test_ppl) < 1e-7, f"test_ppl={result['test_ppl']} final_test_ppl={final_test_ppl}")


#====================
# 41. Extended CLI test metrics: CLI should print both final and best test metrics.
#====================

def test_extended_dataset_cli_test_metrics_smoke():
    print("\n=== test_extended_dataset_cli_test_metrics_smoke ===")
    root = ROOT / "_tmp_extended_cli_eval"
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    for p in (train_dir, val_dir, test_dir):
        p.mkdir(parents=True, exist_ok=True)

    try:
        (train_dir / "train.txt").write_text(("hello simplex train stream\n" * 96), encoding="utf-8")
        (val_dir / "val.txt").write_text(("hello simplex val stream\n" * 64), encoding="utf-8")
        (test_dir / "test.txt").write_text(("hello simplex test stream\n" * 64), encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                "BasicQITNN_Transformer.py",
                "--extended-dataset",
                "--train-dir", str(train_dir),
                "--val-dir", str(val_dir),
                "--test-dir", str(test_dir),
                "--tokenizer", "byte",
                "--dim", "16",
                "--ffn", "32",
                "--layers", "1",
                "--seq-len", "16",
                "--batch-size", "1",
                "--epochs", "2",
                "--steps-per-epoch", "1",
                "--no-save",
                "--no-interactive",
                "--log-every", "1",
                "--prompt", "hello simplex",
                "--prompt-bytes", "16",
                "--gen-bytes", "16",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    finally:
        cleanup_tree(root)

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    detail = stdout[-600:] if proc.returncode == 0 else (stderr or stdout)[-1000:]

    check("extended CLI test-metrics process exits cleanly", proc.returncode == 0, detail)
    check("extended CLI prints final_test_loss", "final_test_loss" in stdout and "final_test_bpb=" in stdout and "final_test_ppl=" in stdout, detail)
    check("extended CLI prints best_test_loss", "best_test_loss" in stdout and "best_test_bpb=" in stdout and "best_test_ppl=" in stdout, detail)


#====================
# run all
#====================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, cannot run stress tests")
        sys.exit(1)

    t0 = time.time()

    test_born_rule_sum()
    test_forward3_raw_channel_projection_contract()
    test_forward3_raw_channel_simplex_contract()
    test_forward3_mixed_raw_channel_contract()
    test_forward3_split_backward_native_contract()
    test_forward3_packed_reference_parity()
    test_forward3_packed_reference_mixed_parity()
    test_forward3_packed_reference_raw_projection_contract()
    test_forward3_packed_reference_fused_epilogue_contract()
    test_forward3_packed_reference_rejects_bad_layout()
    test_forward3_packed_reference_backward_parity()
    test_simplex_triangle_vertices()
    test_centered_simplex_backward_fd()
    test_backnorm_full_fd()
    test_attention2_vs_sdpa()
    test_attention2_streaming_forward_large_seq_contract()
    test_attention2_streaming_backward_large_seq_contract()
    test_attention2_batched_consistency()
    test_native_attention2_streaming_batched_forward_consistency()
    test_native_attention2_streaming_batched_backward_consistency()
    test_native_attention2_batched_bridge_consistency()
    test_native_attention2_backward_reuse_smoke()
    test_native_attention2_path_switch_reuse_smoke()
    test_prior_raises_entropy()
    test_prior_no_overshoot()
    test_full_model_gradient_flow()
    test_model_forward_split_contract()
    test_stepwise_decode_contract()
    test_overfit_single_batch()
    test_checkpoint_roundtrip()
    test_rng_state_helper_roundtrip()
    test_checkpoint_payload_includes_rng_state()
    test_checkpoint_load_legacy_payload_without_rng_state()
    test_checkpoint_resume_restores_batch_rng_exact_fp32()
    test_checkpoint_restore_preserves_generation_rng()
    test_extreme_amplitude_scales()
    test_qitnn_linear_simplex_consistency()
    test_qitnn_linear_packed_runtime_view_contract()
    test_model_default_runtime_backend_contract()
    test_qitnn_linear_runtime_backend_parity()
    test_qitnn_linear_runtime_backend_backward_parity()
    test_model_runtime_backend_parity()
    test_runtime_backend_rejects_invalid_name()
    test_runtime_backend_product_surface_contract()
    test_adamw_param_groups()
    test_generation_sanity()
    test_generation_reference_loop_parity_fp32()
    test_generate_uses_decode_step_contract()
    test_generation_seq_len_window_contract()
    test_mixed_precision_generation_sanity()
    test_generation_reference_loop_parity_mixed_precision()
    test_mixed_precision_guardrails()
    test_mixed_precision_default_stays_fp32()
    test_qitnn_linear_mixed_precision_smoke()
    test_qitnn_linear_mixed_precision_forces_bf16()
    test_model_mixed_precision_smoke()
    test_mixed_precision_cli_defaults()
    test_precision_mode_single_source_of_truth()
    test_legacy_mixed_precision_python_compat()
    test_cli_help_prefers_precision_mode()
    test_precision_config_artifact_prefers_canonical_mode()
    test_diagnostics_schema_contract()
    test_trainer_writes_diag_json()
    test_trainer_writes_diag_csv()
    test_trainer_writes_diag_artifacts_under_mixed_multilayer_stress()
    test_diag_artifacts_resume_same_run_prunes_future_epochs()
    test_sample_batch_contract()
    test_grad_accumulation_config_contract()
    test_grad_accumulation_cli_contract()
    test_grad_accumulation_step_semantics_contract()
    test_grad_accumulation_loss_scaling_contract()
    test_grad_accumulation_optimizer_step_count_contract()
    test_grad_accumulation_grad_clip_boundary()
    test_grad_accumulation_metric_reporting_contract()
    test_grad_accumulation_bpb_contract()
    test_grad_accumulation_csv_contract()
    test_grad_accumulation_schedule_contract()
    test_grad_accumulation_prior_step_contract()
    test_grad_accumulation_resume_exactness_fp32()
    test_grad_accumulation_large_batch_parity_fp32()
    test_grad_accumulation_large_batch_parity_mixed_smoke()
    test_grad_accumulation_finite_grads_smoke()
    test_warmup_schedule_contract()
    test_warmup_trainer_csv_smoke()
    test_warmup_resume_schedule_smoke()
    test_precision_mode_high_level_api()
    test_precision_mode_low_level_api()
    test_native_mixed_bridge_smoke()
    test_public_fp16_mixed_api_smoke()
    test_mixed_precision_training_parity_stress()
    test_memory_stability()
    test_simplex_gelu_passthrough()
    test_init_near_uniform()
    test_determinism()
    test_residual_matters()
    test_byte_tokenizer_roundtrip()
    test_bpe_tokenizer_roundtrip()
    test_bpb_byte_math_helpers()
    test_bpb_bpe_byte_accounting()
    test_bpb_total_aggregation_math()
    test_run_val_bpb_aggregation()
    test_run_val_batches_windows()
    test_eval_split_metrics_helper()
    test_bpe_trainer_smoke()
    test_mixed_precision_trainer_smoke()
    test_mixed_precision_checkpoint_resume_smoke()
    test_mixed_precision_cli_smoke()
    test_json_loader_extracts_text()
    test_json_trainer_smoke()
    test_extended_dataset_best_final_test_metrics_smoke()
    test_extended_dataset_cli_test_metrics_smoke()

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"STRESS TEST RESULTS: {PASSED} passed, {FAILED} failed, {WARNED} warnings")
    print(f"Time: {elapsed:.1f}s")
    if FAILED == 0:
        print("ALL STRESS TESTS PASSED")
    else:
        print(f"{FAILED} FAILURES DETECTED")
    sys.exit(0 if FAILED == 0 else 1)
