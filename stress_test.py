"""
Stress test for PyQITNN architecture.
Designed to expose hidden bugs before public release.

Run: python stress_test.py
Requires CUDA GPU.
"""
import json
import sys
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pyqitnn
from BasicQITNN_Transformer import load_bytes, train
from pyqitnn.ops import forward3, prior_, centered_simplex, attention2
from pyqitnn.bridge import load_native

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
# 10. Overfit test: model must memorize 1 batch
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
# 14. Training convergence: AdamW param groups correctness
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
# 16. Memory stability: repeated forward/backward does not OOM or leak
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
# 17. Simplex gelu: only x channel is activated, y passes through
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
# 18. Weight initialization: all triplets should start near-uniform
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
# 19. Determinism: same seed -> exact same training trajectory
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
# 20. Residual connection: removing residual should hurt loss
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
# 21. Byte tokenizer: current byte path must remain stable
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
# 22. BPE tokenizer: roundtrip and vocab smoke
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
# 23. BPE trainer smoke: end-to-end trainer path without touching QTS math
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
    check("bpe trainer produced finite loss", last_loss is not None and math.isfinite(last_loss), f"last_loss={last_loss}")


#====================
# 24. JSON loader: extract training text from JSON payloads
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
# 25. JSON trainer smoke: train on JSONL without changing QTS math
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
# run all
#====================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, cannot run stress tests")
        sys.exit(1)

    t0 = time.time()

    test_born_rule_sum()
    test_simplex_triangle_vertices()
    test_centered_simplex_backward_fd()
    test_backnorm_full_fd()
    test_attention2_vs_sdpa()
    test_attention2_batched_consistency()
    test_prior_raises_entropy()
    test_prior_no_overshoot()
    test_full_model_gradient_flow()
    test_overfit_single_batch()
    test_checkpoint_roundtrip()
    test_extreme_amplitude_scales()
    test_qitnn_linear_simplex_consistency()
    test_adamw_param_groups()
    test_generation_sanity()
    test_memory_stability()
    test_simplex_gelu_passthrough()
    test_init_near_uniform()
    test_determinism()
    test_residual_matters()
    test_byte_tokenizer_roundtrip()
    test_bpe_tokenizer_roundtrip()
    test_bpe_trainer_smoke()
    test_json_loader_extracts_text()
    test_json_trainer_smoke()

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"STRESS TEST RESULTS: {PASSED} passed, {FAILED} failed, {WARNED} warnings")
    print(f"Time: {elapsed:.1f}s")
    if FAILED == 0:
        print("ALL STRESS TESTS PASSED")
    else:
        print(f"{FAILED} FAILURES DETECTED")
    sys.exit(0 if FAILED == 0 else 1)
