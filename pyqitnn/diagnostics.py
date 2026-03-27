from __future__ import annotations

from collections.abc import Iterable, Iterator
import math

import torch


#====================
# constants
#====================

MAX_SHANNON_H = math.log2(3.0)
QITNN_DIAG_SCHEMA_VERSION = 1
QITNN_DIAG_STAT_KEYS = (
    "count",
    "p_neg",
    "p_zero",
    "p_pos",
    "h",
    "eff",
    "collapse_pct",
    "sum_n",
    "sum_z",
    "sum_p",
    "rms_n",
    "rms_z",
    "rms_p",
    "p0_gt_04_pct",
    "p0_lt_01_pct",
    "maxp_gt_08_pct",
    "h_gt_13_pct",
    "var_h",
)
QITNN_DIAG_CSV_HEADER = (
    "epoch",
    "full",
    "layer_name",
    "layer_label",
    "role",
    *QITNN_DIAG_STAT_KEYS,
)


#====================
# label formatting
#====================

def short_qitnn_label(name: str) -> str:
    if not name.startswith("blocks."):
        return name
    parts = name.split(".")
    idx = parts[1]
    tail = parts[2]
    short = {
        "q_proj": "wq",
        "k_proj": "wk",
        "v_proj": "wv",
        "o_proj": "wo",
        "ff1": "ff1",
        "ff2": "ff2",
    }
    return f"L{idx} {short.get(tail, tail)}"


#====================
# per-layer statistics from raw amplitudes
#====================

@torch.no_grad()
def qitnn_diag_stats(
    a_neg: torch.Tensor,
    a_zero: torch.Tensor,
    a_pos: torch.Tensor,
) -> dict[str, float]:
    a = a_neg.detach().to(dtype=torch.float64)
    b = a_zero.detach().to(dtype=torch.float64)
    c = a_pos.detach().to(dtype=torch.float64)

    a2, b2, c2 = a * a, b * b, c * c
    z = a2 + b2 + c2
    ok = z > 1e-20
    n = int(ok.sum().item())

    if n <= 0:
        return {
            "count": 0.0, "p_neg": 0.0, "p_zero": 0.0, "p_pos": 0.0,
            "h": 0.0, "eff": 1.0, "collapse_pct": 0.0,
            "sum_n": 0.0, "sum_z": 0.0, "sum_p": 0.0,
            "rms_n": 0.0, "rms_z": 0.0, "rms_p": 0.0,
            "p0_gt_04_pct": 0.0, "p0_lt_01_pct": 0.0,
            "maxp_gt_08_pct": 0.0, "h_gt_13_pct": 0.0, "var_h": 0.0,
        }

    pn = torch.zeros_like(z)
    pz = torch.zeros_like(z)
    pp = torch.zeros_like(z)
    pn[ok] = a2[ok] / z[ok]
    pz[ok] = b2[ok] / z[ok]
    pp[ok] = c2[ok] / z[ok]

    # per-element Shannon entropy
    h = torch.zeros_like(z)
    for p in (pn, pz, pp):
        h = h - torch.where(p > 1e-15, p * torch.log2(p), torch.zeros_like(p))

    mx = torch.maximum(torch.maximum(pn, pz), pp)
    mh = float(h[ok].mean().item())
    mh = min(max(mh, 0.0), MAX_SHANNON_H)
    var_h = float((h[ok] * h[ok]).mean().item() - mh * mh)
    if var_h < 0.0:
        var_h = 0.0

    nf = float(n)
    to_pct = lambda cond: float(100.0 * cond.to(torch.float64).mean().item())

    return {
        "count":          nf,
        "p_neg":          float(pn[ok].mean().item()),
        "p_zero":         float(pz[ok].mean().item()),
        "p_pos":          float(pp[ok].mean().item()),
        "h":              mh,
        "eff":            float(2.0 ** mh),
        "collapse_pct":   to_pct(mx[ok] > 0.9),
        "sum_n":          float(a[ok].sum().item()),
        "sum_z":          float(b[ok].sum().item()),
        "sum_p":          float(c[ok].sum().item()),
        "rms_n":          float(torch.sqrt(a2[ok].mean()).item()),
        "rms_z":          float(torch.sqrt(b2[ok].mean()).item()),
        "rms_p":          float(torch.sqrt(c2[ok].mean()).item()),
        "p0_gt_04_pct":   to_pct(pz[ok] > 0.4),
        "p0_lt_01_pct":   to_pct(pz[ok] < 0.1),
        "maxp_gt_08_pct": to_pct(mx[ok] > 0.8),
        "h_gt_13_pct":    to_pct(h[ok] > 1.3),
        "var_h":          var_h,
    }


def qitnn_diag_record(
    name: str,
    role: str,
    a_neg: torch.Tensor,
    a_zero: torch.Tensor,
    a_pos: torch.Tensor,
) -> dict[str, object]:
    return {
        "name": name,
        "label": short_qitnn_label(name),
        "role": role,
        "stats": qitnn_diag_stats(a_neg, a_zero, a_pos),
    }


def qitnn_diag_snapshot(
    layers: Iterable[tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    epoch: int,
    full: bool,
) -> dict[str, object]:
    records = [
        qitnn_diag_record(name, role, a_neg, a_zero, a_pos)
        for name, role, a_neg, a_zero, a_pos in layers
    ]
    return {
        "schema_version": QITNN_DIAG_SCHEMA_VERSION,
        "epoch": int(epoch),
        "full": bool(full),
        "layer_count": len(records),
        "layers": records,
    }


def format_qitnn_diag_record(
    record: dict[str, object],
    *,
    epoch: int | None = None,
) -> list[str]:
    return format_qitnn_diag(
        str(record["label"]),
        dict(record["stats"]),
        epoch=epoch,
    )


def format_qitnn_diag_snapshot(snapshot: dict[str, object]) -> list[str]:
    epoch = snapshot.get("epoch")
    epoch_value = int(epoch) if epoch is not None else None
    lines: list[str] = []
    for record in snapshot.get("layers", []):
        lines.extend(format_qitnn_diag_record(record, epoch=epoch_value))
    return lines


def iter_qitnn_diag_csv_rows(snapshot: dict[str, object]) -> Iterator[list[object]]:
    epoch = int(snapshot["epoch"])
    full = bool(snapshot["full"])
    for record in snapshot["layers"]:
        stats = dict(record["stats"])
        yield [
            epoch,
            full,
            record["name"],
            record["label"],
            record["role"],
            *[stats[key] for key in QITNN_DIAG_STAT_KEYS],
        ]


#====================
# formatted output
#====================

def format_qitnn_diag(label: str, stats: dict[str, float], *, epoch: int | None = None) -> list[str]:
    title = f"  [{label}]"
    if epoch is not None:
        title = f"{title} (epoch {epoch})"

    return [
        title,
        (
            f"    P-={stats['p_neg']:.3f} P0={stats['p_zero']:.3f} P+={stats['p_pos']:.3f} | "
            f"H={stats['h']:.4f}/{MAX_SHANNON_H:.4f} | eff={stats['eff']:.2f}/3 | "
            f"col={stats['collapse_pct']:.1f}%"
        ),
        (
            f"    ampl: sum_n={stats['sum_n']:.4f} sum_z={stats['sum_z']:.4f} sum_p={stats['sum_p']:.4f} | "
            f"rms={stats['rms_n']:.6f}/{stats['rms_z']:.6f}/{stats['rms_p']:.6f}"
        ),
        (
            f"    dist: P0>0.4={stats['p0_gt_04_pct']:.1f}% P0<0.1={stats['p0_lt_01_pct']:.1f}% "
            f"maxP>0.8={stats['maxp_gt_08_pct']:.1f}% H>1.3={stats['h_gt_13_pct']:.1f}% "
            f"var(H)={stats['var_h']:.4f}"
        ),
    ]


def render_qitnn_diag(
    name: str,
    a_neg: torch.Tensor,
    a_zero: torch.Tensor,
    a_pos: torch.Tensor,
    *,
    epoch: int | None = None,
) -> list[str]:
    record = qitnn_diag_record(name, "unknown", a_neg, a_zero, a_pos)
    return format_qitnn_diag_record(record, epoch=epoch)
