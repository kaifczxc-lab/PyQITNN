# BasicQITNN_Transformer.py

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import importlib.util
import torch
ROOT = Path(__file__).resolve().parent
_TRAINER_DEFAULT_PRECISION_MODE = "qts_fp32_rest_bf16"


def _pick_pyqitnn_import_root() -> str | None:
    # Prefer a locally built extension when present.
    local_pkg = ROOT / "pyqitnn"
    if any(local_pkg.glob("_C*.pyd")) or any(local_pkg.glob("_C*.so")):
        return str(ROOT)

    # Next prefer setuptools build output, which often exists after build_ext.
    build_dir = ROOT / "build"
    if build_dir.exists():
        for candidate in sorted(build_dir.glob("lib*"), reverse=True):
            pkg = candidate / "pyqitnn"
            if any(pkg.glob("_C*.pyd")) or any(pkg.glob("_C*.so")):
                return str(candidate)

    # If pyqitnn is already installed, let site-packages win.
    if importlib.util.find_spec("pyqitnn") is not None:
        return None

    # Last resort: raw source tree import for pure-Python debugging.
    return str(ROOT)


_pyqitnn_root = _pick_pyqitnn_import_root()
if _pyqitnn_root is not None:
    sys.path.insert(0, _pyqitnn_root)
import pyqitnn
from pyqitnn.diagnostics import QITNN_DIAG_CSV_HEADER
from pyqitnn.diagnostics import QITNN_DIAG_SCHEMA_VERSION
from pyqitnn.diagnostics import QITNN_DIAG_STAT_KEYS
from pyqitnn.diagnostics import format_qitnn_diag_snapshot
from pyqitnn.diagnostics import iter_qitnn_diag_csv_rows
from pyqitnn.precision import normalize_precision_mode as _shared_normalize_precision_mode
from pyqitnn.precision import precision_mode_choices as _precision_mode_choices
from pyqitnn.precision import resolve_precision_mode as _shared_resolve_precision_mode
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#====================
# training config
#====================
@dataclass
class TrainConfig:
    #====================
    # data
    # dataset: path to a text/json/jsonl file or a directory with such files
    # set this to your actual data path before running
    #====================
    dataset: str | Path = r"D:\so_data\dearimgui_dataset.json"   # path to a file or directory with training data
    extended_dataset: bool = False   # True = use train_dir/val_dir/test_dir separately
    train_dir: str | Path | None = None
    val_dir: str | Path | None = None
    test_dir: str | Path | None = None
    max_bytes: int = 3_000_000       # max bytes to load per data source
    data_format: str = "auto"         # "auto", "text", "json", "jsonl"
    json_text_fields: str | None = None   # comma-separated preferred text fields, e.g. "text,content"
    tokenizer: str = "bpe"           # "byte" or "bpe"
    tokenizer_path: str | Path | None = None
    tokenizer_vocab_size: int = 2048
    tokenizer_min_frequency: int = 2

    #====================
    # model
    #====================
    dim: int = 128
    ffn: int = 256
    layers: int = 2
    seq_len: int = 256

    #====================
    # training
    #====================
    device: str = "cuda:0"
    seed: int = 7
    batch_size: int = 8
    grad_accum_steps: int = 1      # micro-batches per optimizer step; 1 keeps the legacy trainer contract
    steps: int | None = None         # if set, overrides epochs/steps_per_epoch
    epochs: int = 10
    steps_per_epoch: int = 1000
    grad_clip: float = 1.0
    mixed_precision: bool | None = None   # legacy compatibility knob for older trainer calls
    precision_mode: str | None = None     # canonical precision selector; None resolves to trainer default
    #====================
    # optimizer
    # "adamw" and "SGD"
    #====================
    optimizer: str = "adamw"
    lr_start: float = 0.005          # SGD only
    lr_end: float = 0.001            # SGD only
    lr_schedule: str = "cosine"      # "cosine" or "linear"
    warmup_steps: int = 0            # optimizer steps for linear LR warmup before decay
    #====================
    # zero-boost
    # multiplier for a_zero learning rate
    #====================
    zero_boost: float = 3.5
    zero_boost_qk: float | None = None   # per-role override, None = use zero_boost
    zero_boost_vo: float | None = None
    zero_boost_ff: float | None = None
    #====================
    # adamw (only when optimizer="adamw")
    #====================
    adamw_lr_start: float = 3e-4
    adamw_lr_end: float = 3e-5
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.01
    adamw_trit_floor_step: float | None = 5e-5   # fixed prior step for adamw
    adamw_trit_floor_step_qk: float | None = None
    adamw_trit_floor_step_vo: float | None = None
    adamw_trit_floor_step_ff: float | None = None
    #====================
    # trit-floor prior (prevents ternary collapse)
    # trit_floor_h: entropy threshold in bits (log2(3)*2/3 = 1.084)
    # trit_floor_mul: step = lr * mul (SGD mode; adamw uses adamw_trit_floor_step)
    #====================
    trit_floor_h: float = 1.0840643
    trit_floor_mul_start: float = 0.01
    trit_floor_mul_end: float = 0.001
    trit_floor_mul_qk_start: float | None = None
    trit_floor_mul_qk_end: float | None = None
    trit_floor_mul_vo_start: float | None = None
    trit_floor_mul_vo_end: float | None = None
    trit_floor_mul_ff_start: float | None = None
    trit_floor_mul_ff_end: float | None = None
    #====================
    # entropy regularization (in-graph, backward pass)
    # soft push toward uniform distribution during training
    #====================
    ent_lambda: float = 0.001
    ent_lambda_qk: float | None = None
    ent_lambda_vo: float | None = None
    ent_lambda_ff: float | None = None
    #====================
    # validation
    #====================
    val_split_div: int = 10          # 1/N of data becomes validation
    val_steps: int = 50              # max validation windows per epoch (0 = all, slow on large data)
    #====================
    # generation
    #====================
    temperature: float = 0.65
    top_k: int = 12
    gen_bytes: int = 160
    gen_tokens: int | None = None
    prompt: str = ""                 # empty = use random slice from training data
    prompt_bytes: int = 64
    prompt_tokens: int | None = None
    gen_every: int = 1               # generate sample every N epochs
    #====================
    # logging
    #====================
    log_every: int = 100             # print loss every N steps
    diag_every: int = 2             # full QTS diagnostics every N epochs
    csv_log: str | None = None       # None = auto-create in run directory
    #====================
    # saving
    # save_dir: base folder for all runs (relative to cwd or absolute)
    # checkpoints go to save_dir/run_name/
    #====================
    save_dir: str = "runs"
    run_name: str | None = None      # None = auto-generate run_YYYYMMDD_HHMMSS
    no_save: bool = False            # True = no files written at all
    save_every: int = 25             # periodic checkpoint every N epochs
    save_model_only: bool = False    # True = skip optimizer state in checkpoints
    resume: str | None = None        # path to checkpoint to resume from
    #====================
    # interactive
    #====================
    interactive: bool = False        # force interactive prompt loop after training
    no_interactive: bool = False     # force non-interactive
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def _normalize_precision_mode(value: str) -> str:
    return _shared_normalize_precision_mode(value)


def _resolve_precision_mode_cfg(cfg: TrainConfig) -> tuple[str, bool]:
    return _shared_resolve_precision_mode(
        cfg.precision_mode,
        cfg.mixed_precision,
        default_mode=_TRAINER_DEFAULT_PRECISION_MODE,
    )


def _resolve_grad_accum_steps(value: int | None) -> int:
    grad_accum_steps = 1 if value is None else int(value)
    if grad_accum_steps < 1:
        raise RuntimeError("grad_accum_steps must be >= 1")
    return grad_accum_steps


def _resolve_train_step_plan(cfg: TrainConfig) -> tuple[int, int, int, int]:
    epochs = int(cfg.epochs)
    steps_per_ep = int(cfg.steps_per_epoch)
    if cfg.steps is not None:
        epochs = 1
        steps_per_ep = int(cfg.steps)
    total_steps = max(epochs * steps_per_ep, 1)
    grad_accum_steps = _resolve_grad_accum_steps(cfg.grad_accum_steps)
    return epochs, steps_per_ep, total_steps, grad_accum_steps

#====================
# data loading
#====================
def _iter_data_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file())


def _parse_json_field_spec(spec: str | None) -> set[str] | None:
    if spec is None:
        return None
    keys = {part.strip() for part in spec.split(",") if part.strip()}
    return keys or None


def _resolve_data_format(path: Path, mode: str) -> str:
    fmt = mode.strip().lower()
    if fmt != "auto":
        if fmt not in {"text", "json", "jsonl"}:
            raise RuntimeError("data_format must be one of: auto, text, json, jsonl")
        return fmt

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".json":
        return "json"
    return "text"


def _collect_json_strings(obj, wanted: set[str] | None) -> list[str]:
    out: list[str] = []

    if isinstance(obj, str):
        if obj:
            out.append(obj)
        return out

    if isinstance(obj, list):
        for item in obj:
            out.extend(_collect_json_strings(item, wanted))
        return out

    if isinstance(obj, dict):
        if wanted:
            matched = False
            for key, value in obj.items():
                if key in wanted:
                    matched = True
                    out.extend(_collect_json_strings(value, None))
            if matched:
                return out
        for value in obj.values():
            out.extend(_collect_json_strings(value, wanted))
        return out

    return out


def _json_bytes_to_text_bytes(raw: bytes, source: Path, wanted: set[str] | None) -> bytes:
    text = raw.decode("utf-8", errors="replace")
    pieces: list[str] = []

    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"invalid JSONL in {source} line {lineno}: {e}") from e
            pieces.extend(_collect_json_strings(obj, wanted))
    else:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"invalid JSON in {source}: {e}") from e
        pieces.extend(_collect_json_strings(obj, wanted))

    joined = "\n\n".join(part for part in pieces if part)
    return joined.encode("utf-8", errors="replace")


def load_bytes(path: Path, limit: int, *, data_format: str = "auto", json_text_fields: str | None = None) -> bytes:
    if not path.exists():
        raise RuntimeError(
            f"data not found: {path}\n"
            f"pass a file or directory path via dataset='path/to/data'"
        )
    files = _iter_data_files(path)
    if not files:
        raise RuntimeError(f"no files in {path}")

    wanted_fields = _parse_json_field_spec(json_text_fields)
    parts: list[bytes] = []
    total = 0
    for f in files:
        raw = f.read_bytes()
        if not raw:
            continue
        file_format = _resolve_data_format(f, data_format)
        cooked = raw if file_format == "text" else _json_bytes_to_text_bytes(raw, f, wanted_fields)
        if not cooked:
            continue
        take = min(len(cooked), limit - total)
        parts.append(cooked[:take])
        total += take
        if total >= limit:
            break
    if total < 1024:
        raise RuntimeError("dataset too small (need >= 1024 bytes)")
    return b"".join(parts)
def split_train_val(data: torch.Tensor, seq_len: int, val_div: int):
    min_tok = seq_len + 1
    val_n = max(data.numel() // val_div, min_tok)
    val_n = min(val_n, data.numel() - min_tok)
    train_n = data.numel() - val_n
    return data[:train_n], data[train_n:]


def split_train_val_raw(raw: bytes, val_div: int):
    min_bytes = 1024
    if len(raw) < (2 * min_bytes):
        raise RuntimeError("dataset too small (need >= 2048 bytes for BPE split)")
    val_n = max(len(raw) // val_div, min_bytes)
    val_n = min(val_n, len(raw) - min_bytes)
    train_n = len(raw) - val_n
    return raw[:train_n], raw[train_n:]
#====================
# batch sampling
#====================
def sample_batch(data: torch.Tensor, batch: int, seq_len: int, device: torch.device):
    hi = data.numel() - seq_len - 1
    starts = torch.randint(0, hi + 1, (batch,))
    windows = data.unfold(0, seq_len + 1, 1)
    chunks = windows.index_select(0, starts)
    x = chunks[:, :-1].contiguous()
    y = chunks[:, 1:].contiguous()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
#====================
# validation
#====================
@torch.no_grad()
def run_val(
    model,
    val_data: torch.Tensor,
    seq_len: int,
    device: torch.device,
    max_steps: int,
    *,
    tokenizer,
    tok_bytes_lut: torch.Tensor | None = None,
):
    model.eval()
    n_win = max((val_data.numel() - 1) // seq_len, 0)
    if n_win < 1:
        model.train()
        return 0.0, None, 0
    if 0 < max_steps < n_win:
        n_win = max_steps
    elif max_steps == 0 and n_win > 500:
        print(f"  [warn] val has {n_win} windows, this will be slow. set val_steps=50 to cap it.")
    total_nll_nats = 0.0
    total_target_tokens = 0
    total_target_bytes = 0
    windows = val_data.unfold(0, seq_len + 1, seq_len)[:n_win]
    batch_windows = min(8, n_win)
    for start in range(0, n_win, batch_windows):
        chunk = windows[start : start + batch_windows]
        xv = chunk[:, :-1].contiguous().to(device, non_blocking=True)
        yv = chunk[:, 1:].contiguous()
        _, loss = model(xv, targets=yv.to(device, non_blocking=True))
        target_tokens = int(yv.numel())
        total_nll_nats += float(loss.detach().cpu()) * float(target_tokens)
        total_target_tokens += target_tokens
        total_target_bytes += _count_token_bytes(yv, tokenizer, tok_bytes_lut)
    model.train()
    avg_loss, avg_bpb = summarize_nll_metrics(total_nll_nats, total_target_tokens, total_target_bytes)
    return (0.0 if avg_loss is None else avg_loss), avg_bpb, n_win
#====================
# text generation
#====================
def decode_tokens(tokenizer, tokens: torch.Tensor) -> str:
    return tokenizer.decode(tokens.reshape(-1).tolist())


def safe_console_text(text: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def make_prompt(data: torch.Tensor, text: str, n_units: int, device: torch.device, tokenizer):
    if text:
        ids = tokenizer.encode_text(text)[:n_units]
        if not ids:
            ids = tokenizer.encode_text(" ")
        tok = torch.tensor(ids, dtype=torch.long)
    else:
        tok = data[: min(n_units, data.numel())].detach().cpu()
    if tok.numel() <= 0:
        raise RuntimeError("prompt must contain at least one token")
    return tok.unsqueeze(0).to(device, non_blocking=True)


def _prompt_units(cfg: TrainConfig, tokenizer) -> int:
    if getattr(tokenizer, "kind", "byte") == "byte":
        return int(cfg.prompt_bytes)
    if cfg.prompt_tokens is not None:
        return int(cfg.prompt_tokens)
    return int(cfg.prompt_bytes)


def _gen_units(cfg: TrainConfig, tokenizer) -> int:
    if getattr(tokenizer, "kind", "byte") == "byte":
        return int(cfg.gen_bytes)
    if cfg.gen_tokens is not None:
        return int(cfg.gen_tokens)
    return int(cfg.gen_bytes)


def infer_tokenizer_path(resume: str | None) -> Path | None:
    if not resume:
        return None
    cand = Path(resume).resolve().parent / "tokenizer.json"
    return cand if cand.exists() else None


def generate_text(model, data, *, tokenizer, prompt, prompt_units, gen_units, temperature, top_k, device):
    tok = make_prompt(data, prompt, prompt_units, device, tokenizer)
    out = model.generate(
        tok,
        max_new_tokens=gen_units,
        temperature=temperature,
        top_k=top_k,
        ascii_guard=bool(getattr(tokenizer, "uses_ascii_guard", False)),
    )
    return decode_tokens(tokenizer, out[0])
#====================
# lr schedules
#====================
def lr_linear(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
def lr_cosine(a: float, b: float, t: float) -> float:
    return b + 0.5 * (a - b) * (1.0 + math.cos(math.pi * t))


def _resolve_warmup_steps(value: int | None) -> int:
    if value is None:
        return 0
    warmup = int(value)
    if warmup < 0:
        raise RuntimeError("warmup_steps must be >= 0")
    return warmup


def _schedule_progress(step_index: int, total_steps: int, warmup_steps: int = 0) -> float:
    total = max(int(total_steps), 1)
    idx = min(max(int(step_index), 0), total - 1)
    warmup = _resolve_warmup_steps(warmup_steps)

    if total <= 1 or warmup >= total:
        return 0.0
    if idx < warmup:
        return 0.0

    tail_steps = total - warmup
    tail_idx = idx - warmup
    if tail_steps <= 1:
        return 1.0
    return min(max(tail_idx / float(tail_steps - 1), 0.0), 1.0)


def lr_with_warmup(
    a: float,
    b: float,
    step_index: int,
    total_steps: int,
    *,
    warmup_steps: int = 0,
    schedule_fn=lr_cosine,
) -> float:
    total = max(int(total_steps), 1)
    idx = min(max(int(step_index), 0), total - 1)
    warmup = _resolve_warmup_steps(warmup_steps)
    target = schedule_fn(a, b, _schedule_progress(idx, total, warmup))
    ramp_steps = min(warmup, total)

    if ramp_steps > 0 and idx < ramp_steps:
        return target * float(idx + 1) / float(ramp_steps)
    return target


def loss_to_perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    if math.isnan(loss):
        return float("nan")
    # cross-entropy is in nats, so LM perplexity is exp(loss)
    if loss >= 80.0:
        return float("inf")
    return math.exp(loss)


def nats_to_bits(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isnan(value):
        return float("nan")
    return float(value) / math.log(2.0)


def total_nll_to_bpb(total_nll_nats: float | None, target_bytes: int) -> float | None:
    if total_nll_nats is None:
        return None
    if target_bytes <= 0:
        return None
    total_nll_bits = nats_to_bits(total_nll_nats)
    if total_nll_bits is None:
        return None
    return total_nll_bits / float(target_bytes)


def loss_to_bpb(loss: float | None, target_tokens: int, target_bytes: int) -> float | None:
    if loss is None:
        return None
    if target_tokens <= 0:
        return None
    return total_nll_to_bpb(float(loss) * float(target_tokens), target_bytes)


def summarize_nll_metrics(
    total_nll_nats: float,
    target_tokens: int,
    target_bytes: int,
) -> tuple[float | None, float | None]:
    if target_tokens <= 0:
        return None, None
    avg_loss = float(total_nll_nats) / float(target_tokens)
    return avg_loss, total_nll_to_bpb(total_nll_nats, target_bytes)


def eval_split_metrics(
    model,
    data: torch.Tensor | None,
    seq_len: int,
    device: torch.device,
    max_steps: int,
    *,
    tokenizer,
    tok_bytes_lut: torch.Tensor | None = None,
) -> dict[str, float | int | None]:
    was_training = model.training
    if data is None or data.numel() <= seq_len + 1:
        return {"loss": None, "bpb": None, "ppl": None, "windows": 0}
    avg_loss, avg_bpb, n_win = run_val(
        model,
        data,
        seq_len,
        device,
        max_steps,
        tokenizer=tokenizer,
        tok_bytes_lut=tok_bytes_lut,
    )
    if was_training:
        model.train()
    else:
        model.eval()
    if n_win <= 0:
        return {"loss": None, "bpb": None, "ppl": None, "windows": 0}
    return {
        "loss": avg_loss,
        "bpb": avg_bpb,
        "ppl": loss_to_perplexity(avg_loss),
        "windows": n_win,
    }


def capture_model_state_cpu(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_checkpoint_model_state_cpu(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise RuntimeError(f"checkpoint missing model state: {path}")
    return {k: v.detach().cpu().clone() for k, v in ckpt["model"].items()}


def infer_best_ckpt_path(resume: str | None) -> Path | None:
    if not resume:
        return None
    resume_path = Path(resume).resolve()
    cand = resume_path if resume_path.name == "ckpt_best.pt" else (resume_path.parent / "ckpt_best.pt")
    return cand if cand.exists() else None


def eval_model_state_metrics(
    model,
    state_cpu: dict[str, torch.Tensor] | None,
    data: torch.Tensor | None,
    seq_len: int,
    device: torch.device,
    max_steps: int,
    *,
    tokenizer,
    tok_bytes_lut: torch.Tensor | None = None,
    restore_state_cpu: dict[str, torch.Tensor] | None = None,
) -> dict[str, float | int | None]:
    if state_cpu is None:
        return {"loss": None, "bpb": None, "ppl": None, "windows": 0}
    was_training = model.training
    restore_state = restore_state_cpu if restore_state_cpu is not None else capture_model_state_cpu(model)
    try:
        model.load_state_dict(state_cpu)
        metrics = eval_split_metrics(
            model,
            data,
            seq_len,
            device,
            max_steps,
            tokenizer=tokenizer,
            tok_bytes_lut=tok_bytes_lut,
        )
    finally:
        model.load_state_dict(restore_state)
        if was_training:
            model.train()
        else:
            model.eval()
    return metrics


def _build_token_byte_lut(tokenizer) -> torch.Tensor | None:
    if getattr(tokenizer, "kind", "byte") == "byte":
        return None
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is None or not hasattr(inner, "id_to_token"):
        return None
    vocab_size = int(tokenizer.vocab_size)
    tok_bytes_lut = torch.empty(vocab_size, dtype=torch.int64)
    for tok_id in range(vocab_size):
        piece = inner.id_to_token(tok_id)
        if piece is None:
            raise RuntimeError(f"tokenizer returned no token piece for id={tok_id}")
        tok_bytes_lut[tok_id] = len(piece)
    return tok_bytes_lut


def _count_token_bytes(
    token_ids: torch.Tensor,
    tokenizer,
    tok_bytes_lut: torch.Tensor | None = None,
) -> int:
    if token_ids.numel() <= 0:
        return 0
    if getattr(tokenizer, "kind", "byte") == "byte":
        return int(token_ids.numel())
    if tok_bytes_lut is None:
        tok_bytes_lut = _build_token_byte_lut(tokenizer)
    if tok_bytes_lut is None:
        flat_ids = token_ids.detach().reshape(-1).to(device="cpu", dtype=torch.long).tolist()
        text = tokenizer.decode(flat_ids, skip_special_tokens=False)
        return len(text.encode("utf-8", errors="replace"))
    flat_ids = token_ids.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    max_id = int(flat_ids.max().item())
    min_id = int(flat_ids.min().item())
    if min_id < 0 or max_id >= int(tok_bytes_lut.numel()):
        raise RuntimeError("token id out of range for tokenizer vocab")
    tok_hist = torch.bincount(flat_ids, minlength=int(tok_bytes_lut.numel())).to(dtype=torch.int64)
    return int(torch.dot(tok_hist, tok_bytes_lut.to(dtype=torch.int64)).item())


def tokens_per_sec(tokens: int, dt: float) -> float | None:
    if tokens <= 0:
        return None
    if dt <= 0.0:
        return float("inf")
    return float(tokens) / float(dt)


def fmt_metric(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"
#====================
# rng helpers
#====================
def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            s.detach().cpu().clone()
            for s in torch.cuda.get_rng_state_all()
        ]
    return state


def _restore_rng_state(state: dict[str, object] | None) -> bool:
    if not state:
        return False

    restored = False
    cpu_state = state.get("torch_cpu")
    if cpu_state is not None:
        torch.set_rng_state(torch.as_tensor(cpu_state, dtype=torch.uint8, device="cpu"))
        restored = True

    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is not available")
        cuda_state_list = list(cuda_states)
        if len(cuda_state_list) > torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state for more devices than are currently available"
            )
        for idx, cuda_state in enumerate(cuda_state_list):
            torch.cuda.set_rng_state(
                torch.as_tensor(cuda_state, dtype=torch.uint8, device="cpu"),
                device=idx,
            )
        restored = True

    return restored
#====================
# checkpoints
#====================
def save_ckpt(path: Path, model, opt, epoch: int, step: int, best_val,
              *, best_val_epoch: int = 0, model_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "best_val_loss": best_val,
        "best_val_epoch": int(best_val_epoch),
        "rng_state": _capture_rng_state(),
    }
    if not model_only:
        payload["optimizer"] = opt.state_dict()
    torch.save(payload, str(path))


def load_ckpt(path: Path, model, opt, device, *, restore_rng: bool = True):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if restore_rng:
        _restore_rng_state(ckpt.get("rng_state"))
    return ckpt
#====================
# csv logger
#====================
class CsvLog:
    HEADER = [
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
    def __init__(self, path: str | None):
        self._file = None
        self._writer = None
        if path is None:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        need_header = not p.exists() or p.stat().st_size == 0
        self._file = open(p, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if need_header:
            self._writer.writerow(self.HEADER)

    def row(
        self,
        epoch: int,
        train_loss: float,
        train_ppl: float | None,
        val_loss: float | None,
        val_ppl: float | None,
        train_tok_s: float | None,
        val_tok_s: float | None,
        lr: float,
        dt: float,
        train_bpb: float | None = None,
        val_bpb: float | None = None,
    ):
        if self._writer is None:
            return
        self._writer.writerow([
            epoch,
            fmt_metric(train_loss, 6),
            fmt_metric(train_ppl, 6),
            fmt_metric(val_loss, 6),
            fmt_metric(val_ppl, 6),
            fmt_metric(train_tok_s, 2),
            fmt_metric(val_tok_s, 2),
            f"{lr:.8f}",
            f"{dt:.1f}",
            fmt_metric(train_bpb, 6),
            fmt_metric(val_bpb, 6),
        ])
        self._file.flush()
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class QitnnDiagLog:
    def __init__(self, json_path: str | None, csv_path: str | None, *, start_epoch: int = 1):
        self._json_path = Path(json_path) if json_path is not None else None
        self._csv_path = Path(csv_path) if csv_path is not None else None
        self._csv_file = None
        self._csv_writer = None
        self._snapshots: list[dict[str, object]] = []
        self._start_epoch = int(start_epoch)

        if self._json_path is not None:
            self._json_path.parent.mkdir(parents=True, exist_ok=True)
            if self._json_path.exists() and self._json_path.stat().st_size > 0:
                payload = json.loads(self._json_path.read_text(encoding="utf-8"))
                epochs = payload.get("epochs", [])
                if isinstance(epochs, list):
                    self._snapshots = self._filter_snapshots_before_epoch(epochs)
                    self._write_json_payload()

        if self._csv_path is None:
            return
        p = self._csv_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.stat().st_size > 0:
            kept_rows = self._read_csv_rows_before_epoch(p)
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(QITNN_DIAG_CSV_HEADER)
                writer.writerows(kept_rows)
        need_header = not p.exists() or p.stat().st_size == 0
        self._csv_file = open(p, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if need_header:
            self._csv_writer.writerow(QITNN_DIAG_CSV_HEADER)

    def _filter_snapshots_before_epoch(self, snapshots: list[object]) -> list[dict[str, object]]:
        kept: list[dict[str, object]] = []
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            epoch = snapshot.get("epoch")
            if epoch is None:
                continue
            try:
                epoch_int = int(epoch)
            except (TypeError, ValueError):
                continue
            if epoch_int < self._start_epoch:
                kept.append(snapshot)
        return kept

    def _read_csv_rows_before_epoch(self, path: Path) -> list[list[str]]:
        kept_rows: list[list[str]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epoch_int = int(row.get("epoch", ""))
                except (TypeError, ValueError):
                    continue
                if epoch_int < self._start_epoch:
                    kept_rows.append([row.get(col, "") for col in QITNN_DIAG_CSV_HEADER])
        return kept_rows

    def _write_json_payload(self):
        if self._json_path is None:
            return
        json_payload = {
            "kind": "qitnn_layer_diagnostics",
            "schema_version": QITNN_DIAG_SCHEMA_VERSION,
            "stats_keys": list(QITNN_DIAG_STAT_KEYS),
            "csv_header": list(QITNN_DIAG_CSV_HEADER),
            "epochs": self._snapshots,
        }
        self._json_path.write_text(
            json.dumps(json_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def snapshot(self, payload: dict[str, object]):
        if self._json_path is not None:
            self._snapshots.append(payload)
            self._write_json_payload()
        if self._csv_writer is None:
            return
        for row in iter_qitnn_diag_csv_rows(payload):
            self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


def _build_cli_parser() -> argparse.ArgumentParser:
    D = TrainConfig()
    modes = ", ".join(_precision_mode_choices())
    p = argparse.ArgumentParser(description="pyqitnn transformer training")
    # data
    p.add_argument("--dataset",           type=str,   default=D.dataset)
    p.add_argument("--extended-dataset",  action="store_true")
    p.add_argument("--train-dir",         type=str,   default=None)
    p.add_argument("--val-dir",           type=str,   default=None)
    p.add_argument("--test-dir",          type=str,   default=None)
    p.add_argument("--max-bytes",         type=int,   default=D.max_bytes)
    p.add_argument("--data-format",       type=str,   default=D.data_format, choices=("auto", "text", "json", "jsonl"))
    p.add_argument("--json-text-fields",  type=str,   default=D.json_text_fields)
    p.add_argument("--tokenizer",         type=str,   default=D.tokenizer, choices=("byte", "bpe"))
    p.add_argument("--tokenizer-path",    type=str,   default=None)
    p.add_argument("--tokenizer-vocab-size", type=int, default=D.tokenizer_vocab_size)
    p.add_argument("--tokenizer-min-frequency", type=int, default=D.tokenizer_min_frequency)
    # model
    p.add_argument("--dim",          type=int,   default=D.dim)
    p.add_argument("--ffn",          type=int,   default=D.ffn)
    p.add_argument("--layers",       type=int,   default=D.layers)
    p.add_argument("--seq-len",      type=int,   default=D.seq_len)
    # training
    p.add_argument("--device",           type=str,   default=D.device)
    p.add_argument("--seed",             type=int,   default=D.seed)
    p.add_argument("--batch-size",       type=int,   default=D.batch_size)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=D.grad_accum_steps,
        help=(
            "Gradient accumulation micro-batches per optimizer step. "
            "Use 1 to keep the legacy trainer contract exactly."
        ),
    )
    p.add_argument("--steps",            type=int,   default=None)
    p.add_argument("--epochs",           type=int,   default=D.epochs)
    p.add_argument("--steps-per-epoch",  type=int,   default=D.steps_per_epoch)
    p.add_argument("--grad-clip",        type=float, default=D.grad_clip)
    p.add_argument(
        "--precision-mode",
        type=str,
        default=D.precision_mode,
        help=(
            "Canonical precision selector. "
            f"Supported modes: {modes}. "
            f"When omitted, the standalone trainer defaults to '{_TRAINER_DEFAULT_PRECISION_MODE}'."
        ),
    )
    p.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", default=D.mixed_precision, help=argparse.SUPPRESS)
    p.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false", help=argparse.SUPPRESS)
    # optimizer
    p.add_argument("--optimizer",    type=str,   default=D.optimizer, choices=("sgd", "adamw"))
    p.add_argument("--lr-start",     type=float, default=D.lr_start)
    p.add_argument("--lr-end",       type=float, default=D.lr_end)
    p.add_argument("--lr-schedule",  type=str,   default=D.lr_schedule, choices=("linear", "cosine"))
    p.add_argument("--warmup-steps", type=int,   default=D.warmup_steps)
    # zero-boost
    p.add_argument("--zero-boost",      type=float, default=D.zero_boost)
    p.add_argument("--zero-boost-qk",   type=float, default=None)
    p.add_argument("--zero-boost-vo",   type=float, default=None)
    p.add_argument("--zero-boost-ff",   type=float, default=None)
    # adamw
    p.add_argument("--adamw-lr-start",         type=float, default=D.adamw_lr_start)
    p.add_argument("--adamw-lr-end",           type=float, default=D.adamw_lr_end)
    p.add_argument("--adamw-beta1",            type=float, default=D.adamw_beta1)
    p.add_argument("--adamw-beta2",            type=float, default=D.adamw_beta2)
    p.add_argument("--adamw-eps",              type=float, default=D.adamw_eps)
    p.add_argument("--adamw-weight-decay",     type=float, default=D.adamw_weight_decay)
    p.add_argument("--adamw-trit-floor-step",     type=float, default=D.adamw_trit_floor_step)
    p.add_argument("--adamw-trit-floor-step-qk",  type=float, default=D.adamw_trit_floor_step_qk)
    p.add_argument("--adamw-trit-floor-step-vo",  type=float, default=D.adamw_trit_floor_step_vo)
    p.add_argument("--adamw-trit-floor-step-ff",  type=float, default=D.adamw_trit_floor_step_ff)
    # trit-floor
    p.add_argument("--trit-floor-h",              type=float, default=D.trit_floor_h)
    p.add_argument("--trit-floor-mul-start",      type=float, default=D.trit_floor_mul_start)
    p.add_argument("--trit-floor-mul-end",        type=float, default=D.trit_floor_mul_end)
    p.add_argument("--trit-floor-mul-qk-start",   type=float, default=None)
    p.add_argument("--trit-floor-mul-qk-end",     type=float, default=None)
    p.add_argument("--trit-floor-mul-vo-start",   type=float, default=None)
    p.add_argument("--trit-floor-mul-vo-end",     type=float, default=None)
    p.add_argument("--trit-floor-mul-ff-start",   type=float, default=None)
    p.add_argument("--trit-floor-mul-ff-end",     type=float, default=None)
    # entropy
    p.add_argument("--ent-lambda",      type=float, default=D.ent_lambda)
    p.add_argument("--ent-lambda-qk",   type=float, default=None)
    p.add_argument("--ent-lambda-vo",   type=float, default=None)
    p.add_argument("--ent-lambda-ff",   type=float, default=None)
    # validation
    p.add_argument("--val-split-div",   type=int, default=D.val_split_div)
    p.add_argument("--val-steps",       type=int, default=D.val_steps)
    # generation
    p.add_argument("--temperature",     type=float, default=D.temperature)
    p.add_argument("--top-k",           type=int,   default=D.top_k)
    p.add_argument("--gen-bytes",       type=int,   default=D.gen_bytes)
    p.add_argument("--gen-tokens",      type=int,   default=None)
    p.add_argument("--prompt",          type=str,   default=D.prompt)
    p.add_argument("--prompt-bytes",    type=int,   default=D.prompt_bytes)
    p.add_argument("--prompt-tokens",   type=int,   default=None)
    p.add_argument("--gen-every",       type=int,   default=D.gen_every)
    # logging
    p.add_argument("--log-every",       type=int, default=D.log_every)
    p.add_argument("--diag-every",      type=int, default=D.diag_every)
    p.add_argument("--csv-log",         type=str, default=None)
    # saving
    p.add_argument("--save-dir",        type=str, default=D.save_dir)
    p.add_argument("--run-name",        type=str, default=None)
    p.add_argument("--no-save",         action="store_true")
    p.add_argument("--save-every",      type=int, default=D.save_every)
    p.add_argument("--save-model-only", action="store_true")
    p.add_argument("--resume",          type=str, default=None)
    # interactive
    p.add_argument("--interactive",     action="store_true")
    p.add_argument("--no-interactive",  action="store_true")
    return p
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#====================
# train
#====================
def train(cfg: TrainConfig | None = None, **kwargs) -> dict:
    """Run training. Returns dict with loss/perplexity metrics, run_dir, model.

    Usage:
        # from Python
        result = train(TrainConfig(dataset="my_data.txt", epochs=50))

        # or with keyword args
        result = train(dataset="my_data.txt", epochs=50)

        # from CLI
        python BasicQITNN_Transformer.py --dataset my_data.txt --epochs 50
    """
    if cfg is None:
        cfg = TrainConfig(**kwargs)
    elif kwargs:
        for k, v in kwargs.items():
            setattr(cfg, k, v)

    precision_mode, use_mixed_precision = _resolve_precision_mode_cfg(cfg)
    epochs, steps_per_ep, total_steps, grad_accum_steps = _resolve_train_step_plan(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device(cfg.device)
    if use_mixed_precision and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"precision_mode='{precision_mode}' currently requires CUDA bf16 support on this device. "
            "Use precision_mode='fp32' to keep the trusted fp32 path."
        )
    #====================
    # seed
    #====================
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    #====================
    # resolve per-role overrides
    #====================
    zb_qk = cfg.zero_boost if cfg.zero_boost_qk is None else cfg.zero_boost_qk
    zb_vo = cfg.zero_boost if cfg.zero_boost_vo is None else cfg.zero_boost_vo
    zb_ff = cfg.zero_boost if cfg.zero_boost_ff is None else cfg.zero_boost_ff
    fm_qk_s = cfg.trit_floor_mul_start if cfg.trit_floor_mul_qk_start is None else cfg.trit_floor_mul_qk_start
    fm_qk_e = cfg.trit_floor_mul_end   if cfg.trit_floor_mul_qk_end   is None else cfg.trit_floor_mul_qk_end
    fm_vo_s = cfg.trit_floor_mul_start if cfg.trit_floor_mul_vo_start is None else cfg.trit_floor_mul_vo_start
    fm_vo_e = cfg.trit_floor_mul_end   if cfg.trit_floor_mul_vo_end   is None else cfg.trit_floor_mul_vo_end
    fm_ff_s = cfg.trit_floor_mul_start if cfg.trit_floor_mul_ff_start is None else cfg.trit_floor_mul_ff_start
    fm_ff_e = cfg.trit_floor_mul_end   if cfg.trit_floor_mul_ff_end   is None else cfg.trit_floor_mul_ff_end
    el_qk = cfg.ent_lambda if cfg.ent_lambda_qk is None else cfg.ent_lambda_qk
    el_vo = cfg.ent_lambda if cfg.ent_lambda_vo is None else cfg.ent_lambda_vo
    el_ff = cfg.ent_lambda if cfg.ent_lambda_ff is None else cfg.ent_lambda_ff
    tokenizer_mode = cfg.tokenizer.strip().lower()
    if tokenizer_mode not in {"byte", "bpe"}:
        raise RuntimeError("tokenizer must be 'byte' or 'bpe'")

    sched = lr_cosine if cfg.lr_schedule == "cosine" else lr_linear
    warmup_steps = _resolve_warmup_steps(cfg.warmup_steps)
    interactive = False
    if cfg.no_interactive:
        interactive = False
    elif cfg.interactive:
        interactive = True
    else:
        interactive = sys.stdin.isatty()

    #====================
    # data
    #====================
    test_data: torch.Tensor | None = None
    test_raw: bytes | None = None
    if cfg.extended_dataset:
        # extended mode: separate train/val/test directories
        t_dir = Path(cfg.train_dir) if cfg.train_dir else Path(cfg.dataset)
        train_raw = load_bytes(t_dir, cfg.max_bytes, data_format=cfg.data_format, json_text_fields=cfg.json_text_fields)
        if cfg.val_dir is not None:
            val_raw = load_bytes(Path(cfg.val_dir), cfg.max_bytes, data_format=cfg.data_format, json_text_fields=cfg.json_text_fields)
        else:
            val_raw = b""
            val_data = torch.tensor([], dtype=torch.long)
        if cfg.test_dir is not None:
            test_raw = load_bytes(Path(cfg.test_dir), cfg.max_bytes, data_format=cfg.data_format, json_text_fields=cfg.json_text_fields)
    else:
        # simple mode: one file/folder, auto-split into train/val
        raw = load_bytes(Path(cfg.dataset), cfg.max_bytes, data_format=cfg.data_format, json_text_fields=cfg.json_text_fields)
        if tokenizer_mode == "byte":
            data = torch.tensor(list(raw), dtype=torch.long)
            train_data, val_data = split_train_val(data, cfg.seq_len, cfg.val_split_div)
            train_raw = b""
            val_raw = b""
        else:
            train_raw, val_raw = split_train_val_raw(raw, cfg.val_split_div)

    tokenizer_path = Path(cfg.tokenizer_path) if cfg.tokenizer_path else infer_tokenizer_path(cfg.resume)
    if tokenizer_mode == "byte":
        tokenizer = pyqitnn.load_text_tokenizer("byte")
        if cfg.extended_dataset:
            train_data = torch.tensor(tokenizer.encode_bytes(train_raw), dtype=torch.long)
            val_data = torch.tensor(tokenizer.encode_bytes(val_raw), dtype=torch.long)
            if test_raw is not None:
                test_data = torch.tensor(tokenizer.encode_bytes(test_raw), dtype=torch.long)
    else:
        train_text = train_raw.decode("utf-8", errors="replace")
        val_text = val_raw.decode("utf-8", errors="replace")
        train_texts = None if tokenizer_path is not None and tokenizer_path.exists() else [train_text]
        tokenizer = pyqitnn.load_text_tokenizer(
            "bpe",
            path=tokenizer_path,
            train_texts=train_texts,
            vocab_size=cfg.tokenizer_vocab_size,
            min_frequency=cfg.tokenizer_min_frequency,
        )
        train_data = torch.tensor(tokenizer.encode_text(train_text), dtype=torch.long)
        val_data = torch.tensor(tokenizer.encode_text(val_text), dtype=torch.long) if val_text else torch.tensor([], dtype=torch.long)
        if test_raw is not None:
            test_text = test_raw.decode("utf-8", errors="replace")
            test_data = torch.tensor(tokenizer.encode_text(test_text), dtype=torch.long)
        if cfg.tokenizer_path is not None and not Path(cfg.tokenizer_path).exists():
            tokenizer.save(cfg.tokenizer_path)
    tok_bytes_lut = _build_token_byte_lut(tokenizer)

    min_train_tokens = cfg.seq_len + 2
    if train_data.numel() < min_train_tokens:
        raise RuntimeError(
            f"train token stream too small after {tokenizer.kind} tokenization: "
            f"{train_data.numel()} tokens (need >= {min_train_tokens})"
        )
    #====================
    # model
    #====================
    model = pyqitnn.QITNNSimplexTransformerLM(
        vocab_size=tokenizer.vocab_size,
        dim=cfg.dim,
        ffn_dim=cfg.ffn,
        seq_len=cfg.seq_len,
        layers=cfg.layers,
        ent_lambda_qk=el_qk,
        ent_lambda_vo=el_vo,
        ent_lambda_ff=el_ff,
        precision_mode=precision_mode,
        device=device,
        dtype=torch.float32,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    #====================
    # optimizer
    #====================
    if cfg.optimizer == "sgd":
        lr_start = cfg.lr_start
        lr_end = cfg.lr_end
        opt = torch.optim.SGD(
            model.make_qitnn_optimizer_groups(
                lr=lr_start,
                zero_boost_qk=zb_qk,
                zero_boost_vo=zb_vo,
                zero_boost_ff=zb_ff,
                weight_decay=0.0,
            ),
            momentum=0.0,
        )
    else:
        lr_start = cfg.adamw_lr_start
        lr_end = cfg.adamw_lr_end
        qts_ids: set[int] = set()
        qts_main: list[torch.nn.Parameter] = []
        qts_zero_qk: list[torch.nn.Parameter] = []
        qts_zero_vo: list[torch.nn.Parameter] = []
        qts_zero_ff: list[torch.nn.Parameter] = []
        for _, role, layer in model.iter_qitnn_layers():
            for p in (layer.a_neg, layer.a_pos):
                if id(p) not in qts_ids:
                    qts_ids.add(id(p))
                    qts_main.append(p)
            if id(layer.a_zero) not in qts_ids:
                qts_ids.add(id(layer.a_zero))
                if role == "qk":
                    qts_zero_qk.append(layer.a_zero)
                elif role == "vo":
                    qts_zero_vo.append(layer.a_zero)
                else:
                    qts_zero_ff.append(layer.a_zero)
        other = [p for p in model.parameters() if id(p) not in qts_ids]
        groups = [
            {"params": qts_main,    "lr": lr_start,           "lr_scale": 1.0,    "weight_decay": 0.0},
            {"params": qts_zero_qk, "lr": lr_start * zb_qk,  "lr_scale": zb_qk,  "weight_decay": 0.0},
            {"params": qts_zero_vo, "lr": lr_start * zb_vo,  "lr_scale": zb_vo,  "weight_decay": 0.0},
            {"params": qts_zero_ff, "lr": lr_start * zb_ff,  "lr_scale": zb_ff,  "weight_decay": 0.0},
            {"params": other,       "lr": lr_start,           "lr_scale": 1.0,    "weight_decay": cfg.adamw_weight_decay},
        ]
        opt = torch.optim.AdamW(
            [g for g in groups if g["params"]],
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            eps=cfg.adamw_eps,
        )
    #====================
    # resume from checkpoint
    #====================
    start_ep = 1
    global_step = 0
    best_val = None
    best_val_ep = 0
    best_state_cpu = None
    if cfg.resume:
        ckpt = load_ckpt(Path(cfg.resume), model, opt, device)
        start_ep = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val = ckpt.get("best_val_loss", None)
        best_val_ep_raw = ckpt.get("best_val_epoch", 0)
        best_val_ep = 0 if best_val_ep_raw is None else int(best_val_ep_raw)
        best_ckpt_path = infer_best_ckpt_path(cfg.resume)
        if best_ckpt_path is not None:
            best_state_cpu = load_checkpoint_model_state_cpu(best_ckpt_path)
        print(f"resumed from {cfg.resume}  epoch={start_ep - 1}  step={global_step}")
    #====================
    # run directory setup
    #====================
    run_dir: Path | None = None
    saving = not cfg.no_save
    if saving:
        run_name = cfg.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(cfg.save_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_asset = tokenizer.save(run_dir / "tokenizer.json")
        diag_json_path = run_dir / "diagnostics.json"
        diag_csv_path = run_dir / "diagnostics_layers.csv"
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in asdict(cfg).items()}
        config_dict["run_dir"] = str(run_dir)
        config_dict["params"] = n_params
        config_dict["grad_accum_steps"] = grad_accum_steps
        config_dict["optimizer_steps_per_epoch"] = steps_per_ep
        config_dict["optimizer_total_steps"] = total_steps
        config_dict["precision_mode_requested"] = config_dict.get("precision_mode")
        config_dict["precision_mode"] = precision_mode
        config_dict["precision_mode_resolved"] = precision_mode
        config_dict["legacy_mixed_precision_input"] = cfg.mixed_precision
        config_dict["tokenizer_summary"] = tokenizer.summary()
        config_dict["tokenizer_asset"] = str(tokenizer_asset)
        config_dict["diagnostics_json"] = str(diag_json_path)
        config_dict["diagnostics_csv"] = str(diag_csv_path)
        (run_dir / "config.json").write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        diag_json_path = None
        diag_csv_path = None
    #====================
    # csv log
    #====================
    csv_path = cfg.csv_log
    if csv_path is None and run_dir is not None:
        csv_path = str(run_dir / "metrics.csv")
    log = CsvLog(csv_path if saving else None)
    diag_log = QitnnDiagLog(
        str(diag_json_path) if diag_json_path is not None else None,
        str(diag_csv_path) if diag_csv_path is not None else None,
        start_epoch=start_ep,
    )
    #====================
    # print config
    #====================
    print(f"tokenizer     {tokenizer.kind}")
    print(f"vocab_size    {tokenizer.vocab_size}")
    print(f"data_format   {cfg.data_format}")
    print(f"train_tokens  {train_data.numel()}")
    print(f"val_tokens    {val_data.numel()}")
    if test_data is not None:
        print(f"test_tokens   {test_data.numel()}")
    print(f"params        {n_params:,}")
    print(f"optimizer     {cfg.optimizer}")
    print(f"grad_accum    {grad_accum_steps}")
    print(f"mixed_prec    {use_mixed_precision}")
    print(f"precision_mode {precision_mode}")
    print(f"lr            {lr_start} -> {lr_end}  ({cfg.lr_schedule})")
    print(f"seed          {cfg.seed}")
    print(f"seq_len       {cfg.seq_len}")
    if run_dir is not None:
        print(f"run_dir       {run_dir}")
        print(f"csv_log       {csv_path}")
        print(f"save_every    {cfg.save_every}  model_only={cfg.save_model_only}")
    else:
        print(f"saving        disabled (--no-save)")
    #====================
    # training loop
    #====================
    first_loss = None
    first_bpb = None
    last_loss = None
    last_bpb = None
    best_train = None
    best_train_ep = 0
    last_epoch_train_loss = None
    last_epoch_train_bpb = None
    last_epoch_train_ppl = None
    last_epoch_val_loss = None
    last_epoch_val_bpb = None
    last_epoch_val_ppl = None
    last_epoch_train_tok_s = None
    last_epoch_val_tok_s = None
    best_val_bpb = None
    model.train()
    for ep in range(start_ep, epochs + 1):
        ep_n = 0
        ep_tokens = 0
        ep_target_bytes = 0
        ep_total_nll_nats = 0.0
        t0 = time.time()
        for step in range(1, steps_per_ep + 1):
            # global_step is the optimizer-step cursor across resume.
            # Micro-steps remain transient and never cross checkpoint boundaries.
            global_step += 1
            step_idx = global_step - 1
            frac = _schedule_progress(step_idx, total_steps, warmup_steps)
            lr_now = lr_with_warmup(
                lr_start,
                lr_end,
                step_idx,
                total_steps,
                warmup_steps=warmup_steps,
                schedule_fn=sched,
            )
            fm_qk  = sched(fm_qk_s, fm_qk_e, frac)
            fm_vo  = sched(fm_vo_s, fm_vo_e, frac)
            fm_ff  = sched(fm_ff_s, fm_ff_e, frac)
            for g in opt.param_groups:
                g["lr"] = lr_now * float(g.get("lr_scale", 1.0))
            opt.zero_grad(set_to_none=True)
            step_total_nll_nats = 0.0
            step_tokens = 0
            step_target_bytes = 0
            for _ in range(grad_accum_steps):
                x, y = sample_batch(train_data, cfg.batch_size, cfg.seq_len, device)
                _, loss = model(x, targets=y)
                (loss / float(grad_accum_steps)).backward()
                micro_loss = float(loss.detach().cpu())
                micro_tokens = int(y.numel())
                micro_target_bytes = _count_token_bytes(y, tokenizer, tok_bytes_lut)
                step_total_nll_nats += micro_loss * float(micro_tokens)
                step_tokens += micro_tokens
                step_target_bytes += micro_target_bytes
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if cfg.optimizer == "adamw" and cfg.adamw_trit_floor_step is not None:
                base = cfg.adamw_trit_floor_step
                ps_qk = base if cfg.adamw_trit_floor_step_qk is None else cfg.adamw_trit_floor_step_qk
                ps_vo = base if cfg.adamw_trit_floor_step_vo is None else cfg.adamw_trit_floor_step_vo
                ps_ff = base if cfg.adamw_trit_floor_step_ff is None else cfg.adamw_trit_floor_step_ff
            else:
                ps_qk = lr_now * fm_qk
                ps_vo = lr_now * fm_vo
                ps_ff = lr_now * fm_ff
            model.apply_qitnn_prior(
                step_qk=ps_qk,
                step_vo=ps_vo,
                step_ff=ps_ff,
                entropy_floor=cfg.trit_floor_h,
            )
            # Step-level metrics must reflect the full accumulated optimizer step,
            # not the scaled backward loss used inside each micro-step.
            step_loss, step_bpb = summarize_nll_metrics(step_total_nll_nats, step_tokens, step_target_bytes)
            if step_loss is None:
                step_loss = 0.0
            if first_loss is None:
                first_loss = step_loss
            last_loss = step_loss
            if first_bpb is None:
                first_bpb = step_bpb
            last_bpb = step_bpb
            ep_total_nll_nats += step_total_nll_nats
            ep_n += 1
            ep_tokens += step_tokens
            ep_target_bytes += step_target_bytes
            if step % cfg.log_every == 0 or step == steps_per_ep:
                run_train_loss, run_train_bpb = summarize_nll_metrics(ep_total_nll_nats, ep_tokens, ep_target_bytes)
                if run_train_loss is None:
                    run_train_loss = 0.0
                run_train_ppl = loss_to_perplexity(run_train_loss)
                run_train_tok_s = tokens_per_sec(ep_tokens, time.time() - t0)
                print(
                    f"  [{step}] "
                    f"train_loss={run_train_loss:.11f}  "
                    f"train_bpb={fmt_metric(run_train_bpb, 4)}  "
                    f"train_ppl={fmt_metric(run_train_ppl, 4)}  "
                    f"tok/s={fmt_metric(run_train_tok_s, 1)}  "
                    f"lr={lr_now:.8f}"
                )
        #====================
        # epoch end
        #====================
        ep_time = time.time() - t0
        avg_train, train_bpb = summarize_nll_metrics(ep_total_nll_nats, ep_tokens, ep_target_bytes)
        if avg_train is None:
            avg_train = 0.0
        train_ppl = loss_to_perplexity(avg_train)
        train_tok_s = tokens_per_sec(ep_tokens, ep_time)
        last_epoch_train_loss = avg_train
        last_epoch_train_bpb = train_bpb
        last_epoch_train_ppl = train_ppl
        last_epoch_train_tok_s = train_tok_s
        if best_train is None or avg_train < best_train:
            best_train = avg_train
            best_train_ep = ep
        t_val = time.time()
        avg_val, val_bpb, n_val = run_val(
            model,
            val_data,
            cfg.seq_len,
            device,
            cfg.val_steps,
            tokenizer=tokenizer,
            tok_bytes_lut=tok_bytes_lut,
        )
        val_time = time.time() - t_val
        val_loss = avg_val if n_val > 0 else None
        val_ppl = loss_to_perplexity(val_loss)
        val_tok_s = tokens_per_sec(n_val * cfg.seq_len, val_time)
        last_epoch_val_loss = val_loss
        last_epoch_val_bpb = val_bpb if n_val > 0 else None
        last_epoch_val_ppl = val_ppl
        last_epoch_val_tok_s = val_tok_s
        new_best = n_val > 0 and (best_val is None or avg_val < best_val)
        if new_best:
            best_val = avg_val
            best_val_bpb = val_bpb
            best_val_ep = ep
            best_state_cpu = capture_model_state_cpu(model)
        bt = best_train
        bv = best_val
        best_train_ppl = loss_to_perplexity(bt)
        best_val_ppl = loss_to_perplexity(bv)
        print(
            f"epoch {ep}/{epochs}  "
            f"train_loss={avg_train:.11f}  "
            f"train_bpb={fmt_metric(train_bpb, 4)}  "
            f"train_ppl={fmt_metric(train_ppl, 4)}  "
            f"val_loss={fmt_metric(val_loss, 10)}  "
            f"val_bpb={fmt_metric(val_bpb, 4)}  "
            f"val_ppl={fmt_metric(val_ppl, 4)}"
        )
        print(
            f"  steps train={ep_n} val={n_val}  "
            f"tok/s train={fmt_metric(train_tok_s, 1)} val={fmt_metric(val_tok_s, 1)}  "
            f"best_train={fmt_metric(bt, 11)}@{best_train_ep} "
            f"(ppl={fmt_metric(best_train_ppl, 4)})  "
            f"best_val={fmt_metric(bv, 10)}@{best_val_ep} "
            f"(bpb={fmt_metric(best_val_bpb, 4)})  "
            f"(ppl={fmt_metric(best_val_ppl, 4)})  "
            f"time={ep_time:.1f}s"
        )
        diag_snapshot = model.collect_qitnn_diagnostics(epoch=ep, full=(ep % cfg.diag_every == 0))
        for line in format_qitnn_diag_snapshot(diag_snapshot):
            print(line)
        diag_log.snapshot(diag_snapshot)
        log.row(
            ep,
            avg_train,
            train_ppl,
            val_loss,
            val_ppl,
            train_tok_s,
            val_tok_s,
            lr_now,
            ep_time,
            train_bpb=train_bpb,
            val_bpb=val_bpb,
        )
        # Checkpoints are emitted only after completed optimizer steps.
        # No partial accumulation state is serialized or restored.
        # save best checkpoint
        if run_dir is not None and new_best:
            save_ckpt(run_dir / "ckpt_best.pt", model, opt, ep, global_step, best_val,
                      best_val_epoch=best_val_ep,
                      model_only=cfg.save_model_only)
            print(f"  saved best -> {run_dir / 'ckpt_best.pt'}")
        # periodic checkpoint
        if run_dir is not None and cfg.save_every > 0 and ep % cfg.save_every == 0:
            ckpt_path = run_dir / f"ckpt_ep{ep}.pt"
            save_ckpt(ckpt_path, model, opt, ep, global_step, best_val,
                      best_val_epoch=best_val_ep,
                      model_only=cfg.save_model_only)
            print(f"  saved {ckpt_path}")
        # mid-training generation
        if cfg.gen_every > 0 and ep % cfg.gen_every == 0:
            txt = generate_text(
                model, train_data,
                tokenizer=tokenizer,
                prompt=cfg.prompt,
                prompt_units=_prompt_units(cfg, tokenizer),
                gen_units=_gen_units(cfg, tokenizer),
                temperature=cfg.temperature,
                top_k=cfg.top_k, device=device,
            )
            print(f"  [gen@{ep}] {safe_console_text(txt)}")
    #====================
    # final save
    #====================
    if run_dir is not None:
        save_ckpt(run_dir / "ckpt_final.pt", model, opt, epochs, global_step, best_val,
                  best_val_epoch=best_val_ep,
                  model_only=cfg.save_model_only)
        print(f"saved final -> {run_dir / 'ckpt_final.pt'}")
    #====================
    # test evaluation
    #====================
    best_test_loss = None
    best_test_bpb = None
    best_test_ppl = None
    final_test_loss = None
    final_test_bpb = None
    final_test_ppl = None
    final_state_cpu = capture_model_state_cpu(model)
    final_test_metrics = eval_split_metrics(
        model,
        test_data,
        cfg.seq_len,
        device,
        cfg.val_steps,
        tokenizer=tokenizer,
        tok_bytes_lut=tok_bytes_lut,
    )
    best_test_metrics = eval_model_state_metrics(
        model,
        best_state_cpu,
        test_data,
        cfg.seq_len,
        device,
        cfg.val_steps,
        tokenizer=tokenizer,
        tok_bytes_lut=tok_bytes_lut,
        restore_state_cpu=final_state_cpu,
    )
    final_test_windows = int(final_test_metrics["windows"])
    best_test_windows = int(best_test_metrics["windows"])
    if final_test_windows > 0:
        final_test_loss = final_test_metrics["loss"]
        final_test_bpb = final_test_metrics["bpb"]
        final_test_ppl = final_test_metrics["ppl"]
        print(
            f"final_test_loss     {final_test_loss:.10f}  "
            f"final_test_bpb={fmt_metric(final_test_bpb, 4)}  "
            f"final_test_ppl={fmt_metric(final_test_ppl, 4)}  "
            f"({final_test_windows} windows)"
        )
    if best_test_windows > 0:
        best_test_loss = best_test_metrics["loss"]
        best_test_bpb = best_test_metrics["bpb"]
        best_test_ppl = best_test_metrics["ppl"]
        print(
            f"best_test_loss      {best_test_loss:.10f}  "
            f"best_test_bpb={fmt_metric(best_test_bpb, 4)}  "
            f"best_test_ppl={fmt_metric(best_test_ppl, 4)}  "
            f"({best_test_windows} windows)"
        )
    test_loss = final_test_loss
    test_bpb = final_test_bpb
    test_ppl = final_test_ppl
    log.close()
    diag_log.close()
    print("first_loss", first_loss)
    print("first_bpb", first_bpb)
    print("last_loss", last_loss)
    print("last_bpb", last_bpb)
    print("first_ppl", loss_to_perplexity(first_loss))
    print("last_ppl", loss_to_perplexity(last_loss))
    #====================
    # generation
    #====================
    if cfg.prompt or not interactive:
        txt = generate_text(
            model, train_data,
            tokenizer=tokenizer,
            prompt=cfg.prompt,
            prompt_units=_prompt_units(cfg, tokenizer),
            gen_units=_gen_units(cfg, tokenizer),
            temperature=cfg.temperature,
            top_k=cfg.top_k, device=device,
        )
        print("generated_text_begin")
        print(safe_console_text(txt))
        print("generated_text_end")
    #====================
    # interactive mode
    #====================
    if interactive:
        print("interactive_prompt_ready")
        while True:
            try:
                user_input = input("prompt> ")
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            txt = generate_text(
                model, train_data,
                tokenizer=tokenizer,
                prompt=user_input,
                prompt_units=_prompt_units(cfg, tokenizer),
                gen_units=_gen_units(cfg, tokenizer),
                temperature=cfg.temperature,
                top_k=cfg.top_k, device=device,
            )
            print("generated_text_begin")
            print(safe_console_text(txt))
            print("generated_text_end")
    return {
        "first_loss": first_loss,
        "first_bpb": first_bpb,
        "first_ppl": loss_to_perplexity(first_loss),
        "last_loss": last_loss,
        "last_bpb": last_bpb,
        "last_ppl": loss_to_perplexity(last_loss),
        "best_val": best_val,
        "best_val_bpb": best_val_bpb,
        "best_val_ppl": loss_to_perplexity(best_val),
        "best_test_loss": best_test_loss,
        "best_test_bpb": best_test_bpb,
        "best_test_ppl": best_test_ppl,
        "final_test_loss": final_test_loss,
        "final_test_bpb": final_test_bpb,
        "final_test_ppl": final_test_ppl,
        "test_loss": test_loss,
        "test_bpb": test_bpb,
        "test_ppl": test_ppl,
        "last_epoch_train_loss": last_epoch_train_loss,
        "last_epoch_train_bpb": last_epoch_train_bpb,
        "last_epoch_train_ppl": last_epoch_train_ppl,
        "last_epoch_val_loss": last_epoch_val_loss,
        "last_epoch_val_bpb": last_epoch_val_bpb,
        "last_epoch_val_ppl": last_epoch_val_ppl,
        "last_epoch_train_tok_s": last_epoch_train_tok_s,
        "last_epoch_val_tok_s": last_epoch_val_tok_s,
        "run_dir": str(run_dir) if run_dir else None,
        "diagnostics_json": str(diag_json_path) if diag_json_path is not None else None,
        "diagnostics_csv": str(diag_csv_path) if diag_csv_path is not None else None,
        "precision_mode": precision_mode,
        "mixed_precision": use_mixed_precision,
        "grad_accum_steps": grad_accum_steps,
        "optimizer_steps_per_epoch": steps_per_ep,
        "optimizer_total_steps": total_steps,
        "model": model,
        "tokenizer": tokenizer,
    }
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#====================
# CLI
#====================
def _parse_cli(args: list[str] | None = None) -> TrainConfig:
    arg_list = sys.argv[1:] if args is None else list(args)
    p = _build_cli_parser()
    a = p.parse_args(args=arg_list)
    return TrainConfig(**{
        k.replace("-", "_"): v for k, v in vars(a).items()
    })
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if __name__ == "__main__":
    train(_parse_cli())
