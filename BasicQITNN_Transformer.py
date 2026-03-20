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
    max_bytes: int = 50_000_000       # max bytes to load per data source
    data_format: str = "auto"         # "auto", "text", "json", "jsonl"
    json_text_fields: str | None = None   # comma-separated preferred text fields, e.g. "text,content"
    tokenizer: str = "bpe"           # "byte" or "bpe"
    tokenizer_path: str | Path | None = None
    tokenizer_vocab_size: int = 4096
    tokenizer_min_frequency: int = 2

    #====================
    # model
    #====================
    dim: int = 64
    ffn: int = 128
    layers: int = 2
    seq_len: int = 256

    #====================
    # training
    #====================
    device: str = "cuda:0"
    seed: int = 7
    batch_size: int = 4
    steps: int | None = None         # if set, overrides epochs/steps_per_epoch
    epochs: int = 20
    steps_per_epoch: int = 2000
    grad_clip: float = 1.0
    #====================
    # optimizer
    # "adamw" and "SGD"
    #====================
    optimizer: str = "adamw"
    lr_start: float = 0.005          # SGD only
    lr_end: float = 0.001            # SGD only
    lr_schedule: str = "cosine"      # "cosine" or "linear"
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
    gen_every: int = 0               # generate sample every N epochs
    #====================
    # logging
    #====================
    log_every: int = 100             # print loss every N steps
    diag_every: int = 10             # full QTS diagnostics every N epochs
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
    x = torch.empty((batch, seq_len), dtype=torch.long)
    y = torch.empty((batch, seq_len), dtype=torch.long)
    for i, s in enumerate(starts.tolist()):
        chunk = data[s : s + seq_len + 1]
        x[i] = chunk[:-1]
        y[i] = chunk[1:]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
#====================
# validation
#====================
@torch.no_grad()
def run_val(model, val_data: torch.Tensor, seq_len: int, device: torch.device, max_steps: int):
    model.eval()
    n_win = max((val_data.numel() - 1) // seq_len, 0)
    if n_win < 1:
        model.train()
        return 0.0, 0
    if 0 < max_steps < n_win:
        n_win = max_steps
    elif max_steps == 0 and n_win > 500:
        print(f"  [warn] val has {n_win} windows, this will be slow. set val_steps=50 to cap it.")
    total = 0.0
    for i in range(n_win):
        s = i * seq_len
        chunk = val_data[s : s + seq_len + 1]
        xv = chunk[:-1].unsqueeze(0).to(device, non_blocking=True)
        yv = chunk[1:].unsqueeze(0).to(device, non_blocking=True)
        _, loss = model(xv, targets=yv)
        total += float(loss.detach().cpu())
    model.train()
    return total / n_win, n_win
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


def loss_to_perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    if math.isnan(loss):
        return float("nan")
    # cross-entropy is in nats, so LM perplexity is exp(loss)
    if loss >= 80.0:
        return float("inf")
    return math.exp(loss)


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
# checkpoints
#====================
def save_ckpt(path: Path, model, opt, epoch: int, step: int, best_val,
              *, model_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "best_val_loss": best_val,
    }
    if not model_only:
        payload["optimizer"] = opt.state_dict()
    torch.save(payload, str(path))
def load_ckpt(path: Path, model, opt, device):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
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
        ])
        self._file.flush()
    def close(self):
        if self._file is not None:
            self._file.close()
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device(cfg.device)
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
    interactive = False
    if cfg.no_interactive:
        interactive = False
    elif cfg.interactive:
        interactive = True
    else:
        interactive = sys.stdin.isatty()

    epochs = cfg.epochs
    steps_per_ep = cfg.steps_per_epoch
    if cfg.steps is not None:
        epochs = 1
        steps_per_ep = cfg.steps
    total_steps = max(epochs * steps_per_ep, 1)
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
    if cfg.resume:
        ckpt = load_ckpt(Path(cfg.resume), model, opt, device)
        start_ep = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val = ckpt.get("best_val_loss", None)
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
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in asdict(cfg).items()}
        config_dict["run_dir"] = str(run_dir)
        config_dict["params"] = n_params
        config_dict["tokenizer_summary"] = tokenizer.summary()
        config_dict["tokenizer_asset"] = str(tokenizer_asset)
        (run_dir / "config.json").write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    #====================
    # csv log
    #====================
    csv_path = cfg.csv_log
    if csv_path is None and run_dir is not None:
        csv_path = str(run_dir / "metrics.csv")
    log = CsvLog(csv_path if saving else None)
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
    print(f"lr            {lr_start} -> {lr_end}  ({cfg.lr_schedule})")
    print(f"seed          {cfg.seed}")
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
    last_loss = None
    best_train = None
    best_train_ep = 0
    last_epoch_train_loss = None
    last_epoch_train_ppl = None
    last_epoch_val_loss = None
    last_epoch_val_ppl = None
    last_epoch_train_tok_s = None
    last_epoch_val_tok_s = None
    model.train()
    for ep in range(start_ep, epochs + 1):
        ep_loss = 0.0
        ep_n = 0
        ep_tokens = 0
        t0 = time.time()
        for step in range(1, steps_per_ep + 1):
            global_step += 1
            frac = 0.0 if total_steps <= 1 else (global_step - 1) / (total_steps - 1)
            lr_now = sched(lr_start, lr_end, frac)
            fm_qk  = sched(fm_qk_s, fm_qk_e, frac)
            fm_vo  = sched(fm_vo_s, fm_vo_e, frac)
            fm_ff  = sched(fm_ff_s, fm_ff_e, frac)
            for g in opt.param_groups:
                g["lr"] = lr_now * float(g.get("lr_scale", 1.0))
            x, y = sample_batch(train_data, cfg.batch_size, cfg.seq_len, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y)
            loss.backward()
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
            train_loss = float(loss.detach().cpu())
            if first_loss is None:
                first_loss = train_loss
            last_loss = train_loss
            ep_loss += train_loss
            ep_n += 1
            ep_tokens += int(y.numel())
            if step % cfg.log_every == 0 or step == steps_per_ep:
                run_train_loss = ep_loss / ep_n
                run_train_ppl = loss_to_perplexity(run_train_loss)
                run_train_tok_s = tokens_per_sec(ep_tokens, time.time() - t0)
                print(
                    f"  [{step}] "
                    f"train_loss={run_train_loss:.11f}  "
                    f"train_ppl={fmt_metric(run_train_ppl, 4)}  "
                    f"tok/s={fmt_metric(run_train_tok_s, 1)}  "
                    f"lr={lr_now:.8f}"
                )
        #====================
        # epoch end
        #====================
        ep_time = time.time() - t0
        avg_train = ep_loss / max(ep_n, 1)
        train_ppl = loss_to_perplexity(avg_train)
        train_tok_s = tokens_per_sec(ep_tokens, ep_time)
        last_epoch_train_loss = avg_train
        last_epoch_train_ppl = train_ppl
        last_epoch_train_tok_s = train_tok_s
        if best_train is None or avg_train < best_train:
            best_train = avg_train
            best_train_ep = ep
        t_val = time.time()
        avg_val, n_val = run_val(model, val_data, cfg.seq_len, device, cfg.val_steps)
        val_time = time.time() - t_val
        val_loss = avg_val if n_val > 0 else None
        val_ppl = loss_to_perplexity(val_loss)
        val_tok_s = tokens_per_sec(n_val * cfg.seq_len, val_time)
        last_epoch_val_loss = val_loss
        last_epoch_val_ppl = val_ppl
        last_epoch_val_tok_s = val_tok_s
        new_best = n_val > 0 and (best_val is None or avg_val < best_val)
        if new_best:
            best_val = avg_val
            best_val_ep = ep
        bt = best_train
        bv = best_val
        best_train_ppl = loss_to_perplexity(bt)
        best_val_ppl = loss_to_perplexity(bv)
        print(
            f"epoch {ep}/{epochs}  "
            f"train_loss={avg_train:.11f}  "
            f"train_ppl={fmt_metric(train_ppl, 4)}  "
            f"val_loss={fmt_metric(val_loss, 10)}  "
            f"val_ppl={fmt_metric(val_ppl, 4)}"
        )
        print(
            f"  steps train={ep_n} val={n_val}  "
            f"tok/s train={fmt_metric(train_tok_s, 1)} val={fmt_metric(val_tok_s, 1)}  "
            f"best_train={fmt_metric(bt, 11)}@{best_train_ep} "
            f"(ppl={fmt_metric(best_train_ppl, 4)})  "
            f"best_val={fmt_metric(bv, 10)}@{best_val_ep} "
            f"(ppl={fmt_metric(best_val_ppl, 4)})  "
            f"time={ep_time:.1f}s"
        )
        for line in model.format_qitnn_diagnostics(epoch=ep, full=(ep % cfg.diag_every == 0)):
            print(line)
        log.row(ep, avg_train, train_ppl, val_loss, val_ppl, train_tok_s, val_tok_s, lr_now, ep_time)
        # save best checkpoint
        if run_dir is not None and new_best:
            save_ckpt(run_dir / "ckpt_best.pt", model, opt, ep, global_step, best_val,
                      model_only=cfg.save_model_only)
            print(f"  saved best -> {run_dir / 'ckpt_best.pt'}")
        # periodic checkpoint
        if run_dir is not None and cfg.save_every > 0 and ep % cfg.save_every == 0:
            ckpt_path = run_dir / f"ckpt_ep{ep}.pt"
            save_ckpt(ckpt_path, model, opt, ep, global_step, best_val,
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
                  model_only=cfg.save_model_only)
        print(f"saved final -> {run_dir / 'ckpt_final.pt'}")
    #====================
    # test evaluation
    #====================
    test_loss = None
    test_ppl = None
    if test_data is not None and test_data.numel() > cfg.seq_len + 1:
        avg_test, n_test = run_val(model, test_data, cfg.seq_len, device, cfg.val_steps)
        if n_test > 0:
            test_loss = avg_test
            test_ppl = loss_to_perplexity(test_loss)
            print(
                f"test_loss     {test_loss:.10f}  "
                f"test_ppl={fmt_metric(test_ppl, 4)}  "
                f"({n_test} windows)"
            )
    log.close()
    print("first_loss", first_loss)
    print("last_loss", last_loss)
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
        "first_ppl": loss_to_perplexity(first_loss),
        "last_loss": last_loss,
        "last_ppl": loss_to_perplexity(last_loss),
        "best_val": best_val,
        "best_val_ppl": loss_to_perplexity(best_val),
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "last_epoch_train_loss": last_epoch_train_loss,
        "last_epoch_train_ppl": last_epoch_train_ppl,
        "last_epoch_val_loss": last_epoch_val_loss,
        "last_epoch_val_ppl": last_epoch_val_ppl,
        "last_epoch_train_tok_s": last_epoch_train_tok_s,
        "last_epoch_val_tok_s": last_epoch_val_tok_s,
        "run_dir": str(run_dir) if run_dir else None,
        "model": model,
        "tokenizer": tokenizer,
    }
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#====================
# CLI
#====================
def _parse_cli() -> TrainConfig:
    D = TrainConfig()
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
    p.add_argument("--steps",            type=int,   default=None)
    p.add_argument("--epochs",           type=int,   default=D.epochs)
    p.add_argument("--steps-per-epoch",  type=int,   default=D.steps_per_epoch)
    p.add_argument("--grad-clip",        type=float, default=D.grad_clip)
    # optimizer
    p.add_argument("--optimizer",    type=str,   default=D.optimizer, choices=("sgd", "adamw"))
    p.add_argument("--lr-start",     type=float, default=D.lr_start)
    p.add_argument("--lr-end",       type=float, default=D.lr_end)
    p.add_argument("--lr-schedule",  type=str,   default=D.lr_schedule, choices=("linear", "cosine"))
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

    a = p.parse_args()
    return TrainConfig(**{
        k.replace("-", "_"): v for k, v in vars(a).items()
    })
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if __name__ == "__main__":
    train(_parse_cli())
