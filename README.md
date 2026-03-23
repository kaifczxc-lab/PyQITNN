# PyQITNN

A PyTorch library implementing quantum-inspired ternary neural network layers.
Runs on standard NVIDIA GPUs; no quantum hardware is required.

```python
import torch
import pyqitnn

model = pyqitnn.QITNNSimplexTransformerLM(
    vocab_size=256,
    dim=64,
    ffn_dim=128,
    seq_len=128,
    layers=2,
    precision_mode="fp32",
    device="cuda:0",
)
tokens = torch.randint(0, 256, (2, 128), device="cuda:0")
targets = torch.randint(0, 256, (2, 128), device="cuda:0")
logits, loss = model(tokens, targets=targets)
loss.backward()
```

---

## What this is

Every linear projection stores three amplitude vectors `(a_neg, a_zero, a_pos)` instead of
one weight matrix. A Born-rule normalization converts amplitudes to ternary probabilities
`(P-, P0, P+)`. The result propagates through the network as a full 2D centered simplex
state `[x | y]`, not a collapsed scalar.

This gives the network two independent degrees of freedom per output coordinate - the minimal
complete representation of a ternary probability state.

These are classical amplitudes computed on a GPU. The "quantum-inspired" part is the geometry
and the normalization rule, not the hardware.

---

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.10+ at the source level
- PyTorch 2.0+ with CUDA support
- Local CUDA toolkit when building from source

### Current tested setup

- Windows 11 x86_64
- Python 3.13
- PyTorch 2.10.0+cu126
- Local CUDA toolkit 13.1

The package contains a compiled CUDA extension. Prebuilt wheels are platform- and
Python-version-specific. If no wheel matches your environment, build from source inside
a CUDA-enabled PyTorch environment.

### Install

Install a CUDA-enabled PyTorch build first. Example for CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then install PyQITNN without allowing pip to replace your existing Torch build:

```bash
pip install pyqitnn --no-deps
```

If you want optional BPE/subword tokenizer support:

```bash
pip install pyqitnn[tokenizers] --no-deps
```

### Verify

```python
import pyqitnn

status = pyqitnn.bridge_status()
print(pyqitnn.__version__)        # e.g. 0.3.6
print(status["native_found"])     # True
print(status["native_loadable"])  # True
```

---

## Quickstart

### Single QTS layer

```python
import torch
import pyqitnn

layer = pyqitnn.QITNNLinear(in_dim=64, out_dim=32, device="cuda:0")

x = torch.randn(4, 64, device="cuda:0")
out = layer(x)         # shape: [4, 64] -> packed [x | y] simplex state
print(out.shape)       # torch.Size([4, 64])
```

### Full transformer

```python
import torch
import pyqitnn

model = pyqitnn.QITNNSimplexTransformerLM(
    vocab_size=256,    # byte-level
    dim=64,            # logical feature width
    ffn_dim=128,       # FFN intermediate width
    seq_len=128,       # max sequence length
    layers=2,          # transformer blocks
    precision_mode="fp32",  # explicit trusted baseline
    device="cuda:0",
)

tokens = torch.randint(0, 256, (2, 128), device="cuda:0")
targets = torch.randint(0, 256, (2, 128), device="cuda:0")

logits, loss = model(tokens, targets=targets)
loss.backward()
```

### Training loop

```python
import torch

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(1000):
    opt.zero_grad(set_to_none=True)
    _, loss = model(tokens, targets=targets)
    loss.backward()
    opt.step()

    # Entropy-floor prior. Call only after optimizer.step().
    model.apply_qitnn_prior(
        step_qk=5e-5,
        step_vo=5e-5,
        step_ff=5e-5,
        entropy_floor=1.0840643,
    )
```

### Mixed precision toggle

Core `pyqitnn` modules stay on dense `fp32` unless you opt in explicitly.

```python
model = pyqitnn.QITNNSimplexTransformerLM(
    vocab_size=256,
    dim=64,
    ffn_dim=128,
    seq_len=128,
    layers=2,
    precision_mode="fp32",
    device="cuda:0",
)
```

Set `precision_mode="qts_fp32_rest_bf16"` to enable the conservative mixed path:

- visible activations use CUDA autocast (`bf16`)
- the native extension accepts `bf16/fp16` activations directly; it no longer relies on Python-side `float32` staging for mixed mode
- QITNN master weights stay in `fp32`
- sensitive math stays in `fp32`: Born normalization, backnorm, entropy/prior, and the attention softmax path
- training script / CLI: use `TrainConfig(precision_mode="qts_fp32_rest_bf16")` or `--precision-mode qts_fp32_rest_bf16`
- to force the trusted baseline explicitly from CLI, use `TrainConfig(precision_mode="fp32")`, `--precision-mode fp32`, or legacy `--no-mixed-precision`
- legacy compatibility still exists: `mixed_precision=True` maps to `qts_fp32_rest_bf16`

`BasicQITNN_Transformer.py` currently ships with `TrainConfig.precision_mode="qts_fp32_rest_bf16"` as its standalone trainer default. The lower-level `pyqitnn` modules still default to trusted `fp32` if you omit both `precision_mode` and legacy `mixed_precision`.

### Generation

```python
import torch

prompt = torch.tensor([[72, 101, 108, 108, 111]], device="cuda:0")  # "Hello"
output = model.generate(prompt, max_new_tokens=64, temperature=0.7, top_k=12)

text = bytes(output[0].cpu().tolist()).decode("utf-8", errors="replace")
print(text)
```

### Tokenizer Modes

PyQITNN's QTS math is tokenizer-agnostic. Switching from byte tokens to BPE/subword
tokens does **not** change `forward3`, `backnorm3`, `centered_simplex`, `attention2`,
the 2D simplex state, or the ternary/Born-rule parameterization. It only changes
how raw text is mapped to token ids and what `vocab_size` the embedding/head use.

- `byte`: built in, fixed `vocab_size=256`, no extra dependency
- `bpe`: optional, uses HuggingFace `tokenizers`

When you compare runs across tokenizer choices, use `BPB` (bits per byte) as the
primary metric. `PPL` is still reported, but it is tokenizer-dependent because the
token stream changes. In byte mode the relation is exact:

```text
BPB = loss_nats / ln(2)
PPL = 2 ** BPB
```

```python
import pyqitnn

bpe = pyqitnn.train_bpe_tokenizer(
    ["hello simplex transformer", "born rule ternary attention"],
    vocab_size=320,
    min_frequency=1,
)

model = pyqitnn.QITNNSimplexTransformerLM(
    vocab_size=bpe.vocab_size,
    dim=64,
    ffn_dim=128,
    seq_len=128,
    layers=2,
    precision_mode="fp32",
    device="cuda:0",
)
```

### Trainer data formats

The training script accepts plain text as well as structured JSON corpora.

- `text`: raw file contents
- `json`: parse JSON and extract text fields
- `jsonl` / `ndjson`: parse one JSON record per line
- `auto`: use file extension to choose between text and JSON parsing

For JSON inputs, the trainer can either collect all string leaves recursively or prefer
specific fields such as `text,content`.

---

## Architecture Overview

```text
tokens -> embedding + pos_emb -> [x | y]
                                    |
                          +---------+---------+
                          | QITNNSimplexBlock |  x N layers
                          |                   |
                          |  LN -> Q,K,V (QTS)|
                          |  -> attention2    |
                          |  -> O (QTS)       |
                          |  + residual       |
                          |                   |
                          |  LN -> ff1 (QTS)  |
                          |  -> gelu(x) | y   |
                          |  -> ff2 (QTS)     |
                          |  + residual       |
                          +---------+---------+
                                    |
                          final LN -> head -> logits
```

Every QTS projection replaces a standard `nn.Linear` with three amplitude matrices and
Born-rule normalization. The hidden state is always packed as `[x | y]`, where `x` is the
polarity channel and `y` is the centered zero-state channel.

---

## Core Math

Each QTS projection computes:

```text
C_neg  = input @ A_neg
C_zero = input @ A_zero
C_pos  = input @ A_pos

Z = C_neg^2 + C_zero^2 + C_pos^2

u = (C_pos^2 - C_neg^2) / Z        # polarity, range [-1, +1]
v = C_zero^2 / Z                   # zero-state probability, range [0, 1]
```

Then the centered simplex transform maps `(u, v)` to `(x, y)`:

```text
x = u
y = sqrt(3) * v - 1/sqrt(3)
```

The three pure qutrit states become vertices of an equilateral triangle:

| State  | (u, v)  | (x, y)           |
|--------|---------|------------------|
| \|-1\> | (-1, 0) | (-1, -1/sqrt(3)) |
| \|0\>  | (0, 1)  | (0, 2/sqrt(3))   |
| \|+1\> | (1, 0)  | (1, -1/sqrt(3))  |

Full math derivations are in the [reference](docs/reference.md).

---

## Stability And Training Notes

There are three separate stabilization mechanisms:

### 1. `ent_lambda`

Adds entropy pressure inside the backward path of `forward3`.
Use it when you want the optimization itself to discourage collapsed ternary distributions.

### 2. `prior_()` / `apply_qitnn_prior()`

A post-step entropy-floor correction.
Silent when a triplet is healthy. Only nudges it when entropy drops below the floor.

### 3. Zero-branch learning-rate boost

The zero branch often benefits from a somewhat higher effective learning rate.
The optimizer helpers and training script expose separate handling for `a_zero`.

A healthy training regime is not "perfectly uniform all the time".
The goal is to avoid hard collapse while still allowing the model to specialize.

---

## AdamW Configuration

QTS amplitude parameters need `weight_decay=0`. Standard weight decay fights the
ternary structure and collapses the distribution. Use the trit-floor prior instead.

```python
# Separate QTS params from standard params.
qts_ids = set()
qts_params = []
for _, _, layer in model.iter_qitnn_layers():
    for p in (layer.a_neg, layer.a_zero, layer.a_pos):
        qts_ids.add(id(p))
        qts_params.append(p)

other_params = [p for p in model.parameters() if id(p) not in qts_ids]

opt = torch.optim.AdamW([
    {"params": qts_params, "lr": 3e-4, "weight_decay": 0.0},
    {"params": other_params, "lr": 3e-4, "weight_decay": 0.01},
])
```

---

## Known Limitations

**Hardware:**
- Only `cuda:0` is supported. Multi-GPU requires changes to the CUDA backend.
- Supported precision modes today are `fp32` and `qts_fp32_rest_bf16`.
- `precision_mode="fp32"` keeps the original all-`fp32` path.
- `precision_mode="qts_fp32_rest_bf16"` enables a conservative CUDA `bf16` path for activations while keeping sensitive QITNN math in `fp32`.
- Legacy `mixed_precision=True` is still accepted as a compatibility alias for `qts_fp32_rest_bf16`.
- Do not call `.half()` or `.bfloat16()` on the model. Mixed mode expects fp32 master weights.

**Architecture:**
- Single-head attention only. Multi-head QTS attention is not implemented.
- No dropout. Regularization comes from the entropy prior.
- No gradient checkpointing. Memory scales linearly with layers.
- `seq_len` is fixed at construction time and cannot be changed.
- BPE/subword tokenization is supported at the Python/trainer layer and does not alter the QTS math path.
- Byte mode remains the simplest baseline and the default install path.

**Numerical:**
- cuBLAS GEMM results may differ from `torch.mm` by up to about `1e-2` on large matrices.
  This is expected FP32 accumulation error and does not affect training.
- Attention backward error grows with sequence length due to FP32 accumulation.
  For `seq_len <= 256`, max error is typically below `0.05`.
- Very small `init_std` (`< 1e-5`) can create dead zones where `Z ~ 0` and gradients
  vanish. The default `init_std=0.02` avoids this.
- `prior_()` modifies tensors in-place. Call it only after `optimizer.step()` and
  outside any autograd context.

**Platform:**
- Primary development is on Windows. Linux builds are less exercised.
- macOS is not supported because CUDA is required.

---

## Tests

The repository ships a stress test covering correctness, stability, and convergence:

```bash
python stress_test.py
```

This checks Born-rule invariants, finite-difference gradient correctness, attention
forward/backward vs PyTorch SDPA, prior effectiveness, checkpoint roundtrip,
determinism, memory stability, and more.

---

## Links

- [Full API Reference](https://github.com/kaifczxc-lab/PyQITNN/blob/SiritoriProjects/docs/reference.md)
- [Basic QITNN Transformer](https://github.com/kaifczxc-lab/PyQITNN/blob/SiritoriProjects/BasicQITNN_Transformer.py)
- [GitHub Repository](https://github.com/kaifczxc-lab/PyQITNN)
- [QITNN Architecture Analysis](https://github.com/kaifczxc-lab/qitnn/blob/SiritoriProjects/Analysis-QITNN.md)
- [Original Devlog (Discord, GPU Mode)](https://discord.com/channels/1189498204333543425/1466534042768904356/1476227907327098931)

---

## Disclaimer

This is an experimental library implementing a novel neural network architecture.
The core math, architecture design, debugging, and system integration are the author's
original work, developed with AI assistance for implementation.

The CUDA kernels are optimized for NVIDIA consumer GPUs; RTX 3060 Ti was the primary
development target. They work on other architectures but have not been extensively
benchmarked outside that hardware.

No guarantees of correctness, performance, or suitability for production use.
Constructive feedback is welcome.
