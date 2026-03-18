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

### Verify

```python
import pyqitnn

status = pyqitnn.bridge_status()
print(pyqitnn.__version__)        # e.g. 0.3.2
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

### Generation

```python
import torch

prompt = torch.tensor([[72, 101, 108, 108, 111]], device="cuda:0")  # "Hello"
output = model.generate(prompt, max_new_tokens=64, temperature=0.7, top_k=12)

text = bytes(output[0].cpu().tolist()).decode("utf-8", errors="replace")
print(text)
```

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
- FP32 only. The Born-rule division is numerically sensitive; FP16 would cause instabilities.

**Architecture:**
- Single-head attention only. Multi-head QTS attention is not implemented.
- No dropout. Regularization comes from the entropy prior.
- No gradient checkpointing. Memory scales linearly with layers.
- `seq_len` is fixed at construction time and cannot be changed.
- Byte-level vocabulary (256 tokens). No BPE or subword tokenizer.

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
