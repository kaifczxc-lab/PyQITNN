# PyQITNN API Reference

Complete reference for the `pyqitnn` package. Every public function, class, parameter,
and return value is documented here.

---

## Package overview

```python
import pyqitnn
```

On import, `pyqitnn` calls `prepare_runtime()` which adds CUDA and Torch library
directories to the system DLL search path. This is a one-time side effect.

PyQITNN exposes the following public symbols:

| Symbol | Type | Description |
|--------|------|-------------|
| `forward3` | function | Born-rule ternary projection with autograd |
| `centered_simplex` | function | Centered simplex transform with autograd |
| `attention2` | function | 2D/3D simplex-aware causal attention with autograd |
| `prior_` | function | In-place entropy-floor prior on amplitude triplets |
| `QITNNLinear` | nn.Module | Ternary Born-rule linear layer |
| `QITNNSimplexTransformerLM` | nn.Module | Full transformer language model |
| `bridge_status` | function | Runtime status diagnostics |
| `prepare_runtime` | function | Initialize library search paths |
| `qitnn_diag_stats` | function | Compute per-layer ternary statistics |
| `format_qitnn_diag` | function | Format statistics as printable lines |
| `render_qitnn_diag` | function | Compute + format in one call |
| `short_qitnn_label` | function | Abbreviate layer names for display |
| `__version__` | str | Package version string |

**Requirements:**

- NVIDIA GPU (CUDA, device `cuda:0` only)
- PyTorch >= 2.0 with CUDA support
- Core library default path is full `float32`
- Canonical mixed mode is `precision_mode="qts_fp32_rest_bf16"`
- Legacy `mixed_precision=True` is still accepted as a compatibility alias; prefer `precision_mode`
- Windows or Linux
- Python >= 3.10

---

## Core operations

### forward3

```python
pyqitnn.forward3(
    inp: torch.Tensor,       # [M, in_dim], float32 by default; fp16/bf16 allowed when precision_mode="qts_fp32_rest_bf16"
    a_neg: torch.Tensor,     # [in_dim, out_dim], float32, CUDA, contiguous
    a_zero: torch.Tensor,    # [in_dim, out_dim], float32, CUDA, contiguous
    a_pos: torch.Tensor,     # [in_dim, out_dim], float32, CUDA, contiguous
    *,
    ent_lambda: float = 0.0,
    mixed_precision: bool | None = None,   # legacy compatibility alias
    precision_mode: str | None = None,     # "fp32" | "qts_fp32_rest_bf16"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

The fundamental QTS operation. Computes the full Born-rule ternary projection
from input activations and three amplitude matrices.

**Forward pass:**

```
C_neg  = inp @ a_neg
C_zero = inp @ a_zero
C_pos  = inp @ a_pos

Z = C_neg^2 + C_zero^2 + C_pos^2

u = (C_pos^2 - C_neg^2) / Z     where Z > 1e-12, else 0
v = C_zero^2 / Z                 where Z > 1e-12, else 0
```

**Returns:** `(u, v, cn, cz, cp)` where:

| Return | Shape | Description |
|--------|-------|-------------|
| `u` | [M, out_dim] | Polarity channel. Range [-1, +1]. Equals `P_pos - P_neg`. |
| `v` | [M, out_dim] | Zero-state channel. Range [0, 1]. Equals `P_zero`. |
| `cn` | [M, out_dim] | Raw pre-normalization channel `inp @ a_neg`. |
| `cz` | [M, out_dim] | Raw pre-normalization channel `inp @ a_zero`. |
| `cp` | [M, out_dim] | Raw pre-normalization channel `inp @ a_pos`. |

**Backward pass:**

Gradients flow through both `u` and `v`. The backward kernel (`backnorm3`) computes
analytical derivatives of the Born-rule normalization:

```
Given upstream gradients du, dv and saved cn, cz, cp:

a = cn, b = cz, c = cp
a2 = a^2, b2 = b^2, c2 = c^2
z = a2 + b2 + c2
iz2 = 1 / z^2

dcn = -2a * iz2 * (du*(b2 + 2*c2) + dv*b2)
dcz =  2b * iz2 * (du*(a2 - c2) + dv*(a2 + c2))
dcp =  2c * iz2 * (du*(2*a2 + b2) - dv*b2)
```

When `dv = 0`, these reduce to the original single-channel backward formulas from
the Serenade kernel. The generalization to two channels is what makes the full 2D
simplex state trainable.

After backnorm, standard matrix multiply backward computes:

```
grad_inp  = dcn @ a_neg^T + dcz @ a_zero^T + dcp @ a_pos^T
grad_neg  = inp^T @ dcn
grad_zero = inp^T @ dcz
grad_pos  = inp^T @ dcp
```

**Entropy regularization:**

When `ent_lambda > 0`, the backward kernel adds an entropy gradient that pushes
the ternary distribution toward uniform `(1/3, 1/3, 1/3)`. This is a soft
in-graph regularizer, separate from the `prior_()` hard-floor mechanism.

The entropy gradient for each element:

```
P_neg = a^2/z,  P_zero = b^2/z,  P_pos = c^2/z

dcn -= ent_lambda * 2a * (b2*(log(P_zero) - log(P_neg)) + c2*(log(P_pos) - log(P_neg))) / z^2
dcz -= ent_lambda * 2b * (a2*(log(P_neg) - log(P_zero)) + c2*(log(P_pos) - log(P_zero))) / z^2
dcp -= ent_lambda * 2c * (a2*(log(P_neg) - log(P_pos)) + b2*(log(P_zero) - log(P_pos))) / z^2
```

**Constraints:**

- All tensors must be 2D, CUDA, contiguous, on device cuda:0.
- Default path is full `float32`.
- With `precision_mode="qts_fp32_rest_bf16"` or legacy `mixed_precision=True`, `inp` may be `float16` or `bfloat16`, but QITNN weights stay `float32` and the Born-rule math still computes in `float32`.
- `inp.size(1)` must equal `a_neg.size(0)`.
- `a_neg`, `a_zero`, `a_pos` must have the same shape.

**Example:**

```python
inp = torch.randn(32, 64, device="cuda:0")
a_neg = torch.randn(64, 32, device="cuda:0", requires_grad=True)
a_zero = torch.randn(64, 32, device="cuda:0", requires_grad=True)
a_pos = torch.randn(64, 32, device="cuda:0", requires_grad=True)

u, v, cn, cz, cp = pyqitnn.forward3(inp, a_neg, a_zero, a_pos, ent_lambda=0.01)

# u is the polarity signal, v is the zero-state probability
# cn, cz, cp are the raw channels (useful for diagnostics)
loss = u.sum() + v.sum()
loss.backward()

# a_neg.grad, a_zero.grad, a_pos.grad are now populated
```

---

### centered_simplex

```python
pyqitnn.centered_simplex(
    u: torch.Tensor,    # [M, D], float32 by default; fp16/bf16 allowed when precision_mode="qts_fp32_rest_bf16"
    v: torch.Tensor,    # [M, D], float32 by default; fp16/bf16 allowed when precision_mode="qts_fp32_rest_bf16"
    *,
    mixed_precision: bool | None = None,   # legacy compatibility alias
    precision_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

Maps the raw Born-rule outputs `(u, v)` to the centered simplex basis `(x, y)`.

**Forward:**

```
x = u
y = sqrt(3) * v - 1/sqrt(3)
```

where `sqrt(3) = 1.7320508075688772` and `1/sqrt(3) = 0.5773502691896258`.

**Backward:**

```
grad_u = grad_x
grad_v = sqrt(3) * grad_y
```

The `sqrt(3)` factor is the key gradient scaling. Without it, the y-channel
receives attenuated gradients and trains slower than the x-channel.

**Why this transform:**

A ternary probability state `(P-, P0, P+)` with `P- + P0 + P+ = 1` has two
degrees of freedom. The centered simplex basis maps these to an equilateral
triangle in 2D:

| Pure state | (u, v) | (x, y) |
|------------|--------|--------|
| \|-1\> (P-=1) | (-1, 0) | (-1, -0.5774) |
| \|0\> (P0=1) | (0, 1) | (0, 1.1547) |
| \|+1\> (P+=1) | (1, 0) | (1, -0.5774) |

All three vertices are equidistant (distance = 2). The centroid (uniform
distribution P=1/3 each) maps to the origin (0, 0).

**Returns:** `(x, y)` with the same shape as input.

**Example:**

```python
u = torch.tensor([[1.0], [-1.0], [0.0]], device="cuda:0")
v = torch.tensor([[0.0], [0.0], [1.0]], device="cuda:0")

x, y = pyqitnn.centered_simplex(u, v)
# x: [[ 1.0], [-1.0], [0.0]]
# y: [[-0.5774], [-0.5774], [1.1547]]
```

---

### attention2

```python
pyqitnn.attention2(
    q: torch.Tensor,    # [S, D] or [B, S, D], float32 by default; fp16/bf16 allowed when precision_mode="qts_fp32_rest_bf16"
    k: torch.Tensor,    # same shape as q
    v: torch.Tensor,    # same shape as q
    *,
    mixed_precision: bool | None = None,   # legacy compatibility alias
    precision_mode: str | None = None,
) -> torch.Tensor       # same shape as q
```

Simplex-aware causal self-attention. Operates on packed `[x | y]` tensors where
the first half of the last dimension is the x-channel and the second half is
the y-channel.

**Forward:**

The packed inputs are split:

```
qx, qy = q[..., :D//2], q[..., D//2:]
kx, ky = k[..., :D//2], k[..., D//2:]
vx, vy = v[..., :D//2], v[..., D//2:]
```

Attention scores use both channels:

```
score[i,j] = (qx[i] . kx[j] + qy[i] . ky[j]) / sqrt(D)
```

with causal masking (j > i -> -inf) and softmax normalization.

Output is computed for both channels independently using the same attention weights:

```
ox[i] = sum_j  attn[i,j] * vx[j]
oy[i] = sum_j  attn[i,j] * vy[j]
```

The result is packed back as `[ox | oy]`.

**Batched mode:**

When input is 3D `[B, S, D]`, attention is computed independently for each
batch element through the native batched bridge/CUDA path.

**Backward:**

Full analytical gradients for `dq`, `dk`, `dv` through both channels. The
backward kernel handles the causal mask and softmax Jacobian.

**Constraints:**

- Last dimension must be even (packed x|y format).
- q, k, v must have identical shapes.
- 2D input: `[seq_len, dim]`. 3D input: `[batch, seq_len, dim]`.

**Numerical notes:**

FP32 accumulation error in the attention backward grows with sequence length.
For `seq_len <= 32`, max error vs PyTorch SDPA is typically < 0.01.
For `seq_len = 128`, max error may reach 0.05. This is inherent to the
FP32 kernel and does not affect training convergence.

**Example:**

```python
S, D = 64, 128  # D must be even (64 for x-channel, 64 for y-channel)
q = torch.randn(S, D, device="cuda:0", requires_grad=True)
k = torch.randn(S, D, device="cuda:0", requires_grad=True)
v = torch.randn(S, D, device="cuda:0", requires_grad=True)

out = pyqitnn.attention2(q, k, v)  # [64, 128]
out.sum().backward()
# q.grad, k.grad, v.grad are populated
```

---

### prior_

```python
pyqitnn.prior_(
    a_neg: torch.Tensor,     # [*, *], float32 by default; fp16/bf16 allowed when precision_mode="qts_fp32_rest_bf16"
    a_zero: torch.Tensor,    # same shape as a_neg, MODIFIED IN-PLACE
    a_pos: torch.Tensor,     # same shape as a_neg, MODIFIED IN-PLACE
    *,
    step: float,
    entropy_floor: float,
    mixed_precision: bool | None = None,   # legacy compatibility alias
    precision_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

In-place entropy-floor prior. For each element, computes the Shannon entropy
of the ternary distribution. If entropy is below `entropy_floor`, nudges the
amplitudes toward uniform and renormalizes.

**Mechanism:**

For each triplet `(a, b, c)`:

```
P_neg = a^2 / (a^2 + b^2 + c^2)
P_zero = b^2 / (a^2 + b^2 + c^2)
P_pos = c^2 / (a^2 + b^2 + c^2)

H = -P_neg*log2(P_neg) - P_zero*log2(P_zero) - P_pos*log2(P_pos)
```

If `H < entropy_floor`:
- Each amplitude is nudged toward `sqrt(1/3)` by `step`.
- The triplet is then renormalized to preserve overall scale.

If `H >= entropy_floor`, the triplet is untouched.

**The recommended entropy floor** is `1.0840643`, which corresponds to `log2(3) * 2/3`.
This keeps the distribution moderately spread without forcing full uniformity.

**When to call:**

Always call `prior_()` after `optimizer.step()` and never inside a gradient
computation context. It modifies tensors in-place and is not differentiable.

```python
opt.step()
model.apply_qitnn_prior(
    step_qk=5e-5, step_vo=5e-5, step_ff=5e-5,
    entropy_floor=1.0840643,
)
```

**Why not use weight decay instead:**

Standard weight decay pushes all amplitudes toward zero uniformly, which does
not preserve the ternary structure. The entropy-floor prior specifically targets
collapsed distributions (where one channel dominates) and is silent when the
distribution is healthy. This makes it compatible with AdamW as long as
`weight_decay=0` is set for the QTS amplitude parameters.

**Returns:** the same three tensors (modified in-place).

---

## Modules

### QITNNLinear

```python
pyqitnn.QITNNLinear(
    in_dim: int,
    out_dim: int,
    *,
    ent_lambda: float = 0.0,
    init_std: float = 0.02,
    pack_output: bool = True,
    centered_simplex: bool = True,
    return_triplet: bool = False,
    mixed_precision: bool | None = None,   # legacy compatibility alias
    precision_mode: str | None = None,     # "fp32" | "qts_fp32_rest_bf16"
    device=None,
    dtype=torch.float32,
)
```

A single ternary Born-rule linear layer. Replaces `nn.Linear` with three
amplitude matrices and Born normalization.

**Important:** all inputs must be CUDA tensors. If you create the layer with
`device=None` (default), PyTorch puts the parameters on CPU. You must call
`.cuda()` or `.to("cuda:0")` before the first forward pass, otherwise you get
`RuntimeError: QITNNLinear requires CUDA tensors`.

**Parameters:**

| Parameter | Type | Default | What it does | What happens when you change it |
|-----------|------|---------|--------------|-------------------------------|
| `in_dim` | int | required | Input feature dimension. | Must match the last dim of your input tensor. |
| `out_dim` | int | required | Output feature dimension (logical). | The actual output width is `2 * out_dim` when `pack_output=True` because of the x\|y packing. If you set `out_dim=32`, forward output is `[M, 64]`. |
| `ent_lambda` | float | 0.0 | Entropy regularization strength in the backward pass. | `0.0` = off, no entropy gradient. `0.001` = gentle push toward uniform. `0.01` = noticeable push. `0.1+` = strong, will fight the task gradient. |
| `init_std` | float | 0.02 | Standard deviation for normal initialization of all three amplitude matrices. | `0.02` = safe default, gives near-uniform P-/P0/P+. Don't go below `1e-5` -- amplitudes near zero cause dead gradients (Z -> 0). |
| `pack_output` | bool | True | Pack `[x, y]` into one tensor. | `True`: output is `[M, 2*out_dim]`, one tensor. `False`: output is `(x, y)`, two separate tensors of `[M, out_dim]` each. Pack mode is what the transformer expects. |
| `centered_simplex` | bool | True | Apply the centered simplex transform. | `True`: convert `(u,v)` -> `(x,y)` via `x=u, y=sqrt(3)*v - 1/sqrt(3)`. This gives equilateral triangle geometry. `False`: return raw Born-rule `(u,v)` directly. |
| `return_triplet` | bool | False | Also return the raw channels. | `True`: forward returns `(output, (cn, cz, cp))` -- useful for diagnostics. `False`: forward returns just the output. |
| `mixed_precision` | bool or None | None | Legacy compatibility alias. | `True` maps to `precision_mode="qts_fp32_rest_bf16"`. `False` maps to `precision_mode="fp32"`. Leave `None` when you use `precision_mode` directly. |
| `precision_mode` | str or None | None | Canonical precision selector. | Supported modes are `"fp32"` and `"qts_fp32_rest_bf16"`. The mixed mode keeps QITNN master weights and sensitive Born-rule math in `fp32` while allowing visible activations on the conservative CUDA mixed path. |
| `device` | | None | Device for parameter allocation. | `None` puts params on CPU (you must `.cuda()` later). `"cuda:0"` puts them on GPU directly. |
| `dtype` | | float32 | Data type for parameters. | Keep this at `float32`. Mixed mode expects fp32 master weights; do not call `.half()` or `.bfloat16()` on QITNN parameters. |

**Learnable parameters:**

| Name | Shape | Description |
|------|-------|-------------|
| `a_neg` | [in_dim, out_dim] | Amplitude matrix for the \|-1\> branch. |
| `a_zero` | [in_dim, out_dim] | Amplitude matrix for the \|0\> branch. |
| `a_pos` | [in_dim, out_dim] | Amplitude matrix for the \|+1\> branch. |

Note: there is **no bias**. The Born-rule normalization makes a traditional bias
term meaningless (it would be absorbed into the amplitude scale).

**Forward methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `forward(inp)` | depends on flags | Main forward. See below. |
| `forward_raw(inp)` | `(u, v, cn, cz, cp)` | Raw Born-rule output, no simplex transform. |
| `forward_visible(inp)` | `(x, y, cn, cz, cp)` or `(u, v, cn, cz, cp)` | With or without simplex, depending on `centered_simplex` flag. |

**forward() return value depends on flags:**

| `pack_output` | `return_triplet` | Return type |
|---------------|------------------|-------------|
| True | False | `Tensor [M, 2*out_dim]` |
| True | True | `(Tensor [M, 2*out_dim], (cn, cz, cp))` |
| False | False | `(Tensor [M, out_dim], Tensor [M, out_dim])` |
| False | True | `((left, right), (cn, cz, cp))` |

**Initialization:**

All three amplitude matrices are initialized with `N(0, init_std)`. With
`init_std=0.02`, the resulting ternary distribution starts near-uniform
`(P- ~ P0 ~ P+ ~ 1/3)` because all squared amplitudes have similar expected values.

Do not use very small `init_std` (< 1e-5). When all amplitudes are near zero,
`Z = C_neg^2 + C_zero^2 + C_pos^2` becomes tiny and the division produces
numerically unstable results. The kernel clamps `Z > 1e-12` to avoid division
by zero, but gradients still effectively vanish.

**Example:**

```python
layer = pyqitnn.QITNNLinear(64, 32, ent_lambda=0.01, device="cuda:0")
x = torch.randn(8, 64, device="cuda:0")

# default: packed centered simplex
out = layer(x)            # shape [8, 64]

# raw Born-rule output
u, v, cn, cz, cp = layer.forward_raw(x)

# with diagnostic triplets
layer2 = pyqitnn.QITNNLinear(64, 32, return_triplet=True, device="cuda:0")
packed, (cn, cz, cp) = layer2(x)
```

---

### QITNNSimplexTransformerLM

```python
pyqitnn.QITNNSimplexTransformerLM(
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
    mixed_precision: bool | None = None,   # legacy compatibility alias
    precision_mode: str | None = None,     # "fp32" | "qts_fp32_rest_bf16"
    device=None,
    dtype=torch.float32,
)
```

Complete byte-level autoregressive transformer using QTS projections.

**Parameters:**

| Parameter | Type | Default | What it does | What happens when you change it |
|-----------|------|---------|--------------|-------------------------------|
| `vocab_size` | int | 256 | Vocabulary size. | Default 256 for byte-level. Change only if you're doing something non-standard (e.g. BPE tokenizer). |
| `dim` | int | 64 | Logical feature width. | The packed hidden dimension is `2 * dim`. `dim=64` -> 128 hidden. `dim=384` -> 768 hidden (14.7M params with 2 layers). Bigger = more capacity, more memory, slower. |
| `ffn_dim` | int | 128 | FFN intermediate width (logical). | Packed FFN width is `2 * ffn_dim`. Standard ratio is `ffn_dim = 2 * dim`. |
| `seq_len` | int | 128 | Maximum sequence length. | Positional embeddings are allocated for this length. Longer sequences cost O(seq_len^2) memory in attention. Can't be changed after model creation. |
| `layers` | int | 2 | Number of transformer blocks. | Each block has 6 QITNNLinear layers (Q, K, V, O, FF1, FF2). More layers = deeper model, more params. |
| `ent_lambda_qk` | float | 0.0 | Entropy regularization for Q and K projections. | `0.0` = off. `0.001` = gentle. This is per-role: Q and K share one value. |
| `ent_lambda_vo` | float | 0.0 | Entropy regularization for V and O projections. | Same logic as `ent_lambda_qk`. |
| `ent_lambda_ff` | float | 0.0 | Entropy regularization for FF1 and FF2 projections. | Same logic. |
| `init_std` | float | 0.02 | Standard deviation for all parameter initialization. | Applies to amplitude matrices, embeddings, and output head. |
| `mixed_precision` | bool or None | None | Legacy compatibility alias. | `True` maps to `precision_mode="qts_fp32_rest_bf16"`. `False` maps to `precision_mode="fp32"`. |
| `precision_mode` | str or None | None | Canonical precision selector. | Supported modes are `"fp32"` and `"qts_fp32_rest_bf16"`. The mixed mode keeps visible activations on the conservative CUDA `bf16` path while preserving `fp32` master weights and sensitive math. |

**Internal dimensions:**

```
hidden_dim = dim * 2         # packed [x | y] width
visible_dim = dim * 2        # Q/K/V/O output width (same as hidden)
ff_visible_dim = ffn_dim * 2 # FF1 output width
```

The hidden state is always `[batch, seq_len, hidden_dim]` where `hidden_dim = 2 * dim`.

**Architecture per block:**

```
1. LayerNorm(hidden_dim)
2. Q = QITNNLinear(hidden_dim -> dim)    # output: [B, S, 2*dim]
3. K = QITNNLinear(hidden_dim -> dim)
4. V = QITNNLinear(hidden_dim -> dim)
5. attn_out = attention2(Q, K, V)         # causal, [B, S, 2*dim]
6. O = QITNNLinear(2*dim -> dim)          # output: [B, S, 2*dim]
7. hidden = hidden + O                    # residual
8. LayerNorm(hidden_dim)
9. ff_mid = QITNNLinear(hidden_dim -> ffn_dim)  # output: [B, S, 2*ffn_dim]
10. ff_mid = gelu(ff_mid_x) | ff_mid_y          # simplex gelu (only x-channel)
11. ff_out = QITNNLinear(2*ffn_dim -> dim)       # output: [B, S, 2*dim]
12. hidden = hidden + ff_out                     # residual
```

Step 10 is `simplex_gelu`: GELU is applied only to the x-channel (first half). The
y-channel (second half) passes through unchanged. This preserves the simplex geometry
of the y-channel while adding nonlinearity to x.

**forward:**

```python
model.forward(
    tokens: torch.Tensor,              # [batch, seq_len], long
    *,
    targets: torch.Tensor | None = None,  # [batch, seq_len], long
) -> tuple[torch.Tensor, torch.Tensor | None]
```

Returns `(logits, loss)`. If `targets` is None, `loss` is None.

`logits` shape: `[batch, seq_len, vocab_size]`.

**iter_qitnn_layers:**

```python
model.iter_qitnn_layers() -> Iterator[tuple[str, str, QITNNLinear]]
```

Yields `(name, role, layer)` for every QTS projection in the model.

- `name`: full path like `"blocks.0.q_proj"`, `"blocks.1.ff2"`.
- `role`: one of `"qk"`, `"vo"`, `"ff"`. Used for per-role hyperparameters.
- `layer`: the `QITNNLinear` instance.

**make_qitnn_optimizer_groups:**

```python
model.make_qitnn_optimizer_groups(
    *,
    lr: float,
    zero_boost_qk: float = 1.0,
    zero_boost_vo: float = 1.0,
    zero_boost_ff: float = 1.0,
    weight_decay: float = 0.0,
) -> list[dict]
```

Creates parameter groups for SGD with per-role learning rate scaling for the
`a_zero` parameter. The zero-state channel often needs a higher learning rate
to stay active during training.

Returns a list of dicts compatible with `torch.optim.SGD(groups)`.

Four groups are created:
1. `main_params` (a_neg, a_pos) — base LR, no weight decay
2. `qk_zero` — LR * zero_boost_qk
3. `vo_zero` — LR * zero_boost_vo
4. `ff_zero` — LR * zero_boost_ff

Note: The training script creates equivalent groups for AdamW manually, also
respecting per-role zero-boost. See the training section below.

**apply_qitnn_prior:**

```python
model.apply_qitnn_prior(
    *,
    step_qk: float = 0.0,
    step_vo: float = 0.0,
    step_ff: float = 0.0,
    entropy_floor: float = 0.0,
) -> None
```

Applies the entropy-floor prior to all QTS layers. Each role (qk, vo, ff) gets
its own step size.

Call this after `optimizer.step()`. Never inside autograd.

**format_qitnn_diagnostics:**

```python
model.collect_qitnn_diagnostics(
    *,
    epoch: int,
    full: bool = False,
) -> dict[str, object]
```

Returns the raw structured snapshot for the selected QTS layers.

The returned payload has a stable schema:
- `schema_version`
- `epoch`
- `full`
- `layer_count`
- `layers` (list of `{name, label, role, stats}`)

When `full=False`, only a representative subset of layers is included (last block's
FFN, first and last block's V/O). When `full=True`, all QTS layers are included.

This is the same payload used by the trainer to write `diagnostics.json` and
`diagnostics_layers.csv`.

```python
model.format_qitnn_diagnostics(
    *,
    epoch: int,
    full: bool = False,
) -> list[str]
```

Returns formatted diagnostic lines for QTS layers.

This method formats the raw snapshot from `collect_qitnn_diagnostics()`. It does
not compute a separate diagnostics path.

**generate:**

```python
model.generate(
    tokens: torch.Tensor,       # [batch, prompt_len], long
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 16,
    ascii_guard: bool = True,
) -> torch.Tensor               # [batch, prompt_len + max_new_tokens]
```

Autoregressive byte-level generation. Appends tokens one at a time.

When `ascii_guard=True`, non-printable bytes (except NUL, TAB, LF, CR) are
masked out before sampling. This keeps generated text readable.

When `temperature=0`, greedy decoding is used (argmax).

---

## Diagnostics

### qitnn_diag_stats

```python
pyqitnn.qitnn_diag_stats(
    a_neg: torch.Tensor,
    a_zero: torch.Tensor,
    a_pos: torch.Tensor,
) -> dict[str, float]
```

Computes comprehensive statistics for one set of amplitude matrices.
All computation is done in float64 for precision.

**Returned keys:**

| Key | Description |
|-----|-------------|
| `count` | Number of elements with `Z > 1e-20`. |
| `p_neg` | Mean P- across all elements. |
| `p_zero` | Mean P0 across all elements. |
| `p_pos` | Mean P+ across all elements. |
| `h` | Mean Shannon entropy (bits). Max is `log2(3) = 1.585`. |
| `eff` | Effective number of states: `2^h`. Range [1, 3]. |
| `collapse_pct` | Percentage of elements where `max(P-, P0, P+) > 0.9`. |
| `sum_n` | Sum of all a_neg values. |
| `sum_z` | Sum of all a_zero values. |
| `sum_p` | Sum of all a_pos values. |
| `rms_n` | RMS of a_neg. |
| `rms_z` | RMS of a_zero. |
| `rms_p` | RMS of a_pos. |
| `p0_gt_04_pct` | Percentage of elements where P0 > 0.4. |
| `p0_lt_01_pct` | Percentage of elements where P0 < 0.1. |
| `maxp_gt_08_pct` | Percentage of elements where max(P) > 0.8. |
| `h_gt_13_pct` | Percentage of elements where H > 1.3 bits. |
| `var_h` | Variance of per-element entropy. |

**Healthy values after training:**

- `h` should be 1.0-1.5 (not too collapsed, not stuck at uniform).
- `eff` should be 2.0-2.8.
- `collapse_pct` should be < 5%.
- `p_neg`, `p_zero`, `p_pos` should all be > 0.1.

**Warning signs:**

- `eff` stuck at ~1.83 throughout training: distribution is collapsing to binary (two states dominating, third dying). Try increasing `adamw_trit_floor_step` or adding `ent_lambda`.
- `eff` stuck at ~2.99: nothing is specializing, model can't differentiate channels. Try reducing `ent_lambda` or `trit_floor_h`.
- `collapse_pct` > 20%: too many elements have one dominant channel. Prior step is too small.

**Example:**

```python
stats = pyqitnn.qitnn_diag_stats(layer.a_neg, layer.a_zero, layer.a_pos)
print(f"entropy: {stats['h']:.4f}, effective states: {stats['eff']:.2f}")
print(f"collapse: {stats['collapse_pct']:.1f}%")
```

---

### format_qitnn_diag

```python
pyqitnn.format_qitnn_diag(
    label: str,
    stats: dict[str, float],
    *,
    epoch: int | None = None,
) -> list[str]
```

Formats a stats dict (from `qitnn_diag_stats`) into printable lines.

Returns 4 lines:
1. Label and epoch.
2. Probabilities, entropy, effective states, collapse rate.
3. Amplitude sums and RMS values.
4. Distribution tail statistics.

---

### render_qitnn_diag

```python
pyqitnn.render_qitnn_diag(
    name: str,
    a_neg: torch.Tensor,
    a_zero: torch.Tensor,
    a_pos: torch.Tensor,
    *,
    epoch: int | None = None,
) -> list[str]
```

Convenience function. Computes stats and formats them in one call.
Uses `short_qitnn_label(name)` to abbreviate the layer name.

---

### short_qitnn_label

```python
pyqitnn.short_qitnn_label(name: str) -> str
```

Converts full layer paths to short display labels.

```python
pyqitnn.short_qitnn_label("blocks.0.q_proj")  # -> "L0 wq"
pyqitnn.short_qitnn_label("blocks.1.ff2")     # -> "L1 ff2"
pyqitnn.short_qitnn_label("blocks.0.v_proj")  # -> "L0 wv"
pyqitnn.short_qitnn_label("blocks.0.o_proj")  # -> "L0 wo"
```

---

## Bridge utilities

### prepare_runtime

```python
pyqitnn.prepare_runtime() -> dict[str, object]
```

Adds CUDA and Torch library directories to the system search path.
Called automatically when the package is imported (`import pyqitnn`).

On Windows, uses `os.add_dll_directory()` to register:
- The Torch `lib/` directory (contains cublas, cudnn, etc.)
- The CUDA Toolkit `bin/x64/` directory when present (falls back to `bin/`)

On Linux, prepends to `PATH` and `LD_LIBRARY_PATH`.

Returns a dict with the resolved paths:

```python
{
    "torch_lib_dir": ".../site-packages/torch/lib",
    "cuda_bin_dir": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/x64",
}
```

### bridge_status

```python
pyqitnn.bridge_status() -> dict[str, object]
```

Returns diagnostic information about the compiled extension. Actually attempts
to load the native module (not just check for the file).

```python
{
    "stage": "python-wrapper",
    "native_found": True,         # was _C.pyd/.so found on disk?
    "native_loadable": True,      # did it import successfully?
    "native_module_path": "C:/path/to/pyqitnn/_C.cp313-win_amd64.pyd",
    "load_error": None,           # error string if import failed
}
```

If `native_found` is True but `native_loadable` is False, check `load_error` --
common causes are CUDA version mismatch or missing DLLs.

If `native_found` is False, the CUDA extension was not compiled. Re-run
`build_win.bat` from a repo checkout, or `pip install .` from the `torch_bridge/`
directory.

---

## Training: `TrainConfig` and `train()`

The training pipeline is in `BasicQITNN_Transformer.py`. It can be used from
Python code or from the command line.

All settings are defined in the `TrainConfig` dataclass. Every field has a default
value. You only need to override what you want to change.

### Usage from Python

The training script is not installed as a package. It lives alongside the
`pyqitnn` package in the `torch_bridge/` directory.

```python
import sys
sys.path.insert(0, "path/to/torch_bridge")
from BasicQITNN_Transformer import train, TrainConfig
```

Or just run it from the `torch_bridge/` directory:

```python
# if your working directory is torch_bridge/
from BasicQITNN_Transformer import train, TrainConfig

# option 1: dataclass with named fields
cfg = TrainConfig(
    dataset="my_data/corpus.txt",
    optimizer="adamw",
    adamw_lr_start=3e-4,
    adamw_lr_end=3e-5,
    adamw_trit_floor_step=5e-5,
    lr_schedule="cosine",
    warmup_steps=200,
    epochs=100,
    steps_per_epoch=1000,
    grad_clip=1.0,
    gen_every=20,
    no_interactive=True,
)
result = train(cfg)

# option 2: keyword arguments directly
result = train(dataset="my_data/corpus.txt", optimizer="adamw", epochs=100)

# option 3: separate train/val/test directories
result = train(
    extended_dataset=True,
    train_dir="data/train",
    val_dir="data/val",
    test_dir="data/test",
    optimizer="adamw",
    epochs=100,
)
```

`train()` returns a dict:

`loss` values here are cross-entropy in nats per target token. `BPB` is bits per target
byte and is the recommended metric when you want to compare byte-tokenized and
BPE-tokenized runs on the same corpus. `PPL` is still reported, but it remains
tokenizer-dependent. In byte mode:

```text
BPB = loss / ln(2)
PPL = 2 ** BPB
```

| Key | Type | Description |
|-----|------|-------------|
| `first_loss` | float | Training loss of the very first step. |
| `first_bpb` | float or None | Bits per byte of the very first step, computed from the target stream. |
| `first_ppl` | float or None | Perplexity of the very first step (`exp(first_loss)`). |
| `last_loss` | float | Training loss of the last step. |
| `last_bpb` | float or None | Bits per byte of the last training step. |
| `last_ppl` | float or None | Perplexity of the last step (`exp(last_loss)`). |
| `best_val` | float or None | Best validation loss seen during training. |
| `best_val_bpb` | float or None | BPB corresponding to `best_val`. |
| `best_val_ppl` | float or None | Perplexity corresponding to `best_val`. |
| `best_test_loss` | float or None | Test loss of the best-validation checkpoint, evaluated at the end when test data is available. The returned `model` still remains the final in-memory model. |
| `best_test_bpb` | float or None | Test BPB of the best-validation checkpoint. |
| `best_test_ppl` | float or None | Test perplexity of the best-validation checkpoint. |
| `final_test_loss` | float or None | Test loss of the final model state at the end of training. |
| `final_test_bpb` | float or None | Test BPB of the final model state. |
| `final_test_ppl` | float or None | Test perplexity of the final model state. |
| `test_loss` | float or None | Backward-compatible alias of `final_test_loss`. |
| `test_bpb` | float or None | Backward-compatible alias of `final_test_bpb`. |
| `test_ppl` | float or None | Backward-compatible alias of `final_test_ppl`. |
| `last_epoch_train_loss` | float or None | Average train loss of the final epoch. |
| `last_epoch_train_bpb` | float or None | Average train BPB of the final epoch. |
| `last_epoch_train_ppl` | float or None | Average train perplexity of the final epoch. |
| `last_epoch_val_loss` | float or None | Average validation loss of the final epoch, or `None` when validation has no windows. |
| `last_epoch_val_bpb` | float or None | Average validation BPB of the final epoch. |
| `last_epoch_val_ppl` | float or None | Average validation perplexity of the final epoch. |
| `last_epoch_train_tok_s` | float or None | Training throughput in tokens/sec for the final epoch. |
| `last_epoch_val_tok_s` | float or None | Validation throughput in tokens/sec for the final epoch. |
| `run_dir` | str or None | Path to the run directory with checkpoints and logs. None if `no_save=True`. |
| `diagnostics_json` | str or None | Path to the saved structured layer-diagnostics JSON artifact, or `None` when saving is disabled. |
| `diagnostics_csv` | str or None | Path to the saved layer-level diagnostics CSV artifact, or `None` when saving is disabled. |
| `model` | QITNNSimplexTransformerLM | The trained model instance on GPU. It is returned as-is; call `model.eval()` yourself before inference if you want eval mode. |

### Usage from CLI

```bash
cd torch_bridge
python BasicQITNN_Transformer.py --optimizer adamw --epochs 100 --dataset my_data/corpus.txt
```

With separate data splits:

```bash
python BasicQITNN_Transformer.py \
    --extended-dataset \
    --train-dir data/train \
    --val-dir data/val \
    --test-dir data/test \
    --optimizer adamw \
    --epochs 100
```

All `TrainConfig` fields map to CLI flags with dashes instead of underscores
(e.g. `adamw_lr_start` -> `--adamw-lr-start`). Boolean fields use `--flag-name`
to set True (e.g. `--extended-dataset`, `--no-save`, `--no-interactive`).

---

### TrainConfig reference

Every field is documented below with its type, default value, what it does,
and what happens when you change it.

#### Data

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | str or Path | `r"YourPathToDataset"` | Path to a file or directory with training data. If it's a file, that file is used directly. If it's a directory, all files inside are concatenated as raw bytes. When `extended_dataset=False` (default), the data is auto-split into train/val by `val_split_div`. |
| `extended_dataset` | bool | `False` | Data loading mode switch. `False` = simple mode: load `dataset`, auto-split into train/val. `True` = extended mode: use `train_dir`/`val_dir`/`test_dir` separately, no auto-split. |
| `train_dir` | str, Path, or None | `None` | Path to training data (extended mode only). If None in extended mode, falls back to `dataset`. |
| `val_dir` | str, Path, or None | `None` | Path to validation data (extended mode only). If None in extended mode, validation is skipped (empty tensor). |
| `test_dir` | str, Path, or None | `None` | Path to test data (extended mode only). If None, no test evaluation. If set, the trainer evaluates both the final model state and the best-validation checkpoint on the test split, returning `final_test_*` and `best_test_*`. The legacy `test_*` keys remain as backward-compatible aliases of `final_test_*`. |
| `max_bytes` | int | `50_000_000` | Maximum bytes to load from each data source. If a file or directory is larger, loading stops at this limit. Applies separately to train, val, and test. |

**Two data modes:**

1. **Simple mode** (`extended_dataset=False`, default): set `dataset` to your file
   or directory. The last `1/val_split_div` fraction becomes validation automatically.

   ```python
   train(dataset="my_corpus.txt")  # 90% train, 10% val
   ```

2. **Extended mode** (`extended_dataset=True`): set `train_dir`, `val_dir`, `test_dir`
   separately. No auto-splitting. You control exactly what goes where.

   ```python
   train(
       extended_dataset=True,
       train_dir="data/train",
       val_dir="data/val",
       test_dir="data/test",
   )
   ```

#### Model architecture

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `dim` | int | `256` | Logical feature width. | Hidden dimension is `2 * dim`. `dim=64` -> small model (~0.6M params). `dim=256` -> medium (~6.5M). `dim=512` -> large (~26M). More = better quality but needs more data and time. |
| `ffn` | int | `512` | FFN intermediate dimension (logical). | Actual FFN width is `2 * ffn`. Standard ratio: `ffn = 2 * dim`. Wider FFN = more capacity per block. |
| `layers` | int | `2` | Number of transformer blocks. | Each block has 6 QITNNLinear layers (Q, K, V, O, FF1, FF2). `layers=2` = 12 QTS layers total. `layers=4` = 24. More layers = deeper but slower. |
| `seq_len` | int | `256` | Maximum sequence length in tokens (bytes). | Positional embeddings are allocated for this length. Attention cost is O(seq_len^2). `128` for quick experiments, `256-512` for production. |

#### Training

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `device` | str | `"cuda:0"` | PyTorch device string. | Only `cuda:0` is supported. The CUDA extension checks `get_device() == 0`. |
| `seed` | int | `7` | Random seed. | Sets both `torch.manual_seed` and `torch.cuda.manual_seed_all`. Same seed = same training trajectory. |
| `batch_size` | int | `4` | Sequences per micro-batch. | This is the per-micro-batch memory knob. Effective batch = `batch_size * grad_accum_steps`. Increase `batch_size` directly when VRAM allows it; otherwise keep it smaller and use accumulation. |
| `grad_accum_steps` | int | `1` | Micro-batches per optimizer step. | `1` keeps the legacy trainer contract exactly. `N > 1` accumulates `N` micro-batches before one `optimizer.step()`. This increases effective batch without changing optimizer-step counters: `steps`, `steps_per_epoch`, `warmup_steps`, `global_step`, prior, checkpoints, and resume all remain measured in optimizer steps. |
| `epochs` | int | `20` | Number of training epochs. | Each epoch runs `steps_per_epoch` steps. Ignored if `steps` is set. `5` for quick tests, `50-100` for real training. |
| `steps_per_epoch` | int | `1000` | Optimizer steps per epoch. | Total optimizer steps = `epochs * steps_per_epoch`. Each optimizer step may contain multiple micro-batches when `grad_accum_steps > 1`. |
| `steps` | int or None | `None` | Total optimizer-step override. | When set, overrides `epochs` and `steps_per_epoch`. Runs exactly N optimizer steps in a single "epoch". Micro-batch count still scales with `grad_accum_steps`. |
| `grad_clip` | float | `1.0` | Maximum gradient norm. | `0.0` = no clipping. `1.0` = recommended for AdamW, prevents gradient explosions. Values like `0.5` are more aggressive. |
| `precision_mode` | str or None | `None` | Canonical trainer precision selector. | Supported modes are `"fp32"` and `"qts_fp32_rest_bf16"`. When omitted, the standalone trainer resolves this field to the conservative mixed path `qts_fp32_rest_bf16`. Use `precision_mode="fp32"` to force the trusted baseline explicitly. |
| `mixed_precision` | bool or None | `None` | Legacy compatibility alias. | `True` maps to `precision_mode="qts_fp32_rest_bf16"`. `False` maps to `precision_mode="fp32"`. This field is kept only for backward compatibility with older launch scripts, is hidden from CLI help, and should not be the primary product knob. |

#### Optimizer

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `optimizer` | str | `"adamw"` | Which optimizer. | `"adamw"` = recommended, adaptive LR per parameter. `"sgd"` = simpler but needs more tuning. AdamW uses its own LR settings (see below), SGD uses `lr_start`/`lr_end`. |
| `lr_start` | float | `0.005` | Starting LR for SGD. | Ignored when `optimizer="adamw"`. |
| `lr_end` | float | `0.001` | Final LR for SGD. | Ignored when `optimizer="adamw"`. |
| `lr_schedule` | str | `"cosine"` | LR decay curve. | `"cosine"` = half-cosine, slow start/end, fast middle. Better for longer runs. `"linear"` = straight line from start to end. Applies to both SGD and AdamW LR. |
| `warmup_steps` | int | `0` | Linear LR warmup length in optimizer steps. | `0` preserves the legacy schedule exactly. `N > 0` ramps LR from `0` to the configured start LR over the first `N` optimizer steps, then the selected `lr_schedule` begins its normal decay toward the end LR. `grad_accum_steps` does not rescale this counter. If `warmup_steps` exceeds the full run, the whole run becomes a clean ramp to the start LR. |

#### AdamW settings

These only apply when `optimizer="adamw"`:

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `adamw_lr_start` | float | `3e-4` | Starting LR for AdamW. | `3e-4` is a safe general default. `1e-4` for larger models (dim > 256). `1e-3` can work for tiny models but risks instability. |
| `adamw_lr_end` | float | `3e-5` | Final LR for AdamW. | Decays from `adamw_lr_start` to this over training. `3e-5` = 10x decay. Set equal to `adamw_lr_start` for constant LR. |
| `adamw_beta1` | float | `0.9` | First moment decay. | Standard value. Don't change unless you know what you're doing. |
| `adamw_beta2` | float | `0.95` | Second moment decay. | `0.95` is slightly more aggressive than PyTorch's default `0.999`. Helps with QTS training stability. |
| `adamw_eps` | float | `1e-8` | Numerical stability epsilon. | Standard value. |
| `adamw_weight_decay` | float | `0.01` | Weight decay for non-QTS params. | Applied to embeddings, layernorm, output head. QTS amplitude params always get `weight_decay=0` (standard decay destroys ternary structure). |
| `adamw_trit_floor_step` | float or None | `5e-5` | Fixed prior step size for AdamW. | When set, overrides `lr * trit_floor_mul` for the prior. `5e-5` is recommended. AdamW's LR is small (~3e-4), so `lr * mul` gives tiny prior steps that can't fix collapse. `None` = use `lr * trit_floor_mul` formula (often too small). |
| `adamw_trit_floor_step_qk` | float or None | `None` | Per-role override for Q/K. | If None, uses `adamw_trit_floor_step`. Set separately if Q/K layers need different prior strength. |
| `adamw_trit_floor_step_vo` | float or None | `None` | Per-role override for V/O. | Same logic. |
| `adamw_trit_floor_step_ff` | float or None | `None` | Per-role override for FF1/FF2. | Same logic. |

#### Zero-boost

The `a_zero` amplitude matrix often needs a higher learning rate than `a_neg` and `a_pos`
to keep the zero-state channel active during training. Zero-boost multiplies the LR
for `a_zero` parameters.

This works for both SGD and AdamW. With SGD, the model's
`make_qitnn_optimizer_groups()` creates the groups. With AdamW, the training script
creates equivalent per-role groups manually.

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `zero_boost` | float | `3.5` | LR multiplier for all `a_zero` parameters. | `1.0` = no boost, `a_zero` trains at same speed as `a_neg`/`a_pos`. `3.5` = 3.5x faster. Higher values push the zero channel harder but can cause instability. |
| `zero_boost_qk` | float or None | `None` | Override for Q/K projections. | If None, uses `zero_boost`. Set to a different value if Q/K layers need different zero-boost (rare). |
| `zero_boost_vo` | float or None | `None` | Override for V/O projections. | Same logic. |
| `zero_boost_ff` | float or None | `None` | Override for FF1/FF2 projections. | Same logic. |

#### Trit-floor prior

The entropy-floor prior prevents ternary distribution collapse. After each optimizer
step, it checks every amplitude triplet's Shannon entropy. If entropy is below the
floor, it nudges the triplet toward uniform and renormalizes.

For SGD, the prior step size is `lr * trit_floor_mul` (decays with LR).
For AdamW, use `adamw_trit_floor_step` instead (fixed step, recommended).

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `trit_floor_h` | float | `1.0840643` | Entropy floor threshold (bits). | Equals `log2(3) * 2/3`. Triplets with entropy above this are untouched. Below = nudged toward uniform. Lower = allows more specialization. Higher = forces more spread. Max possible is `log2(3) = 1.585`. |
| `trit_floor_mul_start` | float | `0.01` | Prior step multiplier at training start (SGD). | Actual step = `lr * mul`. Only matters for SGD; AdamW uses `adamw_trit_floor_step` instead. |
| `trit_floor_mul_end` | float | `0.001` | Prior step multiplier at training end (SGD). | Decays from start to end following LR schedule. |
| `trit_floor_mul_qk_start` | float or None | `None` | Per-role override for Q/K. | If None, uses `trit_floor_mul_start`. |
| `trit_floor_mul_qk_end` | float or None | `None` | Per-role override. | If None, uses `trit_floor_mul_end`. |
| `trit_floor_mul_vo_start` | float or None | `None` | Per-role override for V/O. | Same. |
| `trit_floor_mul_vo_end` | float or None | `None` | Per-role override. | Same. |
| `trit_floor_mul_ff_start` | float or None | `None` | Per-role override for FF1/FF2. | Same. |
| `trit_floor_mul_ff_end` | float or None | `None` | Per-role override. | Same. |

#### Entropy regularization (in-graph)

A separate, soft entropy penalty that works inside the backward pass. Unlike the
prior (which is post-step and hard-floor), this adds a gradient that continuously
pushes the distribution toward uniform. Can be combined with the prior.

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `ent_lambda` | float | `0.001` | Entropy regularization strength. | `0.0` = off. `0.001` = gentle push, doesn't hurt learning. `0.01` = noticeable, can slow convergence on short runs. `0.1+` = too strong, fights task gradient and degrades loss. |
| `ent_lambda_qk` | float or None | `None` | Override for Q/K projections. | If None, uses `ent_lambda`. |
| `ent_lambda_vo` | float or None | `None` | Override for V/O projections. | Same. |
| `ent_lambda_ff` | float or None | `None` | Override for FF1/FF2 projections. | Same. |

**Interaction between prior and ent_lambda:**

The prior (`trit_floor_h` + step) and `ent_lambda` both fight distribution collapse,
but differently:
- **Prior** = hard floor. Only activates when entropy drops below threshold. Post-step, not differentiable.
- **ent_lambda** = soft gradient. Always active (when > 0), pushes toward uniform during backward. Differentiable.

For short training (5-10 epochs), use only the prior (`ent_lambda=0` or very small).
For long training (50+ epochs), combining both works well.

#### Validation

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `val_split_div` | int | `10` | Fraction of data for validation. | `10` = 10% validation, 90% training. `5` = 20% validation. Only used in simple mode (`extended_dataset=False`). In extended mode, you provide `val_dir` directly. |
| `val_steps` | int | `50` | Max validation windows. | `0` = evaluate all available windows. Set to e.g. `50` to cap validation time on large datasets. Each window is `seq_len` tokens. |

#### Generation

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `temperature` | float | `0.65` | Sampling temperature. | `0.0` = greedy argmax (most deterministic). `0.5` = fairly conservative. `0.65` = good default for readable byte-level text. `1.0` = full entropy, more random. `> 1.0` = very random. |
| `top_k` | int | `12` | Top-k sampling filter. | Only the top K most probable tokens are sampled from. `12` = conservative. `50` = more diverse. `0` = no filtering (sample from full distribution). |
| `gen_bytes` | int | `160` | Bytes to generate per sample. | How many bytes the model generates after the prompt. |
| `prompt` | str | `""` | Prompt text for generation. | If empty, the first `prompt_bytes` bytes of training data are used as the prompt. If set, this text (encoded as UTF-8) is the prompt. |
| `prompt_bytes` | int | `64` | Max bytes from the prompt. | Truncates the prompt to this many bytes before feeding to the model. |
| `gen_every` | int | `0` | Generate a sample every N epochs. | `0` = only generate at the end. `5` = generate every 5 epochs (useful for monitoring quality during training). |

#### Logging

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `log_every` | int | `100` | Print training metrics every N steps. | `100` = print at steps 100, 200, etc. within each epoch. Logs include `train_loss`, `train_bpb`, `train_ppl`, throughput, and LR. Lower = more verbose output. |
| `diag_every` | int | `10` | Full QTS diagnostics every N epochs. | `10` = show all-layer entropy/probability stats every 10 epochs. On other epochs, only a summary subset is printed. `1` = every epoch (verbose). |
| `csv_log` | str or None | `None` | Path to epoch-level CSV log file. | `None` = auto-creates `metrics.csv` inside the run directory (if saving is enabled). The CSV stores `train_loss`, `train_bpb`, `train_ppl`, `val_loss`, `val_bpb`, `val_ppl`, throughput, LR, and time. Raw layer diagnostics are written separately to `diagnostics_layers.csv`. Set `csv_log` to a custom path like `"logs/experiment.csv"` only for the epoch-level metric log. |

#### Saving

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `save_dir` | str | `"runs"` | Base directory for runs. | Each run creates a subdirectory inside this. |
| `run_name` | str or None | `None` | Subdirectory name. | `None` = auto-generates `run_YYYYMMDD_HHMMSS`. Set to e.g. `"experiment_1"` for a memorable name. Full path: `{save_dir}/{run_name}/`. |
| `no_save` | bool | `False` | Disable all file output. | `True` = no checkpoints, no `metrics.csv`, no `diagnostics.json`, no `diagnostics_layers.csv`, and no `config.json`. The model still trains, you just get it back in memory only. Good for quick tests. |
| `save_every` | int | `25` | Periodic checkpoint every N epochs. | `25` = save at epochs 25, 50, etc. `0` = only save best and final. Files are named `ckpt_ep{N}.pt`. |
| `save_model_only` | bool | `False` | Slim checkpoints. | `True` = checkpoints contain only model weights (smaller files, can't resume training). `False` = checkpoints include optimizer state too (needed for `resume`). |
| `resume` | str or None | `None` | Resume from checkpoint. | Path to a `.pt` file. Loads model weights, optimizer state, epoch/global_step cursors, best-validation metadata, and RNG state. Training continues from the next completed optimizer-step boundary without replay drift. |

**What gets saved in a run directory:**

```
runs/run_20260317_143022/
  config.json        # all TrainConfig values + param count
  metrics.csv        # epoch, train_loss, train_ppl, val_loss, val_ppl, train_tok_s, val_tok_s, lr, time_s, train_bpb, val_bpb
  diagnostics.json   # per-epoch structured QTS layer snapshots
  diagnostics_layers.csv  # one raw layer row per epoch snapshot
  ckpt_best.pt       # best model by val_loss (auto-updated)
  ckpt_ep25.pt       # periodic checkpoint
  ckpt_ep50.pt
  ckpt_final.pt      # final model after training completes
```

#### Interactive mode

| Field | Type | Default | What it does | What happens when you change it |
|-------|------|---------|--------------|-------------------------------|
| `interactive` | bool | `False` | Force interactive prompt loop after training. | `True` = after training finishes, opens a `prompt>` loop where you can type text and see generation. |
| `no_interactive` | bool | `False` | Force non-interactive. | `True` = never open the prompt loop. When calling `train()` from Python code, always set this to `True` so the script doesn't hang waiting for input. |

If neither flag is set, interactive mode is auto-detected based on whether
stdin is a TTY (terminal). Scripts and notebooks get non-interactive by default.

### Recommended configurations

**Quick experiment (1 minute):**

```python
from BasicQITNN_Transformer import train, TrainConfig

result = train(
    dataset="my_data.txt",
    dim=64,
    ffn=128,
    epochs=5,
    no_interactive=True,
)
```

**Standard AdamW training (the most common setup):**

```python
result = train(
    dataset="my_data.txt",
    optimizer="adamw",
    adamw_lr_start=3e-4,
    adamw_lr_end=3e-5,
    adamw_trit_floor_step=5e-5,
    lr_schedule="cosine",
    warmup_steps=200,
    dim=384,
    ffn=768,
    seq_len=256,
    epochs=100,
    steps_per_epoch=1000,
    grad_clip=1.0,
    run_name="adamw_100ep",
    no_interactive=True,
)
```

**With separate train/val/test splits:**

```python
result = train(
    extended_dataset=True,
    train_dir="data/train",
    val_dir="data/val",
    test_dir="data/test",
    optimizer="adamw",
    adamw_lr_start=3e-4,
    adamw_trit_floor_step=5e-5,
    lr_schedule="cosine",
    epochs=100,
    csv_log="logs/experiment.csv",
    no_interactive=True,
)
```

**Small model for small data (< 1 MB):**

```python
result = train(
    dataset="small_corpus.txt",
    dim=64,
    ffn=128,
    layers=2,
    seq_len=128,
    batch_size=4,
    optimizer="adamw",
    adamw_lr_start=3e-4,
    adamw_lr_end=3e-5,
    adamw_trit_floor_step=5e-5,
    lr_schedule="cosine",
    epochs=50,
    steps_per_epoch=400,
    grad_clip=1.0,
    no_interactive=True,
)
```

**Larger effective batch without a larger micro-batch footprint:**

```python
result = train(
    dataset="my_data.txt",
    batch_size=2,
    grad_accum_steps=4,
    optimizer="adamw",
    adamw_lr_start=3e-4,
    adamw_lr_end=3e-5,
    warmup_steps=100,
    steps=1000,
    no_interactive=True,
)
```

This keeps the per-micro-batch footprint at `2` sequences while exposing an
effective batch of `8` sequences per optimizer step. `steps=1000` and
`warmup_steps=100` are still counted in optimizer steps, not in micro-steps.

**Large model for large data (> 10 MB):**

```python
result = train(
    dataset="large_corpus/",
    dim=512,
    ffn=1024,
    layers=4,
    seq_len=512,
    batch_size=2,
    max_bytes=10_000_000,
    optimizer="adamw",
    adamw_lr_start=1e-4,
    adamw_lr_end=1e-5,
    adamw_trit_floor_step=3e-5,
    lr_schedule="cosine",
    epochs=200,
    steps_per_epoch=800,
    grad_clip=1.0,
    save_every=50,
    no_interactive=True,
)
```

**Resume from checkpoint:**

```python
result = train(
    dataset="my_data.txt",
    resume="runs/adamw_100ep/ckpt_final.pt",
    optimizer="adamw",
    adamw_lr_start=1e-5,
    epochs=50,
    run_name="adamw_finetune",
    no_interactive=True,
)
```

---

## Building a custom model from scratch

If you want to use QTS layers in your own architecture instead of the built-in
transformer:

```python
import torch
from torch import nn
import pyqitnn

class MyModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # QTS layer: in_dim -> hidden_dim (output is 2*hidden_dim packed)
        self.qts1 = pyqitnn.QITNNLinear(in_dim, hidden_dim, device="cuda:0")

        # standard linear after QTS (input is 2*hidden_dim)
        self.linear = nn.Linear(hidden_dim * 2, out_dim, device="cuda:0")

    def forward(self, x):
        # x: [batch, in_dim], must be CUDA
        h = self.qts1(x)          # [batch, 2*hidden_dim] packed [x|y]
        return self.linear(h)      # [batch, out_dim]

model = MyModel(64, 32, 10)
x = torch.randn(8, 64, device="cuda:0")
out = model(x)  # [8, 10]
```

For training, remember to:
1. Set `weight_decay=0` for QTS parameters when using AdamW.
2. Call `prior_()` after each optimizer step if you want entropy stabilization.
3. All inputs must be CUDA tensors on `cuda:0`.

---

## Known limitations

1. **Single device only.** All tensors must be on `cuda:0`. The extension
   explicitly checks `get_device() == 0` for every input. No multi-GPU support.

2. **Mixed precision is conservative.** Canonical mixed mode is
   `precision_mode="qts_fp32_rest_bf16"`. Legacy `mixed_precision=True`
   still maps to the same path. This keeps QITNN master weights, Born
   normalization, backnorm, entropy/prior, and the attention softmax path in
   `fp32` while allowing visible activations on the CUDA `bf16` path. Do not
   call `.half()` or `.bfloat16()` on the model itself.

3. **Windows and Linux only.** No macOS support (requires NVIDIA CUDA).

4. **No CPU fallback.** All operations require CUDA. Creating a model with
   `device=None` and calling forward will raise
   `RuntimeError: QITNNLinear requires CUDA tensors`.

5. **cuBLAS GEMM tolerance.** The three matrix multiplications in `forward3` use
   cuBLAS, which may produce slightly different results than `torch.mm`. Max
   observed difference is ~1e-2 on 512x256 matrices. Standard FP32 behavior.

6. **Dead zone at Z=0.** When all three raw channels are near zero, Born
   normalization produces `u=0, v=0`. The kernel clamps `Z > 1e-12` but gradients
   vanish. Default `init_std=0.02` avoids this.

7. **In-place prior and autograd.** `prior_()` modifies tensors in-place.
   Calling it before `backward()` will corrupt the autograd graph. Always call
   after `optimizer.step()`.

8. **Attention backward accumulation.** FP32 gradient accumulation error grows
   with sequence length. For `seq_len > 256`, error may become noticeable. Does
   not affect training convergence in practice.

9. **Attention kernel scope.** The 3D batched path for `attention2` is native
   and batched, but it is still a custom causal kernel rather than a
   FlashAttention-style fused implementation. Large `seq_len` still carries the
   expected quadratic cost.

---
