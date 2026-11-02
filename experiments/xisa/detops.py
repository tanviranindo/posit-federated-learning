"""Order-independent (exact-accumulation) training stack for the pilot CNN.

Every accumulation is made exact, so its result does not depend on the
summation order a BLAS library, SIMD width or thread count picks:

* GEMM (Ozaki-style error-free splitting): float32 operands are split row-wise
  (A) and column-wise (B) into slices on a power-of-two grid narrow enough that
  every slice product, and every partial sum of a dot product, is exact in
  float64. Ordinary float64 BLAS then returns the exact result in any order;
  slice results are combined in a fixed order and rounded once to float32.
* Reductions: the same splitting, summed in float64 (exact in any order).
* Everything else is element-wise and correctly rounded: sqrt in float64 then
  rounded, exp by a fixed float64 polynomial with exact power-of-two scaling,
  SGD with separate multiply and add.

The forward and backward passes of AdaptiveCNN (x86_64 config, dropout off)
are written out by hand, so autograd's own reductions are never used. The
model reads and writes the same state dict as the PyTorch module.
"""

import math

import numpy as np
import torch

F32, F64 = torch.float32, torch.float64
SLICES = 2


def _pow2(e: torch.Tensor) -> torch.Tensor:
    """Exact float64 2**e for integer e in [-1022, 1023], built from the bits."""
    return ((e.to(torch.int64) + 1023) << 52).view(F64)


def _split(x: torch.Tensor, dim: int, s: int) -> list[torch.Tensor]:
    """Slices of x (float64) on per-row grids 2**(e - s), 2**(e - 2s), ...

    e is the exponent of the row maximum along dim (|x| < 2**e), so each slice
    holds integers of at most s bits times the row's grid unit.
    """
    top = torch.frexp(x.abs().amax(dim=dim, keepdim=True)).exponent
    out, rest = [], x
    for i in range(1, SLICES + 1):
        sigma = _pow2(top - i * s + 52) * 1.5
        hi = (rest + sigma) - sigma
        out.append(hi)
        rest = rest - hi
    return out


def _bits(k: int) -> int:
    return max(1, math.ceil(math.log2(max(k, 2))))


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """float32 a @ b, identical on every platform. a: [M, K], b: [K, N]."""
    s = (53 - _bits(a.shape[1])) // 2
    sa = _split(a.to(F64), 1, s)
    sb = _split(b.to(F64), 0, s)
    out = None
    for i in range(SLICES):
        for j in range(SLICES - i):          # drop terms below 2**-(2s) of the row scale
            term = sa[i] @ sb[j]
            out = term if out is None else out + term
    return out.to(F32)


def rowsum(x: torch.Tensor) -> torch.Tensor:
    """Exact-order sum of each row of a float32 [R, M] tensor, as float64."""
    s = 52 - _bits(x.shape[1])
    out = None
    for hi in _split(x.to(F64), 1, s):
        part = hi.sum(dim=1)
        out = part if out is None else out + part
    return out


def colsum(x: torch.Tensor) -> torch.Tensor:
    return rowsum(x.t())


def exp(x: torch.Tensor) -> torch.Tensor:
    """exp of float64 x <= 0 from fixed float64 operations (Cody-Waite + Taylor)."""
    x = x.clamp_min(-700.0)
    k = torch.round(x * 1.4426950408889634)
    r = (x - k * 6.93147180369123816490e-01) - k * 1.90821492927058770002e-10
    p = torch.full_like(r, 1.0 / math.factorial(13))
    for n in range(12, -1, -1):
        p = p * r
        p = p + 1.0 / math.factorial(n)
    return p * _pow2(k)


def sqrt32(x: torch.Tensor) -> torch.Tensor:
    """Correctly rounded float32 sqrt (float64 sqrt, then one rounding)."""
    return torch.from_numpy(np.sqrt(x.to(F64).numpy())).to(F32)


def _c(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=F32)


EPS, MOM = _c(1e-5), _c(0.1)
ONE_MINUS_MOM = _c(0.9)


# ---------------------------------------------------------------- layers

def conv3x3(x, w, b):
    n, c, h, wd = x.shape
    xp = torch.nn.functional.pad(x, (1, 1, 1, 1))
    cols = torch.stack([xp[:, :, ky:ky + h, kx:kx + wd] for ky in range(3) for kx in range(3)], 2)
    cols = cols.permute(0, 3, 4, 1, 2).reshape(n * h * wd, c * 9)
    wm = w.reshape(w.shape[0], -1)
    y = gemm(cols, wm.t()) + b
    return y.reshape(n, h, wd, -1).permute(0, 3, 1, 2).contiguous(), (cols, wm, x.shape)


def conv3x3_back(dy, cache, w_shape):
    cols, wm, (n, c, h, wd) = cache
    d = dy.permute(0, 2, 3, 1).reshape(n * h * wd, -1)
    dw = gemm(d.t().contiguous(), cols).reshape(w_shape)
    db = colsum(d).to(F32)
    dcols = gemm(d, wm).reshape(n, h, wd, c, 9).permute(0, 3, 4, 1, 2)
    dxp = torch.zeros(n, c, h + 2, wd + 2, dtype=F32)
    for k in range(9):
        ky, kx = divmod(k, 3)
        dxp[:, :, ky:ky + h, kx:kx + wd] += dcols[:, :, k]
    return dxp[:, :, 1:h + 1, 1:wd + 1].contiguous(), dw, db


def _per_channel(x):
    return x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)


def _bc(v):
    return v.reshape(1, -1, 1, 1)


def bn_train(x, w, b, state, prefix):
    m = x.numel() // x.shape[1]
    mean = (rowsum(_per_channel(x)) / m).to(F32)
    d = x - _bc(mean)
    var64 = rowsum(_per_channel(d * d)) / m
    var = var64.to(F32)
    std = sqrt32(var + EPS)
    xhat = d / _bc(std)
    y = xhat * _bc(w) + _bc(b)
    rm, rv = prefix + "running_mean", prefix + "running_var"
    state[rm] = state[rm] * ONE_MINUS_MOM + mean * MOM
    state[rv] = state[rv] * ONE_MINUS_MOM + (var64 * m / (m - 1)).to(F32) * MOM
    state[prefix + "num_batches_tracked"] = state[prefix + "num_batches_tracked"] + 1
    return y, (xhat, std, w, m)


def bn_back(dy, cache):
    xhat, std, w, m = cache
    db = rowsum(_per_channel(dy)).to(F32)
    dw = rowsum(_per_channel(dy * xhat)).to(F32)
    dxhat = dy * _bc(w)
    m1 = (rowsum(_per_channel(dxhat)) / m).to(F32)
    m2 = (rowsum(_per_channel(dxhat * xhat)) / m).to(F32)
    t = dxhat - _bc(m1)
    t = t - xhat * _bc(m2)
    return t / _bc(std), dw, db


def bn_eval(x, w, b, rm, rv):
    return (x - _bc(rm)) / _bc(sqrt32(rv + EPS)) * _bc(w) + _bc(b)


def relu(x):
    return torch.where(x > 0, x, torch.zeros_like(x))


def relu_back(dy, x):
    return torch.where(x > 0, dy, torch.zeros_like(dy))


def _quads(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)


def _unquads(q):
    n, c, h, w, _ = q.shape
    return q.reshape(n, c, h, w, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(n, c, 2 * h, 2 * w)


def maxpool2(x):
    q = _quads(x)
    best, idx = q[..., 0], torch.zeros(q.shape[:-1], dtype=torch.int64)
    for k in range(1, 4):
        up = q[..., k] > best
        best = torch.where(up, q[..., k], best)
        idx = torch.where(up, torch.full_like(idx, k), idx)
    return best, idx


def maxpool2_back(dy, idx):
    hit = torch.arange(4).reshape(1, 1, 1, 1, 4) == idx.unsqueeze(-1)
    return _unquads(torch.where(hit, dy.unsqueeze(-1), torch.zeros(1, dtype=F32)))


QUARTER = _c(0.25)


def avgpool2(x):
    q = _quads(x)
    return (((q[..., 0] + q[..., 1]) + q[..., 2]) + q[..., 3]) * QUARTER


def avgpool2_back(dy):
    return _unquads((dy * QUARTER).unsqueeze(-1).expand(*dy.shape, 4))


def linear(x, w, b):
    return gemm(x, w.t().contiguous()) + b


def linear_back(dy, x, w):
    return gemm(dy, w), gemm(dy.t().contiguous(), x), colsum(dy).to(F32)


def softmax_xent(z, y):
    """Mean cross-entropy; returns (loss as float, dloss/dz)."""
    t = z - z.max(dim=1, keepdim=True).values
    e = exp(t.to(F64))
    s = e[:, 0]
    for j in range(1, e.shape[1]):
        s = s + e[:, j]
    p = e / s.unsqueeze(1)
    loss = float(-torch.log(p[torch.arange(len(y)), y]).mean())
    onehot = torch.nn.functional.one_hot(y, e.shape[1]).to(F32)
    return loss, (p.to(F32) - onehot) / _c(float(len(y)))


# ---------------------------------------------------------------- model

BLOCKS = ("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")


def forward(state, x, train: bool):
    """Logits and (if train) the cache for backward. Updates BN running stats in state."""
    cache = {}
    h = x
    for i, (cv, bn) in enumerate(BLOCKS):
        c, cache[cv] = conv3x3(h, state[cv + ".weight"], state[cv + ".bias"])
        if train:
            z, cache[bn] = bn_train(c, state[bn + ".weight"], state[bn + ".bias"], state, bn + ".")
        else:
            z = bn_eval(c, state[bn + ".weight"], state[bn + ".bias"],
                        state[bn + ".running_mean"], state[bn + ".running_var"])
        a = relu(z)
        cache["relu" + cv] = z
        if i < 2:
            h, cache["pool" + cv] = maxpool2(a)
        else:
            h = avgpool2(a)
    h = h.reshape(h.shape[0], -1)
    for fc in ("fc1", "fc2"):
        cache[fc] = h
        z = linear(h, state[fc + ".weight"], state[fc + ".bias"])
        cache["relu" + fc] = z
        h = relu(z)
    cache["fc3"] = h
    return linear(h, state["fc3.weight"], state["fc3.bias"]), cache


def backward(state, cache, dz):
    grads = {}
    for fc in ("fc3", "fc2", "fc1"):
        dz, grads[fc + ".weight"], grads[fc + ".bias"] = linear_back(dz, cache[fc], state[fc + ".weight"])
        if fc != "fc1":
            dz = relu_back(dz, cache["relu" + ("fc2" if fc == "fc3" else "fc1")])
    dh = dz.reshape(-1, 128, 4, 4)
    for i, (cv, bn) in reversed(list(enumerate(BLOCKS))):
        da = avgpool2_back(dh) if i == 2 else maxpool2_back(dh, cache["pool" + cv])
        dzz = relu_back(da, cache["relu" + cv])
        dc, grads[bn + ".weight"], grads[bn + ".bias"] = bn_back(dzz, cache[bn])
        dh, grads[cv + ".weight"], grads[cv + ".bias"] = conv3x3_back(dc, cache[cv], state[cv + ".weight"].shape)
    return grads


def sgd_step(state, grads, bufs, lr, momentum=0.9, weight_decay=1e-4):
    """torch.optim.SGD semantics (momentum, weight decay) with separate mul/add."""
    lr, mom, wd = _c(lr), _c(momentum), _c(weight_decay)
    for k, g in grads.items():
        p = state[k]
        d = g + p * wd
        bufs[k] = d.clone() if k not in bufs else bufs[k] * mom + d
        state[k] = p - bufs[k] * lr


def to_state(np_state: dict) -> dict:
    return {k: torch.from_numpy(v.copy()) for k, v in np_state.items()}


def to_numpy(state: dict) -> dict:
    return {k: v.numpy().copy() for k, v in state.items()}
