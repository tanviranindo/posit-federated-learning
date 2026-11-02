#!/usr/bin/env python3
"""Self-checks for detops: exactness, order independence, agreement with PyTorch.

Usage: python experiments/xisa/detops_check.py
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

import detops
from common import exact_init, make_model, setup_torch, state_to_numpy


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


def main() -> None:
    setup_torch(1)
    g = torch.Generator().manual_seed(0)

    # GEMM: close to float64 reference; bitwise unchanged when K is permuted.
    for m, k, n in ((64, 27, 32), (300, 2048, 512), (32, 8192, 64)):
        a = torch.randn(m, k, generator=g) * torch.exp(torch.randn(m, 1, generator=g))
        b = torch.randn(k, n, generator=g)
        c = detops.gemm(a, b)
        perm = torch.randperm(k, generator=g)
        same = torch.equal(c, detops.gemm(a[:, perm].contiguous(), b[perm].contiguous()))
        print(f"gemm {m}x{k}x{n}: rel err vs fp64 {rel(c, a.double() @ b.double()):.1e}, "
              f"permuted-K identical {same}")
        assert same

    x = torch.randn(7, 5000, generator=g)
    perm = torch.randperm(5000, generator=g)
    assert torch.equal(detops.rowsum(x), detops.rowsum(x[:, perm].contiguous()))
    print(f"rowsum rel err {rel(detops.rowsum(x), x.double().sum(1)):.1e}, permuted identical")

    t = -torch.rand(100000, generator=g, dtype=torch.float64) * 50
    ulps = ((detops.exp(t) - torch.exp(t)).abs() / (torch.exp(t) * 2.0 ** -52)).max()
    print(f"exp max error {float(ulps):.2f} ulp (float64)")
    assert ulps < 4

    v = torch.rand(65536, generator=g) * 100
    assert torch.equal(detops.sqrt32(v), torch.from_numpy(
        np.array([np.float32(math.sqrt(float(u))) for u in v.numpy()])))
    print("sqrt32 correctly rounded on 65,536 inputs")

    # Full model: forward and gradients vs PyTorch autograd.
    model = make_model()
    exact_init(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.01)
    model.train()
    xb = torch.randn(32, 3, 32, 32, generator=g)
    yb = torch.randint(0, 10, (32,), generator=g)
    state = detops.to_state(state_to_numpy(model))   # before PyTorch updates BN stats
    logits = model(xb)
    F.cross_entropy(logits, yb).backward()
    dlogits, cache = detops.forward(state, xb, train=True)
    _, dz = detops.softmax_xent(dlogits, yb)
    grads = detops.backward(state, cache, dz)
    print(f"logits rel diff {rel(dlogits, logits.detach()):.1e}")
    # Conv biases feed BatchNorm, so their true gradient is 0 and both stacks
    # return rounding noise; compare those in absolute terms.
    noise = max(float(grads[n].abs().max()) for n in grads if n.startswith("conv") and n.endswith("bias"))
    print(f"conv bias gradients (exactly 0 in theory): max |g| {noise:.1e}")
    worst = max((rel(grads[n], p.grad), n) for n, p in model.named_parameters()
                if not (n.startswith("conv") and n.endswith("bias")))
    print(f"worst gradient rel diff {worst[0]:.1e} ({worst[1]})")
    assert rel(dlogits, logits.detach()) < 1e-4 and worst[0] < 1e-3 and noise < 1e-6
    for k in ("bn1.running_mean", "bn3.running_var"):
        print(f"{k} rel diff {rel(state[k], model.state_dict()[k]):.1e}")
    assert all(rel(state[k], model.state_dict()[k]) < 1e-5 for k in state if "running" in k)
    print("all checks passed")


if __name__ == "__main__":
    main()
