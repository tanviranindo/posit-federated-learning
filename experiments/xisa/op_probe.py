#!/usr/bin/env python3
"""H1: which operations give different bits on different platforms?

Runs a fixed list of PyTorch operations on exact, platform-independent inputs
and saves every output tensor. compare.py ops then reports per-operation ULP
distances between two platforms.

Usage: python experiments/xisa/op_probe.py --threads 1 [--out experiments/results/xisa]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (env_info, exact_init, exact_uniform, load_cifar, make_model,
                    REPO_ROOT, results_dir, setup_torch, sha256_bytes, write_json)


def t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.copy())


def probe(data: dict) -> dict:
    rng = np.random.Generator(np.random.PCG64(2026))
    out = {}

    # Element-wise controls: IEEE 754 requires these to be correctly rounded.
    a, b = t(exact_uniform(rng, (1 << 16,))), t(exact_uniform(rng, (1 << 16,), 3))
    out["ew_add"] = a + b
    out["ew_mul"] = a * b
    out["ew_div"] = a / (b.abs() + 1)
    out["ew_sqrt"] = (a.abs() + 1).sqrt()
    out["ew_fma_pattern"] = a * b + a  # may be fused into FMA by a compiler

    # Reductions: order depends on SIMD width and threading.
    big = t(exact_uniform(rng, (1 << 20,), 4))
    out["red_sum"] = big.sum().reshape(1)
    out["red_mean"] = big.mean().reshape(1)
    out["red_var"] = big.var().reshape(1)
    out["red_norm"] = big.norm().reshape(1)
    out["red_sum_dim"] = big.reshape(1024, 1024).sum(dim=1)

    # Transcendentals: libm / SLEEF implementations differ.
    x = t(exact_uniform(rng, (1 << 16,), 3))
    out["tr_exp"] = x.exp()
    out["tr_log"] = (x.abs() + 1e-3).log()
    out["tr_tanh"] = x.tanh()
    out["tr_sigmoid"] = torch.sigmoid(x)
    logits = t(exact_uniform(rng, (512, 10), 3))
    out["tr_softmax"] = F.softmax(logits, dim=1)
    out["tr_log_softmax"] = F.log_softmax(logits, dim=1)

    # Dense linear algebra.
    m1 = t(exact_uniform(rng, (256, 2048)))
    m2 = t(exact_uniform(rng, (2048, 512), -5))
    out["gemm_256x2048x512"] = m1 @ m2
    m3 = t(exact_uniform(rng, (32, 512)))
    m4 = t(exact_uniform(rng, (512, 256), -4))
    out["gemm_32x512x256"] = m3 @ m4

    # Convolutions and BatchNorm at the model's own shapes.
    img = t(data["x_train"][:32])
    w1 = t(exact_uniform(rng, (32, 3, 3, 3), -2))
    out["conv_3to32"] = F.conv2d(img, w1, padding=1)
    feat = t(exact_uniform(rng, (32, 64, 8, 8)))
    w3 = t(exact_uniform(rng, (128, 64, 3, 3), -5))
    out["conv_64to128"] = F.conv2d(feat, w3, padding=1)
    out["bn_train"] = F.batch_norm(feat, None, None, training=True)
    out["avgpool_4x4"] = F.adaptive_avg_pool2d(feat, (4, 4))

    # One full forward/backward pass of the paper's model on one minibatch.
    model = make_model()
    exact_init(model)
    model.train()
    xb, yb = t(data["x_train"][:32]), torch.from_numpy(data["y_train"][:32])
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    out["model_logits"] = logits.detach()
    out["model_loss"] = loss.detach().reshape(1)
    for name, p in model.named_parameters():
        out[f"model_grad.{name}"] = p.grad.detach()

    return {k: v.contiguous().numpy() for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "xisa_cifar10.npz"))
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "results" / "xisa"))
    ap.add_argument("--tag", default=None, help="results subfolder (default: platform tag)")
    args = ap.parse_args()

    setup_torch(args.threads)
    data = load_cifar(args.data)
    out_dir = results_dir(args.out, args.tag)

    t0 = time.time()
    outputs = probe(data)
    elapsed = time.time() - t0

    stem = f"ops_t{args.threads}"
    np.savez(out_dir / f"{stem}.npz", **outputs)
    write_json(out_dir / f"{stem}.json", {
        "env": env_info(),
        "input_hash": data["input_hash"],
        "elapsed_s": round(elapsed, 3),
        "hashes": {k: sha256_bytes(v) for k, v in outputs.items()},
    })
    print(f"{out_dir / stem}: {len(outputs)} ops in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
