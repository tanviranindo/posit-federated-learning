#!/usr/bin/env python3
"""Mitigation M1/M2/M3: FedAvg with the exact-accumulation stack (detops).

Same model, initial weights, data split, minibatch order and FedAvg as
fl_trajectory.py, but every client trains with detops' hand-written forward,
backward and SGD. Records each client's and the global state hash every round
(M1: identical on every platform?), the wall time per local epoch for both
stacks (M2), and test accuracy (M3).

Usage: python experiments/xisa/det_trajectory.py --epochs 5 --rounds 3 [--bench]
"""

import argparse
import time

import numpy as np
import torch

import detops
from common import (env_info, exact_init, load_cifar, make_model, portable_args, REPO_ROOT,
                    results_dir, setup_torch, sha256_bytes, state_hash, state_to_numpy,
                    write_json)
from fl_trajectory import client_indices, fedavg, local_train, split_hash


def local_train_det(global_state: dict, data: dict, idx: np.ndarray, *, order_seed: int,
                    round_idx: int, client: int, epochs: int, batch: int, lr: float) -> dict:
    """detops counterpart of fl_trajectory.local_train (same batches, same SGD)."""
    state = detops.to_state(global_state)
    bufs = {}
    rng = np.random.Generator(np.random.PCG64([order_seed, round_idx, client]))
    x_all = torch.from_numpy(data["x_train"])
    y_all = torch.from_numpy(data["y_train"])
    for _ in range(epochs):
        order = idx[rng.permutation(len(idx))]
        for start in range(0, len(order), batch):
            b = torch.from_numpy(order[start:start + batch])
            logits, cache = detops.forward(state, x_all[b], train=True)
            _, dz = detops.softmax_xent(logits, y_all[b])
            detops.sgd_step(state, detops.backward(state, cache, dz), bufs, lr)
    return detops.to_numpy(state)


def evaluate_det(state: dict, data: dict) -> tuple[np.ndarray, np.ndarray]:
    s = detops.to_state(state)
    x = torch.from_numpy(data["x_test"])
    logits = torch.cat([detops.forward(s, x[i:i + 500], train=False)[0]
                        for i in range(0, len(x), 500)])
    return logits.numpy(), logits.argmax(1).numpy().astype(np.int8)


def bench(glob: dict, data: dict, idx: np.ndarray, args) -> dict:
    """Seconds per local epoch for the standard and the deterministic stack."""
    kw = dict(order_seed=args.order_seed, round_idx=1, client=0, epochs=1,
              batch=args.batch, lr=args.lr)
    out = {}
    for name, fn, reps in (("standard", local_train, 3), ("deterministic", local_train_det, 1)):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(glob, data, idx, **kw)
            times.append(time.perf_counter() - t0)
        out[name] = min(times)
    out["ratio"] = out["deterministic"] / out["standard"]
    print(f"bench s/epoch: standard {out['standard']:.2f}, deterministic "
          f"{out['deterministic']:.2f} ({out['ratio']:.1f}x)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--split", choices=["iid", "dirichlet"], default="iid")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--order-seed", type=int, default=7)
    ap.add_argument("--bench", action="store_true", help="time one epoch of each stack first")
    ap.add_argument("--label", default="")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "xisa_cifar10.npz"))
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "results" / "xisa"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    setup_torch(args.threads)
    data = load_cifar(args.data)
    parts = client_indices(data["y_train"], args.clients, args.split, args.alpha)
    weights = [len(p) / sum(len(q) for q in parts) for p in parts]
    run_dir = results_dir(args.out, args.tag) / (
        f"det_t{args.threads}_o{args.order_seed}_e{args.epochs}{args.label}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = make_model()
    exact_init(model)
    glob = state_to_numpy(model)
    timing = bench(glob, data, parts[0], args) if args.bench else None

    preds, history = {}, []
    t_start = time.time()
    for r in range(args.rounds + 1):
        client_hashes = []
        if r > 0:
            t0 = time.perf_counter()
            states = [local_train_det(glob, data, parts[k], order_seed=args.order_seed,
                                      round_idx=r, client=k, epochs=args.epochs,
                                      batch=args.batch, lr=args.lr)
                      for k in range(args.clients)]
            epoch_s = (time.perf_counter() - t0) / (args.clients * args.epochs)
            client_hashes = [state_hash(s) for s in states]
            glob = fedavg(states, weights)
        logits, pred = evaluate_det(glob, data)
        preds[f"r{r}"] = pred
        history.append({
            "round": r,
            "state_hash": state_hash(glob),
            "client_hashes": client_hashes,
            "logits_hash": sha256_bytes(logits),
            "accuracy": float((pred == data["y_test"]).mean()),
            "s_per_local_epoch": round(epoch_s, 3) if r > 0 else None,
            "elapsed_s": round(time.time() - t_start, 2),
        })
        print(f"round {r:3d} acc {history[-1]['accuracy']:.4f} "
              f"hash {history[-1]['state_hash'][:12]} ({history[-1]['elapsed_s']}s)", flush=True)

    np.savez(run_dir / "preds.npz", **preds)
    write_json(run_dir / "summary.json", {
        "env": env_info(), "args": portable_args(vars(args)),
        "input_hash": data["input_hash"], "split_hash": split_hash(parts),
        "bench": timing, "history": history,
    })


if __name__ == "__main__":
    main()
